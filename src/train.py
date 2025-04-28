import os
import yaml
import torch
import numpy as np
from typing import Dict, Any
from tqdm import tqdm
import wandb
from datetime import datetime

from environment.energy_env import EnergyResilienceEnv
from agents.marl_agent import MARLAgent, SolarAgent, GridAgent, BatteryAgent

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config: Dict[str, Any]) -> None:
    if config['logging']['wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            config=config
        )

def save_checkpoint(agents, env, episode, save_dir='checkpoints'):
    import os
    from datetime import datetime
    
    # Create save directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a timestamped directory for this checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(save_dir, f'checkpoint_ep{episode}_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save each agent's state
    for i, agent in enumerate(agents):
        agent_path = os.path.join(checkpoint_dir, f'agent_{i}.pt')
        agent.save(agent_path)
    
    # Save environment state
    env_state = {
        'battery_charge': env.battery_charge,
        'daily_solar_generated': env.daily_solar_generated,
        'solar_energy_storage': env.solar_energy_storage,
        'available_solar': env.available_solar,
        'time_step': env.time_step,
        'metrics': env.metrics
    }
    torch.save(env_state, os.path.join(checkpoint_dir, 'env_state.pt'))
    
    print(f"\nCheckpoint saved at episode {episode} to {checkpoint_dir}")
    return checkpoint_dir

def load_checkpoint(checkpoint_dir, config):
    import os
    
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    
    # Load agents
    agents = []
    agent_types = [SolarAgent, GridAgent, BatteryAgent]
    
    for i, agent_type in enumerate(agent_types):
        agent_path = os.path.join(checkpoint_dir, f'agent_{i}.pt')
        if not os.path.exists(agent_path):
            raise ValueError(f"Agent checkpoint {agent_path} not found")
        
        agent = agent_type(config['network'])
        agent.load(agent_path)
        agents.append(agent)
    
    # Load environment state
    env = EnergyResilienceEnv(config['environment'])
    env_state_path = os.path.join(checkpoint_dir, 'env_state.pt')
    if os.path.exists(env_state_path):
        env_state = torch.load(env_state_path)
        env.battery_charge = env_state['battery_charge']
        env.daily_solar_generated = env_state['daily_solar_generated']
        env.solar_energy_storage = env_state['solar_energy_storage']
        env.available_solar = env_state['available_solar']
        env.time_step = env_state['time_step']
        env.metrics = env_state['metrics']
    
    print(f"\nCheckpoint loaded from {checkpoint_dir}")
    return agents, env

def train(config: Dict[str, Any], load_checkpoint_dir: str = None) -> None:
    # Setup environment and agents
    if load_checkpoint_dir:
        print(f"Loading checkpoint from {load_checkpoint_dir}")
        agents, env = load_checkpoint(load_checkpoint_dir, config)
    else: 
        env = EnergyResilienceEnv(config['environment'])
        agents = [
            SolarAgent(config['network']),
            GridAgent(config['network']),
            BatteryAgent(config['network'])
        ]

    # Initialize communication channel with default values
    communication_channel = {
        'solar_agent': {
            'solar_available': 0,
            'current_generation': 0
        },
        'grid_agent': {
            'grid_pricing': 0.4,  # Default to morning price
            'current_demand': 0.5  # Increased from 1.5 to 3.0 kW
        },
        'battery_agent': {
            'current_charge': 0,
            'available_capacity': 12.5,  # Full battery capacity
            'min_reserve': 2.5,
            'investment_cost': 0.001
        },
        'energy_usage': {
            'solar': 0.0,
            'grid': 0.0,
            'battery': 0.0
        }
    }

    for episode in range(config['training']['num_episodes']):
        # Reset communication channel at start of episode
        communication_channel.update({
            'solar_agent': {
                'solar_available': 5.0,  # Increased from 0 to 5 kW
                'current_generation': 0
            },
            'grid_agent': {
                'grid_pricing': 0.4,
                'current_demand': 3.0  # Increased from 1.5 to 3.0 kW
            },
            'battery_agent': {
                'current_charge': 6.25,  # Start at 50% capacity instead of 0
                'available_capacity': 12.5,
                'min_reserve': 2.5,
                'investment_cost': 0.001
            },
            'energy_usage': {
                'solar': 0.0,
                'grid': 0.0,
                'battery': 0.0
            }
        })

        initial_observations, _ = env.reset()
        
        # Reset daily energy tracking for all agents
        for agent in agents:
            agent.reset_daily_energy()

        # Format observations for each agent with float32 type
        observations = {
            f'agent_{i}': torch.FloatTensor([
                initial_observations['solar_energy_storage'][0],
                initial_observations['grid_pricing'][0],
                initial_observations['household_demand'][0],
                env.battery_charge  # Add battery state as fourth observation
            ])
            for i in range(config['environment']['n_agents'])
        }

        episode_reward = 0
        episode_length = 0
        
        # Initialize session tracking
        morning_energy = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        afternoon_energy = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        evening_energy = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        
        morning_costs = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        afternoon_costs = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        evening_costs = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}

        for step in range(env.max_steps):
            actions = []
            # Get actions from each agent with communication
            for i, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(observations[f'agent_{i}'])  # Ensure float32 type
                action = agent.get_action(obs_tensor, communication_channel)
                
                # Convert action to scalar value regardless of source type
                if isinstance(action, torch.Tensor):
                    action = action.item()  # Convert single tensor value to scalar
                elif isinstance(action, np.ndarray):
                    action = float(action.flatten()[0])  # Convert first element to scalar
                elif isinstance(action, (np.float32, np.float64)):
                    action = float(action)
                    
                actions.append(action)

            # Step the environment with the list of actions
            next_observations, reward, done, truncated, info = env.step(actions)
            done = done or truncated
            
            # Track energy usage and costs by session
            current_hour = (step // 4) % 24
            power_usage = info['power_by_session']
            costs = info['costs_by_session']
            
            if 6 <= current_hour < 9:  # Morning session
                if power_usage['morning']:
                    morning_energy['solar'] += power_usage['morning']['solar']
                    morning_energy['grid'] += power_usage['morning']['grid']
                    morning_energy['battery'] += abs(power_usage['morning']['battery'])
                if costs['morning']:
                    morning_costs['solar'] += costs['morning']['solar']
                    morning_costs['grid'] += costs['morning']['grid']
                    morning_costs['battery'] += costs['morning']['battery']
            elif 9 <= current_hour < 16:  # Afternoon session
                if power_usage['afternoon']:
                    afternoon_energy['solar'] += power_usage['afternoon']['solar']
                    afternoon_energy['grid'] += power_usage['afternoon']['grid']
                    afternoon_energy['battery'] += abs(power_usage['afternoon']['battery'])
                if costs['afternoon']:
                    afternoon_costs['solar'] += costs['afternoon']['solar']
                    afternoon_costs['grid'] += costs['afternoon']['grid']
                    afternoon_costs['battery'] += costs['afternoon']['battery']
            elif 16 <= current_hour < 21:  # Evening session
                if power_usage['evening']:
                    evening_energy['solar'] += power_usage['evening']['solar']
                    evening_energy['grid'] += power_usage['evening']['grid']
                    evening_energy['battery'] += abs(power_usage['evening']['battery'])
                if costs['evening']:
                    evening_costs['solar'] += costs['evening']['solar']
                    evening_costs['grid'] += costs['evening']['grid']
                    evening_costs['battery'] += costs['evening']['battery']
            
            # Update energy usage for each agent
            total_solar = morning_energy['solar'] + afternoon_energy['solar'] + evening_energy['solar']
            total_grid = morning_energy['grid'] + afternoon_energy['grid'] + evening_energy['grid']
            total_battery = morning_energy['battery'] + afternoon_energy['battery'] + evening_energy['battery']
            
            # Update energy usage in communication channel
            communication_channel['energy_usage'] = {
                'solar': total_solar,
                'grid': total_grid,
                'battery': total_battery
            }
            
            # Update each agent's energy tracking
            agents[0].update_daily_energy(total_solar, 'solar')
            agents[1].update_daily_energy(total_grid, 'grid')
            agents[2].update_daily_energy(total_battery, 'battery')
            
            episode_reward += reward
            episode_length += 1

            # Update agents with stronger emphasis on negative rewards
            for i, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(observations[f'agent_{i}'])  # Ensure float32 type
                # Convert single action value to tensor properly
                action_value = actions[i]
                if isinstance(action_value, (np.ndarray, np.float32, np.float64)):
                    action_value = float(action_value)
                action = torch.FloatTensor([action_value])  # Ensure float32 type
                agent_reward = torch.FloatTensor([reward])  # Ensure float32 type
                
                # Scale up negative rewards
                if reward < 0:
                    agent_reward = agent_reward * 2.0
                
                agent.update_policy(obs_tensor, action, agent_reward)

            # Update observations with float32 type
            observations = {
                f'agent_{i}': torch.FloatTensor([
                    next_observations['solar_energy_storage'][0],
                    next_observations['grid_pricing'][0],
                    next_observations['household_demand'][0],
                    env.battery_charge  # Add battery state as fourth observation
                ])
                for i in range(config['environment']['n_agents'])
            }

            if done:
                # Calculate total daily cost
                total_daily_cost = (
                    sum(morning_costs.values()) + 
                    sum(afternoon_costs.values()) + 
                    sum(evening_costs.values())
                )
                            
               
                
                # Print total energy usage for the day
                total_solar = morning_energy['solar'] + afternoon_energy['solar'] + evening_energy['solar']
                total_grid = morning_energy['grid'] + afternoon_energy['grid'] + evening_energy['grid']
                total_battery = morning_energy['battery'] + afternoon_energy['battery'] + evening_energy['battery']
                total_energy = total_solar + total_grid + total_battery
                
                print(f"\nDay {episode + 1}")
                print(f"Energy: {total_energy:.1f} kWh (Solar: {total_solar/total_energy*100:.0f}%, Grid: {total_grid/total_energy*100:.0f}%, Battery: {total_battery/total_energy*100:.0f}%)")
                print(f"Cost: ${total_daily_cost:.2f} | Battery: {env.battery_charge:.1f} kWh")
                print("-" * 50)

                break
       

        # Save model checkpoint periodically
        if (episode + 1) % config['training'].get('save_interval', 100) == 0:
            save_checkpoint(agents, env, episode + 1)


def evaluate(env: EnergyResilienceEnv, agents: list, config: Dict[str, Any]) -> None:
    eval_rewards = []
    eval_lengths = []

    for episode in range(1):  # only run 1 episode for debugging
        try:
            initial_observations, _ = env.reset()
        except Exception as e:
            print("💥 Error during env.reset():", e)
            return

        # Format observations for each agent
        observations = {
            f'agent_{i}': np.array([
                initial_observations['solar_energy_storage'][0],
                initial_observations['grid_pricing'][0],
                initial_observations['household_demand'][0],
                initial_observations['time_step'][0]
            ])
            for i in range(config['environment']['n_agents'])
        }

        print("🔍 Initial observations:", observations.keys())

        episode_reward = 0
        episode_length = 0

        for step in range(5):  # only a few steps
            actions = []
            for i, agent in enumerate(agents):
                key = f'agent_{i}'
                if key not in observations:
                    raise KeyError(f"❌ Missing observation for {key}. Full observations: {observations}")
                obs_tensor = torch.FloatTensor(observations[key])
                action = agent.get_action(obs_tensor, deterministic=True)
                actions.append(action.numpy())

            next_observations, rewards, done, truncated, info = env.step(actions)
            episode_reward += np.mean(rewards)
            episode_length += 1

            # Reformat next_observations for each agent
            observations = {
                f'agent_{i}': np.array([
                    next_observations['solar_energy_storage'][0],
                    next_observations['grid_pricing'][0],
                    next_observations['household_demand'][0],
                    next_observations['time_step'][0]
                ])
                for i in range(config['environment']['n_agents'])
            }

            if done or truncated:
                break

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

    print(f"✅ Eval result: mean reward = {np.mean(eval_rewards)}, mean length = {np.mean(eval_lengths)}")


if __name__ == "__main__":
    # Load configuration
    config = load_config('configs/training_config.yaml')
    
    # Setup logging
    setup_logging(config)
    
    # Start training
    train(config)
    
    # Close wandb if used
    if config['logging']['wandb']:
        wandb.finish()