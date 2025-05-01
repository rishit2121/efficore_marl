import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np
import random

class MARLAgent(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.observation_dim = config.get('observation_dim', 4)
        self.action_dim = config.get('action_dim', 1)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.time_step = 0
        
        # Policy network with continuous output
        self.policy_net = nn.Sequential(
            nn.Linear(self.observation_dim + 3, self.hidden_dim),  # Added 3 for energy usage info
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.observation_dim + 3, self.hidden_dim),  # Added 3 for energy usage info
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Learning parameters
        self.base_lr = 0.1
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr, weight_decay=0.01)
        
        # Experience buffer
        self.buffer = []
        self.buffer_size = 10000
        self.batch_size = 64
        self.gamma = 0.99
        
        # Energy constraints
        self.max_daily_energy = 30.0  # kWh
        self.current_daily_energy = 0.0
        self.energy_usage = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        
        # Learning parameters
        self.target_energy = 30.0
        self.energy_tolerance = 0.5
        self.energy_penalty_scale = 1000.0
        
        # Track learning progress
        self.energy_history = []
        self.reward_history = []
        self.best_reward = float('-inf')
        self.patience = 2
        self.patience_counter = 0
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def get_action(self, observations: torch.Tensor, communication_channel=None) -> torch.Tensor:
        if communication_channel is None:
            communication_channel = {}
            
        # Share energy usage information
        communication_channel['energy_usage'] = self.energy_usage
        
        # Get total energy usage from other agents
        total_energy = sum(self.energy_usage.values())
        energy_deviation = total_energy - self.target_energy
        
        # Add energy usage information to observations
        energy_info = torch.tensor([
            self.energy_usage['solar'],
            self.energy_usage['grid'],
            self.energy_usage['battery']
        ], dtype=torch.float32)
        
        # Ensure observations are float32
        observations = observations.to(torch.float32)
        extended_obs = torch.cat([observations, energy_info])
        
        with torch.no_grad():
            action = self.policy_net(extended_obs)
            
            # Add minimal noise for exploration
            noise = torch.randn_like(action, dtype=torch.float32) * 0.01
            action = torch.clamp(action + noise, -1, 1)
            
            # Calculate target step energy (30 kWh / 96 steps)
            target_step_energy = 30.0 / 96.0
            
            # Predict the energy this action will produce
            predicted_energy = action.abs() * self.max_daily_energy / 96.0  # Approximate scaling
            
            # Compute deviation just for this action
            deviation = predicted_energy.sum().item() - target_step_energy
            
            if deviation > 0:
                # If predicted energy too high, scale down slightly
                scale_factor = max(0.5, 1 - 0.5 * (deviation / target_step_energy))
                action = action * scale_factor
            elif deviation < -0.1 * target_step_energy:
                # If predicted energy too low, encourage scaling up
                scale_factor = min(1.5, 1 + 0.5 * (-deviation / target_step_energy))
                action = action * scale_factor
            
            action = torch.clamp(action, -1, 1)
            
        return action
    
    def update_policy(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        # Increment time step
        self.time_step += 1
        
        # Store experience with energy usage information
        energy_info = torch.tensor([
            self.energy_usage['solar'],
            self.energy_usage['grid'],
            self.energy_usage['battery']
        ], dtype=torch.float32)
        
        # Ensure all tensors are float32
        observations = observations.to(torch.float32)
        actions = actions.to(torch.float32)
        rewards = rewards.to(torch.float32)
        
        extended_obs = torch.cat([observations, energy_info])
        
        self.buffer.append({
            'observations': extended_obs,
            'actions': actions,
            'rewards': rewards
        })
        
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        obs_batch = torch.stack([x['observations'] for x in batch])
        act_batch = torch.stack([x['actions'] for x in batch])
        rew_batch = torch.stack([x['rewards'] for x in batch])
        
        # Calculate returns with extremely strong emphasis on negative rewards
        returns = []
        R = 0
        for r in reversed(rew_batch):
            if r < 0:
                R = r + self.gamma * 0.01 * R  # Extremely strong discount for negative rewards
            else:
                R = r + self.gamma * 0.1 * R  # Very reduced retention of positive rewards
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages with extremely strong emphasis on negative rewards
        with torch.no_grad():
            values = self.value_net(obs_batch)
            advantages = returns - values.squeeze()
            
            # Scale advantages based on reward magnitude and energy usage
            reward_magnitude = abs(rew_batch.mean().item())
            energy_deviation = abs(sum(self.energy_usage.values()) - self.target_energy)
            
            # Extremely aggressive advantage scaling
            if reward_magnitude > 100 or energy_deviation > 2:  # Bad state
                advantage_scale = 1000.0  # Increased from 100.0
            elif reward_magnitude > 50 or energy_deviation > 1:  # Moderately bad state
                advantage_scale = 500.0  # Increased from 50.0
            else:  # Near optimal state
                advantage_scale = 1.0
            
            # Scale up advantages for negative rewards
            negative_mask = rew_batch.squeeze() < 0
            advantages[negative_mask] *= advantage_scale
            
            # Add immediate energy constraint penalty
            if energy_deviation > self.energy_tolerance:
                energy_penalty = -self.energy_penalty_scale * (energy_deviation - self.energy_tolerance)
                advantages += energy_penalty
        
        # Update policy with more conservative learning
        for _ in range(5):
            # Calculate new action probabilities
            new_actions = self.policy_net(obs_batch)
            
            # Calculate policy loss with stronger penalty for negative rewards
            policy_loss = -torch.min(
                (new_actions - act_batch).pow(2) * advantages.unsqueeze(-1),
                torch.clamp(new_actions - act_batch, -0.1, 0.1).pow(2) * advantages.unsqueeze(-1)
            ).mean()
            
            # Calculate value loss
            value_loss = (self.value_net(obs_batch).squeeze() - returns).pow(2).mean()
            
            # Add immediate energy constraint loss
            energy_loss = torch.tensor(0.0, dtype=torch.float32)
            if sum(self.energy_usage.values()) > self.target_energy + self.energy_tolerance:
                energy_loss = torch.tensor(self.energy_penalty_scale * (sum(self.energy_usage.values()) - self.target_energy), dtype=torch.float32)
            
            # Total loss with balanced weights
            loss = policy_loss + 0.5 * value_loss + energy_loss
            
            # Update learning rate based on performance
            current_reward = rewards.mean().item()
            if current_reward < self.best_reward or energy_deviation > self.energy_tolerance:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    # Reduce learning rate more aggressively
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    self.patience_counter = 0
            else:
                self.best_reward = current_reward
                self.patience_counter = 0
            
            # Update with stronger gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)  # More conservative gradient clipping
            self.optimizer.step()
        
        # Track progress
        self.energy_history.append(sum(self.energy_usage.values()))
        self.reward_history.append(rewards.mean().item())
        
        # Clear buffer
        self.buffer = []
    
    def update_daily_energy(self, energy_used: float, source: str):
        self.energy_usage[source] = energy_used
        self.current_daily_energy = sum(self.energy_usage.values())
    
    def reset_daily_energy(self):
        self.current_daily_energy = 0.0
        self.energy_usage = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        self.time_step = 0  # Reset time step
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'energy_usage': self.energy_usage,
            'current_daily_energy': self.current_daily_energy,
            'time_step': self.time_step,
            'energy_history': self.energy_history,
            'reward_history': self.reward_history,
            'best_reward': self.best_reward,
            'patience_counter': self.patience_counter
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.energy_usage = checkpoint['energy_usage']
        self.current_daily_energy = checkpoint['current_daily_energy']
        self.time_step = checkpoint['time_step']
        self.energy_history = checkpoint['energy_history']
        self.reward_history = checkpoint['reward_history']
        self.best_reward = checkpoint['best_reward']
        self.patience_counter = checkpoint['patience_counter']

class SolarAgent(MARLAgent):
    def __init__(self, config):
        super().__init__(config)
        self.max_power = 5.0  # Maximum solar power in kW
        self.efficiency = 0.9  # Solar panel efficiency
        self.base_action_scale = 0.5  # Base scaling for solar actions

    def get_action(self, observations: torch.Tensor, communication_channel=None) -> torch.Tensor:
        if communication_channel is None:
            communication_channel = {}
            
        # Get current state information
        time_step = int(observations[3].item())
        current_hour = (time_step // 4) % 24
        solar_potential = self._get_solar_potential(current_hour)
        
        # Share solar information
        communication_channel['solar_agent'] = {
            'solar_available': solar_potential,
            'current_generation': self.energy_usage['solar']
        }
        
        # Get base action from policy network
        base_action = super().get_action(observations, communication_channel)
        
        # Scale action based on solar potential and time of day
        if solar_potential > 0:
            # During peak solar hours (9 AM - 4 PM), maximize solar usage
            if 9 <= current_hour < 16:
                scaled_action = torch.clamp(base_action + 0.8, -1, 1)  # Strong bias towards using solar
            else:
                scaled_action = torch.clamp(base_action + 0.5, -1, 1)  # Moderate bias
            
            # Scale by solar potential and base action scale
            scaled_action = scaled_action * (solar_potential / self.max_power) * self.base_action_scale
        else:
            scaled_action = torch.zeros_like(base_action)
        
        return scaled_action

    def _get_solar_potential(self, hour: int) -> float:
        if 5 <= hour < 19:  # Solar hours (5 AM to 7 PM)
            position = (hour - 5) / 14
            return self.max_power * np.exp(-((position - 0.5) ** 2) / 0.1)
        return 0.0

class GridAgent(MARLAgent):
    def __init__(self, config):
        super().__init__(config)
        self.peak_price = 0.58672
        self.off_peak_price = 0.46366
        self.peak_hours = set(range(7, 23))  # Peak hours from 7 AM to 11 PM
        self.base_action_scale = 0.3  # Base scaling for grid actions

    def get_action(self, observations: torch.Tensor, communication_channel=None) -> torch.Tensor:
        if communication_channel is None:
            communication_channel = {}
            
        # Get current state information
        time_step = int(observations[3].item())
        current_hour = (time_step // 4) % 24
        grid_price = observations[1].item()
        
        # Get info from other agents
        solar_info = communication_channel.get('solar_agent', {})
        battery_info = communication_channel.get('battery_agent', {})
        
        # Calculate remaining demand after solar and battery
        solar_available = solar_info.get('solar_available', 0)
        battery_discharge = max(0, battery_info.get('current_charge', 0) - battery_info.get('min_reserve', 2.5))
        
        # Get base action from policy network
        base_action = super().get_action(observations, communication_channel)
        
        # Scale action based on time of day and energy usage
        if current_hour in self.peak_hours:
            scaled_action = torch.clamp(base_action - 0.5, -1, 1) * self.base_action_scale  # Minimize grid usage during peak
        else:
            scaled_action = torch.clamp(base_action + 0.2, -1, 1) * self.base_action_scale  # Allow more grid usage during off-peak
        
        return scaled_action

class BatteryAgent(MARLAgent):
    def __init__(self, config):
        super().__init__(config)
        self.battery_capacity = 12.5  # 12.5 kWh capacity
        self.battery_charge = self.battery_capacity * 0.6  # Start at 60% charge
        self.min_reserve = 2.5        # Minimum 2.5 kWh reserve for emergencies
        self.charge_threshold = 0.6   # Target to maintain 60% charge when possible
        self.power_rating = 5.0       # Maximum charge/discharge rate in kW
        self.efficiency = 0.9         # Charging/discharging efficiency
        self.base_action_scale = 0.4  # Base scaling for battery actions

    def get_action(self, observations: torch.Tensor, communication_channel=None) -> torch.Tensor:
        if communication_channel is None:
            communication_channel = {}
            
        # Get current battery state
        battery_state = observations[3].item()
        self.battery_charge = battery_state
        available_capacity = self.battery_capacity - battery_state
        grid_price = observations[1].item()
        time_step = int(observations[3].item())
        current_hour = (time_step // 4) % 24

        # Share battery status
        communication_channel['battery_agent'] = {
            'current_charge': battery_state,
            'available_capacity': available_capacity,
            'min_reserve': self.min_reserve
        }
        
        # Get info from other agents
        solar_info = communication_channel.get('solar_agent', {})
        solar_available = solar_info.get('solar_available', 0)
        
        # Get base action from policy network
        base_action = super().get_action(observations, communication_channel)
        
        # Scale action based on battery state and time of day
        if self.battery_charge < self.min_reserve:
            # Emergency charging needed
            if solar_available > 0:
                scaled_action = torch.clamp(base_action + 0.9, -1, 1) * self.base_action_scale  # Strong bias towards charging from solar
            else:
                scaled_action = torch.clamp(base_action + 0.7, -1, 1) * self.base_action_scale  # Charge from grid if needed
        elif current_hour in range(7, 23):  # Peak hours
            if self.battery_charge > self.min_reserve * 1.5:
                scaled_action = torch.clamp(base_action - 0.8, -1, 1) * self.base_action_scale  # Bias towards discharging
            elif solar_available > 0 and self.battery_charge < self.battery_capacity * 0.8:
                scaled_action = torch.clamp(base_action + 0.6, -1, 1) * self.base_action_scale  # Charge from solar if available
            else:
                scaled_action = base_action * self.base_action_scale
        else:  # Off-peak hours
            if self.battery_charge < self.battery_capacity * 0.7:
                scaled_action = torch.clamp(base_action + 0.5, -1, 1) * self.base_action_scale  # Charge from grid
            elif solar_available > 0 and grid_price > 0.4:
                scaled_action = torch.clamp(base_action + 0.4, -1, 1) * self.base_action_scale  # Charge from solar if cheaper
            else:
                scaled_action = base_action * self.base_action_scale
        
        return scaled_action

    def _calculate_action_utilities(self, state):
        current_time = state['time_step'][0]
        current_part = ['morning', 'afternoon', 'evening'][int(current_time) % 3]
        grid_price = state['grid_pricing'][0]
        demand = state['household_demand'][0]

        # Solar utility calculation - make it very attractive during high-price periods
        solar_utility = 1.0  # Base utility
        if current_part == 'morning':
            solar_utility *= 1.5  # Higher utility in morning
        elif current_part == 'afternoon':
            solar_utility *= 2.0  # Highest utility in afternoon
        elif current_part == 'evening':
            solar_utility *= 1.2  # Still useful but less available

        # Grid utility calculation - inverse to price
        grid_utility = 1.0 / (grid_price + 0.1)  # Add 0.1 to avoid division by zero

        # Battery utility based on charge level and time of day
        battery_charge = state.get('battery_level', [0.5])[0]  # Default to 50% if not available
        
        # Battery is more valuable during evening (high price) period
        battery_utility = 0.5
        if current_part == 'evening':
            battery_utility = 1.5 if battery_charge > 0.3 else 0.2  # High utility if charged
        elif current_part == 'morning':
            battery_utility = 0.3 if battery_charge < 0.8 else 0.1  # Prefer charging
        else:  # afternoon
            battery_utility = 0.8 if battery_charge < 0.9 else 0.2  # Balance charging/discharging

        return solar_utility, grid_utility, battery_utility
 