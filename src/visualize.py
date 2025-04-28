import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.environment.energy_env import EnergyResilienceEnv

def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def display_session_costs(info):
    """Display costs and energy usage for the session"""
    print("\nMorning Session:")
    print(f"Energy Required: {info['energy_required']['morning']:.2f} kWh")
    print(f"Solar Agent Cost: ${info['costs']['solar']['morning']:.2f}")
    print(f"Grid Agent Cost: ${info['costs']['grid']['morning']:.2f}")
    print(f"Battery Agent Cost: ${info['costs']['battery']['morning']:.2f}")
    
    print("\nAfternoon Session:")
    print(f"Energy Required: {info['energy_required']['afternoon']:.2f} kWh")
    print(f"Solar Agent Cost: ${info['costs']['solar']['afternoon']:.2f}")
    print(f"Grid Agent Cost: ${info['costs']['grid']['afternoon']:.2f}")
    print(f"Battery Agent Cost: ${info['costs']['battery']['afternoon']:.2f}")
    
    print("\nEvening Session:")
    print(f"Energy Required: {info['energy_required']['evening']:.2f} kWh")
    print(f"Solar Agent Cost: ${info['costs']['solar']['evening']:.2f}")
    print(f"Grid Agent Cost: ${info['costs']['grid']['evening']:.2f}")
    print(f"Battery Agent Cost: ${info['costs']['battery']['evening']:.2f}")
    
    total_cost = sum([sum(costs.values()) for costs in info['costs'].values()])
    total_energy = sum(info['energy_required'].values())
    print(f"\nTotal Cost: ${total_cost:.2f}")
    print(f"Total Energy Required: {total_energy:.2f} kWh")

def main():
    # Load configuration
    config = load_config('configs/training_config.yaml')
    
    # Create environment
    env = EnergyResilienceEnv(config)
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Run simulation and create visualizations
    for step in range(10):  # Run for 10 steps
        # Generate random actions for power and battery control
        power_action = np.random.uniform(0, 100)  # Random power between 0-100 MW
        battery_action = np.random.uniform(-1, 1)  # Random battery action between -1 and 1
        
        # Step the environment (now handling 5 return values)
        obs, reward, done, truncated, info = env.step({
            'power_control': power_action,
            'battery_control': battery_action
        })
        
        # Calculate power flow before visualization
        env.network.lpf()  # Linear power flow calculation
        
        # Render current state
        env.render()
        
        # Save network state visualization
        plt.savefig(f'visualizations/network_step_{step}.png')
        plt.close()
        
        # Plot and save power flow
        env.network_visualizer.plot_power_flow(f'visualizations/power_flow_step_{step}.png')
        
        # Plot and save bus voltages
        env.network_visualizer.plot_bus_voltages(f'visualizations/voltages_step_{step}.png')
        
        # Display session costs
        display_session_costs(info)
        
        if done or truncated:
            break
    
    # Create animation of network changes
    env.network_visualizer.create_animation('visualizations/network_animation.gif')
    
    # Plot performance metrics
    env.network_visualizer.plot_metrics('visualizations/metrics.png')
    
    print("Visualizations created successfully in the 'visualizations' directory")

if __name__ == "__main__":
    main()