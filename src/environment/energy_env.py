import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Any
import pypsa
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging

# Configure logging to suppress PyPSA INFO messages
logging.getLogger('pypsa').setLevel(logging.WARNING)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.network_visualization import NetworkVisualizer

class EnergyResilienceEnv(gym.Env):

    # Environment for simulating energy resilience in critical infrastructure.
    # This environment hass a multi-agent system where each agent controls
    # different energy sources.
    
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Environment configuration
        self.config = config
        self.n_agents = config.get('n_agents', 3)
        self.time_step = 0
        self.max_steps = 96  # 24 hours * 4 (15-minute intervals)
        self.blackout = False
        self.episode_count = 0  # Add episode counter
        
        # Load solar generation data
        self.solar_data = pd.read_csv('data/archive (2)/Plant_1_Generation_Data.csv')
        self.solar_data['DATE_TIME'] = pd.to_datetime(self.solar_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
        
        # Scale down solar generation to match residential installation
        # Assuming average residential solar installation is 5kW (vs power plant)
        self.solar_scaling_factor = 5.0 / self.solar_data['AC_POWER'].max()
        self.solar_data['AC_POWER_kWh'] = self.solar_data['AC_POWER'] * self.solar_scaling_factor * 0.25  # Convert kW to kWh for 15-min interval
        
        # Solar investment costs
        self.solar_installation_cost = 15000  # $15,000 initial investment
        self.solar_lifespan_years = 25
        self.daily_solar_cost = self.solar_installation_cost / (self.solar_lifespan_years * 365)  # Daily amortized cost
        
        # Update battery configuration
        self.battery_capacity = 12.5     # 12.5 kWh capacity
        self.battery_charge = self.battery_capacity * 0.6  # Start at 60% charge
        self.battery_reserve_threshold = 0.2 * self.battery_capacity
        self.battery_efficiency = 0.9    # 90% charging/discharging efficiency
        self.battery_investment_cost = 0.001  # $0.001 per kWh for battery investment/maintenance
        self.battery_power_rating = 5    # 5kW maximum power rate
        
        # Solar tracking
        self.daily_solar_generated = 0   # Track daily solar generation
        self.solar_energy_storage = 0    # Initialize solar energy storage
        self.available_solar = 0         # Track available solar for current interval
        
        # Solar cost and limits
        self.solar_cost_per_kwh = 0.02   # $0.02/kWh for solar energy
        self.solar_maintenance_cost = 0.005  # $0.005/kWh for maintenance
        
        # Grid pricing (in $/kWh)
        self.peak_price = 0.58672  # Peak price during 4-9 PM
        self.off_peak_price = 0.46366  # Off-peak price for all other hours
        self.peak_hours = set(range(16, 21))  # Peak hours from 4 PM to 9 PM
        
        # Initialize other components
        self.network = pypsa.Network()
        self._initialize_network()
        self._initialize_pricing_and_demand()
        
        # Initialize visualizer
        self.visualizer = NetworkVisualizer(self.network)
        
        # Create output directory for visualizations
        self.viz_dir = config.get('viz_dir', 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Define observation and action spaces
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # Metrics tracking
        self.metrics = {
            'energy_cost': [],
            'resilience_score': [],
            'reliability_score': []
        }
        
        # Add communication between agents to share information
        self.communication_channel = {
            'resilience_agent': None,
            'cost_agent': None
        }
        
        # Initialize cumulative tracking
        self.cumulative_energy = {
            'solar': 0.0,
            'grid': 0.0,
            'battery': 0.0
        }
        self.cumulative_costs = {
            'solar': 0.0,
            'grid': 0.0,
            'battery': 0.0,
            'penalty': 0.0
        }
    
    def _initialize_network(self):
        """Initialize the power network with solar, grid, and battery components"""
        # Add buses
        self.network.add("Bus", "main_bus", v_nom=20, x=0, y=0)
        self.network.add("Bus", "load_bus", v_nom=20, x=1, y=0)
        self.network.add("Bus", "solar_bus", v_nom=20, x=-1, y=0)
        self.network.add("Bus", "battery_bus", v_nom=20, x=0, y=1)
        
        # Add components
        self.network.add("Generator", "solar_panel", bus="solar_bus", p_nom=10, marginal_cost=25)  # Limited to 10kWh/day
        self.network.add("Generator", "grid", bus="main_bus", p_nom=100, marginal_cost=40)
        self.network.add("StorageUnit", "battery", bus="battery_bus", p_nom=5, marginal_cost=2, max_hours=2.5)  # 5kW power rating, 12.5kWh capacity
        self.network.add("Load", "household", bus="load_bus", p_set=50)
        
        # Add lines
        self.network.add("Line", "line1", bus0="main_bus", bus1="load_bus", x=0.1)
        self.network.add("Line", "line2", bus0="solar_bus", bus1="main_bus", x=0.1)
        self.network.add("Line", "line4", bus0="battery_bus", bus1="main_bus", x=0.1)
    
    def _initialize_pricing_and_demand(self):
        """Initialize pricing and demand profiles for the day"""
        # Updated grid pricing for different periods ($/kWh)
        self.pricing = {
            'morning': 0.28,    # Updated morning rate
            'afternoon': 0.15,  # Updated afternoon rate
            'evening': 0.47     # Updated evening rate
        }

        # Base demand profile that can be scaled
        self.base_demand_profile = {
            'morning': 10,    # Base morning demand
            'afternoon': 8,   # Base afternoon demand
            'evening': 12     # Base evening demand
        }
        
        # Add random variation to demand profile
        variation = np.random.uniform(0.8, 1.2)  # Allow 20% variation
        self.demand_profile = {
            period: base_demand * variation
            for period, base_demand in self.base_demand_profile.items()
        }

        # Solar generation potential (kWh) for different periods
        self.solar_potential = {
            'morning': 4.0,     # Morning potential
            'afternoon': 6.0,   # Peak generation potential
            'evening': 2.0      # Reduced evening potential
        }
    
    def _get_current_pricing_and_demand(self, time_step: int):
        """Get current pricing and demand based on the 15-minute interval"""
        # Convert time_step to hour (0-23)
        current_hour = (time_step // 4) % 24
        
        # Determine if current hour is peak or off-peak
        is_peak = current_hour in self.peak_hours
        current_price = self.peak_price if is_peak else self.off_peak_price
        
        # Base demand profile with noise
        if 0 <= current_hour < 6:
            base_demand = 0.5  # Night base demand
        elif 6 <= current_hour < 9:
            base_demand = 1.5  # Morning peak base demand
        elif 9 <= current_hour < 16:
            base_demand = 1.25  # Midday base demand
        elif 16 <= current_hour < 21:
            base_demand = 2.0  # Evening peak base demand
        else:
            base_demand = 1.0  # Night base demand
        
        # Add random noise to demand (20% variation)
        noise = np.random.normal(0, 0.2)  # Gaussian noise with 20% std dev
        current_demand = base_demand * (1 + noise)
        
        # Ensure demand is never negative
        current_demand = max(0.1, current_demand)
        
        # Convert kW to kWh for 15-minute interval
        current_demand = current_demand * 0.25  # 15 minutes = 0.25 hours
        
        return current_price, current_demand

    def _get_solar_potential(self, time_step: int) -> float:
        """Get solar generation potential for current 15-minute interval using real data"""
        # Get current date and time
        current_date = self.solar_data['DATE_TIME'].iloc[0].date()
        current_time = datetime.strptime(f"{time_step//4:02d}:{(time_step%4)*15:02d}", "%H:%M").time()
        current_datetime = datetime.combine(current_date, current_time)
        
        # Find matching data point
        matching_data = self.solar_data[
            (self.solar_data['DATE_TIME'].dt.date == current_date) &
            (self.solar_data['DATE_TIME'].dt.hour == current_time.hour) &
            (self.solar_data['DATE_TIME'].dt.minute == current_time.minute)
        ]
        
        if not matching_data.empty:
            # Return average AC power in kWh for the interval
            return matching_data['AC_POWER_kWh'].mean()
        else:
            # If no data found for this time, return 0
            return 0.0

    def _get_observation_space(self) -> gym.spaces.Dict:
        """Define the observation space for the environment"""
        return gym.spaces.Dict({
            'power_flow': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents,)),
            'battery_level': gym.spaces.Box(low=0, high=100, shape=(self.n_agents,)),
            'demand': gym.spaces.Box(low=0, high=np.inf, shape=(self.n_agents,)),
            'time_step': gym.spaces.Box(low=0, high=self.max_steps, shape=(1,))
        })
    
    def _get_action_space(self) -> gym.spaces.Dict:
        """Define the action space for the environment"""
        return gym.spaces.Dict({
            'power_control': gym.spaces.Box(low=-1, high=1, shape=(self.n_agents,)),
            'battery_control': gym.spaces.Box(low=-1, high=1, shape=(self.n_agents,))
        })
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.time_step = 0
        self.episode_count += 1  # Increment episode counter
        self.network = pypsa.Network()
        self._initialize_network()
        self._initialize_pricing_and_demand()
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.clear()
        
        # Reset battery and solar tracking
        self.battery_charge = self.battery_capacity * 0.6  # Reset to 60% charge
        self.daily_solar_generated = 0
        self.solar_energy_storage = 0
        
        # Reset session costs and power tracking
        self.session_costs = {
            'solar': 0.0,
            'grid': 0.0,
            'battery': 0.0,
            'penalty': 0.0
        }
        
        # Reset cumulative tracking
        self.cumulative_energy = {
            'solar': 0.0,
            'grid': 0.0,
            'battery': 0.0
        }
        self.cumulative_costs = {
            'solar': 0.0,
            'grid': 0.0,
            'battery': 0.0,
            'penalty': 0.0
        }
        
        # Initialize power tracking
        self.morning_power = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        self.afternoon_power = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        self.evening_power = {'solar': 0.0, 'grid': 0.0, 'battery': 0.0}
        
        return self._get_observation(), {}
    
    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """Apply actions to the environment"""
        # Get current interval info
        current_price, current_demand = self._get_current_pricing_and_demand(self.time_step)
        solar_potential = self._get_solar_potential(self.time_step)
        
        # Extract control variables and add noise for exploration
        solar_control = actions.get('solar_control', 0) + np.random.normal(0, 0.1)
        grid_control = actions.get('grid_control', 0) + np.random.normal(0, 0.1)
        battery_control = actions.get('battery_control', 0) + np.random.normal(0, 0.1)
        
        # Scale controls to [0,1] range
        solar_control = (solar_control + 1) / 2  # Scale from [-1,1] to [0,1]
        grid_control = (grid_control + 1) / 2
        battery_control = (battery_control + 1) / 2
        
        # Calculate demand allocation based on controls and available resources
        # Solar gets priority and is scaled by potential
        demand_from_solar = min(solar_potential, current_demand * solar_control)
        
        # Remaining demand is split between grid and battery based on their controls
        remaining_demand = current_demand - demand_from_solar
        if remaining_demand > 0:
            # During evening peak hours, prioritize battery discharge
            current_hour = (self.time_step // 4) % 24
            if 16 <= current_hour < 21:  # Evening peak hours
                battery_scale = 1.5  # Increase battery contribution during peak
                grid_scale = 0.5    # Reduce grid contribution during peak
            else:
                battery_scale = 1.0
                grid_scale = 1.0
            
            # Calculate total control weight
            total_control = (grid_control * grid_scale + battery_control * battery_scale)
            if total_control > 0:
                demand_from_grid = remaining_demand * (grid_control * grid_scale / total_control)
                demand_from_battery = remaining_demand * (battery_control * battery_scale / total_control)
            else:
                # If no control preference, split evenly
                demand_from_grid = remaining_demand / 2
                demand_from_battery = remaining_demand / 2
        else:
            demand_from_grid = 0
            demand_from_battery = 0
        
        # Set power setpoints in the network
        # Convert from kW to MW for PyPSA
        self.network.generators.at['solar_panel', 'p_set'] = demand_from_solar / 1000  # kW to MW
        self.network.generators.at['grid', 'p_set'] = demand_from_grid / 1000  # kW to MW
        self.network.storage_units.at['battery', 'p_set'] = demand_from_battery / 1000  # kW to MW
        
        # Run power flow calculation
        self.network.lpf()
        
        # Get actual power flows after network calculation
        solar_usage = max(0, self.network.generators.at['solar_panel', 'p_set'] * 1000)  # MW to kW
        grid_usage = max(0, self.network.generators.at['grid', 'p_set'] * 1000)  # MW to kW
        battery_power = self.network.storage_units.at['battery', 'p_set'] * 1000  # MW to kW
        battery_throughput = abs(battery_power)
        
        # Handle solar usage - use available solar potential
        solar_used = min(solar_usage, solar_potential)
        unmet_solar = solar_usage - solar_used
        self.daily_solar_generated += solar_used
        
        # Calculate solar cost (only maintenance, investment cost is daily)
        solar_cost = solar_used * self.solar_maintenance_cost
        
        # Handle battery usage
        max_discharge = min(self.battery_power_rating, self.battery_charge)
        battery_used = min(battery_throughput, max_discharge)
        unmet_battery = battery_throughput - battery_used
        self.battery_charge -= battery_used
        
        # Calculate battery cost
        battery_cost = battery_used * self.battery_investment_cost
        
        # Handle grid usage
        if self.blackout:
            grid_used = 0
            unmet_grid = demand_from_grid
        else:
            grid_used = grid_usage
            unmet_grid = 0
        
        # Calculate grid cost using current price
        grid_cost = grid_used * current_price
        
        # Handle backup strategy for unmet demand
        total_unmet = unmet_solar + unmet_battery + unmet_grid
        backup_used = min(self.battery_charge, total_unmet)
        self.battery_charge -= backup_used
        backup_cost = backup_used * self.battery_investment_cost
        total_unmet -= backup_used
        
        # Apply penalty for unmet demand
        penalty_per_kwh = 5.0
        penalty_cost = total_unmet * penalty_per_kwh
        
        # Handle battery charging - use remaining solar after meeting demand
        remaining_solar = solar_potential - solar_used
        if remaining_solar > 0 and self.battery_charge < self.battery_capacity:
            max_charge = min(
                remaining_solar,
                self.battery_capacity - self.battery_charge,
                self.battery_power_rating
            )
            self.battery_charge += max_charge
            self.daily_solar_generated += max_charge
            charge_cost = max_charge * self.solar_maintenance_cost
            solar_cost += charge_cost
        
        # Calculate total cost for this step
        step_cost = solar_cost + grid_cost + battery_cost + backup_cost + penalty_cost
        
        # Return step costs for tracking
        step_costs = {
            'solar': solar_cost,
            'grid': grid_cost,
            'battery': battery_cost + backup_cost,
            'penalty': penalty_cost
        }
        
        return solar_used, grid_used, battery_used, step_costs

    def step(self, actions: list) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one time step in the environment"""
        # Convert agent actions list to dict
        actions_dict = {
            'solar_control': actions[0],  # Solar/Resilience agent
            'grid_control': actions[1],   # Cost agent
            'battery_control': actions[2]  # Battery agent
        }
        
        # Apply actions and get energy usage
        solar_usage, grid_usage, battery_power, step_costs = self._apply_actions(actions_dict)
        self.network.lpf()  # Run power flow
        
        # Update cumulative energy usage
        self.cumulative_energy['solar'] += solar_usage
        self.cumulative_energy['grid'] += grid_usage
        self.cumulative_energy['battery'] += abs(battery_power)
        
        # Update cumulative costs
        for source, cost in step_costs.items():
            self.cumulative_costs[source] += cost
        
        # Calculate reward components
        reward, reward_components = self._calculate_reward()
        metrics = self._update_metrics()
        
        # Increment time step and check if done
        self.time_step += 1
        done = self.time_step >= self.max_steps
        truncated = False
        
        # Track power usage by session
        current_hour = (self.time_step // 4) % 24
        if 6 <= current_hour < 9:  # Morning session
            self.morning_power = {
                'solar': solar_usage,
                'grid': grid_usage,
                'battery': battery_power
            }
            self.morning_costs = step_costs
        elif 9 <= current_hour < 16:  # Afternoon session
            self.afternoon_power = {
                'solar': solar_usage,
                'grid': grid_usage,
                'battery': battery_power
            }
            self.afternoon_costs = step_costs
        elif 16 <= current_hour < 21:  # Evening session
            self.evening_power = {
                'solar': solar_usage,
                'grid': grid_usage,
                'battery': battery_power
            }
            self.evening_costs = step_costs
        
        # Return info with detailed costs and energy usage
        info = {
            'costs': {
                'solar': self.cumulative_costs['solar'],
                'grid': self.cumulative_costs['grid'],
                'battery': self.cumulative_costs['battery'],
                'total': sum(self.cumulative_costs.values())
            },
            'energy_usage': {
                'solar': self.cumulative_energy['solar'],
                'grid': self.cumulative_energy['grid'],
                'battery': self.cumulative_energy['battery']
            },
            'power_by_session': {
                'morning': self.morning_power if hasattr(self, 'morning_power') else None,
                'afternoon': self.afternoon_power if hasattr(self, 'afternoon_power') else None,
                'evening': self.evening_power if hasattr(self, 'evening_power') else None
            },
            'costs_by_session': {
                'morning': self.morning_costs if hasattr(self, 'morning_costs') else None,
                'afternoon': self.afternoon_costs if hasattr(self, 'afternoon_costs') else None,
                'evening': self.evening_costs if hasattr(self, 'evening_costs') else None
            },
            'reward_components': reward_components,
            'battery_state': self.battery_charge,
            'time_of_day': self.time_step % 96
        }
        
        return self._get_observation(), reward, done, truncated, info

    def _calculate_reward(self) -> Tuple[float, Dict[str, float]]:
        """Calculate reward considering multiple objectives"""
        current_price, current_demand = self._get_current_pricing_and_demand(self.time_step)
        solar_potential = self._get_solar_potential(self.time_step)
        
        # Calculate energy usage and costs for this step
        solar_usage = max(0, self.network.generators.at['solar_panel', 'p_set'] * 1000)  # MWh to kWh
        grid_usage = max(0, self.network.generators.at['grid', 'p_set'] * 1000)  # MWh to kWh
        battery_power = self.network.storage_units.at['battery', 'p_set'] * 1000  # MWh to kWh
        battery_throughput = abs(battery_power)
        
        # Calculate step energy usage
        step_energy = solar_usage + grid_usage + battery_throughput
        
        # Calculate daily totals
        daily_demand = sum(self.demand_profile.values())  # Use actual demand profile total
        daily_supply = self.cumulative_energy['solar'] + self.cumulative_energy['grid'] + self.cumulative_energy['battery']
        
        # Calculate target energy per step based on actual demand
        target_step_energy = current_demand  # Use current demand as target
        energy_deviation = step_energy - target_step_energy
        
        # Immediate penalty for energy usage - using bounded exponential
        if energy_deviation > 0:
            # Bounded exponential penalty
            energy_penalty = -10000 * min(np.exp(min(energy_deviation / target_step_energy, 5)), 1000)
        else:
            # Small reward for staying under target
            energy_penalty = 100 * abs(energy_deviation)
        
        # 2. Daily Energy Usage Penalty (Primary)
        daily_energy_deviation = daily_supply - daily_demand
        if daily_energy_deviation > 0:
            # Bounded exponential penalty
            daily_penalty = -100000 * min(np.exp(min(daily_energy_deviation / daily_demand, 5)), 1000)
        else:
            daily_penalty = 0
        
        # 3. Source-specific penalties
        # Solar penalty for over-usage
        if solar_usage > solar_potential and solar_potential > 0:
            solar_penalty = -100000 * min(np.exp(min(solar_usage / solar_potential, 5)), 1000)
        else:
            solar_penalty = 0
        
        # Grid penalty for over-usage
        if grid_usage > current_demand and current_demand > 0:
            grid_penalty = -100000 * min(np.exp(min(grid_usage / current_demand, 5)), 1000)
        else:
            grid_penalty = 0
        
        # Battery penalty for over-usage
        if battery_throughput > self.battery_power_rating and self.battery_power_rating > 0:
            battery_penalty = -100000 * min(np.exp(min(battery_throughput / self.battery_power_rating, 5)), 1000)
        else:
            battery_penalty = 0
        
        # 4. Reserve Penalty
        if self.battery_charge < self.battery_reserve_threshold and self.battery_reserve_threshold > 0:
            reserve_penalty = -100000 * min(np.exp(min((self.battery_reserve_threshold - self.battery_charge) / self.battery_reserve_threshold, 5)), 1000)
        else:
            reserve_penalty = 0
        
        # 5. Demand Satisfaction Component (Secondary)
        demand_satisfaction = min(1.0, daily_supply / daily_demand)
        demand_reward = 50 * demand_satisfaction
        
        # 6. Cost Efficiency Component (Tertiary)
        solar_cost = solar_usage * (self.solar_cost_per_kwh + self.solar_maintenance_cost)
        grid_cost = grid_usage * current_price
        battery_storage_cost = battery_throughput * self.battery_investment_cost
        total_cost = solar_cost + grid_cost + battery_storage_cost
        
        # Cost penalty scaled by energy usage
        cost_penalty = -total_cost * (1 + (step_energy / target_step_energy))
        
        # 7. Solar Utilization Component
        solar_utilization = solar_usage / solar_potential if solar_potential > 0 else 0
        solar_reward = 20 * solar_utilization
        
        # 8. Battery Health Component
        battery_health = self.battery_charge / self.battery_capacity
        battery_reward = 10 * battery_health
        
        # 9. Peak Hour Management
        if current_price == self.peak_price:
            peak_penalty = -50 * grid_usage
        else:
            peak_penalty = 0
        
        # Combine all components with normalized weights
        reward = (
            energy_penalty +  # Primary component (most prominent)
            daily_penalty +   # Primary component (most prominent)
            solar_penalty +   # Source-specific penalties
            grid_penalty +
            battery_penalty +
            reserve_penalty + # Reserve penalty
            0.01 * demand_reward +  # Secondary components (reduced weights)
            0.01 * cost_penalty +
            0.01 * solar_reward +
            0.01 * battery_reward +
            0.01 * peak_penalty
        )
        
        # Return both total reward and individual components
        reward_components = {
            'energy_penalty': energy_penalty + daily_penalty,  # Combined energy penalties
            'solar_penalty': solar_penalty,
            'grid_penalty': grid_penalty,
            'battery_penalty': battery_penalty,
            'reserve_penalty': reserve_penalty,
            'demand_reward': demand_reward,
            'cost_penalty': cost_penalty,
            'solar_reward': solar_reward,
            'battery_reward': battery_reward,
            'peak_penalty': peak_penalty
        }
        
        return reward, reward_components

    def _update_metrics(self) -> Dict[str, float]:
        """Update metrics for cost, resilience, and reliability"""
        total_cost = sum(self.network.generators['p_set'] * self.network.generators['marginal_cost'])
        unmet_demand = max(0, self.network.loads.at['household', 'p_set'] - 
                           sum(self.network.generators['p_set']))
        reliability = 1 - (unmet_demand / self.network.loads.at['household', 'p_set'])
        
        self.metrics['energy_cost'].append(total_cost)
        self.metrics['resilience_score'].append(-unmet_demand)  # Negative unmet demand
        self.metrics['reliability_score'].append(reliability)
        
        return {
            'energy_cost': total_cost,
            'resilience_score': -unmet_demand,
            'reliability_score': reliability
        }
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation of the environment"""
        # Ensure all values are valid numbers
        solar_storage = max(0, min(self.solar_energy_storage, 100))  # Cap at 100
        grid_price = max(0, min(self._get_current_pricing_and_demand(self.time_step)[0], 100))  # Cap at 100
        demand = max(0, min(self._get_current_pricing_and_demand(self.time_step)[1], 100))  # Cap at 100
        time_step = max(0, min(self.time_step, self.max_steps))  # Cap at max_steps
        
        return {
            'solar_energy_storage': np.array([solar_storage]),
            'grid_pricing': np.array([grid_price]),
            'household_demand': np.array([demand]),
            'time_step': np.array([time_step])
        }
    
    # def render(self, mode='human'):
    #     """Render the current state of the environment"""
    #     if mode == 'human':
    #         # Calculate power flow before visualization
    #         self.network.lpf()
            
    #         # Plot current network state
    #         self.visualizer.plot_network(
    #             title=f"Power Network - Step {self.time_step}",
    #             save_path=os.path.join(self.viz_dir, f"network_step_{self.time_step}.png")
    #         )
            
    #         # Plot power flow
    #         self.visualizer.plot_power_flow(
    #             save_path=os.path.join(self.viz_dir, f"power_flow_step_{self.time_step}.png")
    #         )
            
    #         # Plot bus voltages
    #         self.visualizer.plot_bus_voltages(
    #             save_path=os.path.join(self.viz_dir, f"voltages_step_{self.time_step}.png")
    #         )
    
    # def create_animation(self, steps: int = None):
    #     """Create an animation of the network over time"""
    #     if steps is None:
    #         steps = self.max_steps
        
    #     self.visualizer.create_animation(
    #         steps=steps,
    #         save_path=os.path.join(self.viz_dir, "network_animation.gif")
    #     )
    
    # def plot_metrics(self):
    #     """Plot environment metrics over time"""
    #     plt.figure(figsize=(12, 8))
        
    #     for metric_name, values in self.metrics.items():
    #         plt.plot(values, label=metric_name)
        
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Value')
    #     plt.title('Environment Metrics Over Time')
    #     plt.legend()
    #     plt.grid(True)
        
    #     plt.savefig(os.path.join(self.viz_dir, "metrics.png"))
    #     plt.close()