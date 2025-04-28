# Energy Management Multi-Agent Reinforcement Learning (MARL)

This project implements a Multi-Agent Reinforcement Learning (MARL) system for managing energy resources in a residential setting. The system uses three agents to control solar, grid, and battery power sources to optimize energy usage, costs, and reliability.

## Project Overview

The system consists of three main components:
1. **Environment** (`src/environment/energy_env.py`): Simulates the energy system with solar panels, grid connection, and battery storage
2. **Agents** (`src/agents/marl_agent.py`): Three specialized agents for controlling different energy sources
3. **Training** (`src/train.py`): Handles the training process and model evaluation

### Key Features
- Dynamic demand modeling with time-of-day variations
- Realistic solar generation patterns
- Battery storage management
- Peak/off-peak pricing
- Multi-agent coordination
- Cost optimization
- Energy resilience

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/efficore_marl.git
cd efficore_marl
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
efficore_marl/
├── src/
│   ├── agents/
│   │   └── marl_agent.py      # Agent implementations
│   ├── environment/
│   │   └── energy_env.py      # Environment simulation
│   └── train.py               # Training script
├── configs/
│   └── training_config.yaml   # Training configuration
├── data/                      # Data directory
├── checkpoints/              # Model checkpoints
└── requirements.txt          # Project dependencies
```

## Configuration

The training configuration is defined in `configs/training_config.yaml`. Key parameters include:
- Number of episodes
- Learning rates
- Network architecture
- Environment parameters
- Agent-specific settings

## Usage

1. Training the model:
```bash
python src/train.py
```

2. Evaluating a trained model:
```bash
python src/train.py --load_checkpoint path/to/checkpoint
```

## Agent Descriptions

### Solar Agent
- Controls solar power generation and usage
- Optimizes solar utilization during peak generation hours
- Manages solar-to-battery charging

### Grid Agent
- Controls grid power consumption
- Minimizes grid usage during peak pricing hours
- Balances cost and reliability

### Battery Agent
- Manages battery storage and discharge
- Maintains reserve capacity
- Coordinates with solar and grid agents

## Environment Details

The environment simulates:
- 15-minute time steps
- Dynamic demand patterns
- Solar generation based on time of day, modeled using a bell curve derived from real solar production data
- Battery storage with capacity limits
- Grid connection with time-varying pricing based on PG&E utility rates
- Power flow calculations using PyPSA

### Data Sources
- **Solar Production**: Modeled using a bell curve derived from real solar production data, capturing the typical daily generation pattern with peak production around midday
- **Utility Costs**: Based on PG&E's time-of-use (TOU) rates, including:
  - Peak hours (4 PM - 9 PM): Higher rates
  - Off-peak hours: Lower rates
  - Seasonal variations in pricing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyPSA for power system modeling
- PyTorch for deep learning implementation
- Gymnasium for reinforcement learning environment interface 