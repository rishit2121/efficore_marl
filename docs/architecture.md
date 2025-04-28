# Multi-Agent Energy Resilience System Architecture

## Overview

This document describes the architecture of the Multi-Agent Energy Resilience System, which is designed to optimize energy management in critical infrastructure facilities using Multi-Agent Reinforcement Learning (MARL).

## System Components

### 1. Environment Module

The environment module (`src/environment/energy_env.py`) simulates the energy system of a critical infrastructure facility. Key features:

- Power network simulation using PyPSA
- Multi-agent observation and action spaces
- Reward calculation based on energy efficiency and resilience
- Partial observability handling
- Metrics tracking for energy cost, resilience, and reliability

### 2. Agent Module

The agent module (`src/agents/marl_agent.py`) implements the MARL agents:

- Policy and value networks for each agent
- Action selection and evaluation
- Experience collection and learning
- Model saving and loading

### 3. Training Module

The training module (`src/train.py`) handles the training process:

- Environment and agent initialization
- Training loop implementation
- Periodic evaluation
- Model checkpointing
- Logging and visualization

### 4. Utility Modules

#### Data Processing (`src/utils/data_processing.py`)
- Data preprocessing and normalization
- Sequence creation for time series data
- Dataset and DataLoader creation
- Metrics calculation
- Agent data preparation

#### Visualization (`src/utils/visualization.py`)
- Training curve plotting
- Energy metrics visualization
- Agent action analysis
- Correlation heatmaps
- Distribution plots

## Data Flow

1. **Environment Interaction**
   - Agents receive observations from the environment
   - Agents select actions based on their policies
   - Environment updates state based on actions
   - Rewards and new observations are returned

2. **Training Process**
   - Collect experience from environment interactions
   - Update agent policies using collected experience
   - Evaluate performance periodically
   - Save model checkpoints

3. **Evaluation Process**
   - Load trained models
   - Run evaluation episodes
   - Calculate and log metrics
   - Generate visualizations

## Configuration

The system is configured using YAML files in the `configs/` directory:

- `training_config.yaml`: Training parameters
- Environment settings
- Network architecture
- Training hyperparameters
- Logging configuration

## Dependencies

- PyTorch: Deep learning framework
- PyPSA: Power network simulation
- Gymnasium: Environment interface
- Pandas: Data processing
- Matplotlib/Seaborn: Visualization
- Wandb: Experiment tracking

## Project Structure

```
.
├── configs/           # Configuration files
├── data/             # Data storage
├── docs/             # Documentation
├── notebooks/        # Analysis notebooks
├── src/              # Source code
│   ├── agents/       # Agent implementations
│   ├── environment/  # Environment simulation
│   ├── models/       # Model architectures
│   └── utils/        # Utility functions
└── tests/            # Test files
```

## Usage

1. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training**
   ```bash
   python src/train.py --config configs/training_config.yaml
   ```

3. **Evaluation**
   ```bash
   python src/evaluate.py --config configs/evaluation_config.yaml
   ```

4. **Analysis**
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

## Future Improvements

1. **Enhanced Environment**
   - More realistic power network modeling
   - Additional failure scenarios
   - Dynamic demand patterns

2. **Advanced Agent Architectures**
   - Attention mechanisms
   - Hierarchical reinforcement learning
   - Transfer learning capabilities

3. **System Features**
   - Distributed training
   - Real-time monitoring
   - Automated hyperparameter optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Run tests
5. Submit a pull request

## License

[License information to be added] 