import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import pandas as pd
import os

def plot_training_curves(rewards: List[float], 
                        eval_rewards: List[float],
                        save_path: str = None) -> None:
    """
    Plot training and evaluation reward curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Training Reward')
    plt.plot(eval_rewards, label='Evaluation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_energy_metrics(metrics: Dict[str, List[float]],
                       save_path: str = None) -> None:
    """
    Plot energy-related metrics over time
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Energy Metrics Over Time')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_agent_actions(actions: Dict[str, np.ndarray],
                      save_path: str = None) -> None:
    """
    Plot actions taken by each agent
    """
    plt.figure(figsize=(12, 8))
    
    for agent_id, agent_actions in actions.items():
        plt.plot(agent_actions, label=f'Agent {agent_id}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Action Value')
    plt.title('Agent Actions Over Time')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_heatmap(data: np.ndarray,
                title: str,
                xlabel: str,
                ylabel: str,
                save_path: str = None) -> None:
    """
    Plot a heatmap of the given data
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_distribution(data: np.ndarray,
                     title: str,
                     xlabel: str,
                     save_path: str = None) -> None:
    """
    Plot the distribution of the given data
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def create_summary_plots(metrics: Dict[str, Any],
                        save_dir: str) -> None:
    """
    Create a comprehensive set of summary plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training curves
    plot_training_curves(
        metrics['training_rewards'],
        metrics['eval_rewards'],
        os.path.join(save_dir, 'training_curves.png')
    )
    
    # Plot energy metrics
    plot_energy_metrics(
        metrics['energy_metrics'],
        os.path.join(save_dir, 'energy_metrics.png')
    )
    
    # Plot agent actions
    plot_agent_actions(
        metrics['agent_actions'],
        os.path.join(save_dir, 'agent_actions.png')
    )
    
    # Plot heatmap of correlation between metrics
    correlation_matrix = pd.DataFrame(metrics['energy_metrics']).corr()
    plot_heatmap(
        correlation_matrix,
        'Metric Correlations',
        'Metric',
        'Metric',
        os.path.join(save_dir, 'metric_correlations.png')
    )
    
    # Plot distribution of rewards
    plot_distribution(
        np.array(metrics['training_rewards']),
        'Training Reward Distribution',
        'Reward',
        os.path.join(save_dir, 'reward_distribution.png')
    ) 