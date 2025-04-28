import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class EnergyDataset(Dataset):
    """
    Dataset class for energy data
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

def preprocess_data(data: pd.DataFrame,
                   target_columns: List[str] = None,
                   scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Preprocess the input data
    Args:
        data: Input DataFrame
        target_columns: List of target column names
        scaler_type: Type of scaler to use ('standard' or 'minmax')
    Returns:
        Tuple of (scaled features, scaled targets, fitted scaler)
    """
    # Separate features and targets
    if target_columns:
        features = data.drop(columns=target_columns)
        targets = data[target_columns]
    else:
        features = data
        targets = None
    
    # Initialize scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Fit and transform features
    scaled_features = scaler.fit_transform(features)
    
    if targets is not None:
        scaled_targets = scaler.fit_transform(targets)
        return scaled_features, scaled_targets, scaler
    
    return scaled_features, None, scaler

def create_sequences(data: np.ndarray,
                    sequence_length: int,
                    target_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from time series data
    Args:
        data: Input time series data
        sequence_length: Length of input sequences
        target_length: Length of target sequences
    Returns:
        Tuple of (input sequences, target sequences)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length - target_length + 1):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length:i + sequence_length + target_length])
    
    return np.array(sequences), np.array(targets)

def create_data_loaders(features: np.ndarray,
                       targets: np.ndarray,
                       batch_size: int,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders
    """
    # Calculate split indices
    n_samples = len(features)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Split data
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    
    val_features = features[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    
    test_features = features[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    
    # Create datasets
    train_dataset = EnergyDataset(train_features, train_targets)
    val_dataset = EnergyDataset(val_features, val_targets)
    test_dataset = EnergyDataset(test_features, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def calculate_metrics(predictions: np.ndarray,
                     targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics for model evaluation
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def prepare_agent_data(observations: Dict[str, np.ndarray],
                      actions: Dict[str, np.ndarray],
                      rewards: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data for agent training
    """
    # Convert observations to tensor
    obs_tensor = torch.FloatTensor(np.stack(list(observations.values())))
    
    # Convert actions to tensor
    act_tensor = torch.FloatTensor(np.stack(list(actions.values())))
    
    # Convert rewards to tensor
    rew_tensor = torch.FloatTensor(rewards)
    
    return obs_tensor, act_tensor, rew_tensor 