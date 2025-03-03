import torch
import numpy as np
from typing import Tuple

def combine_datasets(base_data, fine_tune_data):
    """Combine base and fine-tune datasets while maintaining train/val splits."""
    # Combine training data
    x_train = torch.cat([base_data['train']['x'], fine_tune_data['train']['x']], dim=0)
    y_train = torch.cat([base_data['train']['y'], fine_tune_data['train']['y']], dim=0)
    
    # Combine validation data
    x_val = torch.cat([base_data['val']['x'], fine_tune_data['val']['x']], dim=0)
    y_val = torch.cat([base_data['val']['y'], fine_tune_data['val']['y']], dim=0)
    
    return {
        'train': {'x': x_train, 'y': y_train},
        'val': {'x': x_val, 'y': y_val}
    }

def get_base_data(x_min=0, x_max=4, n_samples=1000, val_ratio=0.1, noise_std=0.1):
    """Generate base training data with sin(2x)*cos(3x) function."""
    # Generate x values
    x = np.linspace(x_min, x_max, n_samples)
    
    # Generate y values with nonlinear function and noise
    y = np.sin(2*x) * np.cos(3*x) + 0.2 * np.tanh(x**2) + noise_std * np.random.randn(n_samples)
    
    # Convert to torch tensors
    x = torch.FloatTensor(x.reshape(-1, 1))
    y = torch.FloatTensor(y.reshape(-1, 1))
    
    # Split into train/val
    n_val = int(n_samples * val_ratio)
    all_indices = np.arange(n_samples)
    val_indices = np.random.choice(all_indices, size=n_val, replace=False)
    train_indices = np.array([i for i in all_indices if i not in val_indices])
    
    return {
        'train': {'x': x[train_indices], 'y': y[train_indices]},
        'val': {'x': x[val_indices], 'y': y[val_indices]}
    }

def get_fine_tune_data(x_min=2, x_max=3, n_samples=1000, val_ratio=0.1, noise_std=0.1):
    """Generate fine-tuning data with a different function in [2,3] range."""
    # Generate x values
    x = np.linspace(x_min, x_max, n_samples)
    
    # Different function for fine-tuning: combination of exponential and oscillation
    y = np.sin(2*x) * np.cos(3*x) + 0.2 * np.sin(16*x) + 0.2 * np.tanh(x**2) + 0.01 * np.random.randn(n_samples)
    
    # Convert to torch tensors
    x = torch.FloatTensor(x.reshape(-1, 1))
    y = torch.FloatTensor(y.reshape(-1, 1))
    
    # Split into train/val
    n_val = int(n_samples * val_ratio)
    all_indices = np.arange(n_samples)
    val_indices = np.random.choice(all_indices, size=n_val, replace=False)
    train_indices = np.array([i for i in all_indices if i not in val_indices])
    
    return {
        'train': {'x': x[train_indices], 'y': y[train_indices]},
        'val': {'x': x[val_indices], 'y': y[val_indices]}
    } 