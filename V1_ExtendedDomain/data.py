import torch
import numpy as np
from typing import Tuple

def split_data(x: torch.Tensor, y: torch.Tensor, 
               train_ratio=0.7, val_ratio=0.15) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                                       Tuple[torch.Tensor, torch.Tensor],
                                                       Tuple[torch.Tensor, torch.Tensor]]:
    """Split data into training, validation and test sets."""
    n = len(x)
    indices = torch.randperm(n)
    
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return (x[train_indices], y[train_indices]), \
           (x[val_indices], y[val_indices]), \
           (x[test_indices], y[test_indices])

def get_nonlinear_data(x_min, x_max, n_samples=1000, val_ratio=0.1, noise_std=0.1):
    """Generate nonlinear data within specified x range with validation ratio."""
    # Generate x values
    x = np.linspace(x_min, x_max, n_samples)
    
    # Generate y values with nonlinear function and noise
    y = np.sin(2*x) * np.cos(3*x) + 0.2 * np.tanh(x**2) + noise_std * np.random.randn(n_samples)
    
    # Convert to torch tensors
    x = torch.FloatTensor(x.reshape(-1, 1))
    y = torch.FloatTensor(y.reshape(-1, 1))
    
    # Calculate number of validation samples (10%)
    n_val = int(n_samples * val_ratio)
    
    # Randomly select validation indices
    all_indices = np.arange(n_samples)
    val_indices = np.random.choice(all_indices, size=n_val, replace=False)
    train_indices = np.array([i for i in all_indices if i not in val_indices])
    
    return {
        'train': {
            'x': x[train_indices],
            'y': y[train_indices]
        },
        'val': {
            'x': x[val_indices],
            'y': y[val_indices]
        }
    }

def get_dataset(dataset_type: str, n_points: int = 100):
    """Factory function to get different types of datasets."""
    datasets = {
        'linear': lambda: generate_linear_data(n_points=n_points),
        'nonlinear': lambda: generate_nonlinear_data(n_points=n_points),
        'classification': lambda: generate_classification_data(n_points=n_points)
    }
    return datasets[dataset_type]() 