"""
PyTorch dataset for preflop poker training.
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class PreflopDataset(Dataset):
    """
    PyTorch dataset for preflop poker strategy learning.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.

        Args:
            features: Numpy array of shape (n_samples, n_features)
            targets: Numpy array of shape (n_samples, n_actions)
        """
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, targets)
        """
        return self.features[idx], self.targets[idx]
