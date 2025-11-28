"""
Data preprocessing for neural network training.
Handles feature encoding and dataset preparation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split


class PreflopDataPreprocessor:
    """
    Preprocesses CFR training data for neural network training.
    """

    def __init__(self):
        """Initialize the preprocessor with encoding mappings."""
        self.position_encoding = {'btn': 0, 'bb': 1}

        # Rank encoding (2=0, 3=1, ..., A=12)
        self.rank_encoding = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
        }

        # Action encoding for history
        self.action_encoding = {
            'fold': 0, 'check': 1, 'call': 2,
            'raise_2bb': 3, 'raise_3bb': 4, 'raise_4bb': 5, 'all_in': 6
        }

    def parse_hand(self, hand_str: str) -> Tuple[int, int, int]:
        """
        Parse a hand string into encoded features.

        Args:
            hand_str: Hand notation like 'AKs', 'QQ', '72o'

        Returns:
            Tuple of (rank1, rank2, is_suited)
        """
        if len(hand_str) < 2:
            raise ValueError(f"Invalid hand: {hand_str}")

        rank1 = self.rank_encoding[hand_str[0]]
        rank2 = self.rank_encoding[hand_str[1]]

        # Ensure rank1 >= rank2
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1

        # Check if suited
        if len(hand_str) == 2:
            # Pocket pair
            is_suited = 0  # Pairs are neither suited nor offsuit
        elif hand_str[2] == 's':
            is_suited = 1
        else:
            is_suited = 0

        return rank1, rank2, is_suited

    def encode_action_history(self, history_str: str, max_length: int = 10) -> np.ndarray:
        """
        Encode action history as a fixed-length sequence.

        Args:
            history_str: Action history like 'raise_2bb_call'
            max_length: Maximum sequence length

        Returns:
            Numpy array of encoded actions (padded/truncated to max_length)
        """
        # Handle NaN or empty values
        if pd.isna(history_str) or not history_str or history_str == '':
            actions = []
        else:
            actions = str(history_str).split('_')
            # Reconstruct multi-word actions
            reconstructed = []
            i = 0
            while i < len(actions):
                if i + 1 < len(actions) and actions[i] == 'raise':
                    # Combine 'raise' with next part
                    reconstructed.append(f"raise_{actions[i+1]}")
                    i += 2
                elif i + 1 < len(actions) and actions[i] == 'all':
                    reconstructed.append('all_in')
                    i += 2
                else:
                    reconstructed.append(actions[i])
                    i += 1
            actions = reconstructed

        # Encode actions
        encoded = [self.action_encoding.get(a, 0) for a in actions[:max_length]]

        # Pad with -1 (no action)
        while len(encoded) < max_length:
            encoded.append(-1)

        return np.array(encoded, dtype=np.int32)

    def preprocess_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a dataframe of training examples.

        Args:
            df: DataFrame with columns: position, hand, action_history, stack_bb,
                prob_fold, prob_check, prob_call, prob_raise_2bb, prob_raise_3bb,
                prob_raise_4bb, prob_all_in

        Returns:
            Tuple of (features, targets)
            - features: numpy array of shape (n_samples, n_features)
            - targets: numpy array of shape (n_samples, 7) for action probabilities
        """
        n_samples = len(df)

        # Feature dimensions:
        # - position: 1 (encoded as 0 or 1)
        # - rank1: 1 (0-12)
        # - rank2: 1 (0-12)
        # - is_suited: 1 (0 or 1)
        # - stack_bb: 1 (normalized)
        # - action_history: 10 (encoded sequence)
        # Total: 15 features
        features = np.zeros((n_samples, 15), dtype=np.float32)

        # Target dimensions: 7 action probabilities
        targets = np.zeros((n_samples, 7), dtype=np.float32)

        for i, row in df.iterrows():
            # Encode position
            features[i, 0] = self.position_encoding[row['position']]

            # Encode hand
            rank1, rank2, is_suited = self.parse_hand(row['hand'])
            features[i, 1] = rank1
            features[i, 2] = rank2
            features[i, 3] = is_suited

            # Normalize stack size (divide by 200 to get 0-1 range for typical stacks)
            features[i, 4] = row['stack_bb'] / 200.0

            # Encode action history
            history_encoded = self.encode_action_history(row['action_history'])
            features[i, 5:15] = history_encoded

            # Extract target probabilities
            targets[i, 0] = row['prob_fold']
            targets[i, 1] = row['prob_check']
            targets[i, 2] = row['prob_call']
            targets[i, 3] = row['prob_raise_2bb']
            targets[i, 4] = row['prob_raise_3bb']
            targets[i, 5] = row['prob_raise_4bb']
            targets[i, 6] = row['prob_all_in']

        return features, targets

    def load_and_split(self, csv_path: str, test_size: float = 0.2,
                       val_size: float = 0.1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load data from CSV and split into train/validation/test sets.

        Args:
            csv_path: Path to CSV file
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set

        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing
            (features, targets) tuple
        """
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} examples from {csv_path}")

        # Preprocess all data
        features, targets = self.preprocess_dataframe(df)

        # Split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )

        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=42
        )

        print(f"Train set: {len(X_train)} examples")
        print(f"Validation set: {len(X_val)} examples")
        print(f"Test set: {len(X_test)} examples")

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
