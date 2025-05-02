import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional

class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for time series data.
    """
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        feature_cols: List[int] = None,
        target_cols: List[int] = None,
        normalize: bool = True,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ):
        """
        Initialize the TimeSeriesDataset.
        
        Args:
            data: The time series data
            seq_len: Sequence/lookback length
            pred_len: Prediction horizon length
            feature_cols: Column indices to use as features
            target_cols: Column indices to use as targets (if None, same as feature_cols)
            normalize: Whether to normalize the data
            mean: Pre-computed mean for normalization
            std: Pre-computed standard deviation for normalization
        """
        # Store data with shape [samples, features]
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Handle feature and target columns
        if feature_cols is None:
            self.feature_cols = list(range(data.shape[1]))
        else:
            self.feature_cols = feature_cols
            
        if target_cols is None:
            self.target_cols = self.feature_cols
        else:
            self.target_cols = target_cols
            
        # Extract features and targets
        self.features = self.data[:, self.feature_cols]
        self.targets = self.data[:, self.target_cols]
        
        # Normalize if required
        self.normalize = normalize
        if normalize:
            if mean is None or std is None:
                self.mean = np.mean(self.features, axis=0)
                self.std = np.std(self.features, axis=0)
                # Avoid division by zero
                self.std = np.where(self.std == 0, 1, self.std)
            else:
                self.mean = mean
                self.std = std
                
            self.features = (self.features - self.mean) / self.std
        
        # Calculate effective length
        self.effective_len = len(self.data) - self.seq_len - self.pred_len + 1

    def __len__(self):
        return self.effective_len

    def __getitem__(self, idx):
        # Get sequence
        x_start = idx
        x_end = idx + self.seq_len
        x = self.features[x_start:x_end]
        
        # Get target
        y_start = x_end
        y_end = y_start + self.pred_len
        y = self.targets[y_start:y_end]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def get_normalization_stats(self):
        """Return normalization statistics (mean and std)"""
        if self.normalize:
            return self.mean, self.std
        else:
            return None, None


class TimeSeriesDataLoader:
    """
    Data loader for time series datasets
    """
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        seq_len: int = 96,
        pred_len: int = 24,
        feature_cols: List[int] = None,
        target_cols: List[int] = None,
        normalize: bool = True,
        split_ratio: List[float] = [0.7, 0.1, 0.2],
        shuffle: bool = True,
    ):
        """
        Initialize the TimeSeriesDataLoader.
        
        Args:
            data_path: Path to the CSV data file
            batch_size: Batch size for DataLoader
            seq_len: Sequence/lookback length
            pred_len: Prediction horizon length
            feature_cols: Column indices to use as features
            target_cols: Column indices to use as targets
            normalize: Whether to normalize the data
            split_ratio: Train/val/test split ratio
            shuffle: Whether to shuffle the data
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.normalize = normalize
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        
        # Read and preprocess data
        self._load_data()
        self._create_datasets()
        self._create_dataloaders()
        
    def _load_data(self):
        """Load and preprocess data from CSV file"""
        # Read CSV file
        df = pd.read_csv(self.data_path)
        
        # Drop date column if exists
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
            
        # Convert to numpy
        self.data = df.values
        
    def _create_datasets(self):
        """Create train, validation, and test datasets"""
        # Calculate split indices
        total_len = len(self.data)
        train_end = int(total_len * self.split_ratio[0])
        val_end = train_end + int(total_len * self.split_ratio[1])
        
        # Split data
        train_data = self.data[:train_end]
        val_data = self.data[train_end:val_end]
        test_data = self.data[val_end:]
        
        # Create datasets
        self.train_dataset = TimeSeriesDataset(
            train_data, self.seq_len, self.pred_len,
            self.feature_cols, self.target_cols, self.normalize
        )
        
        # Get normalization stats from training set
        mean, std = self.train_dataset.get_normalization_stats()
        
        # Use same normalization for val and test
        self.val_dataset = TimeSeriesDataset(
            val_data, self.seq_len, self.pred_len,
            self.feature_cols, self.target_cols, self.normalize, mean, std
        )
        
        self.test_dataset = TimeSeriesDataset(
            test_data, self.seq_len, self.pred_len,
            self.feature_cols, self.target_cols, self.normalize, mean, std
        )
        
    def _create_dataloaders(self):
        """Create DataLoader objects"""
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, 
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, 
            shuffle=False
        )
        
    def get_dataloaders(self):
        """Return train, validation, and test dataloaders"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_datasets(self):
        """Return train, validation, and test datasets"""
        return self.train_dataset, self.val_dataset, self.test_dataset


class DataLoaderFactory:
    """
    Factory class to create appropriate data loaders for different datasets.
    """
    @staticmethod
    def create_dataloader(
        dataset_name: str,
        batch_size: int = 32,
        seq_len: int = 96,
        pred_len: int = 24,
        base_path: str = "data/data_files",
        normalize: bool = True,
        **kwargs
    ) -> TimeSeriesDataLoader:
        """
        Create a data loader for the specified dataset.
        
        Args:
            dataset_name: Name of the dataset ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                          'electricity', 'traffic', 'weather', 'illness', 'exchange_rate')
            batch_size: Batch size for DataLoader
            seq_len: Sequence/lookback length
            pred_len: Prediction horizon length
            base_path: Base path to the data directory
            normalize: Whether to normalize the data
            **kwargs: Additional arguments for specific datasets
            
        Returns:
            A TimeSeriesDataLoader instance for the specified dataset
        """
        # ETT datasets
        if dataset_name.startswith('ETT'):
            data_path = os.path.join(base_path, 'ETT-small', f"{dataset_name}.csv")
            
            # Define feature and target columns based on ETT format
            # ETT columns are date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
            # OT is usually the target
            feature_cols = kwargs.get('feature_cols', list(range(1, 7)))  # Exclude date (0) and OT (-1)
            target_cols = kwargs.get('target_cols', [7])  # OT column
            
            return TimeSeriesDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                feature_cols=feature_cols,
                target_cols=target_cols,
                normalize=normalize,
                **kwargs
            )
            
        # Electricity dataset
        elif dataset_name == 'electricity':
            data_path = os.path.join(base_path, 'electricity', 'electricity.csv')
            
            # First column is date, last is OT
            num_cols = pd.read_csv(data_path, nrows=1).shape[1]
            feature_cols = kwargs.get('feature_cols', list(range(1, num_cols-1)))  # Exclude date and OT
            target_cols = kwargs.get('target_cols', [num_cols-1])  # OT column
            
            return TimeSeriesDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                feature_cols=feature_cols,
                target_cols=target_cols,
                normalize=normalize,
                **kwargs
            )
            
        # Traffic dataset
        elif dataset_name == 'traffic':
            data_path = os.path.join(base_path, 'traffic', 'traffic.csv')
            
            # First column is date, last is OT
            num_cols = pd.read_csv(data_path, nrows=1).shape[1]
            feature_cols = kwargs.get('feature_cols', list(range(1, num_cols-1)))  # Exclude date and OT
            target_cols = kwargs.get('target_cols', [num_cols-1])  # OT column
            
            return TimeSeriesDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                feature_cols=feature_cols,
                target_cols=target_cols,
                normalize=normalize,
                **kwargs
            )
            
        # Weather dataset
        elif dataset_name == 'weather':
            data_path = os.path.join(base_path, 'weather', 'weather.csv')
            
            # Weather has multiple features
            # Last column (OT) is usually the target
            num_cols = pd.read_csv(data_path, nrows=1).shape[1]
            feature_cols = kwargs.get('feature_cols', list(range(1, num_cols-1)))  # Exclude date and OT
            target_cols = kwargs.get('target_cols', [num_cols-1])  # OT column
            
            return TimeSeriesDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                feature_cols=feature_cols,
                target_cols=target_cols,
                normalize=normalize,
                **kwargs
            )
            
        # Illness dataset
        elif dataset_name == 'illness':
            data_path = os.path.join(base_path, 'illness', 'national_illness.csv')
            
            # Illness dataset specific columns
            # Target is usually % WEIGHTED ILI (column 1)
            feature_cols = kwargs.get('feature_cols', list(range(1, 8)))  # All features except date
            target_cols = kwargs.get('target_cols', [1])  # % WEIGHTED ILI
            
            return TimeSeriesDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                feature_cols=feature_cols,
                target_cols=target_cols,
                normalize=normalize,
                **kwargs
            )
            
        # Exchange rate dataset
        elif dataset_name == 'exchange_rate':
            data_path = os.path.join(base_path, 'exchange_rate', 'exchange_rate.csv')
            
            # Exchange rate dataset structure
            # First column is date
            num_cols = pd.read_csv(data_path, nrows=1).shape[1]
            feature_cols = kwargs.get('feature_cols', list(range(1, num_cols)))  # All features except date
            target_cols = kwargs.get('target_cols', list(range(1, num_cols)))  # All features as target
            
            return TimeSeriesDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                seq_len=seq_len,
                pred_len=pred_len,
                feature_cols=feature_cols,
                target_cols=target_cols,
                normalize=normalize,
                **kwargs
            )
            
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")


# Example usage:
# factory = DataLoaderFactory()
# train_loader, val_loader, test_loader = factory.create_dataloader(
#     dataset_name='ETTh1',
#     batch_size=32,
#     seq_len=96,
#     pred_len=24
# ).get_dataloaders()