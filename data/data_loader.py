import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        """Compute mean and std to be used for normalization"""
        if isinstance(values, np.ndarray):
            values = torch.FloatTensor(values)
            
        if values.dim() == 2:
            # For 2D arrays, compute stats along first dimension
            self.mean = torch.mean(values, dim=0)
            self.std = torch.std(values, dim=0)
        else:
            # For higher dimensional arrays
            dims = list(range(values.dim() - 1))
            self.mean = torch.mean(values, dim=dims)
            self.std = torch.std(values, dim=dims)

    def transform(self, values):
        """Normalize values using stored mean and std"""
        if isinstance(values, np.ndarray):
            values = torch.FloatTensor(values)
            is_numpy = True
        else:
            is_numpy = False

        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
            
        normalized = (values - mean) / (std + self.epsilon)
        
        if is_numpy:
            return normalized.detach().cpu().numpy()
        return normalized

    def fit_transform(self, values):
        """Compute mean, std and normalize values"""
        self.fit(values)
        return self.transform(values)
        
    def inverse_transform(self, normalized_values):
        """Transform normalized data back to original scale"""
        if isinstance(normalized_values, np.ndarray):
            normalized_values = torch.FloatTensor(normalized_values)
            is_numpy = True
        else:
            is_numpy = False
        
        # Ensure all tensors are on the same device
        device = normalized_values.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        original_values = normalized_values * (std + self.epsilon) + mean
        
        if is_numpy:
            return original_values.cpu().numpy()
        return original_values


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, seq_len, pred_len, target_column=None, scaled_data=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = dataframe.copy()
        self.target_column = target_column

        # Process datetime features
        if 'date' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Extract datetime features
            df_stamp = pd.DataFrame()
            df_stamp['hour'] = self.df.date.dt.hour
            df_stamp['day'] = self.df.date.dt.day
            df_stamp['month'] = self.df.date.dt.month
            df_stamp['year'] = self.df.date.dt.year
            df_stamp['dayofweek'] = self.df.date.dt.dayofweek
            self.data_stamp = df_stamp.values
            
            # Drop date column after extracting features
            self.df = self.df.drop(columns=['date'])

        # Select numeric columns for features
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        if scaled_data is not None:
            # Use pre-scaled data if provided
            self.features = scaled_data
            self.targets = scaled_data
        else:
            if target_column and target_column in self.df.columns:
                feature_cols = [col for col in numeric_cols if col != target_column]
                self.features = self.df[feature_cols].values.astype(float)
                self.targets = self.df[target_column].values.astype(float)
            else:
                self.features = self.df[numeric_cols].values.astype(float)
                self.targets = self.features

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len
        
        y_begin = x_end
        y_end = y_begin + self.pred_len
        
        x = self.features[x_begin:x_end]
        y = self.targets[y_begin:y_end]
        
        if hasattr(self, 'data_stamp'):
            seq_x_mark = self.data_stamp[x_begin:x_end]
            seq_y_mark = self.data_stamp[y_begin:y_end]
            return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)
        else:
            return torch.FloatTensor(x), torch.FloatTensor(y)


class TSDataLoader:
    def __init__(self, data_csv_path, seq_len=24, pred_len=24,
                 batch_size=32, device='cpu', train_val_test_split=[0.7, 0.2, 0.1],
                 target_column=None, scale=True):
        self.data_csv_path = data_csv_path
        self.batch_size = batch_size
        self.device = device
        self.train_val_test_split = train_val_test_split
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_column = target_column
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self._load_data()

    def _load_data(self):
        # Load and parse the CSV file
        df = pd.read_csv(self.data_csv_path)
        
        if self.train_val_test_split:
            # Split the data
            train_size = int(len(df) * self.train_val_test_split[0])
            val_size = int(len(df) * self.train_val_test_split[1])
            
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:train_size + val_size]
            test_df = df.iloc[train_size + val_size:]

            # Get numeric columns for scaling
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if self.target_column:
                numeric_cols = [col for col in numeric_cols if col != self.target_column]

            if self.scale:
                # Fit scaler on training data
                train_raw = train_df[numeric_cols].values.astype(float)
                self.scaler.fit(train_raw)
                
                # Transform all datasets
                train_scaled = self.scaler.transform(train_raw)
                val_scaled = self.scaler.transform(val_df[numeric_cols].values.astype(float))
                test_scaled = self.scaler.transform(test_df[numeric_cols].values.astype(float))
                
                # Create datasets with scaled data
                self.train_dataset = TimeSeriesDataset(train_df, self.seq_len, self.pred_len, 
                                                     self.target_column, scaled_data=train_scaled)
                self.val_dataset = TimeSeriesDataset(val_df, self.seq_len, self.pred_len, 
                                                   self.target_column, scaled_data=val_scaled)
                self.test_dataset = TimeSeriesDataset(test_df, self.seq_len, self.pred_len, 
                                                    self.target_column, scaled_data=test_scaled)
            else:
                # Create datasets without scaling
                self.train_dataset = TimeSeriesDataset(train_df, self.seq_len, self.pred_len, self.target_column)
                self.val_dataset = TimeSeriesDataset(val_df, self.seq_len, self.pred_len, self.target_column)
                self.test_dataset = TimeSeriesDataset(test_df, self.seq_len, self.pred_len, self.target_column)
        else:
            if self.scale:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if self.target_column:
                    numeric_cols = [col for col in numeric_cols if col != self.target_column]
                
                raw_data = df[numeric_cols].values.astype(float)
                scaled_data = self.scaler.fit_transform(raw_data)
                self.train_dataset = TimeSeriesDataset(df, self.seq_len, self.pred_len, 
                                                     self.target_column, scaled_data=scaled_data)
            else:
                self.train_dataset = TimeSeriesDataset(df, self.seq_len, self.pred_len, self.target_column)
            
            self.val_dataset = None
            self.test_dataset = None

    def get_data_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0
        )

        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=0
            )
        else:
            val_loader = None
            
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=0
            )
        else:
            test_loader = None

        return train_loader, val_loader, test_loader
