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
    def __init__(self, dataframe, seq_len, pred_len, target_column=None, scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = dataframe.copy()
        self.target_column = target_column
        self.scale = scale  # Store whether we scaled the data

        datetime_columns = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]) or isinstance(self.df[col].iloc[0], str):
                datetime_columns.append(col)

        self.data_stamp = None
        if datetime_columns:
            date_col = datetime_columns[0]
            if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                self.df[date_col] = pd.to_datetime(self.df[date_col])

            date_features = self.df[date_col].copy()
            self.df = self.df.drop(columns=datetime_columns)
            df_stamp = pd.DataFrame()
            df_stamp['date'] = date_features
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp['dayofweek'] = df_stamp.date.apply(
                lambda row: row.dayofweek, 1)
            self.data_stamp = df_stamp.drop(['date'], axis=1).values

        if target_column and target_column not in self.df.columns:
            self.features = self.df.values
            self.targets = self.features
        elif target_column:
            self.features = self.df.drop(columns=[target_column]).values
            self.targets = self.df[target_column].values
        else:
            self.features = self.df.values
            self.targets = self.features

        if scale:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = None

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len

        y_begin = x_end
        y_end = y_begin + self.pred_len

        x = self.features[x_begin:x_end]
        y = self.targets[y_begin:y_end]

        if self.data_stamp is not None:
            seq_x_mark = self.data_stamp[x_begin:x_end]
            seq_y_mark = self.data_stamp[y_begin:y_end]
            return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)
        else:
            return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, data):
        """Transform normalized data back to original scale"""
        if self.scale and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data


class TSDataLoader:
    def __init__(self, data_csv_path, seq_len=24, pred_len=24,
                 batch_size=32, device='cpu', train_val_test_split=[0.7, 0.2, 0.1],
                 shuffle=True, target_column=None, scale=True):
        self.data_csv_path = data_csv_path
        self.batch_size = batch_size
        self.device = device
        self.train_val_test_split = train_val_test_split
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_column = target_column
        self.scale = scale
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.data_csv_path)
        df = df[:2500]
        if self.train_val_test_split:
            train_size = int(len(df) * self.train_val_test_split[0])
            val_size = int(len(df) * self.train_val_test_split[1])
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:train_size + val_size]
            test_df = df.iloc[train_size + val_size:]

            self.train_dataset = TimeSeriesDataset(
                train_df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
            self.val_dataset = TimeSeriesDataset(
                val_df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
            self.test_dataset = TimeSeriesDataset(
                test_df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
        else:
            self.train_dataset = TimeSeriesDataset(
                df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
            self.val_dataset = None
            self.test_dataset = None

    def get_data_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
            num_workers=63
        )

        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=63
            )
            return train_loader, val_loader
        
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=63
            )
            return train_loader, test_loader

        return train_loader, None
