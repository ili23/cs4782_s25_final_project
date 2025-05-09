import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
import os


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
    def __init__(self, file_path, flag='train', size=None,
                 features=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.file_path = file_path
    
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.file_path, parse_dates=['date'])

        # Calculate borders
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        print("Border1s: ", border1s)
        print("Border2s: ", border2s)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[self.features]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # Implement time features encoding if needed
            data_stamp = df_stamp.drop(['date'], axis=1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # Input sequence (look back): [s_begin, s_end]
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # Output sequence (forecast): [r_begin, r_end]
        r_begin = s_end  # Start immediately after input sequence
        r_end = r_begin + self.pred_len  # Only need pred_len for output
        
        # Get data
        seq_x = self.data_x[s_begin:s_end]  # [seq_len, num_features]
        seq_y = self.data_y[r_begin:r_end]  # [pred_len, num_features]
        seq_x_mark = self.data_stamp[s_begin:s_end]  # [seq_len, num_features]
        seq_y_mark = self.data_stamp[r_begin:r_end]  # [pred_len, num_features]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), \
               torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)

    def __len__(self):
        # We need enough space for input sequence and prediction sequence
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TSDataLoader:
    def __init__(self, file_path, features, batch_size=32, size=None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.size = size
        self.features = features

    def get_data_loaders(self):
        train_dataset = TimeSeriesDataset(
            file_path=self.file_path,
            flag='train',
            size=self.size,
            features=self.features
        )

        val_dataset = TimeSeriesDataset(
            file_path=self.file_path,
            flag='val',
            size=self.size,
            features=self.features
        )

        test_dataset = TimeSeriesDataset(
            file_path=self.file_path,
            flag='test',
            size=self.size,
            features=self.features
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            drop_last=True
        )

        return train_loader, val_loader, test_loader
