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
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        # Define the feature columns
        self.feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Ensure all required features are present
        assert all(feat in df_raw.columns for feat in self.feature_names), \
            f"Missing features. Required features: {self.feature_names}"

        # Calculate borders
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[self.feature_names]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError('features must be either M, MS or S')

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
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), \
               torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TSDataLoader:
    def __init__(self, root_path, batch_size=32, size=None):
        self.root_path = root_path
        self.batch_size = batch_size
        self.size = size

    def get_data_loaders(self):
        train_dataset = TimeSeriesDataset(
            root_path=self.root_path,
            flag='train',
            size=self.size,
            features='M'  # Use multivariate features by default
        )

        val_dataset = TimeSeriesDataset(
            root_path=self.root_path,
            flag='val',
            size=self.size,
            features='M'  # Use multivariate features by default
        )

        test_dataset = TimeSeriesDataset(
            root_path=self.root_path,
            flag='test',
            size=self.size,
            features='M'  # Use multivariate features by default
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0
        )

        return train_loader, val_loader, test_loader
