import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, seq_len, pred_len, target_column=None, scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.df = dataframe.copy()
        self.target_column = target_column

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

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len

        y_begin = x_end
        y_end = y_begin + self.pred_len

        x = self.features[x_begin:x_end]
        y = self.targets[y_begin:y_end]

        # Handle case where y is a single value (not an array)
        if not hasattr(y, "__len__"):
            y = np.array([y])

        # Handle time features
        if self.data_stamp is not None:
            seq_x_mark = self.data_stamp[x_begin:x_end]
            seq_y_mark = self.data_stamp[y_begin:y_end]
            return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)
        else:
            return torch.FloatTensor(x), torch.FloatTensor(y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class TSDataLoader:
    def __init__(self, data_csv_path, seq_len=24, pred_len=24,
                 batch_size=32, device='cpu', train_val_split=True,
                 shuffle=True, target_column=None, scale=True,
                 val_ratio=0.2):
        self.data_csv_path = data_csv_path
        self.batch_size = batch_size
        self.device = device
        self.train_val_split = train_val_split
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_column = target_column
        self.scale = scale
        self.val_ratio = val_ratio
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.data_csv_path)
        if self.train_val_split:
            train_size = int(len(df) * (1 - self.val_ratio))
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:]

            self.train_dataset = TimeSeriesDataset(
                train_df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
            self.val_dataset = TimeSeriesDataset(
                val_df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
        else:
            self.train_dataset = TimeSeriesDataset(
                df, self.seq_len, self.pred_len, self.target_column, self.scale
            )
            self.val_dataset = None

    def get_data_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True
        )

        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True
            )
            return train_loader, val_loader

        return train_loader
