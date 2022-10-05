from torch.utils.data import Dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, fpath, target_column, **pandas_args):
        self.data = pd.read_csv(fpath, **pandas_args)
        self.y = self.data[target_column]
        self.X = self.data.drop(target_column, axis=1)

    def __getitem__(self, item):
        return self.X.iloc[item], self.y.iloc[item]

    def get_numpy(self):
        return self.X.values, self.y.values


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)