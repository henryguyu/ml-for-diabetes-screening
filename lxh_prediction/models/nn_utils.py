import torch
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, labels=None):
        self.X = X.to_numpy(float) if isinstance(X, pd.DataFrame) else X
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        X = torch.from_numpy(self.X[i]).float()
        y = int(self.labels[i] if self.labels is not None else -1)
        return X, y
