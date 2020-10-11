import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, feature_len):
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(feature_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, X):
        return self.fcs(X)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, labels=None):
        self.X = X
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        X = torch.from_numpy(self.X[i]).float()
        y = int(self.labels[i] if self.labels is not None else -1)
        return X, y
