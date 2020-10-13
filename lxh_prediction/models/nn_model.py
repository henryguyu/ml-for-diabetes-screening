import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = None
        if in_features != out_features:
            self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        if self.fc is not None:
            x = self.fc(x)
        return x


class BaseFC(nn.Module):
    def __init__(
        self, in_features, out_features, activate="ReLU", bn=True, dropout=False
    ):
        super().__init__()
        modules = [nn.Linear(in_features, out_features)]
        if bn:
            modules.append(nn.BatchNorm1d(out_features))
        modules.append(getattr(nn, activate)())
        if dropout:
            modules.append(nn.Dropout())
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class Cell(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activate="ReLU",
        bn=True,
        dropout=False,
        branches=(1,),
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for n_nodes in branches:
            if n_nodes == 0:
                self.branches.append(Identity(in_features, out_features))
                continue
            modules = [BaseFC(in_features, out_features, activate, bn)] + [
                BaseFC(out_features, out_features, activate, bn)
                for _ in range(n_nodes - 1)
            ]
            self.branches.append(nn.Sequential(*modules))
        self.dropout = nn.Dropout if dropout else None

    def forward(self, x):
        x = sum(branch(x) for branch in self.branches)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, feature_len, n_channels=256, n_layers=6, params={}):
        super().__init__()
        modules = [Cell(feature_len, n_channels)]
        for i in range(1, n_layers):
            dropout = params.get("dropout", False) and i == n_layers - 1
            modules.append(
                Cell(
                    n_channels,
                    n_channels,
                    activate=params.get("activate", "ReLU"),
                    bn=params.get("bn", True),
                    dropout=dropout,
                    branches=params.get("branches", (1,)),
                )
            )
        self.fcs = nn.Sequential(*modules)
        self.classifier = nn.Linear(n_channels, 2)

    def forward(self, x):
        x = self.fcs(x)
        logits = self.classifier(x)
        return logits
