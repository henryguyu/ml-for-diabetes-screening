import logging
from copy import deepcopy
from typing import Dict
from ast import literal_eval

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lxh_prediction import metric_utils

from .base_model import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_optimizer(model, opt_name, lr, weight_decay):
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    elif opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(opt_name)
    return optimizer


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
        if isinstance(branches, str):
            branches = literal_eval(branches)

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
        self.dropout = nn.Dropout() if dropout else None

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


class ANNModel(BaseModel):
    def __init__(self, params: Dict = {}, feature_len=None, metric="roc_auc_score"):
        self.params = {
            "num_epoch": 60,
            "lr": 0.04186361307523874,
            "weight_decay": 0.0001,
            "batch_size": 105,
            "enable_lr_scheduler": 1,
            "opt": "Adam",
            "n_channels": 379,
            "n_layers": 5,
            "dropout": 1,
            "activate": "Sigmoid",
            "branches": [1, 0],
        }
        self.params.update(params)
        self.model = None
        if feature_len is not None:
            self.model = self._create_model(feature_len)
        self.metric_fn = getattr(metric_utils, metric)

    def _create_model(self, feature_len):
        return Model(
            feature_len=feature_len,
            n_channels=self.params["n_channels"],
            n_layers=self.params["n_layers"],
            params=self.params,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
        use_gpu=torch.cuda.is_available(),
    ):
        X = X.to_numpy().copy()
        y = y.to_numpy().copy()
        if X_valid is not None:
            X_valid = X_valid.to_numpy().copy()
            y_valid = y_valid.to_numpy().copy()
        params = self.params
        model = self._create_model(feature_len=X.shape[1])
        if use_gpu:
            model = model.cuda()

        optimizer = get_optimizer(
            model, params["opt"], params["lr"], params["weight_decay"]
        )
        lr_scheduler = (
            # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, params["num_epoch"], eta_min=0.0001
            )
            if params["enable_lr_scheduler"]
            else None
        )

        # dataset
        train_data = Dataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=params["batch_size"], shuffle=True, num_workers=1
        )
        valid_loader = None
        if X_valid is not None:
            valid_data = Dataset(X_valid, y_valid)
            valid_loader = torch.utils.data.DataLoader(
                valid_data,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=1,
            )

        # train
        best_epoch = None
        best_auc = None
        best_state_dict = None
        for epoch in tqdm(range(params["num_epoch"]), desc="Train"):
            self._train_one_epoch(model, optimizer, train_loader)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if not valid_loader:
                # logger.info(f"[{epoch}] train_loss: {train_loss:.3f}")
                pass
            else:
                valid_loss, valid_auc = self._valid_one_epoch(model, valid_loader)
                # logger.info(
                #     f"[{epoch}] train_loss: {train_loss:.3f} "
                #     f"test_loss: {valid_loss:.3f}, test_auc: {valid_auc:.4f}"
                # )
                # if lr_scheduler is not None:
                #     lr_scheduler.step(valid_auc)
                if best_auc is None or valid_auc > best_auc:
                    best_epoch = epoch
                    best_auc = valid_auc
                    best_state_dict = deepcopy(model.state_dict())

        if best_epoch is not None:
            logger.info(f"Best model: epoch {best_epoch}, auc: {best_auc}")
            model.load_state_dict(best_state_dict)
        self.model = model

    def _train_one_epoch(self, model, optimizer, train_loader: DataLoader):
        model.train()
        device = next(model.parameters()).device

        running_loss = 0
        counts = 0
        for step, batch in enumerate(train_loader):
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred_logits = model(X)
            loss = nn.functional.cross_entropy(pred_logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            counts += 1
        avg_loss = running_loss / counts
        return avg_loss

    def _valid_one_epoch(self, model, valid_loader: DataLoader):
        pred_logits, y_gt = self._predict_on_dataloader(model, valid_loader)
        loss = nn.functional.cross_entropy(pred_logits, y_gt)

        probs_pred = torch.softmax(pred_logits, dim=1).data.cpu().numpy()[:, 1]
        auc = self.metric_fn(y_gt.data.cpu().numpy(), probs_pred)
        return loss.item(), auc

    def _predict_on_dataloader(self, model: Model, dataloader: DataLoader):
        model.eval()
        pred_logits = []
        targets = []
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in iter(dataloader):
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                y_logits = model(X)
                pred_logits.append(y_logits.data)
                targets.append(y.data)
        return torch.cat(pred_logits), torch.cat(targets)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        dataloader = torch.utils.data.DataLoader(
            Dataset(X),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=1,
        )
        pred_logits, _ = self._predict_on_dataloader(self.model, dataloader)
        probs_pred = torch.softmax(pred_logits, dim=1).data.cpu().numpy()[:, 1]
        if isinstance(X, pd.DataFrame):
            probs_pred = pd.DataFrame(probs_pred, index=X.index, columns=["probs_pred"])
        return probs_pred

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path):
        device = next(self.model.parameters()).device
        data = torch.load(path, map_location=device)
        self.model.load_state_dict(data["state_dict"])
