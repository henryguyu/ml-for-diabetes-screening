import logging
from copy import deepcopy
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lxh_prediction import metric_utils

from .base_model import BaseModel
from .nn_model import Model
from .nn_utils import Dataset

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
