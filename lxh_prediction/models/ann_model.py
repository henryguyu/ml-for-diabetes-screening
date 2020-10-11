import logging
from typing import Dict
from copy import deepcopy

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .nn_utils import Model, Dataset
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ANNModel(BaseModel):
    def __init__(self, params: Dict, feature_len=None):
        self.params = params
        self.model = None
        if feature_len is not None:
            self.model = Model(feature_len=feature_len)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        use_gpu=torch.cuda.is_available(),
    ):
        params = self.params
        model = Model(feature_len=X.shape[1])
        if use_gpu:
            model = model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"],
        )
        lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
            if self.params["enable_lr_scheduler"]
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
        for epoch in range(params["num_epoch"]):
            train_loss = self._train_one_epoch(model, optimizer, train_loader)
            logger.info(f"[{epoch}] train_loss: {train_loss}")
            if valid_loader:
                valid_loss, valid_auc = self._valid_one_epoch(model, valid_loader)
                logger.info(f"[{epoch}] test_loss: {valid_loss}, test_auc: {valid_auc}")
                if lr_scheduler is not None:
                    lr_scheduler.step(valid_auc)
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
        auc = metrics.roc_auc_score(
            y_gt.data.cpu().numpy(), probs_pred, average="macro"
        )
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        dataloader = torch.utils.data.DataLoader(
            Dataset(X,),
            batch_size=self.params["batch_size"],
            shuffle=False,
            num_workers=1,
        )
        pred_logits, _ = self._predict_on_dataloader(self.model, dataloader)
        probs_pred = torch.softmax(pred_logits, dim=1).data.cpu().numpy()[:, 1]
        return probs_pred

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path):
        device = next(self.model.parameters()).device
        data = torch.load(path, map_location=device)
        self.model.load_state_dict(data["state_dict"])
