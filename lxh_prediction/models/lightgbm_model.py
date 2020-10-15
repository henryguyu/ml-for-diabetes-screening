import logging
import pickle as pk
from typing import Dict

import lightgbm as lgb
import numpy as np

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = {
            "boosting": "gbdt",
            "num_boost_round": 100,
            "metric": ["auc"],
            "early_stopping_round": 20,
            "objective": "binary",
        }
        self.params.update(params)
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
    ):
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        if X_valid is not None:
            valid_sets.append(lgb.Dataset(X_valid, label=y_valid))

        logger.info("Start lgb.train...")
        self.model = lgb.train(self.params, train_data, valid_sets=valid_sets)
        logger.info("lgb.train completed!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)

    def feature_importance(self):
        assert self.model is not None
        return self.model.feature_importance()

    def save(self, path):
        with open(path, "wb") as f:
            pk.dump({"model": self.model, "params": self.params}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
        self.model = data["model"]
        self.params = data["params"]
