import logging
import pickle as pk
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = {
            "class_weight": "balanced",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_leaf": 0.25,
        }
        self.params.update(params)
        self.model = None
        self.scaler = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        logger.info("Start RandomForestClassifier fit...")
        self.scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = RandomForestClassifier(**self.params).fit(X_scaled, y.to_numpy())
        logger.info("End RandomForestClassifier fit")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        X_scaled = self.scaler.transform(X)
        probs_pred = self.model.predict(X_scaled)
        return pd.DataFrame(probs_pred, index=X.index, columns=["probs_pred"])

    def feature_importance(self):
        assert self.model is not None
        return np.copy(self.model.coef_).reshape(-1)

    def save(self, path):
        with open(path, "wb") as f:
            pk.dump(
                {"model": self.model, "scaler": self.scaler, "params": self.params}, f
            )

    def load(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.params = data["params"]
