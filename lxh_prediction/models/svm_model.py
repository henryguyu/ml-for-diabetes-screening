import logging
import pickle as pk
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SVMModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = {"kernel": "linear"}
        self.params.update(params)
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        logger.info("Start SVC fit...")
        self.model = SVC(**self.params).fit(X.to_numpy(), y.to_numpy())
        logger.info("End SVC fit")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        probs_pred = self.model.predict(X.to_numpy())
        return pd.DataFrame(probs_pred, index=X.index, columns=["probs_pred"])

    def feature_importance(self):
        assert self.model is not None
        return np.copy(self.model.coef_).reshape(-1)

    def save(self, path):
        with open(path, "wb") as f:
            pk.dump({"model": self.model, "params": self.params}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
        self.model = data["model"]
        self.params = data["params"]
