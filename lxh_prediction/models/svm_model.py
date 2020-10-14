import logging
import pickle as pk
from typing import Dict

from sklearn.svm import SVC
import numpy as np

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SVMModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = params
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
    ):
        logger.info("Start SVC fit...")
        self.model = SVC(**self.params).fit(X, y)
        logger.info("End SVC fit")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(X)

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
