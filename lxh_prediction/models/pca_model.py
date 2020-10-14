import logging
import pickle as pk
from typing import Dict

from sklearn.decomposition import KernelPCA
import numpy as np

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class PCAModel(BaseModel):
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
        logger.info("Start KernelPCA fit...")
        self.model = KernelPCA(**self.params).fit(X, y)
        logger.info("End KernelPCA fit")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return self.model.transform(X)

    def save(self, path):
        with open(path, "wb") as f:
            pk.dump({"model": self.model, "params": self.params}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
        self.model = data["model"]
        self.params = data["params"]
