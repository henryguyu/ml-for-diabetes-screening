import logging
import pickle as pk
from typing import Dict

import pandas as pd
from sklearn.decomposition import KernelPCA

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class PCAModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = params
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        logger.info("Start KernelPCA fit...")
        self.model = KernelPCA(**self.params).fit(X.to_numpy(), y.to_numpy())
        logger.info("End KernelPCA fit")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        probs_pred = self.model.transform(X)
        return pd.DataFrame(
            probs_pred,
            index=X.index,
            columns=[f"pca_feat_{i}" for i in range(probs_pred.shape[1])],
        )

    def save(self, path):
        with open(path, "wb") as f:
            pk.dump({"model": self.model, "params": self.params}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
        self.model = data["model"]
        self.params = data["params"]
