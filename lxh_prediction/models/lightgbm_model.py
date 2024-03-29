import logging
import pickle as pk
from typing import Dict

import lightgbm as lgb
import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    def __init__(self, params: Dict = {}, metric="roc_auc_score"):
        metric_map = {
            "roc_auc_score": "auc",
            "average_precision_score": "average_precision",
        }

        self.params = {
            "boosting": "gbdt",
            "num_boost_round": 100,
            "metric": [metric_map[metric]],
            "early_stopping_round": 20,
            "objective": "binary",
            "num_threads": 1,
            "seed": 1063,
            "deterministic": True,
            "num_leaves": 9,
            "max_bin": 121,
            "max_depth": 64,
            "learning_rate": 0.06521362882101824,
            "lambda_l1": 0.1,
            "lambda_l2": 0.05,
            "feature_fraction": 0.5,
            "min_data_in_bin": 9,
            "bagging_fraction": 0.9,
            "bagging_freq": 4,
            "path_smooth": 0.1,
            "verbose": -1,
        }
        self.params.update(params)
        self.model = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        train_data = lgb.Dataset(X.to_numpy(), label=y.to_numpy())
        valid_sets = [train_data]
        if X_valid is not None:
            valid_sets.append(lgb.Dataset(X_valid.to_numpy(), label=y_valid.to_numpy()))

        logger.info("Start lgb.train...")
        self.model = lgb.train(
            self.params, train_data, valid_sets=valid_sets, verbose_eval=False
        )
        logger.info("lgb.train completed!")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        probs_pred = self.model.predict(X.to_numpy())
        return pd.DataFrame(probs_pred, index=X.index, columns=["probs_pred"])

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
