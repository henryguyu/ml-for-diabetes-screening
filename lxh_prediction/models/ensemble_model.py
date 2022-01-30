import logging
import pickle as pk
from typing import Dict

from autogluon.tabular import TabularPredictor
import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    LABEL_NAME = "label"

    def __init__(
        self,
        params: Dict = {},
        metric="roc_auc_score",
        train_models=("GBM", "RF", "LR", "NN"),
        predict_model="WeightedEnsemble_L2",
        fit_params: Dict = None,
        tune_on_valid=True,
    ):
        metric = metric.replace("_score", "")
        self.params = {
            "label": self.LABEL_NAME,
            "eval_metric": metric,
            "problem_type": "binary",
        }
        self.params.update(params)
        self.model = None
        self.train_models = train_models
        self.predict_model = predict_model
        self.fit_params = fit_params or {}
        self.tune_on_valid = tune_on_valid

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        train_data = X.copy()
        train_data[self.LABEL_NAME] = y
        valid_data = None
        if self.tune_on_valid:
            valid_data = X_valid.copy()
            valid_data[self.LABEL_NAME] = y_valid

        logger.info("Start TabularPredictor.fit...")
        self.model = TabularPredictor(**self.params)
        self.model.fit(
            train_data,
            valid_data,
            hyperparameters={name: {} for name in self.train_models},
            **self.fit_params
        )
        logger.info("TabularPredictor.fit completed!")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        probs_pred = self.model.predict_proba(X, model=self.predict_model).to_numpy()
        return pd.DataFrame(probs_pred[:, 1], index=X.index, columns=["probs_pred"])

    def save(self, path):
        with open(path, "wb") as f:
            pk.dump({"model": self.model, "params": self.params}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pk.load(f)
        self.model = data["model"]
        self.params = data["params"]


class AutoLightGBMModel(EnsembleModel):
    def __init__(self, params: Dict = {}, metric="roc_auc_score"):
        super().__init__(
            params,
            metric,
            train_models=["GBM"],
            predict_model=None,
            fit_params={"presets": "best_quality"},
            tune_on_valid=False,
        )
