import logging
from typing import Callable

import numpy as np
import pandas as pd

from lxh_prediction.data_utils import split_cross_validation

logger = logging.getLogger(__name__)


class BaseModel:
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_valid: pd.DataFrame = None,
        y_valid: pd.DataFrame = None,
    ):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def feature_importance(self):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_folds=5,
    ):
        metrics = []
        cv_indices = []
        cv_probs_pred = []
        for i, batch in enumerate(split_cross_validation(X, y, n_folds=n_folds)):
            logger.info(f"Cross validation: round {i}")
            X_train, y_train, X_test, y_test, = batch
            self.fit(X_train, y_train, X_test, y_test)

            probs_pred = self.predict(X_test)
            metrics.append(metric_fn(y_test.to_numpy(), probs_pred.to_numpy()))

            cv_indices.append(X_test.index)
            cv_probs_pred.append(probs_pred)
        return metrics, cv_probs_pred, cv_indices
