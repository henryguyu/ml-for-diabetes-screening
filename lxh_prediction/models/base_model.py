import logging
from typing import Callable
import numpy as np
from sklearn import metrics

from lxh_prediction.data_utils import split_cross_validation

logger = logging.getLogger(__name__)


class BaseModel:
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
    ):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_folds=5,
    ):
        metrics = []
        for i, batch in enumerate(split_cross_validation(X, y, n_folds=n_folds)):
            logger.info(f"Cross validation: round {i}")
            X_train, y_train, X_test, y_test = batch
            self.fit(X_train, y_train, X_test, y_test)
            probs_pred = self.predict(X_test)
            metrics.append(metric_fn(y_test, probs_pred))
        return metrics

    @staticmethod
    def precision_recall_curve(y_gt, probs_pred, *args, **kwargs):
        return metrics.precision_recall_curve(y_gt, probs_pred, *args, **kwargs)

    @staticmethod
    def average_precision_score(y_gt, probs_pred, *args, **kwargs):
        return metrics.average_precision_score(y_gt, probs_pred, *args, **kwargs)

    @staticmethod
    def roc_curve(y_gt, probs_pred, *args, **kwargs):
        return metrics.roc_curve(y_gt, probs_pred, *args, **kwargs)

    @staticmethod
    def roc_auc_score(y_gt, probs_pred, *args, **kwargs):
        return metrics.roc_auc_score(y_gt, probs_pred, *args, **kwargs)
