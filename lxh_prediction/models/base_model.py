import numpy as np
from sklearn import metrics


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
