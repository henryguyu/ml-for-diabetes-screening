from sklearn import metrics
from sklearn.metrics._ranking import _binary_clf_curve
import numpy as np


def precision_recall_curve(y_gt, probs_pred, *args, **kwargs):
    return metrics.precision_recall_curve(y_gt, probs_pred, *args, **kwargs)


def average_precision_score(y_gt, probs_pred, *args, **kwargs):
    return metrics.average_precision_score(y_gt, probs_pred, *args, **kwargs)


def roc_curve(y_gt, probs_pred, *args, **kwargs):
    return metrics.roc_curve(y_gt, probs_pred, *args, **kwargs)


def roc_auc_score(y_gt, probs_pred, *args, **kwargs):
    return metrics.roc_auc_score(y_gt, probs_pred, *args, **kwargs)


def nag_miss_curve(y_gt, probas_pred, *args, **kwargs):
    fps, tps, thresholds = _binary_clf_curve(y_gt, probas_pred, *args, **kwargs)
    tns = fps[-1] - fps
    fns = tps[-1] - tps

    nag_rate = (tns + fns) / (fps[-1] + tps[-1])
    # nag_rate = (tns) / (fps[0] + tns[0])
    recall = tps / tps[-1]
    miss_rate = 1 - recall

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[nag_rate[sl], 1], np.r_[miss_rate[sl], 1], thresholds[sl]
