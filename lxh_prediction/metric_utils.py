from typing import List, Tuple

import numpy as np
from scipy import interp
from sklearn import metrics
from sklearn.metrics._ranking import (_binary_clf_curve, column_or_1d,
                                      stable_cumsum)


def precision_recall_curve(y_gt, probs_pred, *args, **kwargs):
    return metrics.precision_recall_curve(y_gt, probs_pred, *args, **kwargs)


def average_precision_score(y_gt, probs_pred, *args, **kwargs):
    return metrics.average_precision_score(y_gt, probs_pred, *args, **kwargs)


def roc_curve(y_gt, probs_pred, *args, **kwargs):
    return metrics.roc_curve(y_gt, probs_pred, *args, **kwargs)


def roc_auc_score(y_gt, probs_pred, *args, **kwargs):
    return metrics.roc_auc_score(y_gt, probs_pred, *args, **kwargs)


def binary_clf_curve(y_gt, probas_pred, *args, **kwargs):
    fps, tps, thresholds = _binary_clf_curve(y_gt, probas_pred, *args, **kwargs)
    tns = fps[-1] - fps
    fns = tps[-1] - tps
    return tps, fps, tns, fns, thresholds


def nag_miss_curve(y_gt, probas_pred, *args, **kwargs):
    tps, fps, tns, fns, thresholds = binary_clf_curve(
        y_gt, probas_pred, *args, **kwargs
    )

    nag_rate = (tns + fns) / (fps[-1] + tps[-1])
    # nag_rate = (tns) / (fps[0] + tns[0])
    recall = tps / tps[-1]
    miss_rate = 1 - recall

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[nag_rate[sl], 1], np.r_[miss_rate[sl], 1], thresholds[sl]


def cost_curve_without_FPG(y_gt, probas_pred, *args, **kwargs):
    tps, fps, tns, fns, thresholds = binary_clf_curve(
        y_gt, probas_pred, *args, **kwargs
    )

    costs = (tps + fps) * 60.95 / (fps[-1] + tps[-1])
    recall = tps / tps[-1]
    miss_rate = 1 - recall

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return costs[sl], miss_rate[sl], thresholds[sl]


def cost_curve_with_FPG(y_gt, probas_pred, FPG, *args, **kwargs):
    def _binary_clf_curve_with_FPG(y_gt, probas_pred, y_FPG):
        y_gt = column_or_1d(y_gt)
        probas_pred = column_or_1d(probas_pred)
        y_FPG = column_or_1d(y_FPG)

        desc_score_indices = np.argsort(probas_pred, kind="mergesort")[::-1]
        y_score = probas_pred[desc_score_indices]
        y_gt = y_gt[desc_score_indices]
        y_FPG = y_FPG[desc_score_indices]

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_gt.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_gt)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        FPG_positives = stable_cumsum(y_FPG)[threshold_idxs]
        return fps, tps, FPG_positives, y_score[threshold_idxs]

    y_FPG = (FPG >= 7).astype(int)
    fps, tps, FPG_positives, thresholds = _binary_clf_curve_with_FPG(
        y_gt, probas_pred, y_FPG, *args, **kwargs
    )
    # tns = fps[-1] - fps
    # fns = tps[-1] - tps
    assert (fps[-1] + tps[-1]) == len(y_gt)
    costs = (tps - FPG_positives + fps) * 51.06 / len(y_gt) + 18.19
    recall = tps / tps[-1]
    miss_rate = 1 - recall

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return costs[sl], miss_rate[sl], thresholds[sl]


def mean_curve(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    x_range=(0, 1),
    y_range=(0, 1),
    num_x=101,
    reverse=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get mean curve: x_base, y_mean, y_lower, y_upper

    Args:
        xs (List[np.ndarray]): list of x
        ys (List[np.ndarray]): list of y

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            x_base, y_mean, y_lower, y_upper
    """
    x_base = np.linspace(0, 1, num_x)
    ys_interp = []
    for x, y in zip(xs, ys):
        if reverse:
            x = x[::-1]
            y = y[::-1]
        y = interp(x_base, x, y)
        ys_interp.append(y)
    ys = np.asarray(ys_interp)
    y_mean = ys.mean(axis=0)
    std = ys.std(axis=0)

    y_upper = np.minimum(y_mean + std, y_range[1])
    y_lower = np.maximum(y_mean - std, y_range[0])
    return x_base, y_mean, y_lower, y_upper
