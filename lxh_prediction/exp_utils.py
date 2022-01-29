import os
import pickle
from typing import List, Tuple

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils, models

results = {}
cache_file = os.path.join(cfg.root, "data/results.pk")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        results = pickle.load(f)


def get_cv_preds(
    model_name="LightGBMModel",
    feat_collection="without_FPG",
    update=False,
    out_tests=False,
    resample_train=False,
):
    # Load data
    X, y = data_utils.load_data(cfg.feature_fields[feat_collection])

    key = (model_name, feat_collection)
    if key not in results or update:
        # Load model
        params = cfg.model_params.get(key, {})
        print(f"Using params: {params}")
        model = getattr(models, model_name)(params=params)
        results[key] = model.cross_validate(
            X, y, metric_fn=metric_utils.roc_auc_score, resample_train=resample_train
        )
        save_cv_preds()
    cv_aucs, cv_probs_pred, cv_indices = results[key]

    cv_ys_gt = [y[idx] for idx in cv_indices]
    cv_y_prob: List[Tuple[float, float]] = list(zip(cv_ys_gt, cv_probs_pred))

    # identifiable by Tests
    tests = {"FPG": 7, "P2hPG": 11.1, "HbA1c": 6.5}
    for test_name, thresh in tests.items():
        if test_name in X:
            test_values = [X[test_name][idx] for idx in cv_indices]
            if test_name != "HbA1c" or "ADA" in cfg.label_field:
                for y_prob, subvalue in zip(cv_y_prob, test_values):
                    y_prob[1][subvalue >= thresh] = 1000
            break
    else:
        test_name, test_values = None, None

    if out_tests:
        return cv_y_prob, test_name, test_values
    return cv_y_prob


def save_cv_preds():
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
