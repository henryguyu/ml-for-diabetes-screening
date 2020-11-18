import os
import pickle

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils, models

results = {}
cache_file = os.path.join(cfg.root, "data/results.pk")
results = {}
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        results = pickle.load(f)


def get_cv_preds(
    model_name="LightGBMModel",
    feat_collection="without_FPG",
    update=False,
    out_FPG=False,
):
    # Load data
    X, y = data_utils.load_data(cfg.feature_fields[feat_collection])

    key = (model_name, feat_collection)
    if key not in results or update:
        # Load model
        params = cfg.model_params.get(key, {})
        print(f"Using params: {params}")
        model = getattr(models, model_name)(params=params)
        results[key] = model.cross_validate(X, y, metric_fn=metric_utils.roc_auc_score)
        save_cv_preds()
    cv_aucs, cv_probs_pred, cv_indices = results[key]

    cv_ys_gt = [y[idx] for idx in cv_indices]
    cv_y_prob = list(zip(cv_ys_gt, cv_probs_pred))
    if "FPG" in X:
        FPG = [X["FPG"][idx] for idx in cv_indices]
        for y_prob, subFPG in zip(cv_y_prob, FPG):
            y_prob[1][subFPG >= 7] = 1000

        if out_FPG:
            return cv_y_prob, FPG

    return cv_y_prob


def save_cv_preds():
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)


def reload_module(module):
    import importlib

    importlib.reload(module)
