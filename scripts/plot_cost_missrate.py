# %%
import os
import logging
import pickle

import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils, models
from lxh_prediction.plot import plot_curve, plot_range, plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# %%
results = {}
cache_file = os.path.join(cfg.root, "data/results.pk")
results = {}
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        results = pickle.load(f)


def get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG"):
    # Load data
    X, y = data_utils.load_data(cfg.feature_fields[feat_collection])

    key = (model_name, feat_collection)
    if key not in results:
        # Load model
        model = getattr(models, model_name)()
        results[key] = model.cross_validate(X, y, metric_fn=metric_utils.roc_auc_score)
    cv_aucs, cv_probs_pred, cv_indices = results[key]

    cv_ys_gt = [y[idx] for idx in cv_indices]
    cv_y_prob = list(zip(cv_ys_gt, cv_probs_pred))
    if "FPG" in X:
        FPG = [X["FPG"][idx] for idx in cv_indices]
        return cv_y_prob, FPG

    return cv_y_prob


# %%
def mean_cost_missrate(cv_y_prob, FPG=None):
    ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
    probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
    if FPG is not None:
        FPG = np.concatenate(FPG)
        cost, missrate, _ = metric_utils.cost_curve_with_FPG(ys, probs, FPG)
    else:
        cost, missrate, _ = metric_utils.cost_curve_without_FPG(ys, probs)
    return cost[-1], missrate[-1]


# %%
# ROC without FPG

fig = plt.figure(figsize=(6, 6))

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA")
cost, miss_rate = mean_cost_missrate(cv_y_prob)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="^", label="ADA (no-lab)")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH")
cost, miss_rate = mean_cost_missrate(cv_y_prob)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="s", label="CDS (no-lab)")
# plt.annotate(f"({fpr:.3f}, {tpr:.3f})", (fpr + 0.02, tpr))

# ANN
cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="without_FPG")
costs, miss_rates, _ = zip(
    *(metric_utils.cost_curve_without_FPG(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    miss_rates, costs, y_range=(0, 70)
)
plot_curve(
    x_base, y_mean, ylim=(0, 70), name="ANN (no-lab)", color="royalblue",
)
plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG")
costs, miss_rates, _ = zip(
    *(metric_utils.cost_curve_without_FPG(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    miss_rates, costs, y_range=(0, 70)
)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 70),
    name="LGBM (no-lab)",
    xlabel="Prediction miss rate",
    ylabel="Average cost per patient",
)
plot_range(x_base, y_lower, y_upper)
plt.legend(loc="upper right")


# %%
# ROC without FPG
fig = plt.figure(figsize=(6, 6))

# ADA
cv_y_prob, FPG = get_cv_preds(model_name="ADAModel", feat_collection="ADA_FPG")
cost, miss_rate = mean_cost_missrate(cv_y_prob, FPG)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="^", label="ADA")

# CDS
cv_y_prob, FPG = get_cv_preds(model_name="CHModel", feat_collection="CH_FPG")
cost, miss_rate = mean_cost_missrate(cv_y_prob, FPG)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="s", label="CDS")
# plt.annotate(f"({fpr:.3f}, {tpr:.3f})", (fpr + 0.02, tpr))

# ANN
cv_y_prob, FPG = get_cv_preds(model_name="ANNModel", feat_collection="with_FPG")
costs, miss_rates, _ = zip(
    *(
        metric_utils.cost_curve_with_FPG(ys, probs, FPG[i])
        for i, (ys, probs) in enumerate(cv_y_prob)
    )
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    miss_rates, costs, y_range=(0, 70)
)
plot_curve(
    x_base, y_mean, ylim=(0, 70), name="ANN", color="royalblue",
)
plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob, FPG = get_cv_preds(model_name="LightGBMModel", feat_collection="with_FPG")
costs, miss_rates, _ = zip(
    *(
        metric_utils.cost_curve_with_FPG(ys, probs, FPG[i])
        for i, (ys, probs) in enumerate(cv_y_prob)
    )
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    miss_rates, costs, y_range=(0, 70)
)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 70),
    name="LGBM",
    xlabel="Prediction miss rate",
    ylabel="Average cost per patient",
)
plot_range(x_base, y_lower, y_upper)
plt.legend(loc="upper right")

# %%
with open(cache_file, "wb") as f:
    pickle.dump(results, f)
# %%
