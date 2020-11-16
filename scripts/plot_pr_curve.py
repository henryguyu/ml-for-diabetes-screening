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
    return cv_y_prob


# %%
def mean_precision_recall(cv_y_prob):
    ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
    probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
    precision = np.mean(ys[probs > 0] > 0)
    recall = np.mean(probs[ys > 0] > 0)
    return precision, recall


# %%
# ROC without FPG

fig = plt.figure(figsize=(6, 6))

# ANN
cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="without_FPG")
precisions, recalls, _ = zip(
    *(metric_utils.precision_recall_curve(ys, probs) for ys, probs in cv_y_prob)
)
aps = np.asarray(
    [metric_utils.average_precision_score(ys, probs) for ys, probs in cv_y_prob]
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    recalls, precisions, reverse=True
)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"ANN (no-lab). mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
    color="royalblue",
)
plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG")
precisions, recalls, _ = zip(
    *(metric_utils.precision_recall_curve(ys, probs) for ys, probs in cv_y_prob)
)
aps = np.asarray(
    [metric_utils.average_precision_score(ys, probs) for ys, probs in cv_y_prob]
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    recalls, precisions, reverse=True
)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"LGBM (no-lab). mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
)
plot_range(x_base, y_lower, y_upper)

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA")
p, r = mean_precision_recall(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")

# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(r, p, marker="^", label="ADA (no-lab).")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH")
p, r = mean_precision_recall(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")
# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(r, p, marker="s", label="CDS (no-lab).")
# plt.annotate(f"({fpr:.3f}, {tpr:.3f})", (fpr + 0.02, tpr))

# Random
plot_curve(
    (0, 1),
    (0, 0),
    ylim=(-0.02, 1),
    xlabel="Recall",
    ylabel="Precision",
    color="navy",
    lw=2,
    linestyle="--",
    name="Random",
)
plt.legend(loc="upper right")


# %%

fig = plt.figure(figsize=(6, 6))

# ANN
cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="with_FPG")
precisions, recalls, _ = zip(
    *(metric_utils.precision_recall_curve(ys, probs) for ys, probs in cv_y_prob)
)
aps = np.asarray(
    [metric_utils.average_precision_score(ys, probs) for ys, probs in cv_y_prob]
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    recalls, precisions, reverse=True
)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"ANN. mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
    color="royalblue",
)
plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="with_FPG")
precisions, recalls, _ = zip(
    *(metric_utils.precision_recall_curve(ys, probs) for ys, probs in cv_y_prob)
)
aps = np.asarray(
    [metric_utils.average_precision_score(ys, probs) for ys, probs in cv_y_prob]
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    recalls, precisions, reverse=True
)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"LGBM. mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
)
plot_range(x_base, y_lower, y_upper)

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA_FPG")
p, r = mean_precision_recall(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")

# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(r, p, marker="^", label="ADA.")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH_FPG")
p, r = mean_precision_recall(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")
# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(r, p, marker="s", label="CDS.")
# plt.annotate(f"({fpr:.3f}, {tpr:.3f})", (fpr + 0.02, tpr))

# Random
plot_curve(
    (0, 1),
    (0, 0),
    ylim=(-0.02, 1),
    xlabel="Recall",
    ylabel="Precision",
    color="navy",
    lw=2,
    linestyle="--",
    name="Random",
)
plt.legend(loc="lower left")
# %%
with open(cache_file, "wb") as f:
    pickle.dump(results, f)

# %%
