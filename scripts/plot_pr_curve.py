# %%
import logging

import numpy as np

from lxh_prediction import metric_utils
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.plot import plot_curve, plot_range, plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# %%
def mean_precision_recall(cv_y_prob):
    ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
    probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
    precision = np.mean(ys[probs > 0] > 0)
    recall = np.mean(probs[ys > 0] > 0)
    return precision, recall


# %%
# PR without FPG

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
    name=f"ADA (no-lab). mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
    color="dodgerblue",
    linestyle="--",
)
p, r = mean_precision_recall(cv_y_prob)
plt.scatter(r, p, marker="s", color="dodgerblue")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH")
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
    name=f"CDS (no-lab). mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
    color="darkgreen",
    linestyle="--",
    xlabel="Recall",
    ylabel="Precision",
)
p, r = mean_precision_recall(cv_y_prob)
plt.scatter(r, p, marker="s", color="darkgreen")

plt.legend(loc="upper right")


# %% PR with FPG

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
# p, r = mean_precision_recall(cv_y_prob)
# plt.scatter(r, p, marker="^", label="ADA.")
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
    name=f"ADA. mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
    color="dodgerblue",
    linestyle="--",
)
p, r = mean_precision_recall(cv_y_prob)
plt.scatter(r, p, marker="s", color="dodgerblue")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH_FPG")
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
    name=f"CDS. mAP={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
    color="darkgreen",
    linestyle="--",
    xlabel="Recall",
    ylabel="Precision",
)
p, r = mean_precision_recall(cv_y_prob)
plt.scatter(r, p, marker="s", color="darkgreen")

plt.legend(loc="lower left")
# %%
