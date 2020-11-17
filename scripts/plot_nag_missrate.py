# %%
import logging

import numpy as np

from lxh_prediction import metric_utils
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.plot import plot_curve, plot_range, plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# %%


def mean_nag_missrate(cv_y_prob):
    ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
    probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
    miss_rate = np.mean(probs[ys > 0] <= 0)
    nag_rate = np.mean(probs <= 0)
    return nag_rate, miss_rate


# %%
# NAG-Miss without FPG

fig = plt.figure(figsize=(6, 6))

# # ANN
# cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="without_FPG")
# nag_rates, miss_rates, _ = zip(
#     *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
# )
# x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
# plot_curve(
#     x_base, y_mean, ylim=(0, 1), name="ANN (no-lab)", color="royalblue",
# )
# plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG")
nag_rates, miss_rates, _ = zip(
    *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
plot_curve(x_base, y_mean, ylim=(0, 1), name="LGBM (no-lab)")
plot_range(x_base, y_lower, y_upper)

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA")
nag_rates, miss_rates, _ = zip(
    *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
plot_curve(
    x_base, y_mean, ylim=(0, 1), name="ADA (no-lab)", color="dodgerblue", linestyle="--"
)
nag_rate, miss_rate = mean_nag_missrate(cv_y_prob)
plt.plot((0, 1), (nag_rate, nag_rate), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, nag_rate, marker="s", color="dodgerblue")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH")
nag_rates, miss_rates, _ = zip(
    *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name="CDS (no-lab)",
    color="darkgreen",
    linestyle="--",
    xlabel="Prediction miss rate",
    ylabel="Patients avoiding examination",
)
nag_rate, miss_rate = mean_nag_missrate(cv_y_prob)
plt.plot((0, 1), (nag_rate, nag_rate), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, nag_rate, marker="s", color="darkgreen")


# %%
# Nag-miss with FPG

fig = plt.figure(figsize=(6, 6))

# # ANN
# cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="with_FPG")
# nag_rates, miss_rates, _ = zip(
#     *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
# )
# x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
# plot_curve(
#     x_base, y_mean, ylim=(0, 1), name="ANN", color="royalblue",
# )
# plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="with_FPG")
nag_rates, miss_rates, _ = zip(
    *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
plot_curve(x_base, y_mean, ylim=(0, 1), name="LGBM")
plot_range(x_base, y_lower, y_upper)


# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA_FPG")
nag_rates, miss_rates, _ = zip(
    *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
plot_curve(x_base, y_mean, ylim=(0, 1), name="ADA", color="dodgerblue", linestyle="--")
nag_rate, miss_rate = mean_nag_missrate(cv_y_prob)
plt.plot((0, 1), (nag_rate, nag_rate), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, nag_rate, marker="s", color="dodgerblue")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH_FPG")
nag_rates, miss_rates, _ = zip(
    *(metric_utils.nag_miss_curve(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(miss_rates, nag_rates)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name="CDS",
    color="darkgreen",
    linestyle="--",
    xlabel="Prediction miss rate",
    ylabel="Patients avoiding examination",
)
nag_rate, miss_rate = mean_nag_missrate(cv_y_prob)
plt.plot((0, 1), (nag_rate, nag_rate), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, nag_rate, marker="s", color="darkgreen")


# %%
