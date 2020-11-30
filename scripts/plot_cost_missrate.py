# %%
import logging
import os

import numpy as np
import pandas as pd

from lxh_prediction import config as cfg
from lxh_prediction import metric_utils
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.plot import plot_curve, plot_range, plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# %%
def mean_cost_missrate(cv_y_prob, FPG=None):
    ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
    probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob]) > 0
    if FPG is not None:
        FPG = np.concatenate(FPG)
        cost, missrate, _ = metric_utils.cost_curve_with_FPG(ys, probs, FPG)
    else:
        cost, missrate, _ = metric_utils.cost_curve_without_FPG(ys, probs)
    return cost[-1], missrate[-1]


# %%
# ROC without FPG

fig = plt.figure(figsize=(6, 6))
y_means = {}
x_means = {}


# # ANN
# cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="without_FPG")
# costs, miss_rates, _ = zip(
#     *(metric_utils.cost_curve_without_FPG(ys, probs) for ys, probs in cv_y_prob)
# )
# x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
#     miss_rates, costs, y_range=(0, 70)
# )
# plot_curve(
#     x_base, y_mean, ylim=(0, 70), name="ANN (no-lab)", color="royalblue",
# )
# plot_range(x_base, y_lower, y_upper)
# y_means["ANN (no-lab)"] = y_mean

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG")
costs, miss_rates, _ = zip(
    *(metric_utils.cost_curve_without_FPG(ys, probs) for ys, probs in cv_y_prob)
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    miss_rates, costs, y_range=(0, 70)
)
plot_curve(x_base, y_mean, ylim=(0, 70), name="LGBM (no-lab)")
plot_range(x_base, y_lower, y_upper)
y_means["LGBM (no-lab)"] = y_mean

y_base, x_mean = metric_utils.mean_curve(
    costs, miss_rates, x_range=(0, 70), reverse=True
)[:2]
x_means["LGBM (no-lab)"] = x_mean


# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA")
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
    name="ADA (no-lab)",
    color="dodgerblue",
    linestyle="--",
)
cost, miss_rate = mean_cost_missrate(cv_y_prob)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="s", color="dodgerblue")
print(f"ADA (no-lab): miss_rate: {miss_rate}, cost: {cost}")
y_means["ADA (no-lab)"] = y_mean

y_base, x_mean = metric_utils.mean_curve(
    costs, miss_rates, x_range=(0, 70), reverse=True
)[:2]
x_means["ADA (no-lab)"] = x_mean


# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH")
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
    name="CDS (no-lab)",
    color="darkgreen",
    linestyle="--",
    xlabel="Prediction miss rate",
    ylabel="Average cost per patient",
)
cost, miss_rate = mean_cost_missrate(cv_y_prob)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="s", color="darkgreen")
print(f"CDS (no-lab): miss_rate: {miss_rate}, cost: {cost}")
y_means["CDS (no-lab)"] = y_mean

y_base, x_mean = metric_utils.mean_curve(
    costs, miss_rates, x_range=(0, 70), reverse=True
)[:2]
x_means["CDS (no-lab)"] = x_mean


plt.legend(loc="upper right")

fig.savefig(os.path.join(cfg.root, "data/results/cost_missrate_withoutFPG.pdf"))


# %%
# ROC with FPG
fig = plt.figure(figsize=(6, 6))


# # ANN
# cv_y_prob, FPG = get_cv_preds(
#     model_name="ANNModel", feat_collection="with_FPG", out_FPG=True
# )
# costs, miss_rates, _ = zip(
#     *(
#         metric_utils.cost_curve_with_FPG(ys, probs, FPG[i])
#         for i, (ys, probs) in enumerate(cv_y_prob)
#     )
# )
# x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
#     miss_rates, costs, y_range=(0, 70)
# )
# plot_curve(
#     x_base, y_mean, ylim=(0, 70), name="ANN", color="royalblue",
# )
# plot_range(x_base, y_lower, y_upper)
# y_means["ANN"] = y_mean

# LGBM
cv_y_prob, FPG = get_cv_preds(
    model_name="LightGBMModel", feat_collection="with_FPG", out_FPG=True
)
costs, miss_rates, _ = zip(
    *(
        metric_utils.cost_curve_with_FPG(ys, probs, FPG[i])
        for i, (ys, probs) in enumerate(cv_y_prob)
    )
)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
    miss_rates, costs, y_range=(0, 70)
)
plot_curve(x_base, y_mean, ylim=(0, 70), name="LGBM")
plot_range(x_base, y_lower, y_upper)
y_means["LGBM"] = y_mean

y_base, x_mean = metric_utils.mean_curve(
    costs, miss_rates, x_range=(0, 70), reverse=True
)[:2]
x_means["LGBM"] = x_mean


# ADA
cv_y_prob, FPG = get_cv_preds(
    model_name="ADAModel", feat_collection="ADA_FPG", out_FPG=True
)
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
    x_base, y_mean, ylim=(0, 70), name="ADA", color="dodgerblue", linestyle="--",
)
cost, miss_rate = mean_cost_missrate(cv_y_prob, FPG)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="s", color="dodgerblue")
print(f"ADA: miss_rate: {miss_rate}, miss_rate: {cost}")
y_means["ADA"] = y_mean

y_base, x_mean = metric_utils.mean_curve(
    costs, miss_rates, x_range=(0, 70), reverse=True
)[:2]
x_means["ADA"] = x_mean


# CDS
cv_y_prob, FPG = get_cv_preds(
    model_name="CHModel", feat_collection="CH_FPG", out_FPG=True
)
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
    name="CDS",
    color="darkgreen",
    linestyle="--",
    xlabel="Prediction miss rate",
    ylabel="Average cost per patient",
)
cost, miss_rate = mean_cost_missrate(cv_y_prob, FPG)
plt.plot((0, 1), (cost, cost), color="gray", lw=1, linestyle="--")
plt.plot((miss_rate, miss_rate), (0, 70), color="gray", lw=1, linestyle="--")
plt.scatter(miss_rate, cost, marker="s", color="darkgreen")
print(f"CDS: miss_rate: {miss_rate}, miss_rate: {cost}")
y_means["CDS"] = y_mean

y_base, x_mean = metric_utils.mean_curve(
    costs, miss_rates, x_range=(0, 70), reverse=True
)[:2]
x_means["CDS"] = x_mean


plt.legend(loc="upper right")

fig.savefig(os.path.join(cfg.root, "data/results/cost_missrate_withFPG.pdf"))


# %%
df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
output = os.path.join(cfg.root, "data/results/cost_missrate.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_ymeans.to_csv(output)

df_xmeans = pd.DataFrame(x_means.values(), index=x_means.keys(), columns=y_base)
output = os.path.join(cfg.root, "data/results/cost_missrate_transpose.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_xmeans.to_csv(output)

# %%
