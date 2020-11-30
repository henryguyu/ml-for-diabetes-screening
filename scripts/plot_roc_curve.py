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
def mean_tpr_fpr(cv_y_prob):
    ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
    probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
    tpr = np.mean(probs[ys > 0] > 0)
    fpr = np.mean(probs[ys == 0] > 0)
    return tpr, fpr


# %%
# ROC without FPG

fig = plt.figure(figsize=(6, 6))
y_means = {}
x_means = {}

# # ANN
# cv_y_prob = get_cv_preds(
#     model_name="ANNModel", feat_collection="without_FPG", update=True
# )
# fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
# aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
# x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
# plot_curve(
#     x_base,
#     y_mean,
#     ylim=(0, 1),
#     name=f"ANN (no-lab). auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
#     color="royalblue",
# )
# plot_range(x_base, y_lower, y_upper)
# y_means["ANN (no-lab)"] = y_mean

# LGBM
cv_y_prob = get_cv_preds(
    model_name="LightGBMModel", feat_collection="without_FPG", update=True
)
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
print(aucs)
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"LGBM (no-lab). auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
)
plot_range(x_base, y_lower, y_upper)
y_means["LGBM (no-lab)"] = y_mean

y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
x_means["LGBM (no-lab)"] = x_mean

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA", update=True)
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"ADA (no-lab). auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
    color="dodgerblue",
    linestyle="--",
)
tpr, fpr = mean_tpr_fpr(cv_y_prob)
plt.scatter(fpr, tpr, marker="s", color="dodgerblue")
print(f"ADA (no-lab): fpr: {fpr}, tpr: {tpr}")
y_means["ADA (no-lab)"] = y_mean

y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
x_means["ADA (no-lab)"] = x_mean


# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH", update=True)
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"CDS (no-lab). auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
    color="darkgreen",
    linestyle="--",
)
tpr, fpr = mean_tpr_fpr(cv_y_prob)
plt.scatter(fpr, tpr, marker="s", color="darkgreen")
print(f"CDS (no-lab): fpr: {fpr}, tpr: {tpr}")
y_means["CDS (no-lab)"] = y_mean

y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
x_means["CDS (no-lab)"] = x_mean


# Random
plot_curve(
    (0, 1),
    (0, 1),
    ylim=(0, 1),
    xlabel="False positive rate",
    ylabel="True positive rate",
    color="navy",
    lw=2,
    linestyle="--",
    name="Random",
)

fig.savefig(os.path.join(cfg.root, "data/results/roc_curve_withoutFPG.pdf"))

# %%

fig = plt.figure(figsize=(6, 6))

# # ANN
# cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="with_FPG", update=True)
# fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
# aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
# x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
# plot_curve(
#     x_base,
#     y_mean,
#     ylim=(0, 1),
#     name=f"ANN. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
#     color="royalblue",
# )
# plot_range(x_base, y_lower, y_upper)
# y_means["ANN"] = y_mean

# LGBM
cv_y_prob = get_cv_preds(
    model_name="LightGBMModel", feat_collection="with_FPG", update=True
)
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"LGBM. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
)
plot_range(x_base, y_lower, y_upper)
y_means["LGBM"] = y_mean

y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
x_means["LGBM"] = x_mean

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA_FPG", update=True)
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"ADA. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
    color="dodgerblue",
    linestyle="--",
)
tpr, fpr = mean_tpr_fpr(cv_y_prob)
plt.scatter(fpr, tpr, marker="s", color="dodgerblue")
print(f"ADA: fpr: {fpr}, tpr: {tpr}")
y_means["ADA"] = y_mean

y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
x_means["ADA"] = x_mean


# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH_FPG", update=True)
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"CDS. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
    color="darkgreen",
    linestyle="--",
)
tpr, fpr = mean_tpr_fpr(cv_y_prob)
plt.scatter(fpr, tpr, marker="s", color="darkgreen")
print(f"CDS: fpr: {fpr}, tpr: {tpr}")
y_means["CDS"] = y_mean

y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
x_means["CDS"] = x_mean

# Random
plot_curve(
    (0, 1),
    (0, 1),
    ylim=(0, 1),
    xlabel="False positive rate",
    ylabel="True positive rate",
    color="navy",
    lw=2,
    linestyle="--",
    name="Random",
)

fig.savefig(os.path.join(cfg.root, "data/results/roc_curve_withFPG.pdf"))

# %%
df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
output = os.path.join(cfg.root, "data/results/roc_curve.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_ymeans.to_csv(output)

df_xmeans = pd.DataFrame(x_means.values(), index=x_means.keys(), columns=y_base)
output = os.path.join(cfg.root, "data/results/roc_curve_transpose.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_xmeans.to_csv(output)

# %%
