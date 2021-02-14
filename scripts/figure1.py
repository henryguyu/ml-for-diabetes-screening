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
# Figure 1a, ROC, full, top 5/10/15/20

fig = plt.figure(figsize=(5, 5))
y_means = {}
x_means = {}


def auc_roc_exp(name, feat_collection, color=None):
    cv_y_prob = get_cv_preds(
        model_name="LightGBMModel",
        feat_collection=feat_collection,
        update=True,
        resample_train=False,
    )
    fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
    aucs = np.asarray(
        [metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob]
    )
    print(aucs)
    x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
    color = cfg.color_map[len(y_means)]
    plot_curve(
        x_base,
        y_mean,
        ylim=(0, 1),
        name=f"{name}. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
        color=color,
    )
    plot_range(x_base, y_lower, y_upper)
    y_means[name] = y_mean

    y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
    x_means[name] = x_mean
    return x_base


auc_roc_exp(
    "Full Model", "full_non_lab",
)
auc_roc_exp("Top-20", "top20_non_lab")
auc_roc_exp("Top-15", "top15_non_lab")
auc_roc_exp("Top-10", "top10_non_lab")
x_base = auc_roc_exp("Top-5", "top5_non_lab")
# x_base = auc_roc_exp("ADA", "ADA")
# x_base = auc_roc_exp("CDS", "CH")
# one_exp("Top-3", "top3_non_lab")

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

df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
output = os.path.join(cfg.root, "data/results/figure1_a.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_ymeans.to_csv(output)

fig.savefig(os.path.join(cfg.root, "data/results/figure1_a.pdf"))


# %%
# Figure 1b, auPR, full, top 5/10/15/20

fig = plt.figure(figsize=(5, 5))
y_means = {}
x_means = {}


def auc_pr_exp(name, feat_collection, color=None):
    cv_y_prob = get_cv_preds(
        model_name="LightGBMModel",
        feat_collection=feat_collection,
        update=False,
        resample_train=False,
    )
    precisions, recalls, _ = zip(
        *(metric_utils.precision_recall_curve(ys, probs) for ys, probs in cv_y_prob)
    )
    aps = np.asarray(
        [metric_utils.average_precision_score(ys, probs) for ys, probs in cv_y_prob]
    )
    x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
        recalls, precisions, reverse=True
    )
    color = cfg.color_map[len(y_means)]
    plot_curve(
        x_base,
        y_mean,
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="Precision",
        name=f"{name}. auPR={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
        color=color,
    )

    plot_range(x_base, y_lower, y_upper)
    y_means[name] = y_mean

    y_base, x_mean = metric_utils.mean_curve(precisions, recalls)[:2]
    x_means[name] = x_mean


auc_pr_exp(
    "Full Model", "full_non_lab",
)
auc_pr_exp("Top-20", "top20_non_lab")
auc_pr_exp("Top-15", "top15_non_lab")
auc_pr_exp("Top-10", "top10_non_lab")
x_base = auc_pr_exp("Top-5", "top5_non_lab")
# one_exp("Top-3", "top3_non_lab")

# # Random
# plot_curve(
#     (0, 1),
#     (0.0, 0.0),
#     ylim=(0, 1),
#     xlabel="Recall",
#     ylabel="Precision",
#     color="navy",
#     lw=2,
#     linestyle="--",
#     name="Random",
# )
# plt.legend(loc="upper right")

df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
output = os.path.join(cfg.root, "data/results/figure1_b.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_ymeans.to_csv(output)

fig.savefig(os.path.join(cfg.root, "data/results/figure1_b.pdf"))

# %%
# Figure 1c, ROC, full, top 5/10/15/20

fig = plt.figure(figsize=(5, 5))
y_means = {}
x_means = {}


x_base = auc_roc_exp("FPG Model", "FPG")
auc_roc_exp("2hPG Model", "2hPG")
auc_roc_exp("HbA1c Model", "HbA1c")
# one_exp("Top-3", "top3_non_lab")

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

df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
output = os.path.join(cfg.root, "data/results/figure1_c.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_ymeans.to_csv(output)

fig.savefig(os.path.join(cfg.root, "data/results/figure1_c.pdf"))

# %%
# Figure 1d, auPR, full, top 5/10/15/20

fig = plt.figure(figsize=(5, 5))
y_means = {}
x_means = {}


x_base = auc_pr_exp("FPG Model", "FPG")
auc_pr_exp("2hPG Model", "2hPG")
auc_pr_exp("HbA1c Model", "HbA1c")

df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
output = os.path.join(cfg.root, "data/results/figure1_d.csv")
os.makedirs(os.path.dirname(output), exist_ok=True)
df_ymeans.to_csv(output)

fig.savefig(os.path.join(cfg.root, "data/results/figure1_d.pdf"))

# %%
