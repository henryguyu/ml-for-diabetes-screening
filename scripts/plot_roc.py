# %%
import logging

import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils, models
from lxh_prediction.plot import plot_curve, plot_range, plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# %%
results = {}


def get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG"):
    key = (model_name, feat_collection)
    if key in results:
        return results[key]

    # Load data
    X, y = data_utils.load_data(cfg.feature_fields[feat_collection])

    # Load model
    model = getattr(models, model_name)()
    cv_aucs, cv_probs_pred, cv_indices = model.cross_validate(
        X, y, metric_fn=metric_utils.roc_auc_score
    )
    cv_ys_gt = [y[idx] for idx in cv_indices]
    results[key] = list(zip(cv_ys_gt, cv_probs_pred))
    return results[key]


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

# ANN
cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="without_FPG")
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"ANN (no-lab). auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
    color="royalblue",
)
plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="without_FPG")
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"LGBM (no-lab). auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
)
plot_range(x_base, y_lower, y_upper)

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA")
tpr, fpr = mean_tpr_fpr(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")

# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(fpr, tpr, marker="^", label="ADA (no-lab).")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH")
tpr, fpr = mean_tpr_fpr(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")
# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(fpr, tpr, marker="s", label="CDS (no-lab).")
# plt.annotate(f"({fpr:.3f}, {tpr:.3f})", (fpr + 0.02, tpr))

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

# %%

fig = plt.figure(figsize=(6, 6))

# ANN
cv_y_prob = get_cv_preds(model_name="ANNModel", feat_collection="with_FPG")
fprs, tprs, _ = zip(*(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob))
aucs = np.asarray([metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob])
x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
plot_curve(
    x_base,
    y_mean,
    ylim=(0, 1),
    name=f"ANN. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
    color="royalblue",
)
plot_range(x_base, y_lower, y_upper)

# LGBM
cv_y_prob = get_cv_preds(model_name="LightGBMModel", feat_collection="with_FPG")
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

# ADA
cv_y_prob = get_cv_preds(model_name="ADAModel", feat_collection="ADA_FPG")
tpr, fpr = mean_tpr_fpr(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")
# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(fpr, tpr, marker="^", label="ADA")

# CDS
cv_y_prob = get_cv_preds(model_name="CHModel", feat_collection="CH_FPG")
tpr, fpr = mean_tpr_fpr(cv_y_prob)
# plt.plot((0, 1), (tpr, tpr), color="gray", lw=1, linestyle="--")
# plt.plot((fpr, fpr), (0, 1), color="gray", lw=1, linestyle="--")
plt.scatter(fpr, tpr, marker="s", label="CDS")

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

# %%
