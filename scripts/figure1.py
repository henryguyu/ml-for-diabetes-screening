# %%
import logging

import numpy as np

from lxh_prediction import metric_utils
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.plot import plot_curve, plot_range, ExpFigure, plt

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class auROCNonLab(ExpFigure):
    def run(self, name, model, feat_collection):
        cv_y_prob = get_cv_preds(
            model_name=model,
            feat_collection=feat_collection,
            update=True,
            resample_train=False,
        )
        fprs, tprs, _ = zip(
            *(metric_utils.roc_curve(ys, probs) for ys, probs in cv_y_prob)
        )
        aucs = np.asarray(
            [metric_utils.roc_auc_score(ys, probs) for ys, probs in cv_y_prob]
        )
        print(aucs)
        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(fprs, tprs)
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            ylim=(0, 1),
            name=f"{name}. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
            color=color,
            xlabel="False positive rate",
            ylabel="True positive rate",
        )
        plot_range(x_base, y_lower, y_upper)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
        self.x_means[name] = x_mean


class auPRNonLab(ExpFigure):
    def run(self, name, model, feat_collection):
        cv_y_prob = get_cv_preds(
            model_name=model,
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
        color = self.next_color()
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
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(precisions, recalls)[:2]
        self.x_means[name] = x_mean


# %%
# Figure 1a, ROC, full, top 5/10/15/20
exp = auROCNonLab()
exp.run("Full Model", "LightGBMModel", "full_non_lab")
exp.run("Top-20", "LightGBMModel", "top20_non_lab")
exp.run("Top-15", "LightGBMModel", "top15_non_lab")
exp.run("Top-10", "LightGBMModel", "top10_non_lab")
exp.run("Top-5", "LightGBMModel", "top5_non_lab")
exp.plot()

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

exp.save("figure1_a")


# %%
# Figure 1b, auPR, full, top 5/10/15/20

exp = auPRNonLab()
exp.run("Full Model", "LightGBMModel", "full_non_lab")
exp.run("Top-20", "LightGBMModel", "top20_non_lab")
exp.run("Top-15", "LightGBMModel", "top15_non_lab")
exp.run("Top-10", "LightGBMModel", "top10_non_lab")
exp.run("Top-5", "LightGBMModel", "top5_non_lab")
exp.plot()

plt.legend(loc="upper right")

exp.save("figure1_b")

# %%
# Figure 1c, ROC, full, top 5/10/15/20

exp = auROCNonLab()
#exp.run("Non-lab(AI)", "LightGBMModel", "top20_non_lab")
exp.run("AI+FPG Model", "LightGBMModel", "FPG")
exp.run("AI+2hPG Model", "LightGBMModel", "2hPG")
exp.run("AI+HbA1c Model", "LightGBMModel", "HbA1c")
exp.plot()

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

exp.save("figure1_c")

# %%
# Figure 1d, auPR, full, top 5/10/15/20

exp = auPRNonLab()
#exp.run("Non-lab", "LightGBMModel", "top20_non_lab")
exp.run("AI+FPG Model", "LightGBMModel", "FPG")
exp.run("AI+2hPG Model", "LightGBMModel", "2hPG")
exp.run("AI+HbA1c Model", "LightGBMModel", "HbA1c")
exp.plot()

exp.save("figure1_d")


# %%
