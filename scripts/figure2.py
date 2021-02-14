# %%
import logging

import numpy as np

from lxh_prediction import metric_utils
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.plot import plot_curve, plot_range, plt, ExpFigure

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class auROCExp(ExpFigure):
    @staticmethod
    def mean_tpr_fpr(cv_y_prob):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
        tpr = np.mean(probs[ys > 0] > 0)
        fpr = np.mean(probs[ys == 0] > 0)
        return tpr, fpr

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
            name=f"{name} Model. auROC={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
            color=color,
            zorder=3,
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(tprs, fprs)[:2]
        self.x_means[name] = x_mean

        if "LightGBMM" not in model:
            tpr, fpr = self.mean_tpr_fpr(cv_y_prob)
            print(f"{name}: fpr: {fpr}, tpr: {tpr}")
            plt.axvline(x=fpr, c="gray", ls="--", lw=1, zorder=1)
            plt.axhline(y=tpr, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                fpr, tpr, marker="8", color=color, label=f"{name} Cutoff Point", zorder=4
            )


class auPRExp(ExpFigure):
    @staticmethod
    def mean_precision_recall(cv_y_prob):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
        precision = np.mean(ys[probs > 0] > 0)
        recall = np.mean(probs[ys > 0] > 0)
        return precision, recall

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
            zorder=3,
        )

        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(precisions, recalls)[:2]
        self.x_means[name] = x_mean

        if "LightGBMM" not in model:
            p, r = self.mean_precision_recall(cv_y_prob)
            print(f"{name}: precision: {p}, recall: {r}")
            plt.axvline(x=r, c="gray", ls="--", lw=1, zorder=1)
            plt.axhline(y=p, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                r, p, marker="8", color=color, label=f"{name} Cutoff Point", zorder=4
            )


# %%
# Figure 2c, ROC, ADA/CDS
exp = auROCExp()
exp.run("ADA", "ADAModel", "ADA")
exp.run("CDS", "CHModel", "CH")

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

exp.save("figure2_c")


# %%
# Figure 2d, auPR, ADA/CDS

exp = auPRExp()
exp.run("ADA", "ADAModel", "ADA")
exp.run("CDS", "CHModel", "CH")

plt.legend(loc="upper right")
exp.save("figure2_d")


# %%
# SFigure 2a auROC, ADA/CDS, FPG

exp = auROCExp()
exp.run("FPG Model", "LightGBMModel", "FPG")
exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
exp.run("CDS+FPG", "CHModel", "CH_FPG")

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

exp.save("s_figure2_a")
# %%
# SFigure 2b auPR, ADA/CDS, FPG

exp = auPRExp()
exp.run("FPG Model", "LightGBMModel", "FPG")
exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
exp.run("CDS+FPG", "CHModel", "CH_FPG")

plt.legend(loc="upper left")
exp.save("s_figure2_b")


# %%
# SFigure 2c auROC, ADA/CDS, 2hPG

exp = auROCExp()
exp.run("2hPG Model", "LightGBMModel", "2hPG")
exp.run("ADA+2hPG", "ADAModel", "ADA_2hPG")
exp.run("CDS+2hPG", "CHModel", "CH_2hPG")

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

exp.save("s_figure2_c")
# %%
# SFigure 2d auPR, ADA/CDS, 2hPG

exp = auPRExp()
exp.run("2hPG Model", "LightGBMModel", "2hPG")
exp.run("ADA+2hPG", "ADAModel", "ADA_2hPG")
exp.run("CDS+2hPG", "CHModel", "CH_2hPG")

plt.legend(loc="upper left")
exp.save("s_figure2_d")


# %%
# SFigure 2e auROC, ADA/CDS, HbA1c

exp = auROCExp()
exp.run("HbA1c Model", "LightGBMModel", "HbA1c")
exp.run("ADA+HbA1c", "ADAModel", "ADA_HbA1c")
exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")

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

exp.save("s_figure2_e")
# %%
# SFigure 2f auPR, ADA/CDS, HbA1c

exp = auPRExp()
exp.run("HbA1c Model", "LightGBMModel", "HbA1c")
exp.run("ADA+HbA1c", "ADAModel", "ADA_HbA1c")
exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")

plt.legend(loc="upper left")
exp.save("s_figure2_f")

# %%
