# %%
import logging

import numpy as np
from brokenaxes import brokenaxes

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

    def run(self, name, model, feat_collection, cutoff=False):
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
        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
            fprs, tprs, x_range=self.xlim, y_range=self.ylim
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            ylim=self.ylim,
            xlim=self.xlim,
            name=f"{self.fname(name)}={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
            color=color,
            zorder=3,
            ax=self.ax,
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base
        self.y_base, x_mean = metric_utils.mean_curve(
            tprs, fprs, x_range=self.ylim, y_range=self.xlim
        )[:2]

        self.x_means[name] = x_mean

        if "LightGBMM" not in model and cutoff:
            tpr, fpr = self.mean_tpr_fpr(cv_y_prob)
            print(f"{name}: fpr: {fpr}, tpr: {tpr}")
            plt.axvline(x=fpr, c="gray", ls="--", lw=1, zorder=1)
            plt.axhline(y=tpr, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                fpr,
                tpr,
                marker="8",
                color=color,
                label=f"{name} Cutoff Point",
                zorder=4,
            )
            self.add_point(fpr, tpr)


class auPRExp(ExpFigure):
    @staticmethod
    def mean_precision_recall(cv_y_prob):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
        precision = np.mean(ys[probs > 0] > 0)
        recall = np.mean(probs[ys > 0] > 0)
        return precision, recall

    def run(self, name, model, feat_collection, cutoff=False):
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
            name=f"{self.fname(name)}={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
            color=color,
            zorder=3,
            ax=self.ax,
        )

        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(precisions, recalls)[:2]
        self.x_means[name] = x_mean

        if "LightGBMM" not in model and cutoff:
            p, r = self.mean_precision_recall(cv_y_prob)
            print(f"{name}: precision: {p}, recall: {r}")
            plt.axvline(x=r, c="gray", ls="--", lw=1, zorder=1)
            plt.axhline(y=p, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                r, p, marker="8", color=color, label=f"{name} Cutoff Point", zorder=4
            )
            self.add_point(r, p)


# %%
# Figure 2c, ROC, ADA/CDS
# fig = plt.figure(figsize=(7, 7))
exp = auROCExp()
exp.run("ML Model", "LightGBMModel", "top20_non_lab")
exp.run("ADART", "ADAModel", "ADA", cutoff=True)
exp.run("NCDRS", "CHModel", "CH", cutoff=True)
exp.plot()

# Random
plot_curve(
    (0, 1),
    (0, 1),
    ylim=(0, 1),
    xlabel="1-Specificity",
    ylabel="Sensitivity",
    color="navy",
    lw=2,
    linestyle="--",
    name="Random",
)

exp.save("figure3_b")


# Figure 2d, auPR, ADA/CDS

exp = auPRExp()
exp.run("ML Model", "LightGBMModel", "top20_non_lab")
exp.run("ADART", "ADAModel", "ADA", cutoff=True)
exp.run("NCDRS", "CHModel", "CH", cutoff=True)
exp.plot()

plt.legend(loc="upper right")
exp.save("figure3_c")


# %%
# Figure 2e auROC, ADA/CDS, FPG
# fig = plt.figure(figsize=(7, 7))
# bax = brokenaxes(ylims=((0, 0.1), (0.5, 1)))

exp = auROCExp()
# exp.ylim = (0.5, 1)
# exp.xlim = (0.5, 1)
# exp.run("ML Model", "LightGBMModel", "top20_non_lab")
exp.run("ML+FPG Model", "LightGBMModel", "FPG")
exp.run("NCDRS+FPG Model", "CHModel", "CH_FPG")
# exp.run("ML+2hPG Model", "LightGBMModel", "2hPG")
# exp.run("NCDRS+2hPG Model", "CHModel", "CH_2hPG")
# exp.run("ML+HbA1c Model", "LightGBMModel", "HbA1c")
# exp.run("NCDRS+HbA1c Model", "CHModel", "CH_HbA1c")
# exp.run("CDS", "CHModel", "CH", cutoff=True)
exp.plot()


# Random
plot_curve(
    (0, 1),
    (0, 1),
    xlim=exp.xlim,
    ylim=exp.ylim,
    xlabel="1-Specificity",
    ylabel="Sensitivity",
    color="navy",
    lw=2,
    linestyle="--",
    name="Random",
)

exp.save("figure2_e")
# %%
# Figure 2f auPR, ADA/CDS, FPG

exp = auPRExp()
# exp.run("ML Model", "LightGBMModel", "top20_non_lab")
# exp.run("ML+FPG Model", "LightGBMModel", "FPG")
# exp.run("NCDRS+FPG Model", "CHModel", "CH_FPG")
# exp.run("ML+2hPG Model", "LightGBMModel", "2hPG")
# exp.run("NCDRS+2hPG Model", "CHModel", "CH_2hPG")
exp.run("ML+HbA1c Model", "LightGBMModel", "HbA1c")
exp.run("NCDRS+HbA1c Model", "CHModel", "CH_HbA1c")
# exp.run("CDS", "CHModel", "CH", cutoff=True)
exp.plot()

plt.legend(loc="upper left")
exp.save("figure2_f")


