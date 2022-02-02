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
            update=self.retrain,
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
            name=f"{name}={aucs.mean():.3f} [{aucs.min():.3f}, {aucs.max():.3f}]",
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
            update=self.retrain,
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
            name=f"{name}={aps.mean():.3f} [{aps.min():.3f}, {aps.max():.3f}]",
            color=color,
        )

        plot_range(x_base, y_lower, y_upper)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(precisions, recalls)[:2]
        self.x_means[name] = x_mean


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
            update=self.retrain,
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
            update=self.retrain,
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
