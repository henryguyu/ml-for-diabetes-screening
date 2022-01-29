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


# %%
# Figure 1a, ROC, full, top 5/10/15/20
exp = auROCNonLab(retrain=False)
exp.run("Full Model", "LightGBMModel", "full_non_lab")
exp.run("Top-25 Model", "LightGBMModel", "top25_non_lab")
exp.run("Top-25 EnsembleModel", "EnsembleModel", "top25_non_lab")
exp.run("Top-20 Model", "LightGBMModel", "top20_non_lab")
exp.run("Top-15 Model", "LightGBMModel", "top15_non_lab")
exp.run("Top-10 Model", "LightGBMModel", "top10_non_lab")
exp.run("Top-5 Model", "LightGBMModel", "top5_non_lab")
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

exp.save("figure1_c")


# %%
# Figure 1b, auPR, full, top 5/10/15/20

exp = auPRNonLab(retrain=False)
exp.run("Full Model", "LightGBMModel", "full_non_lab")
exp.run("Top-25 Model", "LightGBMModel", "top25_non_lab")
exp.run("Top-25 EnsembleModel", "EnsembleModel", "top25_non_lab")
exp.run("Top-20 Model", "LightGBMModel", "top20_non_lab")
exp.run("Top-15 Model", "LightGBMModel", "top15_non_lab")
exp.run("Top-10 Model", "LightGBMModel", "top10_non_lab")
exp.run("Top-5 Model", "LightGBMModel", "top5_non_lab")
exp.plot()

plt.legend(loc="upper right")

exp.save("figure1_d")

# %%
# Figure 1c, ROC, full, top 5/10/15/20

exp = auROCNonLab()
# exp.run("Non-lab(AI)", "LightGBMModel", "top20_non_lab")
exp.run("AI+FPG Model", "EnsembleModel", "FPG")
exp.run("AI+FPG Model", "LightGBMModel", "FPG")
# exp.run("AI+2hPG Model", "EnsembleModel", "2hPG")
# exp.run("AI+HbA1c Model", "EnsembleModel", "HbA1c")
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
# exp.run("Non-lab", "LightGBMModel", "top20_non_lab")
exp.run("AI+FPG Model", "AutoGluonModel", "FPG")
exp.run("AI+2hPG Model", "AutoGluonModel", "2hPG")
exp.run("AI+HbA1c Model", "AutoGluonModel", "HbA1c")
exp.plot()

exp.save("figure1_d")


# %%

exp = auROCNonLab()
feature_set = "top20_non_lab"
exp.run("Non-lab(LGBM)", "LightGBMModel", feature_set)
exp.run("Non-lab(ANN)", "ANNModel", feature_set)
exp.run("CDS", "CHModel", "CH")
exp.run("Non-lab(LR) Model", "LogisticRegressionModel", feature_set)
exp.run("Non-lab(SVM) Model", "SVMModel", feature_set)
exp.run("Non-lab(RF) Model", "RandomForestModel", feature_set)
exp.run("Non-lab(AutoGluonModel) Model", "AutoGluonModel", feature_set)
exp.plot()
exp.save("model_test_top20_auc")

# %%
exp = auPRNonLab()
exp.run("Non-lab(LGBM)", "LightGBMModel", feature_set)
exp.run("Non-lab(ANN)", "ANNModel", feature_set)
exp.run("CDS", "CHModel", "CH")
exp.run("Non-lab(LR) Model", "LogisticRegressionModel", feature_set)
exp.run("Non-lab(SVM) Model", "SVMModel", feature_set)
exp.run("Non-lab(RF) Model", "RandomForestModel", feature_set)
exp.run("Non-lab(AutoGluonModel) Model", "AutoGluonModel", feature_set)
exp.plot()
exp.save("model_test_top20_auPR")


# %%
