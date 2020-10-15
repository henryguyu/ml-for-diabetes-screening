import logging
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import models
from lxh_prediction import data_utils
from lxh_prediction.plot import plot_curve
from lxh_prediction import metric_utils


logging.basicConfig(level=logging.INFO)


def train():
    X, y, feat_names = data_utils.load_data(cfg.feature_fields["CH"])

    model = models.CHModel()
    cv_aucs, cv_probs_pred, cv_indices = model.cross_validate(
        X, y, metric_fn=metric_utils.roc_auc_score, feat_names=feat_names
    )
    print(cv_aucs, np.mean(cv_aucs))
    probs_pred, indices = cv_probs_pred[0], cv_indices[0]
    y_test = y[indices]

    costs, miss_rate, _ = metric_utils.cost_curve_without_FPG(y_test, probs_pred)
    print(costs)

    # FPG_test = FPG[indices]
    # costs, miss_rate, _ = metric_utils.cost_curve_with_FPG(y_test, probs_pred, FPG_test)

    plot_curve(
        miss_rate,
        costs,
        ylim=[0, 70],
        xlabel="Prediction miss rate",
        ylabel="Cost",
        # subline=((0, 1), (0, 1)),
    )

    nag_rate, miss_rate, _ = metric_utils.nag_miss_curve(y_test, probs_pred)
    plot_curve(
        miss_rate,
        nag_rate,
        xlabel="Prediction miss rate",
        ylabel="Patients avoiding examination",
        # subline=((0, 1), (0, 1)),
    )

    # precision, recall, _ = metric_utils.precision_recall_curve(y_test, probs_pred)
    # plot_curve(
    #     recall,
    #     precision,
    #     xlabel="Reccall",
    #     ylabel="Precision",
    #     # subline=((0, 1), (0, 1)),
    # )

    roc_auc = metric_utils.roc_auc_score(y_test, probs_pred)
    fpr, tpr, _ = metric_utils.roc_curve(y_test, probs_pred)
    plot_curve(
        fpr,
        tpr,
        name="ROC curve (area = %0.2f)" % roc_auc,
        xlabel="False positive rate",
        ylabel="True positive rate",
        subline=((0, 1), (0, 1)),
    )

    print()
    # print(model.average_precision_score(y_test, probs_pred))

    # model.save("data/ann_with_FPG.pth")


if __name__ == "__main__":
    train()
