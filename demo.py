import logging
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import models
from lxh_prediction import data_utils
from lxh_prediction.plot import plot_curve
from lxh_prediction import metric_utils


logging.basicConfig(level=logging.INFO)


def train():
    X, y, feat_names, FPG = data_utils.load_data(
        cfg.feature_fields["without_FPG"], extra_fields="FPG"
    )
    # X, y, feat_names = data_utils.load_data(
    #     None, filename="data/pca_with_FPG.csv", onehot_fields=[]
    # )
    # X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

    model = models.LightGBMModel(
        {
            "num_leaves": 16,
            "max_bin": 162,
            "max_depth": 256,
            "learning_rate": 0.028753305217484978,
            "lambda_l1": 0.1,
            "lambda_l2": 0.001,
            "feature_fraction": 0.7,
            "min_data_in_bin": 5,
            "bagging_fraction": 0.5,
            "bagging_freq": 4,
            "path_smooth": 0.01,
        }
    )

    # model = models.ANNModel(
    #     {
    #         "lr": 0.04150735339940105,
    #         "weight_decay": 0.0005,
    #         "batch_size": 251,
    #         "enable_lr_scheduler": 0,
    #         "opt": "Adam",
    #         "n_channels": 428,
    #         "n_layers": 5,
    #         "dropout": 1,
    #         "activate": "Tanh",
    #         "branches": [1],
    #     },
    #     feature_len=X.shape[1],
    # )

    # model = models.SVMModel({"kernel": "linear"})
    # model = models.LogisticRegressionModel()

    # model.load("data/ann_with_FPG.pth")

    # model = LogisticRegressionModel({"solver": "saga", "max_iter": 1000})

    # # feat importance
    # model.fit(X, y)
    # try:
    #     feat_importances = list(zip(feat_names, model.feature_importance()))
    #     feat_importances = sorted(feat_importances, key=lambda x: -x[1])
    #     print(feat_importances)
    # except NotImplementedError:
    #     pass

    cv_aucs, cv_probs_pred, cv_indices = model.cross_validate(
        X, y, metric_fn=metric_utils.roc_auc_score
    )
    print(cv_aucs, np.mean(cv_aucs))
    probs_pred, indices = cv_probs_pred[0], cv_indices[0]
    y_test = y[indices]

    # costs, miss_rate, _ = metric_utils.cost_curve_without_FPG(y_test, probs_pred)

    # FPG_test = FPG[indices]
    # costs, miss_rate, _ = metric_utils.cost_curve_with_FPG(y_test, probs_pred, FPG_test)

    # plot_curve(
    #     miss_rate,
    #     costs,
    #     ylim=[0, 70],
    #     xlabel="Prediction miss rate",
    #     ylabel="Cost",
    #     # subline=((0, 1), (0, 1)),
    # )

    # nag_rate, miss_rate, _ = metric_utils.nag_miss_curve(y_test, probs_pred)
    # plot_curve(
    #     miss_rate,
    #     nag_rate,
    #     xlabel="Prediction miss rate",
    #     ylabel="Patients avoiding examination",
    #     # subline=((0, 1), (0, 1)),
    # )

    precision, recall, _ = metric_utils.precision_recall_curve(y_test, probs_pred)
    plot_curve(
        recall,
        precision,
        xlabel="Reccall",
        ylabel="Precision",
        # subline=((0, 1), (0, 1)),
    )

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
