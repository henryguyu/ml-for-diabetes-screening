import logging
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import models
from lxh_prediction import data_utils
from lxh_prediction.plot import plot_curve
from lxh_prediction import metric_utils


logging.basicConfig(level=logging.INFO)


def train():
    X, y, feat_names = data_utils.load_data(cfg.feature_fields["without_FPG"])
    # X, y, feat_names = data_utils.load_data(
    #     None, filename="data/pca_with_FPG.csv", onehot_fields=[]
    # )
    X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

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
    #         "lr": 0.015596326148781257,
    #         "weight_decay": 0.001,
    #         "batch_size": 26,
    #         "enable_lr_scheduler": 0,
    #         "opt": "Adam",
    #         "n_channels": 154,
    #         "n_layers": 5,
    #         "dropout": 0,
    #         "activate": "ReLU",
    #         "branches": [2, 1],
    #     },
    #     feature_len=X_train.shape[1],
    # )

    # model = models.SVMModel({"kernel": "linear"})
    # model = models.LogisticRegressionModel()

    # model.load("data/ann_with_FPG.pth")

    # model = LogisticRegressionModel({"solver": "saga", "max_iter": 1000})

    # cv_aucs = model.cross_validate(X, y, metric_fn=LightGBMModel.roc_auc_score)
    # print(cv_aucs, np.mean(cv_aucs))

    model.fit(X_train, y_train, X_test, y_test)

    # feat importance
    feat_importances = list(zip(feat_names, model.feature_importance()))
    feat_importances = sorted(feat_importances, key=lambda x: -x[1])
    print(feat_importances)

    probs_pred = model.predict(X_test)
    print(y_test.mean())
    print(np.mean((probs_pred > 0) & y_test))

    nag_rate, miss_rate, _ = metric_utils.nag_miss_curve(y_test, probs_pred)
    plot_curve(
        miss_rate,
        nag_rate,
        xlabel="Prediction miss rate",
        ylabel="Patients avoiding examination",
        # subline=((0, 1), (0, 1)),
    )

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
