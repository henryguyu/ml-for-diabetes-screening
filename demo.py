import logging
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction.models import (
    BaseModel,
    LightGBMModel,
    ANNModel,
    LogisticRegressionModel,
)
from lxh_prediction import data_utils
from lxh_prediction.plot import plot_curve


logging.basicConfig(level=logging.INFO)


def train():
    X, y, feat_names = data_utils.load_data(cfg.feature_fields["without_FPG"])
    X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

    # model = LightGBMModel(
    #     {
    #         "boosting": "gbdt",
    #         "num_leaves": 18,
    #         "max_bin": 70,
    #         "max_depth": 64,
    #         "learning_rate": 0.0002039445148616998,
    #         "lambda_l1": 0.0001,
    #         "lambda_l2": 0.001,
    #         "feature_fraction": 1,
    #         "min_data_in_bin": 5,
    #         "bagging_fraction": 0.5,
    #         "bagging_freq": 4,
    #         "path_smooth": 0.0001,
    #     }
    # )

    model = ANNModel(
        {
            "lr": 0.015596326148781257,
            "weight_decay": 0.001,
            "batch_size": 26,
            "enable_lr_scheduler": 0,
            "opt": "Adam",
            "n_channels": 154,
            "n_layers": 5,
            "dropout": 0,
            "activate": "ReLU",
            "branches": [2, 1],
        },
        feature_len=X_train.shape[1],
    )
    # model.load("data/ann_with_FPG.pth")

    # model = LogisticRegressionModel({"solver": "saga", "max_iter": 1000})

    # cv_aucs = model.cross_validate(X, y, metric_fn=LightGBMModel.roc_auc_score)
    # print(cv_aucs, np.mean(cv_aucs))

    model.fit(X_train, y_train, X_test, y_test)

    # # feat importance
    # feat_importances = list(zip(feat_names, model.model.feature_importance()))
    # feat_importances = sorted(feat_importances, key=lambda x: -x[1])
    # print(feat_importances)

    probs_pred = model.predict(X_test)
    print(y_test.mean())

    roc_auc = model.roc_auc_score(y_test, probs_pred)
    fpr, tpr, _ = model.roc_curve(y_test, probs_pred)

    precision, recall, _ = model.precision_recall_curve(y_test, probs_pred)
    plot_curve(
        recall,
        precision,
        xlabel="Reccall",
        ylabel="Precision",
        # subline=((0, 1), (0, 1)),
    )
    plot_curve(
        fpr,
        tpr,
        auc=roc_auc,
        xlabel="False positive rate",
        ylabel="True positive rate",
        subline=((0, 1), (0, 1)),
    )

    print()
    # print(model.average_precision_score(y_test, probs_pred))

    # model.save("data/ann_with_FPG.pth")


if __name__ == "__main__":
    train()
