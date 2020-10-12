import logging

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
    #         "num_boost_round": 100,
    #         "num_leaves": 51,
    #         "max_bin": 31,
    #         "learning_rate": 0.01,
    #         "objective": "binary",
    #         # "num_class": 2,
    #         # "lambda_l1": 0.001,
    #         "lambda_l2": 0.05,
    #         # "feature_fraction": 0.9,
    #         # "min_data_in_bin": 5,
    #         "early_stopping_round": 20,
    #         # "max_depth": 30,
    #         # "bagging_fraction": 0.6,
    #         "boosting": "gbdt",
    #         "metric": ["auc"],
    #     }
    # )

    model = ANNModel(
        {
            "lr": 1e-2,
            "weight_decay": 0.001,
            "batch_size": 256,
            "enable_lr_scheduler": True,
            "num_epoch": 40,
        },
        feature_len=X_train.shape[1],
    )
    # model.load("data/ann_with_FPG.pth")

    # model = LogisticRegressionModel({"solver": "saga", "max_iter": 1000})

    cv_aucs = model.cross_validate(X, y, metric_fn=LightGBMModel.roc_auc_score)
    print(cv_aucs)

    # model.fit(X_train, y_train, X_test, y_test)
    probs_pred = model.predict(X_test)
    print(probs_pred)

    roc_auc = model.roc_auc_score(y_test, probs_pred)
    fpr, tpr, _ = model.roc_curve(y_test, probs_pred)

    precision, recall, _ = model.precision_recall_curve(y_test, probs_pred)
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
