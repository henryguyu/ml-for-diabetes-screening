import logging

import lxh_prediction.config as cfg
from lxh_prediction.models import BaseModel, LightGBMModel, ANNModel
from lxh_prediction import data_utils


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
    model.load("data/ann_without.pth")

    # model.fit(X_train, y_train, X_test, y_test)
    probs_pred = model.predict(X_test)
    print(model.precision_recall_curve(y_test, probs_pred))
    print(model.roc_auc_score(y_test, probs_pred))

    model.save("data/ann_without.pth")


if __name__ == "__main__":
    train()
