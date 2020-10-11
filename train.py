import logging

import lxh_prediction.config as cfg
from lxh_prediction.models import BaseModel, LightGBMModel, ANNModel
from lxh_prediction import data_utils


logging.basicConfig(level=logging.INFO)


def train():
    X, y, feat_names = data_utils.load_data(cfg.feature_fields["with_FPG"])
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
    model.load("data/ann_with_FPG.pth")

    # model.fit(X_train, y_train, X_test, y_test)
    probs_pred = model.predict(X_test)
    roc_auc = model.roc_auc_score(y_test, probs_pred)
    fpr, tpr, _ = model.roc_curve(y_test, probs_pred)

    precision, recall, _ = model.precision_recall_curve(y_test, probs_pred)

    import matplotlib.pyplot as plt

    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    print()
    # print(model.average_precision_score(y_test, probs_pred))

    # model.save("data/ann_with_FPG.pth")


if __name__ == "__main__":
    train()
