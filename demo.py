import logging
import argparse

import lxh_prediction.config as cfg
from lxh_prediction import models
from lxh_prediction import data_utils
from lxh_prediction.plot import plot_curve
from lxh_prediction import metric_utils


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="without_FPG")
    parser.add_argument("--model", type=str, default="LightGBMModel")
    return parser.parse_args()


def train(feat_collection="without_FPG", model_name="LightGBMModel"):
    X, y = data_utils.load_data(cfg.feature_fields[feat_collection])
    # X, y, feat_names = data_utils.load_data(
    #     None, filename="data/pca_with_FPG.csv", onehot_fields=[]
    # )
    # X_train, y_train, X_test, y_test = data_utils.split_data(X, y)
    X_train, y_train, X_test, y_test = next(data_utils.split_cross_validation(X, y))

    model = getattr(models, model_name)()

    model.fit(X_train, y_train, X_test, y_test)

    # # feat importance
    # try:
    #     feat_importances = list(zip(feat_names, model.feature_importance()))
    #     feat_importances = sorted(feat_importances, key=lambda x: -x[1])
    #     print(feat_importances)
    # except NotImplementedError:
    #     pass

    # cv_aucs, cv_probs_pred, cv_indices = model.cross_validate(
    #     X, y, metric_fn=metric_utils.roc_auc_score
    # )
    probs_pred = model.predict(X_test)

    if "FPG" not in X:
        costs, miss_rate, _ = metric_utils.cost_curve_without_FPG(y_test, probs_pred)
    else:
        FPG = X_test["FPG"]
        costs, miss_rate, _ = metric_utils.cost_curve_with_FPG(y_test, probs_pred, FPG)
    plot_curve(
        miss_rate,
        costs,
        ylim=[0, 70],
        xlabel="Prediction miss rate",
        ylabel="Cost",
        # subline=((0, 1), (0, 1)),
        title="Cost vs Miss rate",
    )

    nag_rate, miss_rate, _ = metric_utils.nag_miss_curve(y_test, probs_pred)
    plot_curve(
        miss_rate,
        nag_rate,
        xlabel="Prediction miss rate",
        ylabel="Patients avoiding examination",
        # subline=((0, 1), (0, 1)),
        title="Avoid test vs Miss rate"
    )

    precision, recall, _ = metric_utils.precision_recall_curve(y_test, probs_pred)
    plot_curve(
        recall,
        precision,
        xlabel="Reccall",
        ylabel="Precision",
        # subline=((0, 1), (0, 1)),
        title="Precision vs Recall",
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
        title="ROC curve",
    )

    print()
    # print(model.average_precision_score(y_test, probs_pred))

    # model.save("data/ann_with_FPG.pth")


if __name__ == "__main__":
    args = parse_args()
    train(args.collection, args.model)
