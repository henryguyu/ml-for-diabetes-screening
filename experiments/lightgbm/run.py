import logging
import nni
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction.models import LightGBMModel
from lxh_prediction import data_utils


logging.basicConfig(level=logging.INFO)


def train(params):
    X, y, feat_names = data_utils.load_data(cfg.feature_fields["without_FPG"])
    # X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

    params.update(
        {
            "num_boost_round": 100,
            "metric": ["auc"],
            "early_stopping_round": 20,
            "objective": "binary",
        }
    )
    model = LightGBMModel(params)
    # model.fit(X_train, y_train, X_test, y_test)
    rocs = model.cross_validate(X, y, model.roc_auc_score)[0]

    # probs_pred = model.predict(X_test)
    # roc_auc = model.roc_auc_score(y_test, probs_pred)
    return np.mean(rocs)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    roc_auc = train(params)
    nni.report_final_result(roc_auc)
