import logging
import nni
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction.models import ANNModel
from lxh_prediction import data_utils


logging.basicConfig(level=logging.INFO)


def train(params):
    X, y, feat_names = data_utils.load_data(cfg.feature_fields["with_FPG"])

    params.update({"num_epoch": 60},)
    model = ANNModel(params)
    rocs = model.cross_validate(X, y, model.roc_auc_score)
    return np.mean(rocs)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    roc_auc = train(params)
    nni.report_final_result(roc_auc)
