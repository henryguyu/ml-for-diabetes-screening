import logging
import argparse

import nni
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils
from lxh_prediction.models import LightGBMModel

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="without_FPG")
    parser.add_argument("--metric", type=str, default="roc_auc_score")
    return parser.parse_args()


def train(params, collection="without_FPG", metric="roc_auc_score"):
    X, y = data_utils.load_data(cfg.feature_fields[collection])

    params.update(
        {"metric": ["auc"]}
    )
    model = LightGBMModel(params)
    results = model.cross_validate(X, y, getattr(metric_utils, metric))[0]
    return np.mean(results)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    roc_auc = train(params)
    nni.report_final_result(roc_auc)
