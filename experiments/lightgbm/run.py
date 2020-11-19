import argparse
import logging

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

    params.update({"metric": ["auc"]})
    resample = params.get("resample", 0)
    model = LightGBMModel(params)
    results = model.cross_validate(
        X, y, getattr(metric_utils, metric), resample_train=resample
    )[0]
    return np.mean(results)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    args = parse_args()
    res = train(params, args.collection, args.metric)
    nni.report_final_result(res)
