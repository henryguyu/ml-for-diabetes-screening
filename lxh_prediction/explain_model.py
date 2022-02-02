import shap

import lxh_prediction.config as cfg
from lxh_prediction import data_utils
from lxh_prediction.models import LightGBMModel


def explain_with_shape_lgbm(feat_collection: str):
    # Load data
    X, y = data_utils.load_data(cfg.feature_fields[feat_collection])
    X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

    # Analyze model
    key = ("LightGBMModel", feat_collection)
    model = LightGBMModel(cfg.model_params.get(key, {}))
    model.fit(X_train.iloc[:-1], y_train.iloc[:-1], X_test, y_test)
    explainer = shap.TreeExplainer(model.model)

    # Feature names
    feature_names = list(X.columns)
    name_to_index = {name: i for i, name in enumerate(feature_names)}
    name_maps = {
        "Ahr": "RPR",
        "age": "Age",
        "lwork": "Work",
        "wc": "WC",
        "culutrue": "Education",
        "lusephy": "Years of cellphone use",
        "ASBP": "SBP",
        "ADBP": "DBP",
        "lgetup": "Wake time",
        "hc": "HC",
        "lvigday": "Days of vigorous activity",
        "lvighour": "Hours of vigorous activity",
        "ldrinking": "Drinking",
        "frye0": "Fried food",
        "ntime": "Daytime sleep duration",
        "nigtime": "Nighttime sleep duration",
    }
    for ori, new in name_maps.items():
        feature_names[name_to_index[ori]] = new

    return explainer, X, feature_names
