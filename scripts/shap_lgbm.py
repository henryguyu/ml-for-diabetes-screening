# %%
import shap
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils
from lxh_prediction.models import LightGBMModel

# print the JS visualization code to the notebook
shap.initjs()

# %%
# Load data
feat_collection = "without_FPG"
X, y = data_utils.load_data(cfg.feature_fields[feat_collection])
X_FPG, _ = data_utils.load_data(["FPG"])
X_display = X
X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

# Train model
key = ("LightGBMModel", feat_collection)
model = LightGBMModel(cfg.model_params.get(key, {}))
model.fit(X_train.iloc[:-1], y_train.iloc[:-1], X_test, y_test)

# %%
# Predict
preds = model.predict(X_test)
precisions, recalls, threshs = metric_utils.precision_recall_curve(y_test, preds)
idx = len(recalls) - np.searchsorted(recalls[::-1], 0.8, side="right") - 1
print(precisions[idx], recalls[idx], threshs[idx])
thresh = threshs[idx]
index = y_test.index

preds = preds.iloc[:, 0] >= thresh
TP = index[(preds >= thresh) & (y_test > 0)]
FN = index[(preds < thresh) & (y_test > 0)]

FPG = X_FPG["FPG"]
hard_mask = (FPG < 7) & (y_test > 0)
X_hard = X_test[hard_mask]
y_hard = y_test[hard_mask]

# %%
explainer = shap.TreeExplainer(
    model.model,
    data=shap.sample(X_train, 100),
    model_output="probability",
    feature_perturbation="interventional",
)
shap_values = explainer.shap_values(X)
# %%
idx = X_hard.index[0]
print(y.iloc[idx])
shap.force_plot(explainer.expected_value, shap_values[idx, :], X_display.iloc[idx, :])


# %%
shap.summary_plot(shap_values, X, plot_type="violin")
# %%
name = "Ahr"
shap.dependence_plot(
    name, shap_values, X, display_features=X_display, interaction_index="age"
)

# %%
shap.force_plot(explainer.expected_value, shap_values[:1000, :], X.iloc[:1000, :])

