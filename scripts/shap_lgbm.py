# %%
import shap
import numpy as np

import lxh_prediction.config as cfg
from lxh_prediction import data_utils
from lxh_prediction.models import LightGBMModel

# print the JS visualization code to the notebook
shap.initjs()

# %%
# Load data
feat_collection = "without_FPG"
X, y = data_utils.load_data(cfg.feature_fields[feat_collection])
X_display = X
X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

# Train model
model = LightGBMModel()
model.fit(X_train, y_train, X_test, y_test)

# %%
explainer = shap.TreeExplainer(
    model.model,
    data=X_test,
    model_output="probability",
    feature_perturbation="interventional",
)
shap_values = explainer.shap_values(X)
# %%
idx = 2
print(y.iloc[idx])
shap.force_plot(explainer.expected_value, shap_values[idx, :], X_display.iloc[idx, :])


# %%
shap.summary_plot(shap_values, X, plot_type="dot")
# %%
name = "lusephy"
shap.dependence_plot(
    name, shap_values, X, display_features=X_display, interaction_index="age", xmax=30
)

# %%
shap.force_plot(explainer.expected_value, shap_values[:1000, :], X.iloc[:1000, :])

