# %%
import shap
import numpy as np
import logging

import lxh_prediction.config as cfg
from lxh_prediction import data_utils
from lxh_prediction.models import ANNModel

logging.basicConfig(level=logging.INFO)
# print the JS visualization code to the notebook
shap.initjs()


# %%
# Load data
feat_collection = "without_FPG"
X, y = data_utils.load_data(cfg.feature_fields[feat_collection])
X_display = X
X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

# Train model
model = ANNModel(feature_len=X_train.shape[1])
# model.fit(X_train, y_train, X_test, y_test)
model.load("/tmp/shap_ann")

# %%
explainer = shap.KernelExplainer(model.predict, data=shap.kmeans(X_train, 10))
shap_values = explainer.shap_values(X_test.iloc[:100], nsamples=100)
# %%
idx = 30
print(y_test.iloc[idx])
shap.force_plot(explainer.expected_value, shap_values[idx, :], X_test.iloc[idx, :])


# %%
shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="dot")
# %%
name = "Ahr"
shap.dependence_plot(
    name, shap_values, X_test.iloc[:10], display_features=X_test.iloc[:10]
)

# %%
shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[:10])


# %%
