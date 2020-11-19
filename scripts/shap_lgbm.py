# %%
import numpy as np
import pandas as pd
import shap

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils
from lxh_prediction.models import LightGBMModel
from lxh_prediction.plot import plot_curve, plot_range

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
idx = X_hard.index[1]
print(y.iloc[idx])
shap.force_plot(explainer.expected_value, shap_values[idx, :], X_display.iloc[idx, :])


# %%
shap.summary_plot(shap_values, X, plot_type="violin")
# %%
name = "BMI"
shap.dependence_plot(
    name, shap_values, X, display_features=X_display, interaction_index=None
)

# %%


def ABS_SHAP(df_shap, df):
    # import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop("index", axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = []
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(
        0
    )
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ["Variable", "Corr"]
    corr_df["Sign"] = np.where(corr_df["Corr"] > 0, "red", "blue")

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ["Variable", "SHAP_abs"]
    k2 = k.merge(corr_df, left_on="Variable", right_on="Variable", how="inner")
    k2 = k2.sort_values(by="SHAP_abs", ascending=True)
    k2 = k2.iloc[-20:]
    colorlist = k2["Sign"]
    ax = k2.plot.barh(
        x="Variable", y="SHAP_abs", color=colorlist, figsize=(5, 6), legend=False
    )
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")


ABS_SHAP(shap_values, X)

# %%


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


shap_v = pd.DataFrame(shap_values, columns=X.columns)
P0 = y.mean()
phi0 = np.log(P0 / (1 - P0))

RR = sigmoid(shap_v + phi0) / sigmoid(phi0)

# %%
name = "mike0"
shap.dependence_plot(
    name, RR.values, X, display_features=X_display, interaction_index=None
)

# %%

from scipy.stats import binned_statistic

name = "WHR"
bins = 7
mean, bin_edges, _ = binned_statistic(X[name], RR[name], "mean", bins=bins)
std, _, _ = binned_statistic(X[name], RR[name], "std", bins=bins)
x = (bin_edges[1:] + bin_edges[:-1]) * 0.5
plot_curve(
    x,
    mean,
    xlim=(x.min(), x.max()),
    ylim=(mean.min() - std.max(), mean.max() + std.max()),
)
plot_range(x, mean - std, mean + std)
# %%
