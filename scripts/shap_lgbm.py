# %%
import numpy as np
import pandas as pd
import shap
import os

import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils
from lxh_prediction.models import LightGBMModel
from lxh_prediction.plot import plot_curve, plot_range

# print the JS visualization code to the notebook
shap.initjs()

# %%
# Load data
feat_collection = "top20_non_lab"
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
all_preds = model.predict(X)
preds = model.predict(X_test)
precisions, recalls, threshs = metric_utils.precision_recall_curve(y_test, preds)
idx = len(recalls) - np.searchsorted(recalls[::-1], 0.8, side="right") - 1
print(precisions[idx], recalls[idx], threshs[idx])
thresh = threshs[idx]
index = y_test.index

preds = preds.iloc[:, 0] >= thresh
TP = index[(preds >= thresh) & (y_test > 0)]
FP = index[(preds >= thresh) & (y_test == 0)]
TN = index[(preds < thresh) & (y_test == 0)]
FN = index[(preds < thresh) & (y_test > 0)]

FPG = X_FPG["FPG"]
hard_mask = (FPG < 7) & (y_test > 0)
X_hard = X_test[hard_mask]
y_hard = y_test[hard_mask]

# %%
# explainer = shap.TreeExplainer(
#     model.model,
#     data=shap.sample(X_train, 100),
#     model_output="probability",
#     feature_perturbation="interventional",
# )
explainer = shap.TreeExplainer(model.model)
shap_values = explainer.shap_values(X)[1]
expected_value = explainer.expected_value[1]

feature_names = list(X.columns)
name_to_index = {name: i for i, name in enumerate(feature_names)}
name_maps = {
    "Ahr": "Rhr",
    "age": "Age",
    "lwork": "Work",
    "wc": "WC",
    "culutrue": "Education",
    "lusephy": "Phone",
    "ASBP": "SBP",
    "ADBP": "DBP",
    "lgetup": "Getuptime",
    "hc": "HC",
    "lvigday": "HeavyPAday",
    "lvighour": "HeavyPAhour",
    "ldrinking": "Drinking",
    "frye0": "Fry",
    "ntime": "Naptime",
    "nigtime": "Nitime",
}
for ori, new in name_maps.items():
    feature_names[name_to_index[ori]] = new
# %%
# idx = X_hard.index[2]
idx = TP[24]
print(y.iloc[idx])
shap.force_plot(
    expected_value,
    shap_values[idx, :],
    X_display.iloc[idx, :],
    feature_names=feature_names,
)


# %%
shap.summary_plot(
    shap_values, X, plot_type="dot", feature_names=feature_names,
)
# %%
name = "BMI"
shap.dependence_plot(
    name,
    shap_values,
    X,
    display_features=X_display,
    interaction_index=None,
    feature_names=feature_names,
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
        b = np.corrcoef(shap_v[i], df_v[i])
        print(b)
        corr_list.append(b[1][0])
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
    k2["VarRenamed"] = [name_maps.get(name, name) for name in k2["Variable"]]
    colorlist = k2["Sign"]
    ax = k2.plot.barh(
        x="Variable", y="SHAP_abs", color=colorlist, figsize=(5, 6), legend=False
    )
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    return k2


k2 = ABS_SHAP(shap_values, X)
k2.to_csv(os.path.join(cfg.root, "data/results/shap_abs.csv"))

# %%


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def desigmoid(p):
    return np.log(p / (1 - p))


shap_v = pd.DataFrame(shap_values, columns=X.columns)
# P0 = y.mean()
# phi0 = np.log(P0 / (1 - P0))
phi0 = expected_value

RR = sigmoid(shap_v + phi0) / sigmoid(phi0)

# %%
name = "Phone"
shap.dependence_plot(
    name,
    RR.values,
    X,
    display_features=X_display,
    interaction_index="Age",
    feature_names=feature_names,
)

# %%

from scipy.stats import binned_statistic

name = "wc"
bins = 5
mean, bin_edges, _ = binned_statistic(X[name], RR[name], "mean", bins=bins)
std, _, _ = binned_statistic(X[name], RR[name], "std", bins=bins)
x = (bin_edges[1:] + bin_edges[:-1]) * 0.5
plot_curve(
    x,
    mean,
    xlim=(x.min(), x.max()),
    ylim=(mean.min() - std.max(), mean.max() + std.max()),
    xlabel=name,
    ylabel="Relative risk",
)
plot_range(x, mean - std, mean + std)
# %%
