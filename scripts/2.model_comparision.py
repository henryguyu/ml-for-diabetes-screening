# %%
import lxh_prediction.config as cfg
from lxh_prediction import data_utils, metric_utils, models


# %%
# Load data
feat_collection = "full_non_lab"
X, y = data_utils.load_data(cfg.feature_fields[feat_collection])
X_train, y_train, X_test, y_test = data_utils.split_data(X, y)

# %%
# Load model
model_name = "LogisticRegressionModel"
params = {"max_iter": 1000, "class_weight": "balanced"}
print(f"Using params: {params}")
model = getattr(models, model_name)(params=params)

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print("Train:", metric_utils.roc_auc_score(y_train, y_pred))


y_pred = model.predict(X_test)
print("Test:", metric_utils.roc_auc_score(y_test, y_pred))

# %%
# Load model
model_name = "SVMModel"
params = {"class_weight": "balanced", "kernel": "linear"}
# params = {}
print(f"Using params: {params}")
model = getattr(models, model_name)(params=params)

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print("Train:", metric_utils.roc_auc_score(y_train, y_pred))


y_pred = model.predict(X_test)
print("Test:", metric_utils.roc_auc_score(y_test, y_pred))

# %%
# Load model
model_name = "RandomForestModel"
params = {
    "class_weight": "balanced",
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 0.25,
}
# params = {}
print(f"Using params: {params}")
model = getattr(models, model_name)(params=params)

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print("Train:", metric_utils.roc_auc_score(y_train, y_pred))


y_pred = model.predict(X_test)
print("Test:", metric_utils.roc_auc_score(y_test, y_pred))

# %%
# Load model
model_name = "ANNModel"
params = {}
print(f"Using params: {params}")
model = getattr(models, model_name)(params=params)

model.fit(X_train, y_train, X_test, y_test)
y_pred = model.predict(X_train)
print("Train:", metric_utils.roc_auc_score(y_train, y_pred))


y_pred = model.predict(X_test)
print("Test:", metric_utils.roc_auc_score(y_test, y_pred))
# %%
