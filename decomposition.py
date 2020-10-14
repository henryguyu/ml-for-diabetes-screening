# %%
import pandas as pd

import lxh_prediction.config as cfg
from lxh_prediction import models
from lxh_prediction import data_utils

# %%


def decomp_with_PCA(X, y, feat_names):
    pca = models.PCAModel()
    pca.fit(X, y)
    X_decomp = pca.predict(X)

    label = pd.Series(y, name=cfg.label_field)
    df = pd.DataFrame(X_decomp, columns=[f"feat_{i}" for i in range(X_decomp.shape[1])])
    df = pd.concat([df, label], axis=1)
    return df


for name, fields in cfg.feature_fields.items():
    X, y, feat_names = data_utils.load_data(fields)
    df = decomp_with_PCA(X, y, feat_names)
    df.to_csv(f"data/pca_{name}.csv", index=False)

# %%


def decomp_with_SVM(X, y, feat_names):
    svm = models.SVMModel({"kernel": "linear"})
    svm.fit(X, y)
    feat_importance = svm.feature_importance()
    feat_weights = list(zip(feat_names, feat_importance))
    feat_weights = sorted(feat_weights, key=lambda x: -abs(x[1]))
    print(feat_weights)

    n_sel = X.shape[1] // 2
    sel_names = [x[0] for x in feat_weights[:n_sel]]
    raw_df = pd.DataFrame(X, columns=feat_names)
    df = raw_df[sel_names]

    label = pd.Series(y, name=cfg.label_field)
    df = pd.concat([df, label], axis=1)
    return df


for name, fields in cfg.feature_fields.items():
    X, y, feat_names = data_utils.load_data(fields)
    df = decomp_with_SVM(X, y, feat_names)
    df.to_csv(f"data/svm_{name}.csv", index=False)

# %%
