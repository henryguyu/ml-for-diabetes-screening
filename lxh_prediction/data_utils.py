import numpy as np
import pandas as pd

import lxh_prediction.config as cfg


def read_csv(filename, check_nan=True) -> pd.DataFrame:
    df = pd.read_csv(filename).astype(float)
    if check_nan:
        assert not np.any(df.isnull()), "null value in data"
    return df


def onehotify(df: pd.DataFrame, colname, rm_origin=True):
    onehots = pd.get_dummies(df[colname].astype(int), prefix=colname)
    if rm_origin:
        df = df.drop(columns=colname)
    return pd.concat([df, onehots], axis=1)


def load_data(
    feature_fields,
    filename=cfg.data_file,
    onehot_fields=cfg.onehot_fields,
    label_field=cfg.label_field,
):
    df = read_csv(filename)
    if not feature_fields:
        feature_fields = list(df.columns)
        feature_fields.remove(label_field)

    y = df[label_field].copy()
    X = df[feature_fields].copy()
    for name in onehot_fields:
        if name in X:
            X = onehotify(X, name, rm_origin=True)
    return X, y


def split_data(X: pd.DataFrame, y: pd.DataFrame = None, train_ratio=0.8, seed=1063):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    num_trains = int(len(indices) * train_ratio)
    train_indices = indices[:num_trains]
    test_indices = indices[num_trains:]

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    if y is None:
        return X_train, X_test
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()
    return X_train, y_train, X_test, y_test


def split_cross_validation(
    X: pd.DataFrame, y: pd.DataFrame, n_folds: int = 5, seed=1063
):
    np.random.seed(seed)
    indices = np.random.permutation(len(X)).tolist()
    num = len(X) // n_folds
    for i in range(n_folds):
        start = i * num
        end = (i + 1) * num if i + 1 < n_folds else None
        valid_indices = indices[start:end]
        train_indices = indices[0:start] + (indices[end:] if end is not None else [])
        X_train = X.iloc[train_indices].copy()
        y_train = y.iloc[train_indices].copy()
        X_valid = X.iloc[valid_indices].copy()
        y_valid = y.iloc[valid_indices].copy()
        yield X_train, y_train, X_valid, y_valid
