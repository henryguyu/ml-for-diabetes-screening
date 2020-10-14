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

    y = df[label_field].to_numpy(dtype=int, copy=True)
    X = df[feature_fields]
    for name in onehot_fields:
        if name in X:
            X = onehotify(X, name, rm_origin=True)
    feat_names = list(X.columns)
    X = X.to_numpy(dtype=float, copy=True)
    return X, y, feat_names


def split_data(X: np.ndarray, y: np.ndarray = None, train_ratio=0.8, seed=1063):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    num_trains = int(len(indices) * train_ratio)
    train_indices = indices[:num_trains]
    test_indices = indices[num_trains:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    if y is None:
        return X_train, X_test
    return X_train, y[train_indices], X_test, y[test_indices]


def split_cross_validation(X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed=1063):
    np.random.seed(seed)
    indices = np.random.permutation(len(X)).tolist()
    num = len(X) // n_folds
    for i in range(n_folds):
        valid_indices = indices[i * num : (i + 1) * num]
        train_indices = indices[0 : i * num] + indices[(i + 1) * num :]
        yield X[train_indices], y[train_indices], X[valid_indices], y[valid_indices]
