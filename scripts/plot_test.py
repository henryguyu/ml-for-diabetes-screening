import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

X, y = make_classification(n_samples=500, random_state=100, flip_y=0.3)

kf = KFold(n_splits=5)

tprs = []
base_fpr = np.linspace(0, 1, 101)

plt.figure(figsize=(5, 5))

for i, (train, test) in enumerate(kf.split(X)):
    model = LogisticRegression().fit(X[train], y[train])
    y_score = model.predict_proba(X[test])
    fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])

    # plt.plot(fpr, tpr, "b", alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std


plt.plot(base_fpr, mean_tprs, "b")
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)

plt.plot([0, 1], [0, 1], "r--")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.axes().set_aspect("equal", "datalim")
plt.show()
