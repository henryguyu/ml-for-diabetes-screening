# %%
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.compare_auc_delong_xu import delong_roc_test
from sklearn import metrics
import itertools
import numpy as np


# %%

compare_set = [
    ("CHModel", "CH"),
    ("ADAModel", "ADA"),
    ("LightGBMModel", "top20_non_lab"),
    ("EnsembleModel", "top20_non_lab"),
    ("AutoLightGBMModel", "top20_non_lab"),
    ("LightGBMModel", "full_non_lab"),
]
cv_y_prob_dict = {}
for model_name, feat_collection in compare_set:
    cv_y_prob_dict[(model_name, feat_collection)] = get_cv_preds(
        model_name=model_name,
        feat_collection=feat_collection,
        update=False,
        resample_train=False,
    )

# %%
curve_keys = list(cv_y_prob_dict.keys())
for k1, k2 in itertools.combinations(curve_keys, 2):
    print(k1, k2)
    for (y1, prob1), (y2, prob2) in zip(cv_y_prob_dict[k1], cv_y_prob_dict[k2]):
        assert np.all(y1 == y2)
        y = np.asarray(y1).reshape(-1)
        prob1 = np.asarray(prob1).reshape(-1)
        prob2 = np.asarray(prob2).reshape(-1)
        log10p = delong_roc_test(y, prob1, prob2)
        pvalue = np.power(10, log10p).reshape(-1)[0]
        auc1, auc2 = metrics.roc_auc_score(y, prob1), metrics.roc_auc_score(y, prob2)
        print(f"{auc1:.3f} {auc2:.3f} {pvalue:.3f}")


# %%
