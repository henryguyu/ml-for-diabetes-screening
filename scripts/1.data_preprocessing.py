# %%
import os
import sys

import numpy as np
import pandas as pd
import sklearn.neighbors._base

# https://stackoverflow.com/questions/60145652/no-module-named-sklearn-neighbors-base
sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base  # noqa

from missingpy import MissForest  # noqa
from sklearn.ensemble import IsolationForest  # noqa

from lxh_prediction import config as cfg  # noqa

# %%
src_file = os.path.join(cfg.root, "data/missforest.2.csv")
dst_file = os.path.join(cfg.root, "data/processed_data_0214_origin.csv")
df = pd.read_csv(src_file)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# %%
nan99 = "paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify weich".split()
nan97 = "intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify".split()
for col in nan99:
    df[col][df[col] == 99] = np.nan
for col in nan97:
    df[col][df[col] == 97] = np.nan

# %%
# 样本剔除
fields_required = "age lsex FPG P2hPG HbA1c".split()
drop = np.any(df[fields_required].isnull(), axis=1)
df = df[~drop]

fields_not_one = "glutreat ldia insulin chmed tumor".split()
drop = np.any(df[fields_not_one] == 1, axis=1)
df = df[~drop]
df.index = range(len(df))

# %%
# 睡觉时间
drop = (df["lgotosleep"] >= 12) & (df["lgotosleep"] <= 17)
df = df[~drop]
lgotosleep = df["lgotosleep"].to_numpy()
lgotosleep[lgotosleep < 12] += 24
df["lgotosleep"] = lgotosleep - 24
df["nigtime"] = df["lgetup"] - df["lgotosleep"]
df.index = range(len(df))

# %%
# 异常值去除


def drop_abnormal(arr: np.ndarray, k=1.5) -> np.ndarray:
    valid_arr = arr[~np.isnan(arr)]
    q1, q3 = np.percentile(valid_arr, [25, 75])
    IQR = q3 - q1
    drop = (arr < q1 - k * IQR) | (arr > q3 + k * IQR)
    return drop, q1 - k * IQR, q3 + k * IQR


def drop_range(arr: np.ndarray, low=None, high=None):
    drop = np.zeros(len(arr), dtype=bool)
    if low is not None:
        drop[arr < low] = True
    if high is not None:
        drop[arr > high] = True
    return drop, low, high


def drop_abnormal2(arr: np.ndarray):
    mask = ~np.isnan(arr)
    valid_arr = arr[mask]
    preds = IsolationForest(random_state=0).fit_predict(valid_arr[:, None])
    inliers = valid_arr[preds > 0]

    drop = np.zeros(len(arr), dtype=bool)
    drop[mask] = preds < 0

    return drop, inliers.min(), inliers.max()


# fields = "ASBP ADBP Ahr weight2 height2 BMI wc hc weight20new FPG P2hPG".split()
fields = "nigtime".split()
drop = np.zeros(len(df), dtype=bool)
for name in fields:
    sub_drop, low, high = drop_abnormal(df[name].values, k=2)
    print(name, low, high, sub_drop.sum())
    print(df[name][sub_drop])
    drop |= sub_drop

drop |= drop_range(df["age"].values, high=90)[0]
drop |= drop_range(df["lusephy"].values, high=30)[0]
drop |= drop_range(df["Ahr"].values, high=300)[0]

print(drop.sum())
df = df[~drop]
df.index = range(len(df))

# %%
# select features
cat_fields = "lsex age occupt marriage3 living llivealone dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify culutrue wm prisch junsch sensch juncoll lmi lstroke lcvd ht ldiafamily lsmoking lsmoked lneversmoke ldrinking ldrinked lneverdrink weich weial weime1 weime0 weime3 weime99 etime breakd lunchd dinnerd brehome breeate brerest lunhome luneate lunrest dinhome dineate dinrest grae0 ptae0 poke0 befe0 chie0 fise0 vege0 frue0 juie0 egge0 mike0 beae0 frye0 juce0 sode0 cake0 brne0 saue0 fure0 cofe0 orge0 vite0 tea lntea lteai lwork highintenswork leisphysical lphysactive lvigorous lvigday lmiddle lmidday walk0 walkday lseat1a lseat2a lseat1day lseat2day lgoodday1 lbadday1 lusephone lgest lfchild lfboy lfgirl lmbigb lmbn lmmboy lmmgirl lmchild lmboy lmgirl lfbigb lfbn lfmboy lfmgirl lght lghbs lbrestfe lmenop hypertension".split()
scalar_fields = "lvighour lvigtime lmidhour lmidtime walkhour lwalktime lseat1hou lseat2hou seattime ntime lgotosleep lgetup nigtime lusephy lbftime2 lbftime_sum ASBP ADBP Ahr weight2 height2 wc hc BMI WHR WHtR weight20new".split()
label_fields = "FPG P2hPG HbA1c".split()
calc_fields = "leisphysical lphysactive lvigtime lmidtime lwalktime seattime nigtime lbftime_sum BMI WHR WHtR".split()

calc_fields = set(calc_fields)
cat_fields = [name for name in cat_fields if name not in calc_fields]
scalar_fields = [name for name in scalar_fields if name not in calc_fields]

df_feat = df[cat_fields + scalar_fields]
labels = ((df["FPG"] >= 7.0) | (df["P2hPG"] >= 11.1)).astype(int)
null_rate = df_feat.isnull().sum(1) / df_feat.shape[1]
df = df[(null_rate <= 0.25) | (labels > 0)]
df.index = range(len(df))

df_label = df[label_fields]
df_feat = df[cat_fields + scalar_fields]

# %%
# 填补缺失值
mf = MissForest()
X = mf.fit_transform(df_feat, cat_vars=np.arange(len(cat_fields)))
df_feat = pd.DataFrame(X, columns=df_feat.columns)

# %%
# 计算得到的特征
df_feat["leisphysical"] = np.any(
    df_feat["lmiddle lvigorous walk0".split()] == 1, axis=1
).astype(float)

df_feat["lphysactive"] = np.any(
    df_feat["lmiddle lvigorous".split()] == 1, axis=1
).astype(float)

df_feat["lvigtime"] = df_feat["lvighour"] * df_feat["lvigday"]
df_feat["lmidtime"] = df_feat["lmidhour"] * df_feat["lmidday"]
df_feat["lwalktime"] = df_feat["walkhour"] * df_feat["walkday"]
df_feat["seattime"] = (
    df_feat["lseat1day"] * df_feat["lseat1hou"]
    + df_feat["lseat2day"] * df_feat["lseat2hou"]
)
df_feat["nigtime"] = df_feat["lgetup"] - df_feat["lgotosleep"]
df_feat["lbftime_sum"] = df_feat["lbftime2"] * df_feat["lfchild"]
df_feat["BMI"] = df_feat["weight2"] / (df_feat["height2"] ** 2)
df_feat["WHR"] = df_feat["wc"] / df_feat["hc"]
df_feat["WHtR"] = df_feat["wc"] / (df_feat["height2"] * 100)

df = pd.concat([df_feat, df_label], axis=1)
df["label_WHO"] = ((df["FPG"] >= 7.0) | (df["P2hPG"] >= 11.1)).astype(int)
df["label_ADA"] = (
    (df["FPG"] >= 7.0) | (df["P2hPG"] >= 11.1) | (df["HbA1c"] >= 6.5)
).astype(int)

# %%
# Save
df.to_csv(dst_file, index=False)
# %%
