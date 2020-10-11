# %%
import numpy as np
import pandas as pd
from missingpy import MissForest

# %%
src_file = "data/missforest.2.csv"
dst_file = "data/processed_data.csv"
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
fields_required = "age lsex FPG P2hPG".split()
drop = np.any(df[fields_required].isnull(), axis=1)
df = df[~drop]

fields_not_one = "glutreat ldia insulin chmed tumor".split()
drop = np.any(df[fields_not_one] == 1, axis=1)
df = df[~drop]
df.index = range(len(df))
# %%
# 异常值去除


def drop_abnormal(arr: np.ndarray) -> np.ndarray:
    valid_arr = arr[~np.isnan(arr)]
    q1, q3 = np.percentile(valid_arr, [25, 75])
    IQR = q3 - q1
    drop = (arr < q1 - 1.5 * IQR) | (arr > q3 + 1.5 * IQR)
    return drop


fields = "".split()
drop = np.zeros(len(df), dtype=bool)
for name in fields:
    drop |= drop_abnormal(df[name].values)
df = df[~drop]
df.index = range(len(df))

# %%
# select features
cat_fields = "lsex age occupt marriage3 living llivealone dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify culutrue wm prisch junsch sensch juncoll lmi lstroke lcvd ht ldiafamily lsmoking lsmoked lneversmoke ldrinking ldrinked lneverdrink weich weial weime1 weime0 weime3 weime99 etime breakd lunchd dinnerd brehome breeate brerest lunhome luneate lunrest dinhome dineate dinrest grae0 ptae0 poke0 befe0 chie0 fise0 vege0 frue0 juie0 egge0 mike0 beae0 frye0 juce0 sode0 cake0 brne0 saue0 fure0 cofe0 orge0 vite0 tea lntea lteai lwork highintenswork leisphysical lphysactive lvigorous lvigday lmiddle lmidday walk0 walkday lseat1a lseat2a lseat1day lseat2day lgoodday1 lbadday1 lusephone lgest lfchild lfboy lfgirl lmbigb lmbn lmmboy lmmgirl lmchild lmboy lmgirl lfbigb lfbn lfmboy lfmgirl lght lghbs lbrestfe lmenop hypertension".split()
scalar_fields = "lvighour lvigtime lmidhour lmidtime walkhour lwalktime lseat1hou lseat2hou seattime ntime lgotosleep lgetup nigtime lusephy lbftime2 lbftime_sum ASBP ADBP Ahr weight2 height2 wc hc BMI WHR WHtR weight20new".split()
label_fields = "FPG P2hPG".split()
calc_fields = "leisphysical lphysactive lvigtime lmidtime lwalktime seattime nigtime lbftime_sum BMI WHR WHtR".split()

calc_fields = set(calc_fields)
cat_fields = [name for name in cat_fields if name not in calc_fields]
scalar_fields = [name for name in scalar_fields if name not in calc_fields]

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
df_feat["nigtime"] = 24 - df_feat["lgotosleep"] + df_feat["lgetup"]
df_feat["lbftime_sum"] = df_feat["lbftime2"] * df_feat["lfchild"]
df_feat["BMI"] = df_feat["weight2"] / (df_feat["height2"] ** 2)
df_feat["WHR"] = df_feat["wc"] / df_feat["hc"]
df_feat["WHtR"] = df_feat["wc"] / (df_feat["height2"] * 100)

df = pd.concat([df_feat, df_label], axis=1)
df["label"] = ((df["FPG"] >= 7.0) | (df["P2hPG"] >= 11.1)).astype(int)

# %%
# Save
df.to_csv(dst_file, index=False)
# %%
