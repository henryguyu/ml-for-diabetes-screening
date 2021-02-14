import os
from matplotlib import cm

_this_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(_this_dir)
data_file = os.path.join(root, "data/processed_data_0214.csv")

cat_fields = "lsex age occupt marriage3 living llivealone dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify culutrue wm prisch junsch sensch juncoll lmi lstroke lcvd ht ldiafamily lsmoking lsmoked lneversmoke ldrinking ldrinked lneverdrink weich weial weime1 weime0 weime3 weime99 etime breakd lunchd dinnerd brehome breeate brerest lunhome luneate lunrest dinhome dineate dinrest grae0 ptae0 poke0 befe0 chie0 fise0 vege0 frue0 juie0 egge0 mike0 beae0 frye0 juce0 sode0 cake0 brne0 saue0 fure0 cofe0 orge0 vite0 tea lntea lteai lwork highintenswork leisphysical lphysactive lvigorous lvigday lmiddle lmidday walk0 walkday lseat1a lseat2a lseat1day lseat2day lgoodday1 lbadday1 lusephone lgest lfchild lfboy lfgirl lmbigb lmbn lmmboy lmmgirl lmchild lmboy lmgirl lfbigb lfbn lfmboy lfmgirl lght lghbs lbrestfe lmenop hypertension".split()
scalar_fields = "lvighour lvigtime lmidhour lmidtime walkhour lwalktime lseat1hou lseat2hou seattime ntime lgotosleep lgetup nigtime lusephy lbftime2 lbftime_sum ASBP ADBP Ahr weight2 height2 wc hc BMI WHR WHtR weight20new".split()
# onehot_fields = "occupt living dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify weich weial tea lusephone".split()
important_fields = "Ahr age WHR ASBP lwork BMI wc culutrue lusephy WHtR ADBP lgetup hc seattime lvigday lvighour ldrinking frye0 ntime nigtime".split()
onehot_fields = "".split()
optional_fields = ["FPG", "P2hPG", "HbA1c"]

ada_fields = "age lsex ldiafamily lghbs ht ASBP ADBP lphysactive BMI".split()
ch_fields = "age lsex ldiafamily lghbs ht ASBP wc BMI".split()

feature_fields = {
    "without_FPG": cat_fields + scalar_fields,
    "with_FPG": cat_fields + scalar_fields + optional_fields,
    "imp_wo_FPG": important_fields,
    "imp_with_FPG": important_fields + optional_fields,
    "ADA": ada_fields,
    "CH": ch_fields,
    "ADA_FPG": ada_fields + ["FPG"],
    "CH_FPG": ch_fields + ["FPG"],
    # =======
    "full_non_lab": cat_fields + scalar_fields,
    "top20_non_lab": important_fields,
    "top15_non_lab": important_fields[:15],
    "top10_non_lab": important_fields[:10],
    "top5_non_lab": important_fields[:5],
    "top3_non_lab": important_fields[:3],
    "FPG": important_fields + ["FPG"],
    "2hPG": important_fields + ["P2hPG"],
    "HbA1c": important_fields + ["HbA1c"],
}
label_field = "label_WHO"
color_map = cm.get_cmap("tab10").colors


_LightGBMModel_non_lab = {
    "num_leaves": 9,
    "max_bin": 121,
    "max_depth": 64,
    "learning_rate": 0.06521362882101824,
    "lambda_l1": 0.1,
    "lambda_l2": 0.05,
    "feature_fraction": 0.5,
    "min_data_in_bin": 9,
    "bagging_fraction": 0.9,
    "bagging_freq": 4,
    "path_smooth": 0.1,
}

_LightGBMModel_lab = {
    "num_leaves": 37,
    "max_bin": 212,
    "max_depth": 256,
    "learning_rate": 0.03181350217414469,
    "lambda_l1": 0.005,
    "lambda_l2": 0.001,
    "feature_fraction": 0.5,
    "min_data_in_bin": 3,
    "bagging_fraction": 0.5,
    "bagging_freq": 8,
    "path_smooth": 0.0001,
}

model_params = {
    ("LightGBMModel", "without_FPG"): _LightGBMModel_non_lab,
    ("LightGBMModel", "with_FPG"): _LightGBMModel_lab,
    ("LightGBMModel", "FPG"): _LightGBMModel_lab,
    ("LightGBMModel", "2hPG"): _LightGBMModel_lab,
    ("LightGBMModel", "HbA1c"): _LightGBMModel_lab,
    ("ANNModel", "without_FPG"): {
        "lr": 0.000906390150257859,
        "weight_decay": 0.005,
        "batch_size": 112,
        "enable_lr_scheduler": 0,
        "opt": "Adam",
        "n_channels": 412,
        "n_layers": 3,
        "dropout": 1,
        "bn": 1,
        "activate": "Sigmoid",
        "branches": "[1, 0]",
    },
    ("ANNModel", "with_FPG"): {
        "lr": 0.06927405601990173,
        "weight_decay": 0.0001,
        "batch_size": 114,
        "enable_lr_scheduler": 1,
        "opt": "RMSprop",
        "n_channels": 412,
        "n_layers": 3,
        "dropout": 0,
        "bn": 1,
        "activate": "Tanh",
        "branches": "[2, 1]",
    },
}
