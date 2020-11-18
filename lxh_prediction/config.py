import os

_this_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(_this_dir)
data_file = os.path.join(root, "data/processed_data_1118.4.csv")

cat_fields = "lsex age occupt marriage3 living llivealone dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify culutrue wm prisch junsch sensch juncoll lmi lstroke lcvd ht ldiafamily lsmoking lsmoked lneversmoke ldrinking ldrinked lneverdrink weich weial weime1 weime0 weime3 weime99 etime breakd lunchd dinnerd brehome breeate brerest lunhome luneate lunrest dinhome dineate dinrest grae0 ptae0 poke0 befe0 chie0 fise0 vege0 frue0 juie0 egge0 mike0 beae0 frye0 juce0 sode0 cake0 brne0 saue0 fure0 cofe0 orge0 vite0 tea lntea lteai lwork highintenswork leisphysical lphysactive lvigorous lvigday lmiddle lmidday walk0 walkday lseat1a lseat2a lseat1day lseat2day lgoodday1 lbadday1 lusephone lgest lfchild lfboy lfgirl lmbigb lmbn lmmboy lmmgirl lmchild lmboy lmgirl lfbigb lfbn lfmboy lfmgirl lght lghbs lbrestfe lmenop hypertension".split()
scalar_fields = "lvighour lvigtime lmidhour lmidtime walkhour lwalktime lseat1hou lseat2hou seattime ntime lgotosleep lgetup nigtime lusephy lbftime2 lbftime_sum ASBP ADBP Ahr weight2 height2 wc hc BMI WHR WHtR weight20new".split()
# onehot_fields = "occupt living dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify weich weial tea lusephone".split()
onehot_fields = "".split()
optional_fields = ["FPG"]

ada_fields = "age lsex ldiafamily lghbs ht ASBP ADBP lphysactive BMI".split()
ch_fields = "age lsex ldiafamily lghbs ht ASBP wc BMI".split()

feature_fields = {
    "without_FPG": cat_fields + scalar_fields,
    "with_FPG": cat_fields + scalar_fields + optional_fields,
    "ADA": ada_fields,
    "CH": ch_fields,
    "ADA_FPG": ada_fields + optional_fields,
    "CH_FPG": ch_fields + optional_fields,
}
label_field = "label"

model_params = {
    # ("LightGBMModel", "without_FPG"): {
    #     "num_leaves": 40,
    #     "max_bin": 181,
    #     "max_depth": -1,
    #     "learning_rate": 0.012177823735146441,
    #     "lambda_l1": 0.0005,
    #     "lambda_l2": 0.005,
    #     "feature_fraction": 0.5,
    #     "min_data_in_bin": 9,
    #     "bagging_fraction": 0.7,
    #     "bagging_freq": 4,
    #     "path_smooth": 0.01,
    # }
}
