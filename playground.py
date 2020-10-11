#%%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("data/processed_data.csv")

# %%

onehot_fields = "occupt living dwell paint radiatn chemic fixture nuclear intere upset sleep tired appeti loser focus fret suicd impact relitn hit satify weich weial tea lusephone".split()
onehots = pd.get_dummies(df["occupt"].astype(int), prefix="occupt")
# %%
