import logging
from typing import Dict

import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ADAModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = params
        self.model = None

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X

        # agescore
        df.loc[(df["age"] >= 40) & (df["age"] <= 49), "agescore"] = 1
        df.loc[(df["age"] >= 50) & (df["age"] <= 59), "agescore"] = 2
        df.loc[df["age"] >= 60, "agescore"] = 3

        df["sexscore"] = (df["lsex"] == 0).astype(int)
        df["diafamscore"] = (df["ldiafamily"] == 1).astype(int)
        df["lghbsscore"] = (df["lghbs"] == 1).astype(int)

        df["htscore"] = 0
        df.loc[
            (df["ht"] == 1) | (df["ASBP"] >= 140) | (df["ADBP"] >= 90), "htscore"
        ] = 1

        df["phscore"] = (df["lphysactive"] == 0).astype(int)

        df["BMIscore"] = 0
        df.loc[(df["BMI"] >= 25) & (df["BMI"] <= 29.9), "BMIscore"] = 1
        df.loc[(df["BMI"] >= 30) & (df["BMI"] < 39.9), "BMIscore"] = 2
        df.loc[df["BMI"] >= 39.9, "BMIscore"] = 3

        scores = df[
            [
                "agescore",
                "sexscore",
                "diafamscore",
                "lghbsscore",
                "htscore",
                "phscore",
                "BMIscore",
            ]
        ]
        preds = scores.sum(1) - 5 + 1e-6
        if "FPG" in df:
            preds += 100 * (df["FPG"] >= 6.1)
        if "P2hPG" in df:
            preds += 100 * (df["P2hPG"] >= 11.1)
        if "HbA1c" in df:
            preds += 100 * (df["HbA1c"] >= 6.5)

        return preds
