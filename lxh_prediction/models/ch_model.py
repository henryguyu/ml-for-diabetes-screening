import logging
from typing import Dict

import numpy as np
import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class CHModel(BaseModel):
    def __init__(self, params: Dict = {}):
        self.params = params
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
    ):
        pass

    def predict(self, X: np.ndarray, feat_names) -> np.ndarray:
        df = pd.DataFrame(X, columns=feat_names)

        # agescore
        df["Cagescore"] = 0
        df.loc[(df["age"] >= 25) & (df["age"] <= 34), "Cagescore"] = 4
        df.loc[(df["age"] >= 35) & (df["age"] <= 39), "Cagescore"] = 8
        df.loc[(df["age"] >= 40) & (df["age"] <= 44), "Cagescore"] = 11
        df.loc[(df["age"] >= 45) & (df["age"] <= 49), "Cagescore"] = 12
        df.loc[(df["age"] >= 50) & (df["age"] <= 54), "Cagescore"] = 13
        df.loc[(df["age"] >= 55) & (df["age"] <= 59), "Cagescore"] = 15
        df.loc[(df["age"] >= 60) & (df["age"] <= 64), "Cagescore"] = 16
        df.loc[(df["age"] >= 65) & (df["age"] <= 74), "Cagescore"] = 18
        df.loc[df["age"] >= 75, "Cagescore"] = 18

        df["Csexscore"] = 0
        df.loc[df["lsex"] == 0, "Csexscore"] = 2

        df["Cdiafamscore"] = 0
        df.loc[df["ldiafamily"] == 1, "Cdiafamscore"] = 6

        df["Chtscore"] = 0
        df.loc[(df["ASBP"] >= 110) & (df["ASBP"] <= 119), "Chtscore"] = 1
        df.loc[(df["ASBP"] >= 120) & (df["ASBP"] <= 129), "Chtscore"] = 3
        df.loc[(df["ASBP"] >= 130) & (df["ASBP"] <= 139), "Chtscore"] = 6
        df.loc[(df["ASBP"] >= 140) & (df["ASBP"] <= 149), "Chtscore"] = 7
        df.loc[(df["ASBP"] >= 150) & (df["ASBP"] <= 159), "Chtscore"] = 8
        df.loc[(df["ASBP"] >= 160), "Chtscore"] = 10

        df["Cwaiscore"] = 0
        df.loc[
            (df["wc"] >= 75) & (df["wc"] <= 79.9) & (df["lsex"] == 0), "Cwaiscore"
        ] = 3
        df.loc[
            (df["wc"] >= 70) & (df["wc"] <= 74.9) & (df["lsex"] == 1), "Cwaiscore"
        ] = 3
        df.loc[
            (df["wc"] >= 80) & (df["wc"] <= 84.9) & (df["lsex"] == 0), "Cwaiscore"
        ] = 5
        df.loc[
            (df["wc"] >= 75) & (df["wc"] <= 79.9) & (df["lsex"] == 1), "Cwaiscore"
        ] = 5
        df.loc[
            (df["wc"] >= 85) & (df["wc"] <= 89.9) & (df["lsex"] == 0), "Cwaiscore"
        ] = 7
        df.loc[
            (df["wc"] >= 80) & (df["wc"] <= 84.9) & (df["lsex"] == 1), "Cwaiscore"
        ] = 7
        df.loc[
            (df["wc"] >= 90) & (df["wc"] <= 94.9) & (df["lsex"] == 0), "Cwaiscore"
        ] = 8
        df.loc[
            (df["wc"] >= 85) & (df["wc"] <= 89.9) & (df["lsex"] == 1), "Cwaiscore"
        ] = 8
        df.loc[(df["wc"] >= 95) & (df["lsex"] == 0), "Cwaiscore"] = 10
        df.loc[(df["wc"] >= 90) & (df["lsex"] == 1), "Cwaiscore"] = 10

        df["CBMIscore"] = 0
        df.loc[(df["BMI"] >= 22) & (df["BMI"] <= 23.9), "CBMIscore"] = 1
        df.loc[(df["BMI"] >= 24) & (df["BMI"] < 29.9), "CBMIscore"] = 3
        df.loc[df["BMI"] >= 30, "CBMIscore"] = 5

        scores = df[
            [
                "Cagescore",
                "Csexscore",
                "Cdiafamscore",
                "Chtscore",
                "Cwaiscore",
                "CBMIscore",
            ]
        ]
        return (scores.sum(1) >= 25).to_numpy(dtype=int)
