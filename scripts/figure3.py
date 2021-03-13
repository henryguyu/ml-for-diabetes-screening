# %%
import logging

import numpy as np

from lxh_prediction import metric_utils
from lxh_prediction.exp_utils import get_cv_preds
from lxh_prediction.plot import plot_curve, plot_range, plt, ExpFigure

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class NeedsMissRateExp(ExpFigure):
    @staticmethod
    def mean_pos_missrate(cv_y_prob):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob])
        miss_rate = np.mean(probs[ys > 0] <= 0)
        nag_rate = np.mean(probs <= 0)
        return 1 - nag_rate, miss_rate

    def run(self, name, model, feat_collection, cutoff=False):
        cv_y_prob = get_cv_preds(
            model_name=model,
            feat_collection=feat_collection,
            update=False,
            resample_train=False,
        )
        pos_rates, miss_rates, _ = zip(
            *(metric_utils.pos_miss_curve(ys, probs) for ys, probs in cv_y_prob)
        )
        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
            miss_rates, pos_rates
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            ylim=(0, 1),
            name=f"{name} Model",
            color=color,
            zorder=3,
            xlabel="Prediction Miss rate",
            ylabel="The Needs of OGTT",
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(pos_rates, miss_rates)[:2]
        self.x_means[name] = x_mean

        if "LightGBM" not in model and cutoff:
            pos_rate, miss_rate = self.mean_pos_missrate(cv_y_prob)
            print(f"{name}: miss_rate: {miss_rate}, pos_rate: {pos_rate}")
            plt.axvline(x=miss_rate, c="gray", ls="--", lw=1, zorder=1)
            plt.axhline(y=pos_rate, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                miss_rate,
                pos_rate,
                marker="8",
                color=color,
                label=f"{name} Cutoff Point",
                zorder=4,
            )
            self.add_point(miss_rate, pos_rate)


class CostMissRateExp(ExpFigure):
    MaxCost = 65
    MinCost = 0

    @staticmethod
    def mean_cost_missrate(cv_y_prob, test_name=None, test_values=None):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob]) > 0

        if test_name is not None:
            print(test_name)
            test_values = np.concatenate(test_values)
        cost, missrate, _ = metric_utils.cost_curve(ys, probs, test_name, test_values)
        return cost[-1], missrate[-1]

    def yticks(self):
        return np.linspace(self.MinCost, self.MaxCost, 6)

    def run(self, name, model, feat_collection, cutoff=False):
        cv_y_prob, test_name, test_values = get_cv_preds(
            model_name=model,
            feat_collection=feat_collection,
            update=False,
            resample_train=False,
            out_tests=True,
        )
        if test_values is None:
            test_values = [None] * len(cv_y_prob)
        costs, miss_rates, _ = zip(
            *(
                metric_utils.cost_curve(ys, probs, test_name, test_values[i])
                for i, (ys, probs) in enumerate(cv_y_prob)
            )
        )
        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
            miss_rates,
            costs,
            y_range=(self.MinCost, self.MaxCost),
            # num_x=int((self.MaxCost - self.MinCost) * 10),
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            ylim=(self.MinCost, self.MaxCost),
            name=f"{name} Model",
            color=color,
            zorder=3,
            xlabel="Prediction Miss rate",
            ylabel="Screening Costs",
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(
            costs,
            miss_rates,
            x_range=(self.MinCost, self.MaxCost),
            num_x=int((self.MaxCost - self.MinCost) * 10),
            reverse=True,
        )[:2]
        self.x_means[name] = x_mean

        if "LightGBM" not in model and cutoff:
            costs, miss_rate = self.mean_cost_missrate(
                cv_y_prob, test_name, test_values
            )
            print(f"{name}: miss_rate: {miss_rate}, costs: {costs}")
            plt.axvline(x=miss_rate, c="gray", ls="--", lw=1, zorder=1)
            plt.axhline(y=costs, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                miss_rate,
                costs,
                marker="8",
                color=color,
                label=f"{name} Cutoff Point",
                zorder=4,
            )
            self.add_point(miss_rate, costs)


# %%
# Figure 3a, needs of OGTT/Costs, ADA/CDS
exp = NeedsMissRateExp()
exp.run("FPG", "LightGBMModel", "FPG")
exp.run("2hPG", "LightGBMModel", "2hPG")
exp.run("HbA1c", "LightGBMModel", "HbA1c")
exp.run("CDS+FPG", "CHModel", "CH_FPG", cutoff=True)
exp.run("CDS+2hPG", "CHModel", "CH_2hPG")
exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")
exp.run("CDS", "CHModel", "CH", cutoff=True)
exp.run("Non-lab", "LightGBMModel", "top20_non_lab")
exp.plot()
plt.legend(loc="upper right")
exp.save("figure3_a-1")

exp = CostMissRateExp()
exp.MaxCost = 120

# exp.MinCost = 45
exp.run("FPG", "LightGBMModel", "FPG")
exp.run("2hPG", "LightGBMModel", "2hPG")
exp.run("HbA1c", "LightGBMModel", "HbA1c")
exp.run("CDS+FPG", "CHModel", "CH_FPG", cutoff=True)
exp.run("CDS+2hPG", "CHModel", "CH_2hPG")
exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")
exp.run("CDS", "CHModel", "CH", cutoff=True)
exp.run("Non-lab", "LightGBMModel", "top20_non_lab")
exp.plot()
plt.legend(loc="upper right")
exp.save("figure3_a-2")

# %%
# Figure 4a, needs of OGTT/Costs, ADA/CDS

exp = NeedsMissRateExp()
exp.run("FPG", "LightGBMModel", "FPG")
exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
exp.run("CDS+FPG", "CHModel", "CH_FPG")
exp.plot()
exp.save("figure4_a-1")

exp = CostMissRateExp()
exp.run("FPG", "LightGBMModel", "FPG")
exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
exp.run("CDS+FPG", "CHModel", "CH_FPG")
exp.plot()
exp.save("figure4_a-2")

# %%
# Figure 4b, needs of OGTT/Costs, ADA/CDS

exp = NeedsMissRateExp()
exp.run("2hPG", "LightGBMModel", "2hPG")
exp.run("ADA+2hPG", "ADAModel", "ADA_2hPG")
exp.run("CDS+2hPG", "CHModel", "CH_2hPG")
exp.plot()
exp.save("figure4_b-1")

exp = CostMissRateExp()
exp.MaxCost = 70
exp.MinCost = 45
exp.run("2hPG", "LightGBMModel", "2hPG")
exp.run("ADA+2hPG", "ADAModel", "ADA_2hPG")
exp.run("CDS+2hPG", "CHModel", "CH_2hPG")
exp.plot()
exp.save("figure4_b-2")

# %%
# Figure 4c, needs of OGTT/Costs, ADA/CDS

exp = NeedsMissRateExp()
exp.run("HbA1c", "LightGBMModel", "HbA1c")
exp.run("ADA+HbA1c", "ADAModel", "ADA_HbA1c")
exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")
exp.plot()
exp.save("figure4_c-2")

exp = CostMissRateExp()
exp.MaxCost = 150
exp.MinCost = 80
exp.run("HbA1c", "LightGBMModel", "HbA1c")
exp.run("ADA+HbA1c", "ADAModel", "ADA_HbA1c")
exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")
exp.plot()
exp.save("figure4_c-2")

# %%

exp = NeedsMissRateExp()
exp.run("Non-lab", "LightGBMModel", "top20_non_lab")
exp.run("FPG", "LightGBMModel", "FPG")
exp.run("2hPG", "LightGBMModel", "2hPG")
exp.run("HbA1c", "LightGBMModel", "HbA1c")
exp.run("ADA", "ADAModel", "ADA")
exp.run("CDS", "CHModel", "CH")
# exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
# exp.run("CDS+FPG", "CHModel", "CH_FPG")
exp.plot()
plt.legend(loc="upper right")
exp.save("figure4_a-c-1")

exp = CostMissRateExp()
exp.MaxCost = 100
# exp.MinCost = 80
exp.run("Non-lab", "LightGBMModel", "top20_non_lab")
exp.run("FPG", "LightGBMModel", "FPG")
exp.run("2hPG", "LightGBMModel", "2hPG")
exp.run("HbA1c", "LightGBMModel", "HbA1c")
exp.run("ADA", "ADAModel", "ADA")
exp.run("CDS", "CHModel", "CH")
# exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
# exp.run("CDS+FPG", "CHModel", "CH_FPG")
exp.plot()
plt.legend(loc="upper right")
exp.save("figure4_a-c-2")

# exp = CostMissRateExp()
# exp.run("ADA+FPG", "ADAModel", "ADA_FPG")
# exp.run("CDS+FPG", "CHModel", "CH_FPG")
# exp.run("FPG", "LightGBMModel", "FPG")

# exp = CostMissRateExp()
# exp.run("ADA+2hPG", "ADAModel", "ADA_2hPG")
# exp.run("CDS+2hPG", "CHModel", "CH_2hPG")
# exp.run("2hPG", "LightGBMModel", "2hPG")

# exp = CostMissRateExp()
# exp.run("ADA+HbA1c", "ADAModel", "ADA_HbA1c")
# exp.run("CDS+HbA1c", "CHModel", "CH_HbA1c")
# exp.run("HbA1c", "LightGBMModel", "HbA1c")

# %%
