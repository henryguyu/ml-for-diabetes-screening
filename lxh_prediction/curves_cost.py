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
            update=self.retrain,
            resample_train=False,
        )
        pos_rates, miss_rates, _ = zip(
            *(metric_utils.pos_miss_curve(ys, probs) for ys, probs in cv_y_prob)
        )
        miss_rates = 1 - np.asarray(miss_rates)
        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
            miss_rates, pos_rates, x_range=self.xlim, y_range=self.ylim, reverse=True
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            ylim=self.ylim,
            xlim=self.xlim,
            name=f"{name}",
            color=color,
            zorder=3,
            xlabel="Sensitivity",
            ylabel="Proportion requiring confirmatory test",
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(
            pos_rates, miss_rates, y_range=self.xlim, x_range=self.ylim,
        )[:2]
        self.x_means[name] = x_mean

        if "LightGBM" not in model and cutoff:
            pos_rate, miss_rate = self.mean_pos_missrate(cv_y_prob)
            miss_rate = 1 - miss_rate
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
    @staticmethod
    def mean_cost_missrate(cv_y_prob, test_name=None, test_values=None):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob]) > 0

        if test_name is not None:
            print(test_name)
            test_values = np.concatenate(test_values)
        cost, missrate, _ = metric_utils.cost_curve(ys, probs, test_name, test_values)
        return cost[-1], missrate[-1]

    def run(self, name, model, feat_collection, cutoff=False):
        cv_y_prob, test_name, test_values = get_cv_preds(
            model_name=model,
            feat_collection=feat_collection,
            update=self.retrain,
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
        miss_rates = 1 - np.asarray(miss_rates)
        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
            miss_rates,
            costs,
            y_range=self.ylim,
            x_range=self.xlim,
            # num_x=int((self.MaxCost - self.MinCost) * 10),
            reverse=True,
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            ylim=self.ylim,
            xlim=self.xlim,
            name=f"{name}",
            color=color,
            zorder=3,
            xlabel="Sensitivity",
            ylabel="Average screening Costs",
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(
            costs,
            miss_rates,
            x_range=self.ylim,
            y_range=self.xlim,
            num_x=int((self.ylim[1] - self.ylim[0]) * 10),
            reverse=True,
        )[:2]
        self.x_means[name] = x_mean

        if "LightGBM" not in model and cutoff:
            costs, miss_rate = self.mean_cost_missrate(
                cv_y_prob, test_name, test_values
            )
            miss_rate = 1 - miss_rate
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


class SensitivityCostExp(ExpFigure):
    def __init__(self, figure=None, retrain=True, compare_to_all=False):
        super().__init__(figure=figure, retrain=retrain)
        self.compare_to_all = compare_to_all

    @staticmethod
    def mean_cost_missrate(cv_y_prob, test_name=None, test_values=None):
        ys = np.concatenate([y_prob[0] for y_prob in cv_y_prob])
        probs = np.concatenate([y_prob[1] for y_prob in cv_y_prob]) > 0

        if test_name is not None:
            print(test_name)
            test_values = np.concatenate(test_values)
        cost, missrate, _ = metric_utils.cost_curve(ys, probs, test_name, test_values)
        return cost[-1], missrate[-1]

    def run(self, name, model, feat_collection, cutoff=False):
        cv_y_prob, test_name, test_values = get_cv_preds(
            model_name=model,
            feat_collection=feat_collection,
            update=self.retrain,
            resample_train=False,
            out_tests=True,
        )
        if test_values is None:
            test_values = [None] * len(cv_y_prob)
        costs, miss_rates, _ = zip(
            *(
                metric_utils.cost_curve(
                    ys,
                    probs,
                    test_name,
                    test_values[i],
                    compare_to_all=self.compare_to_all,
                )
                for i, (ys, probs) in enumerate(cv_y_prob)
            )
        )
        min_cost = min(map(min, costs))
        print(min_cost)
        sublim = (max(min_cost, self.xlim[0]), self.xlim[1])
        print(sublim)
        # sensitivity = (1 - np.asarray(miss_rates)).tolist()
        sensitivity = list(map(lambda x: 1 - x, miss_rates))
        # future_cost = np.asarray(miss_rates) * 2802

        x_base, y_mean, y_lower, y_upper = metric_utils.mean_curve(
            costs,
            sensitivity,
            x_range=sublim,
            num_x=int((self.xlim[1] - self.xlim[0]) * 10),
            reverse=True,
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean,
            xlim=sublim,
            ylim=self.ylim,
            name=f"{name}",
            color=color,
            zorder=3,
            xlabel="Average screening Costs",
            ylabel="Sensitivity",
        )
        plot_range(x_base, y_lower, y_upper, zorder=2)
        plt.axvline(x=sublim[0], c=color, ls="--", lw=1, zorder=1)
        self.add_point(sublim[0], None)
        self.y_means[name] = y_mean
        self.x_base = x_base

        self.y_base, x_mean = metric_utils.mean_curve(
            sensitivity,
            costs,
            y_range=self.xlim,
            x_range=self.ylim,
            # num_x=int((self.MaxCost - self.MinCost) * 10),
            # reverse=True,
        )[:2]
        self.x_means[name] = x_mean

        if "LightGBM" not in model and cutoff:
            costs, miss_rate = self.mean_cost_missrate(
                cv_y_prob, test_name, test_values
            )
            sensitivity = 1 - miss_rate
            print(f"{name}: miss_rate: {sensitivity}, costs: {costs}")
            plt.axhline(y=sensitivity, c="gray", ls="--", lw=1, zorder=1)
            plt.axvline(x=costs, c="gray", ls="--", lw=1, zorder=1)
            plt.scatter(
                costs,
                sensitivity,
                marker="8",
                color=color,
                label=f"{name} Cutoff Point",
                zorder=4,
            )
            self.add_point(costs, sensitivity)


class FutureCostExp(ExpFigure):
    def __init__(self, figure=None, compare_to_all=True, retrain=True):
        super().__init__(figure=figure, retrain=retrain)
        self.compare_to_all = compare_to_all

    def run(self, name, model, feat_collection, cutoff=False):
        cv_y_prob, test_name, test_values = get_cv_preds(
            model_name=model,
            feat_collection=feat_collection,
            update=self.retrain,
            resample_train=False,
            out_tests=True,
        )
        if test_values is None:
            test_values = [None] * len(cv_y_prob)
        costs, miss_rates, _ = zip(
            *(
                metric_utils.cost_curve(
                    ys,
                    probs,
                    test_name,
                    test_values[i],
                    compare_to_all=self.compare_to_all,
                )
                for i, (ys, probs) in enumerate(cv_y_prob)
            )
        )
        min_cost = min(map(min, costs))
        print(min_cost)
        sublim = (max(min_cost, self.xlim[0]), self.xlim[1])
        print(sublim)
        future_cost_low = list(map(lambda x: x * 2802, miss_rates))
        future_cost_high = list(map(lambda x: x * 5611, miss_rates))

        x_base, y_mean_low, y_lower, y_upper = metric_utils.mean_curve(
            costs,
            future_cost_low,
            x_range=sublim,
            num_x=int((self.xlim[1] - self.xlim[0]) * 10),
            reverse=True,
        )
        color = self.next_color()
        plot_curve(
            x_base,
            y_mean_low,
            xlim=sublim,
            ylim=self.ylim,
            name=f"{name}",
            color=color,
            zorder=3,
            xlabel="Average screening Costs",
            ylabel="Average costs of potential complications",
        )

        x_base, y_mean_high, y_lower, y_upper = metric_utils.mean_curve(
            costs,
            future_cost_high,
            x_range=sublim,
            num_x=int((self.xlim[1] - self.xlim[0]) * 10),
            reverse=True,
        )
        plot_curve(
            x_base,
            y_mean_high,
            xlim=sublim,
            ylim=self.ylim,
            name=None,
            color=color,
            zorder=3,
            xlabel="Average screening Costs",
            ylabel="Average costs of potential complications",
        )

        # print(y_lower)
        plot_range(x_base, y_mean_low, y_mean_high, zorder=2, color=color)
        plt.axvline(x=sublim[0], c=color, ls="--", lw=1, zorder=1)
        self.add_point(sublim[0], None)
        # plot_curve(
        #     (sublim[0], sublim[0]),
        #     (y_mean_low[0], y_mean_high[0]),
        #     ylim=self.ylim,
        #     xlim=sublim,
        #     color=color,
        #     lw=2,
        #     linestyle="--",
        # )

        self.y_means[name] = y_mean_low
        self.x_base = x_base
