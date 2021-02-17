import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import lxh_prediction.config as cfg


def plot_curve(
    x,
    y,
    name="ROC curve",
    xlim=(0, 1),
    ylim=(0, 1.00),
    xlabel="x",
    ylabel="y",
    title=None,
    color="darkorange",
    lw=2,
    **kwargs,
):
    plt.plot(x, y, color=color, lw=lw, label=name, **kwargs)
    plt.xlim(xlim or plt.xlim())
    plt.ylim(ylim or plt.ylim())
    plt.xlabel(xlabel, fontdict={"size": 12})
    plt.ylabel(ylabel, fontdict={"size": 12})
    if title:
        plt.title(title)
    plt.legend(loc="lower right")


def plot_range(x, y_lower, y_upper, color="grey", alpha=0.1, **kwargs):
    plt.fill_between(x, y_lower, y_upper, color=color, alpha=alpha, **kwargs)


class ExpFigure:
    def __init__(self, figure=None):
        if figure is None:
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = figure.fig
            self.ax = figure.ax
        self.y_means = {}
        self.x_means = {}
        self.x_base = None
        self.y_base = None

        self.points = []

    def run(self, name, model, feat_collection):
        raise NotImplementedError

    def add_point(self, x, y):
        self.points.append((x, y))

    def xticks(self):
        return np.linspace(0, 1, 6)

    def yticks(self):
        return np.linspace(0, 1, 6)

    def plot(self):
        def gen_ticks(values, ticks):
            keep = np.ones(len(ticks), dtype=np.bool)
            for v in values:
                dists = np.abs(ticks - v)
                print(v)
                idx = np.argmin(dists)
                if dists[idx] < 0.05 * ticks[-1]:
                    keep[idx] = False
            ticks = ticks[keep]
            ticks = list(ticks) + list(values)
            return ticks, map("{:.2f}".format, ticks)

        xs, ys = list(zip(*self.points)) if len(self.points) > 0 else ([], [])
        plt.xticks(*gen_ticks(xs, ticks=self.xticks()), rotation=45, fontsize=10)
        plt.yticks(*gen_ticks(ys, ticks=self.yticks()), fontsize=10)

        # ax_top = self.ax.secondary_xaxis("top")
        # ax_top.set_xticks(list(map(lambda x: round(x, 3), xs)))

        # ax_right = self.ax.secondary_yaxis("right")
        # ax_right.set_yticks(list(map(lambda x: round(x, 3), ys)))

    def next_color(self):
        return cfg.color_map[len(self.y_means)]

    def save(self, name):
        self.plot()

        x_means, y_means = self.x_means, self.y_means
        x_base, y_base = self.x_base, self.y_base
        df_ymeans = pd.DataFrame(y_means.values(), index=y_means.keys(), columns=x_base)
        output = os.path.join(cfg.root, f"data/results/{name}.csv")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df_ymeans.to_csv(output)

        df_xmeans = pd.DataFrame(x_means.values(), index=x_means.keys(), columns=y_base)
        output = os.path.join(cfg.root, f"data/results/{name}_T.csv")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df_xmeans.to_csv(output)

        self.fig.savefig(os.path.join(cfg.root, f"data/results/{name}.pdf"))

    def fname(self, name):
        name = f"{name} Model" if "+" not in name else name
        return name
