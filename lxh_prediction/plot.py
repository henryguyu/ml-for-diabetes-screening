import os
import pandas as pd

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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc="lower right")


def plot_range(x, y_lower, y_upper, color="grey", alpha=0.2, **kwargs):
    plt.fill_between(x, y_lower, y_upper, color=color, alpha=alpha, **kwargs)


class ExpFigure:
    def __init__(self):
        self.fig = plt.figure(figsize=(5, 5))
        self.y_means = {}
        self.x_means = {}
        self.x_base = None
        self.y_base = None

    def run(self, name, model, feat_collection):
        raise NotImplementedError

    def next_color(self):
        return cfg.color_map[len(self.y_means)]

    def save(self, name):
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
