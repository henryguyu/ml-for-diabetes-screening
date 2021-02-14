import matplotlib.pyplot as plt


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
    **kwargs
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
