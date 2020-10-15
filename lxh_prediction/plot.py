import matplotlib.pyplot as plt


def plot_curve(x, y, name="ROC curve", xlabel="x", ylabel="y", title=None, subline=None):
    plt.figure()
    lw = 2
    plt.plot(x, y, color="darkorange", lw=lw, label=name)
    if subline is not None:
        # subline: [[0, 0], [1, 1]]
        plt.plot(*subline, color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
