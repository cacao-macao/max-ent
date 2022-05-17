import os
import pickle
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np

from src.env import GridworldEnvironment
from src.gridworld import gridworld


def logPlot(figname, xs=None, funcs=[], legends=[None], legends_loc="lower right",
            labels={}, fmt=["--k"], alpha=[1.0]):
    """Plot @funcs as curves on a figure and save the figure as `figname`.

    Args:
        figname (str): Full path to save the figure to a file.
        xs (list[np.Array], optional): List of arrays of x-axis data points.
            Default value is None.
        funcs (list[np.Array], optional): List of arrays of data points. Every array of
            data points from the list is plotted as a curve on the figure.
            Default value is [].
        legends (list[str], optional): A list of labels for every curve that will be
            displayed in the legend. Default value is [None].
        legends_loc (str, optional): Location of the legends. Default value is "lower right".
        labels (dict, optional): A map specifying the labels of the coordinate axes.
            `labels["x"]` specifies the label of the x-axis.
            `labels["y"]` specifies the label of the y-axis.
            Default value is {}.
        fmt (list[str], optional): A list of formating strings for every curve.
            Default value is ["--k"].
        alpha (list[float], optional): A list of line transparency for every curve.
            Default value is [1.0].
    """
    if xs is None:
        xs = [np.arange(len(f)) for f in funcs]
    if len(legends) == 1:
        legends = legends * len(funcs)
    if len(fmt) == 1:
        fmt = fmt * len(funcs)
    if len(alpha) == 1:
        alpha = alpha * len(funcs)

    # Set figure sizes.
    fig, ax = plt.subplots(figsize=(24, 18), dpi=170)
    ax.set_xlabel(labels.get("x"), fontsize=42, labelpad=30)
    ax.set_ylabel(labels.get("y"), fontsize=42, labelpad=30)
    ax.tick_params(axis="both", which="both", labelsize=32)
    ax.grid()

    # Plot curves.
    for x, f, l, c, a in zip(xs, funcs, legends, fmt, alpha):
        ax.plot(x, f, c, label=l, alpha=a, linewidth=0.8)
    ax.legend(loc=legends_loc, fontsize=36)
    fig.savefig(figname)
    plt.close(fig)


# Iterate and plot figures.
for grid in ["SmallGrid", "ConfuseGrid"]:
    # Create the environment
    gridWorldLayoutFunction = getattr(gridworld, "get"+grid)
    gridWorldLayout = gridWorldLayoutFunction()
    env = GridworldEnvironment(gridWorldLayout)

    # Load the training results.
    log_dir = f"../logs/a_{grid}_iters_50001_entreg_0.0"
    log_dir_entropy = f"../logs/a_{grid}_iters_50001_entreg_1.0"
    with open(os.path.join(log_dir, "train_history.pickle"), "rb") as f:
        train_history = pickle.load(f)
    with open(os.path.join(log_dir_entropy, "train_history.pickle"), "rb") as f:
        train_history_entropy = pickle.load(f)

    num_iter = len(train_history)
    returns = [np.sum(train_history[i]["rewards"], axis=1).mean() for i in range(num_iter)]
    returns_entropy = [np.sum(train_history_entropy[i]["rewards"], axis=1).mean()
                        for i in range(num_iter)]

    # Plot return curves.
    file_path_returns = os.path.join(log_dir_entropy, "returns_joined.png")
    logPlot(figname=file_path_returns, xs=[np.arange(num_iter), np.arange(num_iter)],
            funcs=[returns, returns_entropy],
            legends=["mean batch returns (batch_size=32)",
                     "mean batch returns with entropy (batch_size=32)"],
            labels={"x":"Iteration", "y":"Return"},
            fmt=["--b", "--r"], alpha=[0.4, 0.8])

    # Plot entropy curves.
    file_path_entropy = os.path.join(log_dir_entropy, "policy_ent_joined.png")
    policy_entropy = np.array([train_history[i]["policy_entropy"] for i in range(num_iter)])
    policy_entropy_reg = np.array([train_history_entropy[i]["policy_entropy"]
                                    for i in range(num_iter)])
    logPlot(figname=file_path_entropy, xs=[np.arange(num_iter), np.arange(num_iter)],
            funcs=[policy_entropy, policy_entropy_reg],
            legends=["average policy entropy (batch_size=32)",
                     "average policy entropy with entropy reg. (batch_size=32)"],
            legends_loc = "upper right",
            labels={"x":"Iteration", "y":"Return"},
            fmt=["--b", "--r"], alpha=[0.4, 0.8])

#