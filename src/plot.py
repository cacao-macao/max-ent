import os
from pathlib import Path
import pickle
import sys
sys.path.append("..")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.gridworld.graphics import GraphicsGridworldDisplay
from src.gridworld import gridworld
from src.env import GridworldEnvironment
from src.fcnn_policy import FCNNPolicy
from src.pg_agent import PGAgent


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

    plt.rcParams.update({'font.sans-serif':'Times New Roman'})
    fpath = Path(mpl.get_data_path(), "/usr/share/fonts/truetype/cmu/cmunss.ttf")

    # Set figure sizes.
    cm = 1/2.54 # cm to inch
    fontsize = 10
    fig, ax = plt.subplots(figsize=(8.6*cm, 7.3*cm), dpi=330, tight_layout={"pad":0.7})
    ax.set_xlabel(labels.get("x"), fontsize=fontsize, labelpad=2, font=fpath)
    ax.set_ylabel(labels.get("y"), fontsize=fontsize, labelpad=2, font=fpath)
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.grid(which="major", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", linestyle="--", linewidth=0.3)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f"{x/1000:.0f}k" if x > 0 else "0"))

    # Plot curves.
    for x, f, l, c, a in zip(xs, funcs, legends, fmt, alpha):
        ax.plot(x, f, c, label=l, alpha=a, linewidth=0.5)
    ax.legend(loc=legends_loc, fontsize=8)
    fig.savefig(figname)
    plt.close(fig)


# Iterate and plot figures.
for grid in ["SmallGrid", "MazeGrid"]:
    # Create the environment
    gridWorldLayoutFunction = getattr(gridworld, "get"+grid)
    gridWorldLayout = gridWorldLayoutFunction()
    env = GridworldEnvironment(gridWorldLayout)

    # Load the training results.
    log_dir = f"../logs/{grid}_iters_50001"
    log_dir_entropy = f"../logs/{grid}_iters_50001_entreg"
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
            legends=["mean return",
                     "mean return with ent. reg."],
            labels={"x":"Iteration", "y":"Return"},
            fmt=["--b", "--r"], alpha=[0.3, 0.6])

    # Plot entropy curves.
    file_path_entropy = os.path.join(log_dir_entropy, "policy_ent_joined.png")
    policy_entropy = np.array([train_history[i]["policy_entropy"] for i in range(num_iter)])
    policy_entropy_reg = np.array([train_history_entropy[i]["policy_entropy"]
                                    for i in range(num_iter)])
    logPlot(figname=file_path_entropy, xs=[np.arange(num_iter), np.arange(num_iter)],
            funcs=[policy_entropy, policy_entropy_reg],
            legends=["avg policy entropy",
                     "avg policy entropy with ent. reg."],
            legends_loc = "upper right",
            labels={"x":"Iteration", "y":"Return"},
            fmt=["--b", "--r"], alpha=[0.3, 0.6])


# Initialize the policy.
for grid in ["SmallGrid", "MazeGrid"]:
    gridWorldLayoutFunction = getattr(gridworld, "getMazeGrid2")
    gridWorldLayout = gridWorldLayoutFunction()
    gridWorldLayout.setLivingReward(-1)
    gridWorldLayout.setNoise(0.0)
    env = GridworldEnvironment(gridWorldLayout)
    filepath = f"../logs/{grid}_iters_50001/policy.bin"
    policy = FCNNPolicy.load(filepath)
    agent = PGAgent(policy, env)

    display = GraphicsGridworldDisplay(gridWorldLayout, 150, 1)
    display.start()
    display.displayPolicy(agent, message="")
    display.pause()
    display.displayNullValues(agent, message="")
    display.pause()

#