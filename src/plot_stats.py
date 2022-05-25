import argparse
import os
import random
import sys
import time
sys.path.append("..")

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from src.pg_agent import PGAgent
from src.env import GridworldEnvironment
from src.gridworld import gridworld
from src.gridworld.graphics import GraphicsGridworldDisplay
from src.fcnn_policy import FCNNPolicy


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
    help="random seed value")
parser.add_argument("--livingReward", dest="livingReward", type=float, default=-1.,
    help="Reward for living for a time step (default %default)")
parser.add_argument("--noise", dest="noise", type=float, default=0.0,
    help="How often action results in unintended direction (default %default)")
parser.add_argument("--grid", dest="grid", type=str, default="MazeGrid",
    help="Grid to use (case sensitive; options are SmallGrid, MazeGrid, default %default)")
parser.add_argument("--episodes", dest="episodes", type=int, default=32,
    help="Number of episodes of the MDP to run (default %default)")
parser.add_argument("--windowSize", dest="windowSize", type=int, default=150,
    help="Request a window width of X pixels *per grid cell* (default %default)")
parser.add_argument("--speed", dest="speed", type=float, default=1.0,
    help="Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)")
args = parser.parse_args()

args.iters = 20001
args.dropout = 0.0
args.learning_rate = 1e-3
args.lr_decay = 1.0
args.steps = 10
args.clip_grad = 10.0
args.reg = 0.0
args.log_every = 2000


# Fix the random seeds for NumPy and PyTorch, and set print options.
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=~False)


# Create file to log output during training.
log_dir = f"../logs/entropy_statistics"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")
stdout = open(log_file, "w")

# Create the environment
gridWorldLayoutFunction = getattr(gridworld, "get"+args.grid)
gridWorldLayout = gridWorldLayoutFunction()
gridWorldLayout.setLivingReward(args.livingReward)
gridWorldLayout.setNoise(args.noise)
env = GridworldEnvironment(gridWorldLayout)

# Initialize the policy.
input_size = env.shape()
hidden_dims = []
output_size = env.num_actions()


for grid_layout in ["ConfuseGrid"]:
    for ent in np.arange(5.2, -0.01, -0.1):
        policy_network = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)

        # Train the policy-gradient agent.
        agent = PGAgent(policy_network, env)
        tic = time.time()
        agent.train(args.iters, args.episodes, args.steps, args.learning_rate, args.lr_decay,
            args.clip_grad, args.reg, ent, args.log_every, stdout)
        toc = time.time()
        log_dirr = os.path.join(log_dir, f"{grid_layout}/ent_reg_{ent:0.1f}")
        os.makedirs(log_dirr, exist_ok=True)
        agent.save_policy(log_dirr)
        agent.save_history(log_dirr)

print(f"Training took {toc-tic:.3f} seconds.", file=stdout)

# Close the logging file.
stdout.close()



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
    fig, ax = plt.subplots(figsize=(8.6*cm, 7*cm), dpi=330, tight_layout={"pad":0.7})
    ax.set_xlabel(labels.get("x"), fontsize=fontsize, labelpad=6, font=fpath)
    ax.set_ylabel(labels.get("y"), fontsize=fontsize, labelpad=6, font=fpath)
    ax.set_ylim([0, 1.4])
    ax.tick_params(axis="both", which="both", labelsize=fontsize)
    ax.grid(which="major", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", linestyle="--", linewidth=0.3)

    # Plot curves.
    for x, f, l, c, a in zip(xs, funcs, legends, fmt, alpha):
        ax.plot(x, f, c, label=l, alpha=a, linewidth=2)
    ax.legend(loc=legends_loc, fontsize=fontsize)
    fig.savefig(figname)
    plt.close(fig)


gridWorldLayoutFunction = getattr(gridworld, "getMazeGrid")
gridWorldLayout = gridWorldLayoutFunction()
gridWorldLayout.setLivingReward(-1)
gridWorldLayout.setNoise(0.0)
env = GridworldEnvironment(gridWorldLayout)

xs = []
ys = []

for ent in np.arange(5.2, -0.01, -0.1):
    log_dir = f"../logs/entropy_statistics/ConfuseGrid/ent_reg_{ent:0.1f}"
    policy = FCNNPolicy.load(os.path.join(log_dir, "policy.bin"))
    agent = PGAgent(policy, env)

    entropies_list = []
    for state in agent.env.gridWorld.getStates():
        if state == agent.env.gridWorld.grid.terminalState:
            continue
        obs = agent.env._observe(state)
        agent.env._state = state
        probs = (agent.policy(obs, agent.env.actions()) + torch.finfo(torch.float32).eps).detach().cpu().numpy()
        entropy = -np.sum(np.log(probs) * probs)
        entropies_list.append(entropy)

    policy_ent = np.mean(np.array(entropies_list))

    xs.append(ent)
    ys.append(policy_ent)


xs.reverse()
ys.reverse()

file_path = "../logs/entropy_statistics/ConfuseGrid/policy_entropy.png"
logPlot(figname=file_path, xs=[xs], funcs=[ys],
        labels={"x":"Entropy regularization temperature", "y":"Average policy entropy"},
        fmt=["-r"])

#