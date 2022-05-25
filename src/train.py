import argparse
import os
import random
import sys
import time
sys.path.append("..")

import numpy as np
import torch

from src.pg_agent import PGAgent
from src.env import GridworldEnvironment
from src.gridworld import gridworld
from src.gridworld.graphics import GraphicsGridworldDisplay
from src.fcnn_policy import FCNNPolicy


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int, default=0,
    help="random seed value")
parser.add_argument("--entropy_reg", dest="entropy_reg", type=float, default=0.,
    help="Entropy regularization strength")
parser.add_argument("--livingReward", dest="livingReward", type=float, default=-1.,
    help="Reward for living for a time step (default %default)")
parser.add_argument("--noise", dest="noise", type=float, default=0.0,
    help="How often action results in unintended direction (default %default)")
parser.add_argument("--grid", dest="grid", type=str, default="SmallGrid",
    help="Grid to use (case sensitive; options are SmallGrid, MazeGrid, default %default)")
parser.add_argument("--iters", dest="iters", type=int, default=10,
    help="Number of iterations of value iteration (default %default)")
parser.add_argument("--episodes", dest="episodes", type=int, default=32,
    help="Number of episodes of the MDP to run (default %default)")
parser.add_argument("--windowSize", dest="windowSize", type=int, default=150,
    help="Request a window width of X pixels *per grid cell* (default %default)")
parser.add_argument("--speed", dest="speed", type=float, default=1.0,
    help="Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)")
args = parser.parse_args()

args.dropout = 0.0
args.learning_rate = 1e-3
args.lr_decay = 1.0
args.steps = 12
args.clip_grad = 10.0
args.reg = 0.0
args.log_every = 100


# Fix the random seeds for NumPy and PyTorch, and set print options.
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=~False)


# Create file to log output during training.
log_dir = f"../logs/{args.grid}_iters_{args.iters}_entreg_{args.entropy_reg}"#_decayed"
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
policy_network = FCNNPolicy(input_size, hidden_dims, output_size, args.dropout)

# Train the policy-gradient agent.
agent = PGAgent(policy_network, env)
tic = time.time()
agent.train(args.iters, args.episodes, args.steps, args.learning_rate, args.lr_decay,
    args.clip_grad, args.reg, args.entropy_reg, args.log_every, stdout)
toc = time.time()
agent.save_policy(log_dir)
agent.save_history(log_dir)
print(f"Training took {toc-tic:.3f} seconds.", file=stdout)

# Close the logging file.
stdout.close()

# Display resulting policy.
display = GraphicsGridworldDisplay(gridWorldLayout, args.windowSize, args.speed)
display.start()
display.displayPolicy(agent, message="")
display.pause()

#