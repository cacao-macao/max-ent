# Entropy regularization for policy gradient methods

This repository provides source code for running the experiments described in the following
work: (link here). The following modules are implemented:
 * `pg_agent.py` module contains the code for training a policy gradient agent with entropy
regularization
 * `fcnn_policy.py` module contains a PyTorch implementation of a multi-layer perceptron
 * `gridworld` module contains an implementation of the GridWorld as described in
Sutton & Barto. The source code is taken from the UC Berkeley (http://ai.berkeley.edu)
 * `train.py` is a scripts that trains a policy gradient agent to play the gridworld. To
run the training execute the following from the `src` directory:
    ```
    python3 train.py --grid SmallGrid --iters 10001 --episodes 32 --entropy_reg 1.0
    ```
    Logging information with history from the training is saved inside a `logs` directory.

 * `plot.py` is a script that creates plots from the training history. To run this script
execute:
    ```
    python3 plot.py
    ```
    Before running this script the agent must be trained on both `SmallGrid` and
    `ConfuseGrid` with and without entropy regularization.
