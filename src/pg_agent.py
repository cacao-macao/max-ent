import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


class PGAgent:
    """Policy-gradient agent implementation of a reinforcement learning agent.
    The agent uses vanilla policy gradient update to improve its policy.

    Attributes:
        env (Any): Environment object that the agent interacts with.
        policy_network (Any): Policy network object that the agent uses to decide on the
            next action.
        train_history (dict): A dict object used for bookkeeping.
        test_history (dict): A dict object used for bookkeeping.
    """

    def __init__(self, policy_network, env):
        """Initialize policy gradient agent.

        Args:
            env (Any): Environment object.
            policy_network (Any): Policy network object.
        """
        self.policy_network = policy_network
        self.env = env
        self.train_history = {}
        self.test_history = {}

    def policy(self, state, legal=None, temp=1.0):
        """Return a probability distribution over the actions computed by the policy.
        Using the scores returned by the network to compute a boltzmann probability
        distribution over the actions from the action space.

        Args:
            state (np.Array): A numpy array of shape (system_size,), giving the current
                state of the environment.
            legal (list[int], optional): A list of indices of the legal actions for
                the agent. Default value is None, meaning all actions are legal.
            temp (float, optional): Inverse value of the temperature for the boltzmann
                distribution. Default value is 1.0.

        Returns:
            probs (torch.Tensor): Tensor of shape (num_actions,) giving a probability
                distribution over the action space.
        """
        state = torch.from_numpy(state)
        state = state.to(self.policy_network.device)
        logits = self.policy_network(state) * temp
        if legal is not None:
            illegal = list(set(range(self.env.num_actions())) - set(legal))
            logits[illegal] = float("-inf") + torch.finfo(torch.float32).eps
        probs = F.softmax(logits, dim=-1)
        return probs

    def rollout(self, episodes, steps, temp=1.0):
        """Starting from the initial environment state perform a rollout using the
        current policy.

        Args:
            episodes (int): Number of rollout episodes.
            steps (int): Number of steps to rollout the policy.
            temp (float, optional): Inverse value of the temperature for the boltzmann
                distribution. Default value is 1.0.

        Returns:
            states (torch.Tensor): Tensor of shape (b, t, d), giving the states produced
                during policy rollout, where b = number of episodes,
                t = number of time steps, d = size of the environment state.
            actions (torch.Tensor): Tensor of shape (b, t), giving the actions selected by
                the policy during rollout.
            rewards (torch.Tensor): Tensor of shape (b, t), giving the rewards obtained
                during policy rollout.
            masks (torch.Tensor): Tensor of shape (b, t), of boolean values, that masks
                out the part of the trajectory after it has finished.
        """
        device = self.policy_network.device
        d = self.policy_network.input_size

        # Allocate torch tensors to store the data from the rollout.
        states = torch.zeros(size=(episodes, steps, d), dtype=torch.float32, device=device)
        actions = torch.zeros(size=(episodes, steps), dtype=torch.int64, device=device)
        rewards = torch.zeros(size=(episodes, steps), dtype=torch.float32, device=device)
        masks = torch.ones(size=(episodes, steps), dtype=torch.bool, device=device)

        for i in range(episodes):
            # Perform episode rollout starting from the initial state.
            state = self.env.set_random_state()
            # state = self.env.reset()
            for j in range(steps):
                probs = self.policy(state, legal=self.env.actions(), temp=temp)
                states[i, j] = torch.from_numpy(state)
                actions[i, j]  = torch.multinomial(probs, 1)
                state, reward, done = self.env.step(actions[i, j].item())
                rewards[i, j] = reward
                if done: break
            masks[i, j+1:] = False

        return states, actions, rewards, masks

    def train(self, num_iter, episodes, steps, learning_rate, lr_decay=1.0, clip_grad=10.0,
              reg=0.0, entropy_reg=0.0, log_every=1, stdout=sys.stdout):
        """Train the agent using vanilla policy-gradient algorithm.

        Args:
            num_iter (int): Number of iterations to train the agent for.
            episodes (int): Number of episodes used to approximate the gradient of the policy.
            steps (int): Number of steps to rollout the policy for.
            learning_rate (float): Learning rate for gradient decent.
            lr_decay (float, optional): Multiplicative factor of learning rate decay.
                Default value is 1.0.
            clip_grad (float, optional): Threshold for gradient norm during backpropagation.
                Default value is 10.0.
            reg (float, optional): L2 regularization strength. Default value is 0.0.
            entropy_reg (float, optional): Entropy regularization strength.
                Default value is 0.0.
            log_every (int, optional): Every `log_every` iterations write the results to
                the log file. Default value is 100.
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        # Move the neural network to device and prepare for training.
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        print(f"Using device: {device}\n", file=stdout)
        self.policy_network.train()
        self.policy_network = self.policy_network.to(device)

        # Initialize the optimizer and the scheduler.
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
        print(f"Using optimizer:\n{str(optimizer)}\n", file=stdout)

        # Start the training loop.
        for i in range(num_iter):
            tic = time.time()

            # Perform policy rollout.
            states, actions, rewards, masks = self.rollout(episodes, steps)

            # Compute the loss.
            logits = self.policy_network(states)
            episode_entropy = self.entropy_term(logits, actions)
            q_values = self.reward_to_go(rewards)
            q_values += entropy_reg * episode_entropy
            # q_values -= self.reward_baseline(q_values, masks)
            nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
            weighted_nll = torch.mul(masks * nll, q_values)
            loss = torch.mean(weighted_nll)

            # Perform backward pass.
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()

            # Compute average policy entropy.
            probs = F.softmax(logits, dim=-1) + torch.finfo(torch.float32).eps
            avg_policy_ent = -torch.sum(masks * torch.sum(probs*torch.log(probs),axis=-1)) / torch.sum(masks)

            # Book-keeping.
            self.train_history[i] = {
                "rewards"       : rewards.cpu().numpy(),
                "masks"         : masks.cpu().numpy(),
                "exploration"   : episode_entropy.detach().cpu().numpy(),
                "policy_entropy": avg_policy_ent.item(),
                "loss"          : loss.item(),
                "total_norm"    : total_norm.cpu().numpy(),
                "nsolved"       : torch.sum(~masks[:,-1]).item(),
                "nsteps"        : ((~masks[:,-1])*torch.sum(masks, axis=1)).cpu().numpy(),
            }

            toc = time.time()

            # Log results to file.
            if i % log_every == 0:
                stats = self.train_history[i]
                episodes, steps = stats["rewards"].shape
                final_rewards = stats["rewards"][np.arange(episodes), np.sum(stats["masks"], axis=-1) - 1]
                print(f"Iteration ({i}/{num_iter}) took {toc-tic:.3f} seconds.", file=stdout)
                print(f"""\
                Mean final reward:        {np.mean(final_rewards):.4f}
                Mean return:              {np.mean(np.sum(stats["rewards"], axis=1)):.4f}
                Policy entropy:           {stats["policy_entropy"]:.4f}
                Pseudo loss:              {stats["loss"]:.5f}
                Total gradient norm:      {stats["total_norm"]:.5f}
                Solved trajectories:      {stats["nsolved"]} / {episodes}
                Avg steps to solve:       {np.mean(stats["nsteps"][stats["nsteps"].nonzero()]):.3f}
                """, file=stdout)
                # Mean exploration return:  {np.mean(np.sum(stats["exploration"], axis=1)):.4f}

    def reward_to_go(self, rewards):
        """Compute the reward-to-go at every timestep t."""
        return rewards + torch.sum(rewards, keepdims=True, dim=-1) - torch.cumsum(rewards, dim=-1)

    def reward_baseline(self, returns, masks):
        """Compute the baseline as the average return at timestep t."""
        # When working with a batch of episodes, only the active episodes are considered
        # for calculating the baseline. The reward-to-go sum of finished episodes is 0.
        return torch.sum(returns, dim=0) / torch.maximum(
                        torch.sum(masks, dim=0), torch.Tensor([1]).to(self.policy_network.device))

    def entropy_term(self, logits, actions):
        """Compute the entropy regularization term."""
        step_entropy = -F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        episode_entropy = torch.sum(step_entropy, dim=-1, keepdim=True)
        return -0.5 * episode_entropy

    def save_policy(self, filepath):
        """Save the policy as .bin file to disk."""
        self.policy_network.save(os.path.join(filepath, "policy.bin"))

    def save_history(self, filepath):
        """Save the training history and the testing history as pickle dumps."""
        with open(os.path.join(filepath, "train_history.pickle"), "wb") as f:
            pickle.dump(self.train_history, f, protocol=pickle.HIGHEST_PROTOCOL)

#