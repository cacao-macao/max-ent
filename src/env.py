import random
import numpy as np

from src.gridworld.gridworld import GridworldActions


class GridworldEnvironment:

    def __init__(self, gridWorld):
        self.gridWorld = gridWorld
        self._state = None
        self._idxToAct = {0: GridworldActions.NORTH,
                          1: GridworldActions.SOUTH,
                          2: GridworldActions.WEST,
                          3: GridworldActions.EAST,
                          4: GridworldActions.EXIT}
        self._actToIdx = {v: k for k, v in self._idxToAct.items()}
        self.reset()

    def reset(self):
        self._state = self.gridWorld.getStartState()
        obs = self._observe(self._state)
        self._shape = obs.shape
        return obs

    def set_random_state(self):
        self._state = self.gridWorld.getRandomState()
        obs = self._observe(self._state)
        self._shape = obs.shape
        return obs

    def shape(self):
        return self._shape

    def num_actions(self):
        return len(self._idxToAct)

    def actions(self):
        acts = self.gridWorld.getPossibleActions(self._state)
        return list(map(lambda a: self._actToIdx[a], acts))

    def step(self, action):
        action = self._idxToAct[action]
        successors = self.gridWorld.getTransitionStatesAndProbs(self._state, action)
        total_prob = 0.0
        for next_state, prob in successors:
            total_prob += prob
        if total_prob > 1.0:
            raise Exception("Total transition probability more than one.")
        if total_prob < 1.0:
            raise Exception("Total transition probability less than one.")

        rnd = random.random()
        total = 0.0
        for next_state, prob in successors:
            total += prob
            if rnd < total:
                break
        reward = self.gridWorld.getReward(self._state, action, next_state)
        done = self.gridWorld.isTerminal(next_state)
        self._state = next_state
        return self._observe(next_state), reward, done

    def _observe(self, state):
        width = self.gridWorld.grid.width
        height = self.gridWorld.grid.height
        obs = np.zeros(width*height + 1, dtype=np.float32)
        if state == self.gridWorld.grid.terminalState:
            obs[-1] = 1
        else:
            x, y = state
            obs[y*height + x] = 1
        return obs

#