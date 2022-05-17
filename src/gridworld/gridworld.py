# gridworld.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from collections import defaultdict
import optparse
import random
import sys


class GridworldActions:
    NORTH = "north"
    SOUTH = "south"
    WEST = "west"
    EAST = "east"
    EXIT = "exit"


class Gridworld:

    def __init__(self, grid):
        # layout
        if type(grid) == type([]): grid = makeGrid(grid)
        self.grid = grid

        # parameters
        self.livingReward = 0.0
        self.noise = 0.2

    def setLivingReward(self, reward):
        """The (negative) reward for exiting "normal" states.

        Note that in the R+N text, this reward is on entering a state and therefore is not
        clearly part of the state's future rewards.
        """
        self.livingReward = reward

    def setNoise(self, noise):
        """The probability of moving in an unintended direction. Note that taking a `noisy`
        action means deviating left or right from your direction and does not include
        mooving backwards."""
        self.noise = noise

    def getPossibleActions(self, state):
        """Returns list of valid actions for 'state'.

        Note that you can request moves into walls and that "exit" states transition to
        the terminal state under the special action "done".
        """
        if state == self.grid.terminalState:
            return ()
        x, y = state
        if type(self.grid[x][y]) == int:
            return (GridworldActions.EXIT,)
        return (GridworldActions.NORTH, GridworldActions.SOUTH,
                GridworldActions.WEST, GridworldActions.EAST)

    def getStates(self):
        """Return list of all states."""
        # The true terminal state.
        states = [self.grid.terminalState]
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] != '#':
                    state = (x,y)
                    states.append(state)
        return states

    def getStartState(self):
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] == 'S':
                    return (x, y)
        raise Exception('Grid has no start state')

    def getRandomState(self):
        states = []
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if self.grid[x][y] != '#':
                    state = (x,y)
                    states.append(state)
        return random.choice(states)

    def getReward(self, state, action, nextState):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being departed (as in the R+N book
        examples, which more or less use this convention).
        """
        if state == self.grid.terminalState:
            return 0.0
        x, y = state
        cell = self.grid[x][y]
        if type(cell) == int or type(cell) == float:
            return cell
        return self.livingReward

    def isTerminal(self, state):
        """Only the TERMINAL_STATE state is *actually* a terminal state. The other "exit"
        states are technically non-terminals with a single action "exit" which leads to
        the true terminal state. This convention is to make the grids line up with the
        examples in the R+N textbook.
        """
        return state == self.grid.terminalState

    def getTransitionStatesAndProbs(self, state, action):
        """Returns a list of (nextState, prob) pairs representing the states reachable
        from 'state' by taking 'action' along with their transition probabilities.
        """
        if action not in self.getPossibleActions(state):
            raise Exception("Illegal action!")

        if self.isTerminal(state):
            return []

        x, y = state
        if type(self.grid[x][y]) == int or type(self.grid[x][y]) == float:
            termState = self.grid.terminalState
            return [(termState, 1.0)]

        successors = []

        northState = (self._isAllowed(y+1,x) and (x,y+1)) or state
        westState = (self._isAllowed(y,x-1) and (x-1,y)) or state
        southState = (self._isAllowed(y-1,x) and (x,y-1)) or state
        eastState = (self._isAllowed(y,x+1) and (x+1,y)) or state

        if action == 'north' or action == 'south':
            if action == 'north':
                successors.append((northState,1-self.noise))
            else:
                successors.append((southState,1-self.noise))

            massLeft = self.noise
            successors.append((westState,massLeft/2.0))
            successors.append((eastState,massLeft/2.0))

        if action == 'west' or action == 'east':
            if action == 'west':
                successors.append((westState,1-self.noise))
            else:
                successors.append((eastState,1-self.noise))

            massLeft = self.noise
            successors.append((northState,massLeft/2.0))
            successors.append((southState,massLeft/2.0))

        successors = self._aggregate(successors)

        return successors

    def _aggregate(self, successors):
        d = defaultdict(lambda: 0)
        for state, prob in successors:
            d[state] += prob
        return list(d.items())

    def _isAllowed(self, y, x):
        if y < 0 or y >= self.grid.height: return False
        if x < 0 or x >= self.grid.width: return False
        return self.grid[x][y] != '#'


class Grid:
    """
    A 2-dimensional array of immutables backed by a list of lists. Data is accessed via
    grid[x][y] where (x,y) are cartesian coordinates with x horizontal and y vertical.
    The origin (0,0) is in the bottom left corner.

    The __str__ method constructs an output that is oriented appropriately.
    """
    def __init__(self, width, height, initialValue=' '):
        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        self.terminalState = 'TERMINAL_STATE'

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def _getLegacyText(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._getLegacyText())

#-------------------------------------- make utils --------------------------------------#
def makeGrid(gridString):
    width, height = len(gridString[0]), len(gridString)
    grid = Grid(width, height)
    for ybar, line in enumerate(gridString):
        y = height - ybar - 1
        for x, el in enumerate(line):
            grid[x][y] = el
    return grid

#---------------------------------- Gridworld layouts -----------------------------------#
def getCliffGrid():
    grid = [[' ',' ',' ',' ',' '],
            ['S',' ',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return Gridworld(makeGrid(grid))

def getCliffGrid2():
    grid = [[' ',' ',' ',' ',' '],
            [8,'S',' ',' ',10],
            [-100,-100, -100, -100, -100]]
    return Gridworld(grid)

def getDiscountGrid():
    grid = [[' ', ' ', ' ', ' ', ' '],
            [' ', '#', ' ', ' ', ' '],
            [' ', '#',  1 , '#',  10],
            ['S', ' ', ' ', ' ', ' '],
            [-10, -10, -10, -10, -10]]
    return Gridworld(grid)

def getBridgeGrid():
    grid = [[ '#',-100, -100, -100, -100, -100, '#'],
            [  1 , 'S',  ' ',  ' ',  ' ',  ' ',  10],
            [ '#',-100, -100, -100, -100, -100, '#']]
    return Gridworld(grid)

def getBookGrid():
    grid = [[' ', ' ', ' ',  1 ],
            [' ', '#', ' ', -1 ],
            ['S', ' ', ' ', ' ']]
    return Gridworld(grid)

def getMazeGrid():
    grid = [[' ', ' ', ' ',  1 ],
            ['#', '#', ' ', '#'],
            [' ', '#', ' ', ' '],
            [' ', '#', '#', ' '],
            ['S', ' ', ' ', ' ']]
    return Gridworld(grid)

def getTwinGrid():
    grid = [[ 10, ' ', ' ', ' ', ' ', ' ', 10],
            [' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', 'S', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [ 10, ' ', ' ', ' ', ' ', ' ', 10]]
    return Gridworld(grid)

def getConfuseGrid():
    grid = [['S', ' ', ' ', ' ',  10],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [ 10, ' ', ' ', ' ', ' ']]
    return Gridworld(grid)

def getSmallGrid():
    grid = [['S', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ',  10]]
    return Gridworld(grid)

def getZeroGrid():
    grid = [['S', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ',  0 ]]
    return Gridworld(grid)

#