import random
from gym import spaces
import numpy as np

# two-dimensional continuous stochastic h-MDP environment

class ContinuousStochastichMDP2Env:

    def __init__(self):
        self.visited_six = False
        self.current_state = 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([5.0, 5.0]))

    def reset(self):
        self.visited_six = False
        self.current_state = [1.0, 1.0]
        return self.current_state

    def step(self, action):
        if self.current_state[0] > 0.5:
            # If "right" selected
            #if self.current_state[0] < 4.5 or self.current_state[1] < 4.5:
            if action == 3:
                if random.random() > 0.0:
                    self.current_state[0] += 1 #+ random.gauss(0, 0.1)
                else:
                    self.current_state[0] -= 1 #+ random.gauss(0, 0.1)
                self.current_state[0] = min(self.current_state[0], 5.0)
            # If "left" selected
            if action == 2:
                if random.random() > 0.0:
                    self.current_state[1] += 1 #+ random.gauss(0, 0.1)
                else:
                    self.current_state[1] -= 1 #+ random.gauss(0, 0.1)
                self.current_state[1] = min(self.current_state[1], 5.0)
            # If "left" selected
            if action == 1:
                self.current_state[0] -= 1 #+ random.gauss(0, 0.1)
            if action == 0:
                self.current_state[1] -= 1 #+ random.gauss(0, 0.1)
            # If state 6 reached
            if 4.5 >= self.current_state[0] >= 3.5 and 4.5 >= self.current_state[1] >= 3.5:
                #return self.current_state, 1.00, True, None
                self.visited_six = True
        if self.current_state[0] <= 0.5:
            if self.visited_six:
                return self.current_state, 1.00, True, None
            else:
                return self.current_state, 1.00/100.00, True, None
        else:
            return self.current_state, 0.0, False, None
