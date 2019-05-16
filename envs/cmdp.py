import random
from gym import spaces
import numpy as np

class ContinuousStochasticMDPEnv:

    def __init__(self):
        self.current_state = 1.0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([0.0]), high=np.array([5.0]))

    def reset(self):
        self.current_state = 1.0
        return self.current_state

    def step(self, action):
        if 4.5 > self.current_state > 0.5:
            # If "right" selected
            if action == 1:
                self.current_state += random.random() * 2 - 1.0
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
        if self.current_state <= 0.5:
             return self.current_state, 1.00/100.00, True, None
        elif self.current_state >= 4.5:
            return self.current_state, 1.00, True, None
        else:
            return self.current_state, 0.0, False, None
