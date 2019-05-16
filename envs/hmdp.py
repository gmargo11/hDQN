import random
from gym import spaces

class StochastichMDPEnv:

    def __init__(self):
        self.visited_six = False
        self.current_state = 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(6)

    def reset(self):
        self.visited_six = False
        self.current_state = 1
        return self.current_state

    def step(self, action):
        if self.current_state != 0:
            # If "right" selected
            if action == 1:
                if self.current_state < 5 and random.random() < 0.5:
                    self.current_state += 1
                else:
                    self.current_state -= 1
            # If "left" selected
            if action == 0:
                self.current_state -= 1
            # If state 6 reached
            if self.current_state == 5:
                #return self.current_state, 1.00, True, None
                self.visited_six = True
        if self.current_state == 0:
            if self.visited_six:
                return self.current_state, 1.00, True, None
            else:
                return self.current_state, 1.00/100.00, True, None
        else:
            return self.current_state, 0.0, False, None
