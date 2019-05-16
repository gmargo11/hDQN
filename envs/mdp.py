import random
from gym import spaces

class StochasticMDPEnv:

    def __init__(self):
        self.current_state = 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(11)

    def reset(self):
        self.current_state = 1
        return self.current_state

    def step(self, action):
        if self.current_state != 0 and self.current_state != 10:
            # If "right" selected
            if self.current_state < 5:
                if action == 1:
                    if random.random() < 0.5:
                        self.current_state += 1
                    else:
                        self.current_state -= 1
                # If "left" selected
                if action == 0:
                    self.current_state -= 1
            elif self.current_state > 5:
                if action == 1:
                    if random.random() < 0.5:
                        self.current_state -= 1
                    else:
                        self.current_state += 1
                # If "left" selected
                if action == 0:
                    self.current_state += 1
            elif self.current_state == 5:
                self.current_state += 1

        if self.current_state == 10:
            return self.current_state, 1.00, True, None
            #self.visited_six = True
        elif self.current_state == 0:
            return self.current_state, 1.00/100.00, True, None
        else:
            return self.current_state, 0.0, False, None
