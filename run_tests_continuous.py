#from q_learning import QLearningAgent
#from hierarchical_q_learning import hierarchicalQLearningAgent
from DQN import DQNAgent
from hDQN import hDQNAgent

from envs.hmdp import StochastichMDPEnv as hMDP
from envs.mdp import StochasticMDPEnv as MDP

import utils.plotting as plotting
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt



num_trials = 3

stats_q_learning = []
for i in range(num_trials):
    q_agent = QLearningAgent(env=cMDP())
    episode_stats = q_agent.learn()
    stats_q_learning.append(episode_stats)

stats_hq_learning = []
for i in range(num_trials):
    hq_agent = hierarchicalQLearningAgent(env=chMDP())
    episode_stats = hq_agent.learn()
    stats_hq_learning.append(episode_stats)

#########################################################

plt.figure()

plotting.plot_rewards(stats_q_learning, c='g')
plotting.plot_rewards(stats_hq_learning, c='b')

plt.legend(["Q-learning", "Hierarchical Q-learning"])
plt.xlabel("Episode")
plt.ylabel("Extrinsic Reward")
plt.title("Fully Observable Discrete Stochastic Decision Process")

plt.show()