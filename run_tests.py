from q_learning import QLearningAgent
from hierarchical_q_learning import hierarchicalQLearningAgent
from DQN import DQNAgent
from hDQN import hDQNAgent

from envs.hmdp import StochastichMDPEnv as hMDP
from envs.mdp import StochasticMDPEnv as MDP

import utils.plotting as plotting
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt



num_trials = 20

stats_q_learning = []
for i in range(num_trials):
    q_agent = QLearningAgent(env=hMDP(), num_episodes=25000)
    episode_stats = q_agent.learn()
    stats_q_learning.append(episode_stats)

stats_hq_learning = []
for i in range(num_trials):
    hq_agent = hierarchicalQLearningAgent(env=hMDP(), num_episodes=25000)
    episode_stats = hq_agent.learn()
    stats_hq_learning.append(episode_stats)
'''
stats_dqn = []
for i in range(num_trials):
    dqn_agent = DQNAgent(env=hMDP())
    episode_stats = dqn_agent.learn()
    stats_dqn.append(episode_stats)

stats_hdqn = []
for i in range(num_trials):
    hdqn_agent = hDQNAgent(env=hMDP())
    episode_stats = hdqn_agent.learn()
    stats_hdqn.append(episode_stats)
'''


#########################################################

exploit_returns = []
for stat in stats_q_learning:
    exploit_returns += [np.mean(stat.episode_rewards[20000:25000])]

print('Q-learning average exploitative return:', np.mean(exploit_returns))

exploit_returns = []
for stat in stats_hq_learning:
    exploit_returns += [np.mean(stat.episode_rewards[20000:25000])]

print('Hierarchical Q-learning average exploitative return:', np.mean(exploit_returns))


#########################################################

plt.figure()

plotting.plot_rewards(stats_q_learning, c='g')
plotting.plot_rewards(stats_hq_learning, c='b')
'''
plotting.plot_rewards(stats_q_learning, c='r')
plotting.plot_rewards(stats_hq_learning, c='c')
'''

plt.legend(["Q-learning", "Hierarchical Q-learning"])
plt.xlabel("Episode")
plt.ylabel("Extrinsic Reward")
plt.title("Discrete Stochastic Decision Process")

#########################################################

plt.figure()

plotting.plot_visitation_counts(stats_hq_learning)

plt.legend(["s1", "s2", "s3", "s4", "s5", "s6"])
plt.xlabel("Episode")
plt.ylabel("Visit Rate")
plt.title("Discrete Stochastic Decision Process")

#########################################################

plt.figure()

plotting.plot_visitation_counts(stats_q_learning)

plt.legend(["s1", "s2", "s3", "s4", "s5", "s6"])
plt.xlabel("Visit Rate")
plt.ylabel("Extrinsic Reward")
plt.title("Discrete Stochastic Decision Process")

#########################################################
'''
plt.figure()

plotting.plot_visitation_counts(stats_q_learning)

plt.legend(["s1", "s2", "s3", "s4", "s5", "s6"])
plt.xlabel("Visit Rate")
plt.ylabel("Extrinsic Reward")
plt.title("Discrete Stochastic Decision Process")
'''
#########################################################
'''
plt.figure()

plotting.plot_visitation_counts(stats_q_learning)

plt.legend(["s1", "s2", "s3", "s4", "s5", "s6"])
plt.xlabel("Visit Rate")
plt.ylabel("Extrinsic Reward")
plt.title("Discrete Stochastic Decision Process")
'''
#########################################################
'''
plt.figure()

plotting.plot_target_counts(stats_hq_learning)

plt.legend(["s1", "s2", "s3", "s4", "s5", "s6"])
plt.xlabel("Visit Rate")
plt.ylabel("Extrinsic Reward")
plt.title("Discrete Stochastic Decision Process")
'''
#########################################################

plt.show()