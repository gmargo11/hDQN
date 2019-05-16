import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def init_stats(num_epochs=20000, num_states = 6):
    return namedtuple("episode_reward", np.zeros(num_epochs),
                        "visitation_count", np.zeros(num_epochs, num_states))

def plot_rewards(episodes_ydata):
    smoothing_window = 1000

    overall_stats_q_learning = []
    for i in range(num_trials):
        q_agent = QLearningAgent(env=hMDP())
        stats_q_learning = q_agent.learn()
        overall_stats_q_learning.append(pd.Series(stats_q_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    m_stats_q_learning = np.mean(overall_stats_q_learning, axis=0)
    std_stats_q_learning = np.std(overall_stats_q_learning, axis=0)



    overall_stats_hq_learning = []
    for i in range(num_trials):
        hq_agent = hierarchicalQLearningAgent(env=hMDP())
        stats_hq_learning = hq_agent.learn()
        overall_stats_hq_learning.append(pd.Series(stats_hq_learning.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    m_stats_hq_learning = np.mean(overall_stats_hq_learning, axis=0)
    std_stats_hq_learning = np.std(overall_stats_hq_learning, axis=0)



    plt.figure()

    plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c='b')
    plt.plot(range(len(m_stats_hq_learning)), m_stats_hq_learning, c='g')
    plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor='b', facecolor='b')
    plt.fill_between(range(len(std_stats_hq_learning)), m_stats_hq_learning - std_stats_hq_learning, m_stats_hq_learning + std_stats_hq_learning, alpha=0.5, edgecolor='g', facecolor='g')

    plt.legend(["Q-learning", "Hierarchical Q-learning"])
    plt.xlabel("Episode")
    plt.ylabel("Extrinsic Reward")



plt.show()