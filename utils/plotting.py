import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Stats():
    def __init__(self, num_episodes=20000, num_states = 6, continuous=False):
        self.episode_rewards = np.zeros(num_episodes)
        self.episode_lengths = np.zeros(num_episodes)
        if not continuous:
            self.visitation_count = np.zeros((num_states, num_episodes))
            self.target_count = np.zeros((num_states, num_episodes))

def plot_rewards(episodes_ydata, smoothing_window = 1000, c='b'):
    smoothing_window = 1000

    overall_stats_q_learning = []
    for trialdata in episodes_ydata:
        overall_stats_q_learning.append(pd.Series(trialdata.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    m_stats_q_learning = np.mean(overall_stats_q_learning, axis=0)
    std_stats_q_learning = np.std(overall_stats_q_learning, axis=0)

    plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c)
    plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c, facecolor=c)

def plot_visitation_counts(episodes_ydata, smoothing_window = 1000, c=['b', 'g', 'r', 'y', 'k', 'c'], num_states = None):

    if not num_states: 
        num_states = len(episodes_ydata[0].visitation_count)

    overall_stats_q_learning = [[] for i in range(num_states)]
    for trialdata in episodes_ydata:
        for state in range(num_states):
            overall_stats_q_learning[state].append(pd.Series(trialdata.visitation_count[state]).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    
    for state in range(num_states):
        m_stats_q_learning = np.mean(overall_stats_q_learning[state], axis=0)
        std_stats_q_learning = np.std(overall_stats_q_learning[state], axis=0)

        plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c[state])
        plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c[state], facecolor=c[state])

def plot_target_counts(episodes_ydata, smoothing_window = 1000, c=['b', 'g', 'r', 'y', 'k', 'c']):

    num_states = len(episodes_ydata[0].target_count)

    overall_stats_q_learning = [[] for i in range(num_states)]
    for trialdata in episodes_ydata:
        for state in range(num_states):
            overall_stats_q_learning[state].append(pd.Series(trialdata.target_count[state]).rolling(smoothing_window, min_periods=smoothing_window).mean().data)
    
    for state in range(num_states):
        m_stats_q_learning = np.mean(overall_stats_q_learning[state], axis=0)
        std_stats_q_learning = np.std(overall_stats_q_learning[state], axis=0)

        plt.plot(range(len(m_stats_q_learning)), m_stats_q_learning, c=c[state])
        plt.fill_between(range(len(std_stats_q_learning)), m_stats_q_learning - std_stats_q_learning, m_stats_q_learning + std_stats_q_learning, alpha=0.5, edgecolor=c[state], facecolor=c[state])

def plot_q_values(model, observation_space, action_space):

    res = 100

    test_observations = np.linspace(observation_space.low, observation_space.high, res)
    
    print((action_space.n, res))
    q_values = np.zeros((action_space.n, res))

    for action in range(action_space.n):
        for obs in range(res):
            q_values[action, obs] = model.predict(test_observations[obs])[0, action]

        plt.plot(test_observations, q_values[action])

