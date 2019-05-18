import numpy as np
import random
import gym
import matplotlib

from collections import defaultdict
from envs.hmdp import StochastichMDPEnv as hMDP
from envs.mdp import StochasticMDPEnv as MDP
import utils.plotting as plotting


class ExpertAgent():

    def __init__(self, env = hMDP(), num_episodes = 100000, \
                gamma = 0.9, alpha = 0.6, batch_size = 1, epsilon_anneal = 1/50000):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.epsilon_anneal = epsilon_anneal


    def learn(self):

        stats = plotting.Stats(num_episodes=self.num_episodes, num_states=self.env.observation_space.n)

        for i in range(self.num_episodes):
            if i % 1000 == 0:
                    print('Episode ', i)
            s = self.env.reset()
            done = False
            t = 0
            while not done:
                action = 1 # best move in MDP
                s_next, f, done, _ = self.env.step(action)
                stats.episode_rewards[i] += f
                stats.episode_lengths[i] = t
                stats.visitation_count[s_next, i] += 1

                s = s_next
                t += 1

        return stats
        #plotting.plot_episode_stats(stats, smoothing_window=1000)

if __name__ == "__main__":
    agent = ExpertAgent(env=MDP())
    stats = agent.learn()
    print(np.mean(stats.episode_rewards))
    plotting.plot_rewards([stats], smoothing_window=1000)





