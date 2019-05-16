import numpy as np
import random
import gym
import matplotlib

from collections import defaultdict
from envs.hmdp import StochastichMDPEnv as hMDP
from envs.mdp import StochasticMDPEnv as MDP
import utils.plotting as plotting


class QLearningAgent():

    def __init__(self, env = hMDP(), num_episodes = 25000, \
                gamma = 0.9, alpha = 0.6, batch_size = 1, epsilon_anneal = 1/12000):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.epsilon_anneal = epsilon_anneal



    def epsGreedy(self, state, B, eps, Q):
        action_probabilities = np.ones_like(Q[state]) * eps / len(Q[state])
        best_action = np.argmax(Q[state])
        action_probabilities[best_action] += (1.0 - eps)
        action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
        return action

    def QValueUpdate(self, Q, D):
        mini_batch = random.sample(D, self.batch_size)

        for s, action, f, s_next, done in mini_batch:
            target = f
            if not done:
                best_next_action = np.argmax(Q[s_next])
                target = f + self.gamma * Q[s_next][best_next_action]
            delta = target - Q[s][action]
            Q[s][action] += self.alpha * delta

        return Q

    def learn(self):

        stats = plotting.Stats(num_episodes=self.num_episodes, num_states=self.env.observation_space.n)


        D = None
        Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        A = self.env.action_space

        epsilon = 1.0

        for i in range(self.num_episodes):
            if i % 1000 == 0:
                    print('Episode ', i)
                    print(epsilon)
            s = self.env.reset()
            done = False
            t = 0
            while not done:
                action = self.epsGreedy(s, A, epsilon, Q)
                s_next, f, done, _ = self.env.step(action)
                stats.episode_rewards[i] += f
                stats.episode_lengths[i] = t
                stats.visitation_count[s_next, i] += 1

                D = [(s, action, f, s_next, done)]
                Q = self.QValueUpdate(Q, D)
                s = s_next
                t += 1
            epsilon = max(epsilon - self.epsilon_anneal, 0.1) if i < self.num_episodes*0.8 else 0

        return stats
        #plotting.plot_episode_stats(stats, smoothing_window=1000)

if __name__ == "__main__":
    agent = QLearningAgent(env=hMDP())
    stats = agent.learn()
    plotting.plot_rewards([stats], smoothing_window=1000)





