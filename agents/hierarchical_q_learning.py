import numpy as np
import random
import gym
import matplotlib

from collections import defaultdict
from envs.hmdp import StochastichMDPEnv as hMDP
import utils.plotting as plotting


class hierarchicalQLearningAgent():
    def __init__(self, env = hMDP(), meta_goals = [0, 1, 2, 3, 4, 5], num_episodes = 25000, \
                    gamma = 0.9, alpha = 0.6, batch_size = 1, epsilon_anneal = 1/2000, \
                    meta_epsilon_anneal = 1/12000):
        self.env = env
        self.meta_goals = meta_goals
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.epsilon_anneal = epsilon_anneal
        self.meta_epsilon_anneal = meta_epsilon_anneal


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

    def intrinsic_reward(self, state, action, state_next, goal):
        return 1.0 if state_next == goal else 0.0


    def learn(self):

        stats = plotting.Stats(num_episodes=self.num_episodes, num_states=self.env.observation_space.n)


        A = self.env.action_space
        meta_goals = [0, 1, 2, 3, 4, 5]

        D1 = None
        D2 = None
        Q1 = defaultdict(lambda: np.zeros(self.env.action_space.n))
        Q2 = defaultdict(lambda: np.zeros(len(self.meta_goals)))


        epsilon = {}
        for goal in meta_goals:
            epsilon[goal] = 1.0
        epsilon_meta = 1.0

        for i in range(self.num_episodes):
            if i % 1000 == 0:
                    print('Episode ', i)
                    print(epsilon)
                    print(epsilon_meta)
            s = self.env.reset()
            done = False
            goal = self.epsGreedy(s, self.meta_goals, epsilon_meta, Q2)
            #stats.target_count[goal, i] += 1
            epsilon[goal] = max(epsilon[goal] - self.epsilon_anneal, 0.1) if i < self.num_episodes*0.8 else 0
            t = 0
            while not done:
                F = 0
                s0 = s
                r = 0
                while not (done or r > 0):
                    action = self.epsGreedy((s, goal), A, epsilon[goal], Q1)
                    s_next, f, done, _ = self.env.step(action)
                    r = self.intrinsic_reward(s, action, s_next, goal)
                    stats.episode_rewards[i] += f
                    stats.episode_lengths[i] = t
                    stats.visitation_count[s_next, i] += 1

                    D1 = [((s, goal), action, r, (s_next, goal), done)]
                    Q1 = self.QValueUpdate(Q1, D1)
                    F = F + f
                    s = s_next
                    t += 1
                D2 = [(s0, goal, F, s, done)]
                Q2 = self.QValueUpdate(Q2, D2)
                if not done:
                    goal = self.epsGreedy(s, self.meta_goals, epsilon_meta, Q2)
                    stats.target_count[goal, i] += 1
                    epsilon[goal] = max(epsilon[goal] - self.epsilon_anneal, 0.1) if i < self.num_episodes*0.8 else 0

            epsilon_meta = max(epsilon_meta - self.meta_epsilon_anneal, 0.1) if i < self.num_episodes*0.8 else 0


        return stats
        #plotting.plot_episode_stats(stats, smoothing_window=1000)



if __name__ == "__main__":
    agent = hierarchicalQLearningAgent(env=hMDP())
    stats = agent.learn()
    plotting.plot_rewards([stats], smoothing_window=1000)