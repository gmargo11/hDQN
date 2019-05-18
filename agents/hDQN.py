import numpy as np
import random
import gym
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras

from collections import defaultdict, deque
from envs.hmdp import StochastichMDPEnv as hMDP
from envs.chmdp import ContinuousStochastichMDPEnv as chMDP
from envs.chmdp2 import ContinuousStochastichMDP2Env as chMDP2
import utils.plotting as plotting


class hDQNAgent():

    def __init__(self, env = hMDP(), meta_goals = [0, 1, 2, 3, 4, 5], num_episodes = 20000, \
                    gamma = 0.9, batch_size = 32, epsilon_anneal = 1/2000, \
                    meta_epsilon_anneal = 1/12000):
        self.env = env
        self.meta_goals = meta_goals
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_anneal = epsilon_anneal
        self.meta_epsilon_anneal = meta_epsilon_anneal


    def epsGreedy(self, state, B, eps, model):
        Q = model.predict(np.array([state]).reshape(1, -1))[0]
        #print(Q)
        action_probabilities = np.ones_like(Q) * eps / len(Q)
        best_action = np.argmax(Q)
        action_probabilities[best_action] += (1.0 - eps)
        action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
        return action

    def QValueUpdate(self, model, D):
        batch_size = min(self.batch_size, len(D))
        gamma = 0.95
        mini_batch = random.sample(D, batch_size)
        for s, action, f, s_next, done in mini_batch:
            Q_next = model.predict(np.array([s_next]).reshape(1, -1))[0]
            target = f
            if not done:
                best_next_action = np.argmax(Q_next)
                target = f + self.gamma * Q_next[best_next_action]
            target_arr = model.predict(np.array([s]).reshape(1, -1))
            target_arr[0][action] = target
            #print('t', target_arr)
            #print(target_arr[0])
            model.fit(np.array([s]).reshape(1, -1), target_arr, epochs=1, verbose=0)


    def intrinsic_reward(self, state, action, state_next, goal):
        thresh = 0.5
        return 1.0 if np.linalg.norm(np.array(state_next) - np.array(list(goal))) < thresh else 0.0


    def learn(self):

        stats = plotting.Stats(num_episodes=self.num_episodes, continuous=True)

        A = self.env.action_space

        model = Sequential()
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0]+1, activation='relu'))#, kernel_initializer = 'zeros'))
        model.add(Dense(24, activation='relu'))#, kernel_initializer = 'zeros'))
        model.add(Dense(self.env.action_space.n, activation='linear'))#, kernel_initializer = 'zeros'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))

        h_model = Sequential()
        h_model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu'))#, kernel_initializer = 'zeros'))
        h_model.add(Dense(24, activation='relu'))#, kernel_initializer = 'zeros'))
        h_model.add(Dense(len(self.meta_goals), activation='linear'))#, kernel_initializer = 'zeros'))
        h_model.compile(loss='mse', optimizer=Adam(lr=0.0001))


        D1 = None
        D2 = None


        epsilon = {}
        for goal in self.meta_goals:
            epsilon[goal] = 1.0
        epsilon_meta = 1.0

        for i in range(self.num_episodes):
            if i % 1000 == 0:
                    print('Episode ', i)
                    print(epsilon)
                    print(epsilon_meta)
            s = self.env.reset()
            done = False
            goal = self.epsGreedy(s, self.meta_goals, epsilon_meta, h_model)
            epsilon[self.meta_goals[goal]] = max(epsilon[self.meta_goals[goal]] - self.epsilon_anneal, 0.1)
            t = 0
            while not done:
                F = 0
                s0 = s
                r = 0
                while not (done or r > 0):
                    action = self.epsGreedy(s + [goal], range(A.n), epsilon[self.meta_goals[goal]], model)
                    s_next, f, done, _ = self.env.step(action)
                    r = self.intrinsic_reward(s, action, s_next, self.meta_goals[goal])
                    stats.episode_rewards[i] += f
                    stats.episode_lengths[i] = t
                    #stats.visitation_count[s_next, i] += 1

                    D1 = [(s + [goal], action, r, s_next + [goal], done)]
                    print(i, s, action, goal)
                    Q1 = self.QValueUpdate(model, D1)
                    F = F + f
                    s = s_next
                    t += 1
                D2 = [(s0, goal, F, s, done)]
                Q2 = self.QValueUpdate(h_model, D2)
                if not done:
                    goal = self.epsGreedy(s, self.meta_goals, epsilon_meta, h_model)
                    epsilon[self.meta_goals[goal]] = max(epsilon[self.meta_goals[goal]] - self.epsilon_anneal, 0.1)

            epsilon_meta = max(epsilon_meta - self.meta_epsilon_anneal, 0.1)

        return stats
        #plotting.plot_episode_stats(stats, smoothing_window=1000)

if __name__ == "__main__":
    agent = hDQNAgent(env=chMDP2(), num_episodes=12000, meta_goals = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    stats = agent.learn()
    plt.figure()
    plotting.plot_rewards([stats], smoothing_window=1000)
    plt.show()