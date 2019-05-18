import numpy as np
import random
import gym
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras

from collections import defaultdict, deque
from envs.mdp import StochasticMDPEnv as MDP
from envs.cmdp import ContinuousStochasticMDPEnv as cMDP
from envs.chmdp import ContinuousStochastichMDPEnv as chMDP
import utils.plotting as plotting

class DQNAgent():

    def __init__(self, env = cMDP(),  num_episodes = 20000, \
                    gamma = 0.9, batch_size = 1, \
                    epsilon_anneal = 1/2000):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_anneal = epsilon_anneal


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

    def learn(self):

        stats = plotting.Stats(num_episodes=self.num_episodes, continuous=True)

        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu'))#, kernel_initializer = 'zeros'))
        model.add(Dense(24, activation='relu'))#, kernel_initializer = 'zeros'))
        model.add(Dense(self.env.action_space.n, activation='linear'))#, kernel_initializer = 'zeros'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        D = deque(maxlen=2000)
        Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        A = self.env.action_space

        epsilon = 1.0

        for i in range(self.num_episodes):
            if i % 100 == 0:
                print('Episode ', i)
            s = self.env.reset()
            done = False
            t = 0
            while not done:# and t < 500:
                action = self.epsGreedy(s, A, epsilon, model)
                #print(action)
                s_next, f, done, _ = self.env.step(action)
                stats.episode_rewards[i] += f
                stats.episode_lengths[i] = t
                #print(s_next)
                #stats.visitation_count[s_next, i] += 1

                D.append((s, action, f, s_next, done))
                #print(s, epsilon)
                s = s_next
                t += 1
            self.QValueUpdate(model, D)
            epsilon = max(epsilon - self.epsilon_anneal, 0.1)
            print('time', t, ', epsilon ', epsilon)

                
        plt.figure()
        #plotting.plot_q_values(model, self.env.observation_space, self.env.action_space)

        return stats
        #plotting.plot_episode_stats(stats, smoothing_window=100)

if __name__ == "__main__":
    env = chMDP()
    agent = DQNAgent(env=env, num_episodes = 6000)
    stats = agent.learn()

    plt.figure()
    plotting.plot_rewards([stats], smoothing_window=1000)
    plt.show()