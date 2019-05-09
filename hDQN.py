import numpy as np
import random
import gym
import matplotlib

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras

from collections import defaultdict, deque
from envs.hmdp import StochastichMDPEnv as hMDP
import utils.plotting as plotting


def epsGreedy(state, B, eps, model):
    Q = model.predict(np.array([state]).reshape(1, -1))[0]
    #print(Q)
    action_probabilities = np.ones_like(Q) * eps / len(Q)
    best_action = np.argmax(Q)
    action_probabilities[best_action] += (1.0 - eps)
    action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
    return action

def QValueUpdate(model, D):
    batch_size = min(32, len(D))
    gamma = 0.95
    mini_batch = random.sample(D, batch_size)
    for s, action, f, s_next, done in mini_batch:
        Q_next = model.predict(np.array([s_next]).reshape(1, -1))[0]
        target = f
        if not done:
            best_next_action = np.argmax(Q_next)
            target = f + gamma * Q_next[best_next_action]
        target_arr = model.predict(np.array([s]).reshape(1, -1))
        target_arr[0][action] = target
        #print('t', target_arr)
        #print(target_arr[0])
        model.fit(np.array([s]).reshape(1, -1), target_arr, epochs=1, verbose=0)


def intrinsic_reward(state, action, state_next, goal):
    return 1.0 if state_next == goal else 0.0

'''
def QValueUpdateMeta(L, D):
    batch_size = 100
    gamma = 0.9
    mini_batch = random.sample(D, batch_size)
    for s0, g, F, s_next, done in mini_batch:
        target = F
        if not done:
            target = reward + gamma * np.amax(L.predict(s_next)[0])
        target_arr = L.predict(s0)
        target_arr[0][g] = target
        L.fit(s0, target_arr, epochs=1)
'''



num_episodes = 20000
stats = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes)) 

#env = WindyGridworldEnv()
#env = gym.make('CartPole-v0')
env = hMDP()
A = env.action_space
meta_goals = [0, 1, 2, 3, 4, 5]

model = Sequential()
model.add(Dense(24, input_dim=2, activation='relu'))#, kernel_initializer = 'zeros'))
model.add(Dense(24, activation='relu'))#, kernel_initializer = 'zeros'))
model.add(Dense(env.action_space.n, activation='linear'))#, kernel_initializer = 'zeros'))
model.compile(loss='mse', optimizer=Adam(lr=0.0001))

h_model = Sequential()
h_model.add(Dense(24, input_dim=1, activation='relu'))#, kernel_initializer = 'zeros'))
h_model.add(Dense(24, activation='relu'))#, kernel_initializer = 'zeros'))
h_model.add(Dense(len(meta_goals), activation='linear'))#, kernel_initializer = 'zeros'))
h_model.compile(loss='mse', optimizer=Adam(lr=0.0001))


D1 = None
D2 = None


epsilon = {}
for goal in meta_goals:
    epsilon[goal] = 1.0
epsilon_meta = 1.0

for i in range(num_episodes):
    if i % 1000 == 0:
            print('Episode ', i)
            print(epsilon)
            print(epsilon_meta)
    s = env.reset()
    done = False
    goal = epsGreedy(s, meta_goals, epsilon_meta, h_model)
    epsilon[goal] = max(epsilon[goal] - 1 / 5000, 0.1)
    t = 0
    while not done:
        F = 0
        s0 = s
        r = 0
        while not (done or r > 0):
            action = epsGreedy((s, goal), A, epsilon[goal], model)
            s_next, f, done, _ = env.step(action)
            r = intrinsic_reward(s, action, s_next, goal)
            stats.episode_rewards[i] += f
            stats.episode_lengths[i] = t

            D1 = [((s, goal), action, r, (s_next, goal), done)]
            Q1 = QValueUpdate(model, D1)
            F = F + f
            s = s_next
            t += 1
        D2 = [(s0, goal, F, s, done)]
        Q2 = QValueUpdate(h_model, D2)
        if not done:
            goal = epsGreedy(s, meta_goals, epsilon_meta, h_model)
            epsilon[goal] = max(epsilon[goal] - 1 / 1000, 0.1)

    epsilon_meta = max(epsilon_meta - 1 / 12000, 0.1)

plotting.plot_episode_stats(stats, smoothing_window=1000)