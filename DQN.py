import numpy as np
import random
import gym
import matplotlib

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras

from collections import defaultdict, deque
from envs.mdp import StochasticMDPEnv as MDP
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


num_episodes = 500
stats = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes)) 

#env = WindyGridworldEnv()
#env = gym.make('CartPole-v0')
env = MDP()

model = Sequential()
model.add(Dense(24, input_dim=1, activation='relu'))#, kernel_initializer = 'zeros'))
model.add(Dense(24, activation='relu'))#, kernel_initializer = 'zeros'))
model.add(Dense(env.action_space.n, activation='linear'))#, kernel_initializer = 'zeros'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))

D = deque(maxlen=2000)
Q = defaultdict(lambda: np.zeros(env.action_space.n))
A = env.action_space

epsilon = 1.0

for i in range(num_episodes):
    #if i % 100 == 0:
    print('Episode ', i)
    s = env.reset()
    done = False
    t = 0
    while not done:# and t < 500:
        action = epsGreedy(s, A, epsilon, model)
        #print(action)
        s_next, f, done, _ = env.step(action)
        stats.episode_rewards[i] += f
        stats.episode_lengths[i] = t

        D.append((s, action, f, s_next, done))
        #print(s, epsilon)
        s = s_next
        t += 1
    QValueUpdate(model, D)
    epsilon = max(epsilon - 1 / 500, 0.1)
    print('time', t, ', epsilon ', epsilon)

        
plotting.plot_episode_stats(stats, smoothing_window=100)