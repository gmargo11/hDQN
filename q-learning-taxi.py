import numpy as np
import random
import gym
import matplotlib

from collections import defaultdict
from envs.hmdp import StochastichMDPEnv as hMDP
import utils.plotting as plotting



def epsGreedy(state, B, eps, Q):
    action_probabilities = np.ones_like(Q[state]) * epsilon / len(Q[state])
    best_action = np.argmax(Q[state])
    action_probabilities[best_action] += (1.0 - eps)
    action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
    return action

def QValueUpdate(Q, D):
    batch_size = 1
    gamma = 0.9
    alpha = 0.6
    mini_batch = random.sample(D, batch_size)

    for s, action, f, s_next, done in mini_batch:
        target = f
        if not done:
            best_next_action = np.argmax(Q[s_next])
            target = f + gamma * Q[s_next][best_next_action]
        delta = target - Q[s][action]
        Q[s][action] += alpha * delta

    return Q

num_episodes = 10000
stats = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes)) 

#env = WindyGridworldEnv()
#env = hMDP()
env = gym.make('Taxi-v2')

D = None
Q = defaultdict(lambda: np.zeros(env.action_space.n))
A = env.action_space

epsilon = 1.0

for i in range(num_episodes):
    if i % 1000 == 0:
            print('Episode ', i)
            print(epsilon)
    s = env.reset()
    done = False
    t = 0
    while not done:
        action = epsGreedy(s, A, epsilon, Q)
        s_next, f, done, _ = env.step(action)
        stats.episode_rewards[i] += f
        stats.episode_lengths[i] = t

        D = [(s, action, f, s_next, done)]
        Q = QValueUpdate(Q, D)
        s = s_next
        t += 1
    epsilon = max(epsilon - 1 / 7000, 0.1)

        
plotting.plot_episode_stats(stats, smoothing_window=1000)




