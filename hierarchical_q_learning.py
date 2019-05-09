import numpy as np
import random
import gym
import matplotlib

from collections import defaultdict
from envs.hmdp import StochastichMDPEnv as hMDP
import utils.plotting as plotting



def epsGreedy(state, B, eps, Q):
    action_probabilities = np.ones_like(Q[state]) * eps / len(Q[state])
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

def intrinsic_reward(state, action, state_next, goal):
    return 1.0 if state_next == goal else 0.0

num_episodes = 20000
stats = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes)) 

#env = WindyGridworldEnv()
env = hMDP()


A = env.action_space
meta_goals = [0, 1, 2, 3, 4, 5]

D1 = None
D2 = None
Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
Q2 = defaultdict(lambda: np.zeros(len(meta_goals)))


epsilon = {}
for goal in meta_goals:
    epsilon[goal] = 1.0
epsilon_meta = 1.0

for i in range(num_episodes):
    if i % 1000 == 0:
            print('Episode ', i)
            print(epsilon)
            print(epsilon_meta)
            print(Q1)
            print(Q2)
    s = env.reset()
    done = False
    goal = epsGreedy(s, meta_goals, epsilon_meta, Q2)
    epsilon[goal] = max(epsilon[goal] - 1 / 5000, 0.1)
    t = 0
    while not done:
        F = 0
        s0 = s
        r = 0
        while not (done or r > 0):
            action = epsGreedy((s, goal), A, epsilon[goal], Q1)
            s_next, f, done, _ = env.step(action)
            r = intrinsic_reward(s, action, s_next, goal)
            stats.episode_rewards[i] += f
            stats.episode_lengths[i] = t

            D1 = [((s, goal), action, r, (s_next, goal), done)]
            Q1 = QValueUpdate(Q1, D1)
            F = F + f
            s = s_next
            t += 1
        D2 = [(s0, goal, F, s, done)]
        Q2 = QValueUpdate(Q2, D2)
        if not done:
            goal = epsGreedy(s, meta_goals, epsilon_meta, Q2)
            epsilon[goal] = max(epsilon[goal] - 1 / 1000, 0.1)

    epsilon_meta = max(epsilon_meta - 1 / 12000, 0.1)


        
plotting.plot_episode_stats(stats, smoothing_window=1000)




