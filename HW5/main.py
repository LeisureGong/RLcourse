from arguments import get_args
from algo import *
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import scipy.misc

t = str(time.time())

def plot(record, info, h, s, m, n):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    #ax.fill_between(record['steps'], record['min'], record['max'],
    #                color='blue', alpha=0.2)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('Average score per episode')
    import os
    fig.savefig('performance-h{}-s{}-m{}-n{}.png'.format(h, s, m, n))
    plt.close()

'''
def plot_converge(n_num, sample_num, time_num):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(n_num, sample_num, color='blue')
    ax.set_xlabel('n')
    ax.set_ylabel('Count of consumed samples')
    # ax.set_ylim((0,20000))
    ax1 = ax.twinx()
    ax1.plot(n_num, time_num, color='red', label='Consumed time')
    ax1.set_ylabel('Consumed time')
    fig.savefig('lab2/convergence-h20-s5-m500.png')
    plt.close()
'''
    
def main():
    # load hyper parameters
    args = get_args()
    m = args.m
    n = args.n
    start_planning = args.s
    h = args.h
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0]}

    # environment initial
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape
    print(action_shape, observation_shape)

    # agent initial
    # you should finish your agent with QAgent
    # e.g. agent = myQAgent()
    agent = MyQAgent()
    # switch between these two kinds
    dyna = False
    if dyna:
        dynamics_model = DynaModel(8, 8, policy=agent)
    else :
        dynamics_model = NetworkModel(8, 8, policy=agent)
    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99

    flag = False
    sample_step = 0
    sample_num = 0
    total_time = 0
    # start to train your agent(num_updates: 100000/2000=50)
    for i in range(num_updates * 10):
        # an example of interacting with the environment
        obs = envs.reset()
        obs = obs.astype(int)
        done = False
        # for step in range(args.num_steps):
        while not done:
        # for step in range(args.num_steps): # num_steps: 2000
            # Sample actions with epsilon greedy policy
            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)
            # interact with the environment
            obs_next, reward, done, info = envs.step(action)
            sample_step += 1
            obs_next = obs_next.astype(int)
            # add your Q-learning algorithm
            agent.learn(obs, action, reward, obs_next, done)
            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next
            # if done:
            #    obs = envs.reset()
        
        # table-based Model
        if dyna:
            print("\n Not Expected!\n")
            for _ in range(n):
                # sample state and action from model
                s, idx = dynamics_model.sample_state()
                a = dynamics_model.sample_action(s)
                s_ = dynamics_model.predict(s, a)
                r = envs.R(s, a, s_)
                done = envs.D(s, a, s_)
                # add your Q-learning algorithm
                agent.learn(s, a, r, s_, done)
                s = s_
                if done:
                    break
        # neural network based Model
        else:
            # print("\n Not Expected!\n")
            for _ in range(m):
                dynamics_model.train_transition(32)
            if i > start_planning:
                for _ in range(n):
                    s, idx = dynamics_model.sample_state()
                    for _ in range(h): 
                        # use epsilon-greedy selection
                        if np.random.rand() < epsilon:
                            a = envs.action_sample()
                        else:
                            a = agent.select_action(s)
                        s_ = dynamics_model.predict(s, a)
                        r = envs.R(s, a, s_)
                        done = envs.D(s, a, s_)
                        # add your Q-learning algorithm
                        agent.learn(s, a, r, s_, done)
                        s = s_
                        if done:
                            break

        if (i + 1) % (args.log_interval) == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.
                    obs = envs.reset()

            end = time.time()
            print("TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    i, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)))
            record['steps'].append(sample_step)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            plot(record, info, h, start_planning, m, n)
            # Check whether it converges: the condition is min is above 80 for continus 10 times.
            if len(record['min']) > 10:
                last_ten_min = record['min'][-10:]
            else:
                last_ten_min = record['min']
            conver_flag = True
            for min_reward in last_ten_min:
                if min_reward <= 80:
                    conver_flag = False
            if conver_flag and not flag:
                sample_num = sample_step
                total_time = end - start
                flag = True
    print("After " + str(sample_num) + " samples, it converges!")


if __name__ == "__main__":
    main()
