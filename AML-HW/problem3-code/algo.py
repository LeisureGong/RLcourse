import random
from collections import defaultdict
import numpy as np

from abc import abstractmethod


class _SARSAAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.5
        self.discount_factor = 1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def select_action(self, state):
        if np.random.random() < 0.1:
            action = np.random.randint(0, 4)
        else:
            state_action = self.q_table[state]
            action = self.select_max_reward(state_action)
        return action

    def learn(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state][action]
        target = reward + self.discount_factor*self.q_table[next_state][next_action]
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate*(target - predict)

    @staticmethod
    def select_max_reward(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

class _QAgent:
    def __init__(self, actions):
        self.actions = actions
        # 学习率
        self.learning_rate = 0.5
        # 奖励递减值
        self.discount_factor = 1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # 采样，更新QTable
    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table[state][action]
        q_new = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_new - q_predict)

    def select_action(self, state):
        state_action = self.q_table[state]
        action = self.select_max_reward(state_action)
        return action

    # 选择所有状态中，回报最大的action
    @staticmethod
    def select_max_reward(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
