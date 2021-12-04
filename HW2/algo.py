import random
import pandas as pd
import numpy as np
from collections import defaultdict

from abc import abstractmethod


class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class _QAgent:
    def __init__(self, actions):
        self.actions = actions
        # 学习率
        self.learning_rate = 0.4
        # 奖励递减值
        self.discount_factor = 0.9
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
