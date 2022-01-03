import numpy as np
from collections import defaultdict
import random

from abc import abstractmethod
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class MyQAgent(QAgent):
    def __init__(self):
        super(QAgent, self).__init__()
        # init your model
        self.discount_factor = 0.99
        self.learning_rate = 0.1
        self.qtable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    
    def select_action(self, state_pos):
        state = " ".join([str(int(x)) for x in state_pos])
        candidate_actions = self.qtable[state]
        Max = candidate_actions[0]
        max_index = []
        for index, value in enumerate(candidate_actions):
            if value > Max:
                Max = value
                max_index.clear()
                max_index.append(index)
            elif value == Max:
                max_index.append(index)
        return random.choice(max_index)
        
    def learn(self, state_pos, action, reward, next_state_pos, done):
        state = " ".join([str(int(x)) for x in state_pos])
        next_state = " ".join([str(int(x)) for x in next_state_pos])
        new_value = reward + (1 - done) * self.discount_factor * max(self.qtable[next_state])
        self.qtable[state][action] += self.learning_rate * (new_value - self.qtable[state][action])
        self.qtable[state][action] = np.clip(self.qtable[state][action], -100, 100)
'''
    def get_policy(self):
		grid = np.zeros((8, 8), dtype=int)
		for index, state in enumerate(self.qtable):
			state_pos = [int(x) for x in state.split(" ")]
			optimal_action = self.select_action(state_pos)
			grid[state_pos[0]][state_pos[1]] = optimal_action+1
		print(grid)
'''


class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

class DynaModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.transitions = defaultdict(lambda: ["", "", "", ""])
        pass

    def store_transition(self, s, a, r, s_):
        state = " ".join([str(int(x)) for x in s])
        next_state = " ".join([str(int(x)) for x in s_])
        self.transitions[state][a] = next_state

    def sample_state(self):
        states = list(self.transitions.keys())
        idx = random.randint(0,len(states)-1)
        state = [int(x) for x in states[idx].split(" ")]
        return np.array(state), idx

    def sample_action(self, s):
        state = " ".join([str(int(x)) for x in s])
        actions = self.transitions[state]
        candidate_actions = []
        for idx in range(len(actions)):
            if actions[idx] != "":
                candidate_actions.append(idx)
        return random.choice(candidate_actions)

    def predict(self, s, a):
        state = " ".join([str(int(x)) for x in s])
        next_state = self.transitions[state][a]
        return np.array([int(x) for x in next_state.split(" ")])

    def train_transition(self):
        pass


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, units=256, activation=tf.nn.relu)
        self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))
        self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        '''
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)
        '''

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)
        '''
        if len(self.sensitive_index) > 0:
            for _ in range(batch_size):
                idx = np.random.randint(0, len(self.sensitive_index))
                idx = self.sensitive_index[idx]
                s, a, r, s_ = self.buffer[idx]
                s_list.append(s)
                a_list.append([a])
                r_list.append(r)
                s_next_list.append(s_)
        '''
        x_mse = self.sess.run([self.x_mse,  self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])
