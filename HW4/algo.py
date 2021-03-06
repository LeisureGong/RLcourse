import numpy as np
import random
from abc import abstractmethod
import tensorflow as tf
from collections import defaultdict
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


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
        state = str(state.tolist())
        next_state = str(next_state.tolist())
        q_predict = self.q_table[state][action]
        q_new = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_new - q_predict)

    def select_action(self, state):
        state = str(state.tolist())
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

    @abstractmethod
    def predict(self, s, a):
        pass


class DynaModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.transitions = defaultdict(lambda: ["", "", "", ""])

    def store_transition(self, s, a, r, s_):
        state = " ".join([str(int(x)) for x in s])
        next_state = " ".join([str(int(x)) for x in s_])
        self.transitions[state][a] = next_state

    def sample_state(self):
        states = list(self.transitions.keys())
        idx = random.randint(0, len(states) - 1)
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
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

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

        x_mse = self.sess.run([self.x_mse, self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])
