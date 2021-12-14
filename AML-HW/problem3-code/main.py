from algo import _QAgent
from cliff_walking import Env
import numpy as np


def main():
    num_frames = 4000
    num_steps = 200
    num_updates = int(num_frames // num_steps)

    envs = Env(row=4, col=12)
    action_shape = 4
    epsilon = 0.1
    agent = _QAgent(actions=list(range(action_shape)))
    # 保存走过的路径
    list_path = []
    for i in range(num_updates):
        # 每个episode，重置环境
        obs = envs.reset()

        for step in range(num_steps):
            # sample actions with epsilon greedy policy
            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)

            obs_next, reward, done = envs.transition(action)

            list_path.append(obs)
            agent.learn(obs, action, reward, obs_next)

            obs = obs_next
            if done:
                list_path.append(obs)
                print(list_path)
                obs = envs.reset()
                list_path.clear()
        print("第" + str(i+1) + "轮训练over")


if __name__ == "__main__":
    main()
