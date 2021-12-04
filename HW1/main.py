from arguments import get_args
from Dagger import DaggerAgent, ExampleAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
import cv2

# draw performance picture
def plot(record):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    ax1 = ax.twinx()
    ax1.plot(record['steps'], record['query'],
             color='red', label='query')
    ax1.set_ylabel('queries')
    reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
    query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
    patch_set = [reward_patch, query_patch]
    ax.legend(handles=patch_set)
    fig.savefig('performance.png')


# read images in a directory
def read_directory(directory_name):
    array_of_img = []
    for filename in os.listdir(r"./" + directory_name):
        # img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
    return array_of_img


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
    def __init__(self, env_name, num_stacks):
        self.env = gym.make(env_name)
        # num_stacks: the agent acts every num_stacks frames
        # it could be any positive integer
        self.num_stacks = num_stacks
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        reward_sum = 0
        for stack in range(self.num_stacks):
            obs_next, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done:
                self.env.reset()
                return obs_next, reward_sum, done, info
        return obs_next, reward_sum, done, info

    def reset(self):
        return self.env.reset()


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0],
              'query': [0]}
    # query_cnt counts queries to the expert
    query_cnt = 0

    # environment initial
    envs = Env(args.env_name, args.num_stacks)
    # action_shape is the size of the discrete action set, here is 18
    # Most of the 18 actions are useless, find important actions
    # in the tips of the homework introduction document
    action_shape = envs.action_space.n
    # observation_shape is the shape of the observation
    # here is (210,160,3)=(height, weight, channels)
    observation_shape = envs.observation_space.shape
    print(action_shape, observation_shape)

    # agent initial
    # you should finish your agent with DaggerAgent
    # e.g. agent = MyDaggerAgent()
    agent = ExampleAgent()

    # You can play this game yourself for fun
    if args.play_game:
        obs = envs.reset()
        j = 0
        while True:
            im = Image.fromarray(obs)
            im.save('imgs/sampling/' + str(j) + '.png')
            j += 1
            action = int(input('input action'))
            while action < 0 or action >= action_shape:
                action = int(input('re-input action'))
            obs_next, reward, done, _ = envs.step(action)
            obs = obs_next
            if done:
                obs = envs.reset()

    data_set = {'data': [], 'label': []}
    # start train your agent
    for i in range(num_updates):
        # an example of interacting with the environment
        # we init the environment and receive the initial observation
        obs = envs.reset()

        # Initialize policy
        if i == 0:
            images = read_directory('imgs/sampling')
            for imArray in images:
                img_image = Image.fromarray(np.uint8(imArray))
                im = img_image.convert('L')
                ii = np.asarray(im)
                iii = ii.flatten()
                data_set['data'].append(iii)
            # read the label
            with open('labels/label.txt', 'r') as f:
                for label_tmp in f.readlines():
                    data_set['label'].append(label_tmp.strip("\n"))
            agent.update(data_set['data'], data_set['label'])
            continue

        # todo 读取前面失败的次数，模型完成后删除
        # if i < 3:
        #     # read the label
        #     label_path = 'labels/label' + str(i) + '.txt'
        #     with open(label_path, 'r') as f:
        #         for label_tmp in f.readlines():
        #             data_set['label'].append(label_tmp.strip("\n"))
        #     if i == 2:
        #         images = read_directory('imgs/dataset')
        #         for imArray in images:
        #             img_image = Image.fromarray(np.uint8(imArray))
        #             im = img_image.convert('L')
        #             ii = np.asarray(im)
        #             iii = ii.flatten()
        #             data_set['data'].append(iii)
        #         agent.update(data_set['data'], data_set['label'])
        #     continue

        predict_actions = []
        # we get a trajectory with the length of args.num_steps
        for step in range(args.num_steps):
            # Sample actions, diminish gradually
            epsilon = 0.5 ** (step + 1)
            if np.random.rand() < epsilon:
                # we choose a random action
                action = envs.action_space.sample()
            else:
                # we choose a special action according to our model
                im = Image.fromarray(obs)
                grey_pic = im.convert('L')
                grey_array = np.asarray(grey_pic)
                grey_array_data = grey_array.flatten()
                action = agent.select_action(grey_array_data)

            predict_actions.append(action)
            # interact with the environment
            # we input the action to the environments and it returns some information
            # obs_next: the next observation after we do the action
            # reward: (float) the reward achieved by the action
            # down: (boolean)  whether it’s time to reset the environment again.
            #           done being True indicates the episode has terminated.
            obs_next, reward, done, _ = envs.step(action)
            # we view the new observation as current observation
            obs = obs_next
            # if the episode has terminated, we need to reset the environment.
            if done:
                envs.reset()

            # saving observations
            if args.save_img:
                # img_path = 'imgs/' + 'dataset' + str(i) + '/'
                im = Image.fromarray(obs)
                im.save('imgs/dataset/' + str(i) + '-' + str(step + 1) + '.png')
                grey_pic = im.convert('L')
                grey_array = np.asarray(grey_pic)
                grey_array_data = grey_array.flatten()
                data_set['data'].append(grey_array_data)

            # query numbers in each iteration
            if i < 5 and step >= 29:
                query_cnt += 30
                break
            if i < 8 and step >= 49:
                query_cnt += 50
                break
            if i < 10 and step >= 79:
                query_cnt += 80
                break
            if i < 12 and step >= 99:
                query_cnt += 100
                break
            if i < 15 and step >= 119:
                query_cnt += 120
                break
            if i < 120 and step >= 149:
                query_cnt += 150
                break

        # write the predict action result into action.txt
        action_path = 'actions/action' + str(i) + '.txt'
        with open(action_path, 'w') as f:
            for jj in range(len(predict_actions)):
                f.write(str(predict_actions[jj]) + "\r")

        # You need to label the images in 'imgs/' by recording the right actions in label.txt

        # After you have labeled all the images, you can load the labels
        # for training a model
        label_path = 'labels/label' + str(i) + '.txt'
        with open(label_path, 'r') as f:
            for label_tmp in f.readlines():
                data_set['label'].append(label_tmp.strip("\n"))

        # design how to train your model with labeled data
        agent.update(data_set['data'], data_set['label'])

        if (i + 1) % args.log_interval == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            reward_episode_set = []
            reward_episode = 0
            # evaluate your model by testing in the environment
            for step in range(args.test_steps):
                im = Image.fromarray(obs)
                grey_pic = im.convert('L')
                grey_array = np.asarray(grey_pic)
                grey_array_data = grey_array.flatten()
                action = agent.select_action(grey_array_data)
                # you can render to get visual results
                # envs.render()
                obs_next, reward, done, _ = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0
                    envs.reset()
                else:
                    reward_episode_set.append(reward_episode)

            end = time.time()
            print(
                "TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
                    .format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    i, total_num_steps,
                    int(total_num_steps / (end - start)),
                    query_cnt,
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)
                ))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            record['query'].append(query_cnt)
            plot(record)


if __name__ == "__main__":
    main()
