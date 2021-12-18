import torch
from torch import nn
from common.layers import NoisyLinear

class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)

class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions, config):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions
        self.config = config
        use_cuda = True if self.config.device != 'cpu' else False

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )
        if self.config.choice == 'dueling_double_noise':
            self.advantage = nn.Sequential(
                NoisyLinear(self.features_size(), 512, use_cuda=use_cuda),
                nn.LeakyReLU(),
                NoisyLinear(512, self.num_actions, use_cuda=use_cuda)
            )
            self.value = nn.Sequential(
                NoisyLinear(self.features_size(), 512, use_cuda=use_cuda),
                nn.LeakyReLU(),
                NoisyLinear(512, 1, use_cuda=use_cuda)
            )
        elif 'dueling' in self.config.choice:
            self.advantage = nn.Sequential(
                nn.Linear(self.features_size(), 512),
                nn.LeakyReLU(),
                nn.Linear(512, self.num_actions)
            )
            self.value = nn.Sequential(
                nn.Linear(self.features_size(), 512),
                nn.LeakyReLU(),
                nn.Linear(512, 1)
            )
        elif self.config.choice == 'none' or 'double' in self.config.choice:
            self.fc = nn.Sequential(
                nn.Linear(self.features_size(), 512),
                nn.LeakyReLU(),
                nn.Linear(512, self.num_actions)
            )
        elif self.config.choice == 'noise':
            self.fc = nn.Sequential(
                NoisyLinear(self.features_size(), 512, use_cuda=use_cuda),
                nn.LeakyReLU(),
                NoisyLinear(512, self.num_actions, use_cuda=use_cuda)
            )




    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.reshape(batch_size, -1)
        if 'dueling' in self.config.choice:
            x = self.value(x) + self.advantage(x) - self.advantage(x).mean()
        else:
            x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

        # You need to design how to calculate the loss

    s0_action_values = q_values.gather(1, a)
    with torch.no_grad():
        if 'double' in self.config.choice:
            s1_values = self.target_model(s1).gather(1, torch.max(q_next_values, 1)[1].unsqueeze(-1))
        else:
            s1_values = self.target_model(s1).max(1)[0].unsqueeze(-1)
        expected_s0_action_values = s1_values * self.config.gamma * (1 - done) + r

    loss = torch.nn.SmoothL1Loss()(s0_action_values, expected_s0_action_values)