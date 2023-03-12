import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import random

from common.layers import NoisyLinear

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class RainbowCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        """
            定义rainbow组成的网络层
            features: 卷积层，用于提取state的feature特征
            noisy_value：value部分加noisy的网络层
            noisy_advantage：advantage部分加noisy的网络层
        """
        super(RainbowCnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        # Distributional Hyperparameters
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        # 定义网络
        # State：需要经过卷积层，提取feature特征
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Noisy层NoisyLinear
        # DuelingDQN = value+advantage
        self.noisy_value1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)
        self.noisy_advantage1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        """
            前向传播
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        batch_size = x.size(0)
        x = x / 255.
        x = self.features(x)
        x = x.view(batch_size, -1) # reshape成batch_size个行
        # 求得价值函数部分value
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        # 求得优势函数部分advantage
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        # reshape
        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        # Q=V+A-A.mean()
        x = value + advantage - advantage.mean(1, keepdim=True)
        # 对计算出的分布计算softmax，得到大小为action*atoms的矩阵
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        """
            重置噪声
        """
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def feature_size(self):
        """
            获得feature网络层的输出的大小
        """
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
            输入state，获得当前最优的action
        """
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            # 计算动作价值函数的分布 (1,6,51):(1,num_actions,num_atoms)
            dist = self.forward(state).data.cpu()
            dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
            # 取峰值作为action
            action = dist[0].sum(1).max(0)[1].numpy()
        else:
            action = random.randrange(self.num_actions)
        return action