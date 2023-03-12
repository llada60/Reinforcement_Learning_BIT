import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

class NoisyLinear(nn.Module):
    """
        添加噪声的网络层
    """
    def __init__(self, in_features, out_features, use_cuda, std_init=0.4):
        """
            初始化NoisyLinear (mu+sigma*epsilon)
            weight_mu
            weight_sigma
            weight_epsilon
            bias_mu
            bias_sigma
            bias_epsilon
        """
        super(NoisyLinear, self).__init__()
        
        self.use_cuda     = use_cuda
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        # 噪声层中可学习的部分(mu,sigma)
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # 噪声层中噪声部分(epsilon)，不可学习
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        # 每个变量的bias
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        # reset
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        """
            前向传播
        """
        if self.use_cuda:
            weight_epsilon = self.weight_epsilon.cuda()
            bias_epsilon   = self.bias_epsilon.cuda()
        else:
            weight_epsilon = self.weight_epsilon
            bias_epsilon   = self.bias_epsilon
            
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        """
            randomize parameters:(mu,sigma)
        """
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        """
            randomize noise:epsilon
        """
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        """
            随机初始化大小为size的噪声
        """
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x