from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from Agent import *

import os


USE_CUDA = torch.cuda.is_available()

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

# 超参数
num_frames = 1000000 # num of steps
batch_size = 32
gamma = 0.99 # discount

# 保存结果
fig_dir = "./fig/"
figp_dir = "./figp/"

if not os.path.exists("./fig"):
    os.makedirs("./fig")
if not os.path.exists("./figp"):
    os.makedirs("./figp")

agent = Agent(env, num_frames, batch_size, gamma,  figp_dir, fig_dir)
agent.train()






