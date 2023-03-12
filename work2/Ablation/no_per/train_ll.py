import torch

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from Agent import *

from IPython.display import clear_output
import matplotlib.pyplot as plt
import os


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("cuda: ",USE_CUDA)

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

replay_initial = 10000

num_frames = 1000000 # num of steps
batch_size = 32
gamma = 0.99 # discount

model_dir = "./model/"
fig_dir = "./fig/"

if not os.path.exists("./model"):
    os.makedirs("./model")
if not os.path.exists("./fig"):
    os.makedirs("./fig")

agent = Agent(env, num_frames, batch_size, gamma, replay_initial, model_dir, fig_dir)
agent.train()






