import torch
import math
from tqdm import tqdm
from torch import optim, autograd
from IPython.display import clear_output
import matplotlib.pyplot as plt

from Network import DuelingCnnDQN
from NaivePrioritizedBuffer import *

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Agent():
    def __init__(self, env, num_frames, batch_size, gamma, replay_initial, model_dir, fig_dir, p_dir):
        self.fig_dir = fig_dir
        self.model_dir = model_dir
        self.p_dir = p_dir
        self.env = env
        self.num_atoms = 51
        self.Vmin = -10
        self.Vmax = 10

        self.num_frames = num_frames
        self.replay_initial = replay_initial
        self.batch_size = batch_size
        self.gamma = gamma

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.n

        self.replay_buffer  = NaivePrioritizedBuffer(100000)

        self.current_model = DuelingCnnDQN(self.obs_dim, self.act_dim, self.env)
        self.target_model = DuelingCnnDQN(self.obs_dim, self.act_dim, self.env)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.update_target(self.current_model, self.target_model)

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def compute_td_loss(self, batch_size, beta):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))
        weights = Variable(torch.FloatTensor(weights))

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss

    def plot(self, frame_idx, rewards, losses):
        clear_output(True)
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        # plt.show()
        return fig

    def train(self):

        losses = []
        all_rewards = []
        episode_reward = 0
        episode = 1
        loss = 0

        state = self.env.reset()

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 30000

        epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        beta_start = 0.4
        beta_frames = 100000
        beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

        # 进度条
        pbar = tqdm(range(1,self.num_frames+1))

        for frame_idx in pbar:
            epsilon = epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                episode += 1


            if len(self.replay_buffer) > self.replay_initial:
                beta = beta_by_frame(frame_idx)
                loss = self.compute_td_loss(self.batch_size, beta)
                losses.append(loss.item())

            # 模型更新
            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)

            if frame_idx % 50000 == 0:
                figure = self.plot(frame_idx, all_rewards, losses)
                # 保存模型
                torch.save(self.current_model.state_dict(), self.model_dir + "ddqn_%d.pth" % frame_idx)
                # 保存图片
                figure.savefig(self.p_dir + 'priorizeddqn_' + '%d.png' % frame_idx)
                # 保存绘图数据
                reward_tmp = np.array(all_rewards)
                losses_tmp = np.array(losses)
                np.save(self.fig_dir + 'prioritizeddqn_reward_' + '%d.npy' % frame_idx, reward_tmp)  # 保存为.npy格式
                np.save(self.fig_dir + 'prioritizeddqn_loss_' + '%d.npy' % frame_idx, losses_tmp)  # 保存为.npy格式

            pbar.set_description("episode:%d frame_id:%d loss:%4f reward:%4f" % (episode, frame_idx, loss, episode_reward))

