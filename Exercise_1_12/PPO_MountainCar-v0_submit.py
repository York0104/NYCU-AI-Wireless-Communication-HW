import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# Parameters
env_name = 'MountainCar-v0'
gamma = 0.99
render = False
seed = 1
log_interval = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'exp_submit')
PARAM_DIR = os.path.join(BASE_DIR, 'param_submit')
NET_PARAM_DIR = os.path.join(PARAM_DIR, 'net_param')
IMG_DIR = os.path.join(PARAM_DIR, 'img')

env = gym.make(env_name)
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
np.random.seed(seed)
try:
    env.reset(seed=seed)
except TypeError:
    env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 8000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        try:
            self.writer = SummaryWriter(LOG_DIR)
        except Exception:
            self.writer = None

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        os.makedirs(NET_PARAM_DIR, exist_ok=True)
        os.makedirs(IMG_DIR, exist_ok=True)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        actor_path = os.path.join(NET_PARAM_DIR, 'actor_net_' + str(time.time())[:10] + '.pkl')
        critic_path = os.path.join(NET_PARAM_DIR, 'critic_net_' + str(time.time())[:10] + '.pkl')
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                action_prob = self.actor_net(state[index]).gather(1, action[index])

                # ─── YOUR CODE HERE ──────────────────────────────────────────── #
                # calculate the importance ratio
                ratio = action_prob / old_action_log_prob[index]

                # update actor network
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(self.critic_net(state[index]), Gt_index)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                # ─────────────────────────────────────────────────────────────── #

                self.training_step += 1

        del self.buffer[:]


def main():
    agent = PPO()
    for i_epoch in range(1000):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if render:
            env.render()

        for t in count():
            action, action_prob = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            trans = Transition(state, action, action_prob, reward, next_state)
            if render:
                env.render()
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                if agent.writer is not None:
                    agent.writer.add_scalar('Steptime/steptime', t, global_step=i_epoch)
                break


if __name__ == '__main__':
    main()
    print("end")
