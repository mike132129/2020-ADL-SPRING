import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_dim, action_num)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x is the state
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64
                           		)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 10000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions, self.logprob = [], [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions, self.logprob = [], [], []

    def make_action(self, state, test=False):
        # action = self.env.action_space.sample() # Replace this line, 
                                                  # because this function randomly select action from action space
        # Use your model to output distribution over actions and sample from it.
        # state shape = (8, )
        output = self.model(torch.tensor(state).view(1, -1))
        category = torch.distributions.Categorical(output)
        action = category.sample()
        self.logprob.append(category.log_prob(action))

        return action.item()

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        R = 0
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.append(R)

        rewards.reverse()
        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        # baseline = rewards.mean()   
        # rewards = (rewards - baseline) / (rewards.std())
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        loss = 0
        for r, logprob in zip(rewards, self.logprob):
            loss += -r * logprob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        print('Training Starting')
        episode_reward = []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action) # state is a 8-dim vector
                                                               # reward is a float

                self.saved_actions.append(action)
                self.rewards.append(reward)         #store rewards

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            episode_reward.append(avg_reward)
            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                np.save('./reward/pg.npy', np.array(episode_reward))
                break
