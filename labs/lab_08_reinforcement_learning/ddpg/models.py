import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 20)
        self.linear2 = nn.Linear(20 + self.action_dim, 20)
        self.linear3 = nn.Linear(20, 20)
        self.linear4 = nn.Linear(20, 10)
        self.linear5 = nn.Linear(10, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        xa = F.relu(self.linear4(xa))
        qval = self.linear5(xa)

        return qval

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, max_action = 2):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.linear1 = nn.Linear(self.obs_dim, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 10)
        self.linear4 = nn.Linear(10, self.action_dim)

        #self.linear1 = nn.Linear(self.obs_dim, 50)
        #self.linear2 = nn.Linear(50, 30)
        #self.linear3 = nn.Linear(30, 10)
        #self.linear4 = nn.Linear(10, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x =  self.max_action*torch.tanh(self.linear4(x))

        return x
