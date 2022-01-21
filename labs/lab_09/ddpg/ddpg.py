import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from ddpg.models import Critic, Actor
from common.exp_replay import ExperienceReplayLog
from common.noise import OUNoise


class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate, max_action=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.noise = OUNoise(env.action_space)
        self.iter = 0.0
        self.noisy = False
        self.max_action= max_action
        
        print(self.action_dim)
        print(self.obs_dim)
        
        # RL hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        # Initialize critic and actorr networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim,self.max_action).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy target network paramters for critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Set Optimization algorithms
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
        self.replay_buffer = ExperienceReplayLog(buffer_maxlen)        
        
    def get_action(self, obs):
        #print('obs;',obs)
        
        if self.noisy == True:
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor.forward(state)
            action = action.squeeze(0).cpu().detach().numpy()
            action = self.noise.get_action(action,self.iter)
            self.iter = self.iter+1
        
        else:
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor.forward(state)
            action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        
        #Batch updates
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
   
        # Q info updates
        curr_Q = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        expected_Q = reward_batch + self.gamma * next_Q
        
        # Update Critic network
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        
        self.critic_optimizer.step()

        # Update Actor network
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update Actor and Critic target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            