import numpy as np
import random
from OU_process import OUNoise

from p3_model import Actor, Critic

import torch
import torch.optim as optim


LR_ACTOR = 1e-4 # learning rate for actor
LR_CRITIC = 1e-3 # learning rate for critic
WEIGHT_DECAY = 0. # L2 weight decay
EPSILON = 1.0           # explore->exploit noise process added to act step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, tau=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network (local / target)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # inputs size of critic should be 2 times state_size of each agent, so as action_size
        # since we use the concatenated states and actions of both agents
        self.critic_local = Critic(state_size * 2, action_size * 2, seed).to(device)
        self.critic_target = Critic(state_size * 2, action_size * 2, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # initialize the params in target network same as them in local network
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # initialize noise
        self.noise = OUNoise(action_size, seed)

        self.tau = tau
        self.epsilon = EPSILON

    def act_local(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action += self.epsilon * self.noise.sample()
        return np.clip(action, -1., 1.)

    def act_target(self, state):
        """Returns actions for given state as target policy."""
        action = self.actor_target(state).cpu().data.numpy()

        # action += self.epsilon * self.noise.sample()
        return np.clip(action, -1., 1.)

    def update(self):
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for local_params, target_params in zip(local_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1 - tau) * target_params.data)

    def hard_update(self, target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def reset(self):
        self.noise.reset()

