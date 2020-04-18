# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)
#
# class FullyConnectedNetwork(nn.Module):
#
#     def __init__(self, input_size, output_size, hidden_size, seed):
#         super(FullyConnectedNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.hidden_size = hidden_size
#
#         # self.input_layer = torch.nn.BatchNorm1d(state_size)
#         self.input_layer = None
#
#         self.layer = nn.ModuleList()
#         self.normalizer = nn.ModuleList()
#
#         size = input_size
#         for i in self.hidden_size:
#             self.layer.append(nn.Linear(size, i))
#             size = i
#
#         self.layer.append(nn.Linear(size, output_size))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for i in range(len(self.layer)-1):
#             self.layer[i].weight.data.uniform_(*hidden_init(self.layer[i]))
#             self.layer[i].weight.data.uniform_(*hidden_init(self.layer[i]))
#         self.layer[-1].weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, input):
#         """Build a network that maps state -> action values."""
#         x = input
#         if self.input_layer:
#             x = self.input_layer(x)
#
#         for i, l in enumerate(self.layer):
#             x = F.relu(l(x))
#             if i < len(self.normalizer):
#                 x = self.normalizer[i](x)
#         return x
#
#
# class PPOPolicyNetwork(nn.Module):
#
#     def __init__(self, state_size, action_size, hidden_size, device, seed):
#         super(PPOPolicyNetwork, self).__init__()
#
#         self.actor = FullyConnectedNetwork(state_size, action_size, hidden_size, seed)
#         self.value_network = FullyConnectedNetwork(state_size, 1, hidden_size,seed)
#         self.std = nn.Parameter(torch.ones(1, action_size))
#         self.actor.to(device)
#         self.value_network.to(device)
#         self.to(device)
#         self.device = device
#
#     def forward(self, obs, action=None):
#
#         if isinstance(obs, (np.ndarray, np.generic) ):
#             obs = torch.from_numpy(obs).float().to(self.device)
#         else:
#             obs = obs.to(self.device)
#         a = F.tanh(self.actor(obs))
#         v = self.value_network(obs)
#
#         dist = torch.distributions.Normal(a, self.std)
#         if action is None:
#             action = dist.sample()
#         else:
#             action = action.to(self.device)
#         log_prob = dist.log_prob(action)
#         log_prob = torch.sum(log_prob, dim=1, keepdim=True)
#         return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FullyConnectedNetwork(nn.Module):

    def __init__(self, state_size, output_size, hidden_size, output_gate=None):
        super(FullyConnectedNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.output_gate = output_gate

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x


class PPOPolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, hidden_size, device, seed):
        super(PPOPolicyNetwork, self).__init__()
        self.actor = FullyConnectedNetwork(state_size, action_size, hidden_size, F.tanh)
        self.critic = FullyConnectedNetwork(state_size, 1, hidden_size)
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)
        self.device = device

    def forward(self, obs, action=None):
        if isinstance(obs, (np.ndarray, np.generic) ):
            obs = torch.from_numpy(obs).float().to(self.device)
        else:
            obs = obs.to(self.device)

        a = self.actor(obs)
        v = self.critic(obs)

        dist = torch.distributions.Normal(a, self.std)
        if action is None:
            action = dist.sample()
        else:
            action = action.to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, torch.Tensor(np.zeros((log_prob.size(0), 1))), v

