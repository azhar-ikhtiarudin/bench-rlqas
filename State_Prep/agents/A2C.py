import torch.nn as nn
import torch.nn.functional as F
import random
import torch

import copy
from collections import namedtuple, deque
import numpy as np
from itertools import product

from utils import dictionary_of_actions, dict_of_actions_revert_q


class A2C(nn.Module):
    def __init__(self, conf, action_size, state_size, device):

        super().__init__()
        
        # Environment and memory settings
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        

        # Agent hyperparameters
        self.final_gamma = conf['agent']['final_gamma']
        self.actor_lr = conf['agent']['learning_rate']
        self.critic_lr = conf['agent']['learning_rate']
        self.with_angles = conf['agent']['angles']  # Whether state includes angle information
        

        # State size adjustments based on configuration
        self.action_size = action_size
        self.state_size = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size
        
        # Action translation dictionaries
        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        # Model for policy and critic network
        
        neuron_list_actor = conf['agent']['neurons'] #neurons = [1000,1000,1000,1000,1000]
        neuron_list_critic = conf['agent']['neurons'] #neurons = [1000,1000,1000,1000,1000]
        drop_prob = conf['agent']['dropout']

        self.actor = self.unpack_network(neuron_list_actor, drop_prob, type='actor').to(device)
        self.critic = self.unpack_network(neuron_list_critic, drop_prob, type='critic').to(device)


        # list for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)]).to(device)

        # Optimizer and Loss Function
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.device = device
    
    def unpack_network(self, neuron_list, dropout, type):
        layer_list = []
        neuron_list = [self.state_size] + neuron_list

        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Dropout(p=dropout))

        if type=='actor':
            layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        elif type=='critic':
            layer_list.append(nn.Linear(neuron_list[-1], 1))
        
        return nn.Sequential(*layer_list).double()
    

    @staticmethod
    def compute_loss(log_probs, returns, values, advantages):
        
        actor_loss = -torch.mean(log_probs*advantages)
        critic_loss = torch.mean(torch.square(returns - values))
        entropy_loss = -torch.mean(torch.mul(log_probs, torch.exp(log_probs)))
        
        # Total loss and tune hyperparameter
        value_coef = 0.5
        entropy_coef = 0.01 # typical values in the range 0.01 to 0.1, but could be tuned further
        total_loss = actor_loss + value_coef*critic_loss + entropy_coef*entropy_loss

        return total_loss, actor_loss, critic_loss, entropy_loss


if __name__ == '__main__':
    pass