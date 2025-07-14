import torch.nn as nn
import random
import torch
import copy
from collections import namedtuple
import numpy as np
from utils import dictionary_of_actions, dict_of_actions_revert_q

# Define the Dueling Deep Q-Network (Dueling DQN) class
class Dueling_DQN(object):
    """Implementation of a Dueling Q-Network agent."""
    def __init__(self, conf, action_size, state_size, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        memory_size = conf['agent']['memory_size']

        self.final_gamma = conf['agent']['final_gamma']  
        self.epsilon_min = conf['agent']['epsilon_min']
        self.epsilon_decay = conf['agent']['epsilon_decay']
        learning_rate = conf['agent']['learning_rate']
        self.update_target_net = conf['agent']['update_target_net']
        neuron_list = conf['agent']['neurons']
        drop_prob = conf['agent']['dropout']
        self.with_angles = conf['agent']['angles']

        self.action_size = action_size
        self.state_size = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        self.state_size += 1 if conf['agent']['en_state'] else 0
        self.state_size += 1 if conf['agent'].get("threshold_in_state", False) else 0

        # Handle memory reset logic
        self.memory_reset_switch = conf['agent'].get("memory_reset_switch", False)
        self.memory_reset_threshold = conf['agent'].get("memory_reset_threshold", False)
        self.memory_reset_counter = 0

        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        # Initialize policy and target networks as Dueling DQN
        self.policy_net = self.unpack_network(neuron_list, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)]).to(device)
        self.memory = ReplayMemory(memory_size)

        self.epsilon = 1.0  

        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()  
        self.device = device
        self.step_counter = 0  

        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, ill_action):
        state = state.unsqueeze(0)  
        epsilon = False

        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_action:  
                rand_ac = torch.randint(self.action_size, (1,)).item()
            epsilon = True
            return rand_ac, epsilon

        act_values = self.policy_net(state)
        act_values[0][ill_action] = float('-inf')  
        return torch.argmax(act_values[0]).item(), epsilon

    def replay(self, batch_size):
        if self.step_counter % self.update_target_net == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.step_counter += 1

        transitions = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))

        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) * (1 - done_batch) + reward_batch
        expected_state_action_values = expected_state_action_values.view(-1, 1)

        assert state_action_values.shape == expected_state_action_values.shape, "Shape mismatch in loss calculation"

        cost = self.fit(state_action_values, expected_state_action_values)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        assert self.epsilon >= self.epsilon_min, "Epsilon decay issue"
        return cost

    def fit(self, output, target_f):
        self.optim.zero_grad()
        loss = self.loss(output, target_f)
        loss.backward()
        self.optim.step()
        return loss.item()

    def unpack_network(self, neuron_list, p):
        """Dueling DQN with separate Value and Advantage streams."""
        class DuelingNetwork(nn.Module):
            def __init__(self, state_size, action_size, neuron_list, dropout_prob):
                super(DuelingNetwork, self).__init__()
                layers = []
                neuron_list = [state_size] + neuron_list

                for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
                    layers.append(nn.Linear(input_n, output_n))
                    layers.append(nn.LeakyReLU())
                    layers.append(nn.Dropout(p=dropout_prob))

                self.feature_layer = nn.Sequential(*layers)

                # Separate Value and Advantage streams
                last_hidden_size = neuron_list[-1]
                self.value_stream = nn.Sequential(
                    nn.Linear(last_hidden_size, last_hidden_size // 2),
                    nn.LeakyReLU(),
                    nn.Linear(last_hidden_size // 2, 1)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(last_hidden_size, last_hidden_size // 2),
                    nn.LeakyReLU(),
                    nn.Linear(last_hidden_size // 2, action_size)
                )

            def forward(self, x):
                features = self.feature_layer(x)
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                return q_values

        return DuelingNetwork(self.state_size, self.action_size, neuron_list, p)

# Replay Memory remains unchanged
class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clean_memory(self):
        self.memory = []
        self.position = 0

# Entry point
if __name__ == '__main__':
    pass
