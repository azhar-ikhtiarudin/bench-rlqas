import torch
import torch.nn as nn
from utils import dictionary_of_actions, dict_of_actions_revert_q

class PPO(nn.Module):
    def __init__(self, conf, action_size, state_size, device):
        super().__init__()
        
        # Environment and memory settings
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']

         # Agent hyperparameters
        self.final_gamma = conf['agent']['final_gamma']
        self.policy_lr = conf['agent']['learning_rate']
        self.value_lr = conf['agent']['learning_rate']
        self.with_angles = conf['agent']['angles']  # Whether state includes angle information

        self.action_size = action_size
        self.state_size = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size

        # Action translation dictionaries
        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        # Model for policy and critic network
        neuron_list_policy = conf['agent']['neurons'] #neurons = [1000,1000,1000,1000,1000]
        neuron_list_value = conf['agent']['neurons'] #neurons = [1000,1000,1000,1000,1000]
        drop_prob = conf['agent']['dropout']

        self.policy = self.unpack_network(neuron_list_policy, drop_prob, type='policy').to(device)
        self.value = self.unpack_network(neuron_list_value, drop_prob, type='value').to(device)

        # list for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # Optimizer and Loss Function
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.optim_value = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.device=device

    def unpack_network(self, neuron_list, dropout, type):
        """
        Dynamically creates a network architecture for Policy or ValueFunction based on the parameters.

        Args:
            neuron_list (list): List of neuron counts for each layer.
            dropout (float): Dropout probability for regularization.
            type (str): Type of network to create ('policy' or 'value').

        Returns:
            nn.Sequential: The dynamically created network.
        """
        layer_list = []
        neuron_list = [self.state_size] + neuron_list  # Include input size in the list

        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p=dropout))

        if type == 'policy':
            layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        elif type == 'value':
            layer_list.append(nn.Linear(neuron_list[-1], 1))
        else:
            raise ValueError("type must be 'policy' or 'value'.")
        
        return nn.Sequential(*layer_list).double()
    

    @staticmethod
    def compute_loss(log_probs, returns, values, advantages, entropy_coef, clip_range, value_coef):
        policy_loss = -torch.mean(torch.min(
            torch.mul(log_probs, torch.clamp(returns + advantages, 1 - clip_range, 1 + clip_range)),
            torch.mul(log_probs, returns)
        ))
        value_loss = torch.mean(torch.square(returns - values))
        entropy_loss = -torch.mean(torch.mul(log_probs, torch.exp(log_probs)))
        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        return total_loss, policy_loss, value_loss, entropy_loss
    

if __name__ == "__main__":
    pass