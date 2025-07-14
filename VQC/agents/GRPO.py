import torch
import torch.nn as nn
from utils import dictionary_of_actions, dict_of_actions_revert_q

class GRPOagent(nn.Module):
    def __init__(self, conf, action_size, state_size, device):
        super().__init__()
        
        # Environment and memory settings
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']

        # GRPO specific hyperparameters
        #conf['agent']['group_size']  # Number of samples per group
        self.kl_coef = 0.01 #conf['agent']['kl_coef']  # KL divergence coefficient
        self.n_epochs = 10
        # Agent hyperparameters
        self.final_gamma = conf['agent']['final_gamma']
        self.policy_lr = conf['agent']['learning_rate']
        self.with_angles = conf['agent']['angles']

        self.action_size = action_size
        self.state_size = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size

        # Action translation dictionaries
        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        # Model for policy network
        neuron_list_policy = conf['agent']['neurons']
        drop_prob = conf['agent']['dropout']

        self.policy = self.unpack_network(neuron_list_policy, drop_prob).to(device)

        # Reference policy (frozen copy of initial policy)
        self.ref_policy = self.unpack_network(neuron_list_policy, drop_prob).to(device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        # list for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.device = device

    def unpack_network(self, neuron_list, dropout):
        layer_list = []
        neuron_list = [self.state_size] + neuron_list

        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p=dropout))

        layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        
        return nn.Sequential(*layer_list).float()

    def compute_loss(self, log_probs, advantages, old_log_probs, clip_range):
        """Compute the clipped surrogate loss and KL divergence."""
        log_probs = log_probs.to(dtype=torch.float32)  
        advantages = advantages.to(dtype=torch.float32)  
        old_log_probs = old_log_probs.to(dtype=torch.float32) 
        ratio = torch.exp(log_probs - old_log_probs)  # Importance sampling ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)  # Clip ratio
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()  # PPO loss
        kl_div = (old_log_probs - log_probs).mean()  # KL divergence between old and new policies
        total_loss = policy_loss + self.kl_coef * kl_div  # KL penalty in total loss
        return total_loss, policy_loss, kl_div


    def compute_group_advantage(self, rewards):
        """Compute group-normalized advantages."""
        group_mean = rewards.mean(dim=1, keepdim=True)  # Mean across groups (e.g., batch dimension)
        group_std = rewards.std(dim=1, keepdim=True) + 1e-8  # Standard deviation with numerical stability
        advantages = (rewards - group_mean) / group_std
        return advantages


    def update(self, states, actions, rewards, old_log_probs, clip_range):
        """Perform the GRPO update step."""
        states = states.to(dtype=torch.float32)   
        actions = actions.to(dtype=torch.long)   
        rewards = rewards.to(dtype=torch.float32)  
        old_log_probs = old_log_probs.to(dtype=torch.float32)

        for _ in range(self.n_epochs):  # Multiple epochs for PPO updates
            log_probs = self.policy(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            loss, policy_loss, kl_div = self.compute_loss(log_probs=log_probs,
                                                        advantages=rewards,
                                                        old_log_probs=old_log_probs,
                                                        clip_range=clip_range)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), policy_loss.item(), kl_div.item()


if __name__ == "__main__":
    pass
