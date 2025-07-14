import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from utils import dictionary_of_actions, dict_of_actions_revert_q

class TPPO(nn.Module):
    def __init__(self, conf, action_size, state_size, device):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # TPPO specific hyperparameters
        self.delta = conf['agent'].get('delta', 0.01)
        self.alpha = conf['agent'].get('alpha', 0.5)
        self.num_epochs = conf['agent'].get('num_epochs', 10)
        
        num_qubits = conf['env']['num_qubits']
        self.translate = dictionary_of_actions(num_qubits)
        if len(self.translate) != action_size:
            raise ValueError(f"Action size mismatch: expected {action_size}, got {len(self.translate)}")
        
        # Build networks
        neuron_list = conf['agent']['neurons']
        dropout = conf['agent']['dropout']
        self.policy = self.unpack_network(neuron_list, dropout, 'policy')
        self.value = self.unpack_network(neuron_list, dropout, 'value')
        
        # Move networks to device
        self.policy = self.policy.to(device)
        self.value = self.value.to(device)
        
        # Optimizers
        self.policy_lr = conf['agent']['learning_rate']
        self.value_lr = conf['agent']['learning_rate']
        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.optim_value = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

        # Storage for experiences
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []

    def unpack_network(self, neuron_list, dropout, type):
        layer_list = []
        neuron_list = [self.state_size] + neuron_list

        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p=dropout))

        if type == 'policy':
            layer_list.append(nn.Linear(neuron_list[-1], self.action_size))
        elif type == 'value':
            layer_list.append(nn.Linear(neuron_list[-1], 1))
        else:
            raise ValueError("type must be 'policy' or 'value'")
        
        return nn.Sequential(*layer_list).double()
    
    def act(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.double, device=self.device)
        state = state.to(self.device).unsqueeze(0)
        
        if state.shape[1] != self.state_size:
            raise ValueError(f"State size mismatch in act: expected {self.state_size}, got {state.shape[1]}")
        
        with torch.no_grad():
            logits = self.policy(state)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        self.states.append(state.squeeze(0).cpu().numpy())
        self.actions.append(action.item())
        self.old_log_probs.append(log_prob.item())
        
        return action.item(), log_prob.item()

    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        logits = self.policy(states)
        new_log_probs = F.log_softmax(logits, dim=1).gather(1, actions.unsqueeze(1)).squeeze()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        kl = old_log_probs - new_log_probs
        surrogate = ratio * advantages
        
        mask = (kl >= self.delta) & ((advantages > 0) & (ratio >= 1) | 
                                    (advantages < 0) & (ratio <= 1))
        penalty = self.alpha * kl * mask.double()

        policy_loss = -torch.mean(surrogate - penalty)
        value_loss = F.mse_loss(self.value(states).squeeze(), returns)
        entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
        entropy_loss = -entropy.mean()

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        return total_loss, policy_loss, value_loss, entropy_loss

    def update(self):
        """Update policy and value networks."""
        if not self.states:
            return

        # Convert stored experiences to tensors, keeping them on CPU first
        states_np = np.array(self.states)
        actions_np = np.array(self.actions)
        old_log_probs_np = np.array(self.old_log_probs)
        rewards_np = np.array([r.item() if torch.is_tensor(r) else r for r in self.rewards])  # Convert tensors to scalars

        # Move to device after conversion
        states = torch.tensor(states_np, dtype=torch.double).to(self.device)
        actions = torch.tensor(actions_np, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs_np, dtype=torch.double).to(self.device)
        
        # Calculate returns and advantages (simplified version)
        returns = torch.tensor(rewards_np, dtype=torch.double).to(self.device)
        advantages = torch.tensor(rewards_np, dtype=torch.double).to(self.device)

        # Perform multiple epochs of updates
        for _ in range(self.num_epochs):
            total_loss, _, _, _ = self.compute_loss(states, actions, old_log_probs, 
                                                  advantages, returns)
            
            # Update both networks
            self.optim_policy.zero_grad()
            self.optim_value.zero_grad()
            total_loss.backward()
            self.optim_policy.step()
            self.optim_value.step()

        # Clear experience buffer
        self.states, self.actions, self.rewards, self.old_log_probs = [], [], [], []