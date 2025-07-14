import torch.nn as nn
import random
import torch
import copy
from collections import namedtuple, deque
import numpy as np
import heapq
from utils import dictionary_of_actions, dict_of_actions_revert_q

# Define the Deep Q-Network (DQN) class with Prioritized Experience Replay
class DQN_rank(object):
    """Implementation of a DQN agent with Prioritized Experience Replay."""
    def __init__(self, conf, action_size, state_size, device):
        """
        Initializes the DQN agent with Prioritized Experience Replay.
        
        Parameters:
        - conf: Configuration dictionary with agent and environment settings.
        - action_size: Number of possible actions in the environment.
        - state_size: Size of the state vector.
        - device: Computation device (CPU or GPU).
        """
        # Environment and memory settings
        # Environment and memory settings
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        memory_size = conf['agent']['memory_size']
    
        # Agent parameters
        self.final_gamma = conf['agent']['final_gamma']  # Discount factor
        self.epsilon_min = conf['agent']['epsilon_min']  # Minimum epsilon value for exploration
        self.epsilon_decay = conf['agent']['epsilon_decay']  # Rate of epsilon decay
        learning_rate = conf['agent']['learning_rate']  # Learning rate for optimizer
        self.update_target_net = conf['agent']['update_target_net']  # Target network update frequency
        neuron_list = conf['agent']['neurons']  # List of neurons in each layer
        drop_prob = conf['agent']['dropout']  # Dropout probability for regularization
        self.with_angles = conf['agent']['angles']  # Whether state includes angle information
    
        # Prioritized Experience Replay parameters
        self.alpha = conf['agent'].get('alpha', 0.6)  # Priority exponent
        self.beta = conf['agent'].get('beta', 0.4)  # Importance sampling exponent
        self.beta_increment = conf['agent'].get('beta_increment', 0.001)  # Beta increment per step
        self.priority_eps = conf['agent'].get('priority_eps', 1e-6)  # Small constant to avoid zero priority

        # Memory reset logic
        if "memory_reset_switch" in conf['agent'].keys():
            self.memory_reset_switch = conf['agent']["memory_reset_switch"]
            self.memory_reset_threshold = conf['agent']["memory_reset_threshold"]
            self.memory_reset_counter = 0
        else:
            self.memory_reset_switch = False
            self.memory_reset_threshold = False
            self.memory_reset_counter = False

        # State size adjustments based on configuration
        self.action_size = action_size
        self.state_size = state_size if self.with_angles else state_size - self.num_layers * self.num_qubits * 3
        self.state_size = self.state_size + 1 if conf['agent']['en_state'] else self.state_size
        self.state_size = self.state_size + 1 if ("threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]) else self.state_size

        # Action translation dictionaries
        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)

        # Initialize the policy and target networks 
        self.policy_net = self.unpack_network(neuron_list, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()  # Target network is in evaluation mode

        # Initialize gamma and memory
        self.gamma = torch.Tensor([np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)]).to(device)
        self.memory = PrioritizedReplayMemory(memory_size, self.alpha, self.priority_eps)

        self.epsilon = 1.0  # Start with full exploration

         # Optimizer and loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss = torch.nn.SmoothL1Loss()  # Huber loss
        self.device = device
        self.step_counter = 0  # Step counter for updating target network

    # Define the structure of a transition
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def remember(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay memory with maximum priority.
        """
        max_priority = self.memory.max_priority() if len(self.memory) > 0 else 1.0
        self.memory.push(state, action, reward, next_state, done, max_priority)

    def act(self, state, ill_action):
        """
        Selects an action using an epsilon-greedy policy.
            - state: Current state of the environment.
            - ill_action: List of illegal actions to avoid.
            - Returns: The chosen action and whether it was random.
        """
        state = state.unsqueeze(0)  # Add batch dimension
        epsilon = False
    
    # Exploration: Random action with probability epsilon
        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_action:  # Ensure the action is valid
                rand_ac = torch.randint(self.action_size, (1,)).item()
            epsilon = True
            return (rand_ac, epsilon)  # Returns a tuple (action, epsilon)
    
        # Exploitation: Action with maximum Q-value
        act_values = self.policy_net.forward(state)
        act_values[0][ill_action] = float('-inf')  # Mask illegal actions
        return torch.argmax(act_values[0]).item(), epsilon  # Returns a tuple (action, epsilon)

    def replay(self, batch_size):
        """ Trains the network using a batch of samples from prioritized replay memory."""
        if self.step_counter % self.update_target_net == 0:
            # Periodically update target network
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.step_counter += 1

        # Sample a batch of transitions with priorities
        transitions, indices, weights = self.memory.sample(batch_size, self.beta)
        batch = self.Transition(*zip(*transitions))

        # Extract states, actions, rewards, and next states from the batch
        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)

        # Calculate Q-values for current state-action pairs
        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch.unsqueeze(1))

        # Use the target network to calculate the maximum Q-value for the next state
        next_state_values = self.target_net.forward(next_state_batch).max(1)[0].detach()

        # Compute the expected Q-values
        expected_state_action_values = (next_state_values * self.gamma) * (1 - done_batch) + reward_batch
        expected_state_action_values = expected_state_action_values.view(-1, 1)

        # Compute TD errors and update priorities
        td_errors = (expected_state_action_values - state_action_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Optimize the network with importance sampling weights
        cost = self.fit(state_action_values, expected_state_action_values, weights)
        if self.epsilon > self.epsilon_min:
            # Decay epsilon for exploration-exploitation trade-off
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        assert self.epsilon >= self.epsilon_min, "Problem with epsilons"
        return cost

    def fit(self, output, target_f, weights):
        """
        Performs a single optimization step with importance sampling weights.
        """
        self.optim.zero_grad()
        loss = (self.loss(output, target_f) * weights).mean()  # Weighted loss
        loss.backward()
        self.optim.step()
        return loss.item()

    def unpack_network(self, neuron_list, p):
        """
        Dynamically creates a feedforward neural network for Q-value estimation.
       - neuron_list: Defines the number of neurons in each layer.
       - p: Dropout probability to prevent overfitting.
        """
        layer_list = []
        neuron_list = [self.state_size] + neuron_list  # Add input size to neuron list
        for input_n, output_n in zip(neuron_list[:-1], neuron_list[1:]):
            layer_list.append(nn.Linear(input_n, output_n))
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Dropout(p=p))
        layer_list.append(nn.Linear(neuron_list[-1], self.action_size))  # Output layer
        return nn.Sequential(*layer_list)

# Prioritized Replay Memory for storing past transitions
class PrioritizedReplayMemory(object):
    """
    A prioritized replay memory buffer for storing past experiences (transitions).
    """
    def __init__(self, capacity: int, alpha: float, priority_eps: float):
        """
        Initializes the prioritized replay memory with a fixed capacity.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.priority_eps = priority_eps
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)  # Stores priorities
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state, action, reward, next_state, done, priority):
        """
        Saves a transition in the replay memory with a given priority.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(state, action, reward, next_state, done)
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        """
        Samples a batch of transitions based on their priorities.
        """
        # Sort transitions by absolute TD error (|δi|)
        sorted_indices = np.argsort(-self.priorities[:len(self.memory)])  # Descending order
        ranks = np.arange(1, len(sorted_indices) + 1)  # Assign ranks (1 is highest priority)
        priorities = 1 / ranks  # Compute priorities p_i = 1 / rank(i)

        # Normalize priorities to form a probability distribution
        probs = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(sorted_indices, batch_size, p=probs)
        transitions = [self.memory[idx] for idx in indices]

        # Compute importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Updates the priorities of sampled transitions based on their TD errors.
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + self.priority_eps) ** self.alpha

    def max_priority(self):
        """
        Returns the maximum priority in the replay memory.
        """
        return self.priorities.max() if len(self.memory) > 0 else 1.0

    def __len__(self):
        """
        Returns the current size of the replay memory.
        """
        return len(self.memory)

    def clean_memory(self):
        """
        Clears the replay memory.
        """
        self.memory = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
# Entry point of the script
if __name__ == '__main__':
    pass