import numpy as np
import random
import torch
import torch.nn as nn
import sys
import os
import argparse
import pathlib
import copy
from utils import get_config, dictionary_of_actions, dict_of_actions_revert_q
from environment import CircuitEnv
import agents
import time
# import wandb
# os.environ['WANDB_DISABLED'] = 'true'
torch.set_num_threads(1)
import json

class Saver:
    def __init__(self, results_path, experiment_seed):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode, episode_no, eps):
        if mode not in self.stats_file:
            self.stats_file[mode] = {}
        if episode_no not in self.stats_file[mode]:
            self.stats_file[mode][episode_no] = {}
        
        if mode == 'train':
            self.stats_file[mode][episode_no][eps] = {
                'loss_policy': [],
                'kl_divergence': [],
                'actions': [],
                'errors': [],
                'errors_noiseless':[],
                'done_threshold': 0,
                'bond_distance': 0,
                'nfev': [], 
                'opt_ang': [],
                'time' : [],
                'rewards' : []
            }
        elif mode == 'test':
            self.stats_file[mode][episode_no][eps] = {
                'actions': [],
                'errors': [],
                'errors_noiseless':[],
                'done_threshold': 0,
                'bond_distance': 0,
                'nfev': [],
                'opt_ang': [],
                'time' : [],
                'rewards' : []
            }

    def save_file(self):
        with open(f"{self.rpath}/summary_{self.exp_seed}.json", "w") as outfile:
            json.dump(self.stats_file, outfile)

    def validate_stats(self, episode, mode, eps):
        assert len(self.stats_file[mode][episode][eps]['actions']) == len(self.stats_file[mode][episode][eps]['errors'])

    
def modify_state(state,env):
    
        
    if conf['agent']['en_state']:
        
        state = torch.cat((state, torch.tensor(env.prev_energy,dtype=torch.float,device=device).view(1)))
        
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        state = torch.cat((state, torch.tensor(env.done_threshold,dtype=torch.float,device=device).view(1)))
         
    return state


def one_episode_mod(episode_no, env, agent, episodes):

    # Initialize buffers
    t0 = time.time()
    # Collect data for multiple episodes
    all_states, all_actions, all_log_probs, all_rewards = [], [], [], []
    data_collection_eps = 5
    for eps in range(data_collection_eps):  # Collect data for E episodes
        state = env.reset()
        # print(agent.saver)
        agent.saver.get_new_episode('train', episode_no, eps)
        agent.saver.stats_file['train'][episode_no][eps]['bond_distance'] = env.current_bond_distance
        agent.saver.stats_file['train'][episode_no][eps]['done_threshold'] = env.done_threshold

        state = modify_state(state, env)
        states, actions, rewards = [], [], []
        for itr in range(env.num_layers + 1):
            # ... (action selection and environment interaction)
            ill_action_from_env = env.illegal_action_new()
            # state_tensor = state.clone().detach().to(dtype=torch.float32, device=agent.device)
            state_tensor = state.clone().detach().to(dtype=torch.float32, device=agent.device)

            # Get action probabilities from policy network
            with torch.no_grad():
                action_probs = torch.softmax(agent.policy(state_tensor), dim=-1)

                # Mask illegal actions
                if ill_action_from_env is not None:
                    action_probs[ill_action_from_env] = 0
                    action_probs /= action_probs.sum()

                # action = torch.multinomial(action_probs, 1).item()
                # Move action_probs to CPU for multinomial sampling
                action_probs_cpu = action_probs.cpu()

                # Ensure the probabilities are valid
                action_probs_cpu = torch.clamp(action_probs_cpu, min=1e-6)
                action_probs_cpu /= action_probs_cpu.sum()

                try:
                    action = torch.multinomial(action_probs_cpu, 1).item()
                except RuntimeError as e:
                    action = torch.argmax(action_probs_cpu).item()

            # Take action in environment
            next_state, reward, done = env.step(agent.translate[action])

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            all_log_probs.append(torch.log(action_probs[action]))
            state = next_state.clone()
            next_state = modify_state(next_state, env)
            state = next_state.clone()

            # Update stats file
            agent.saver.stats_file['train'][episode_no][eps]['actions'].append(action)
            agent.saver.stats_file['train'][episode_no][eps]['errors'].append(env.error)
            agent.saver.stats_file['train'][episode_no][eps]['errors_noiseless'].append(env.error_noiseless)
            agent.saver.stats_file['train'][episode_no][eps]['rewards'].append(env.current_reward)
            agent.saver.stats_file['train'][episode_no][eps]['time'].append(time.time() - t0)
            

            if done:
                if episode_no % 1 == 0:
                    print("Episode: {}/{}, eps: {} Steps: {}, Reward: {:.2f}"
                          .format(episode_no, episodes, eps, itr, reward), flush=True)
                break
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)

    # Normalize advantages across all episodes
    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
    advantages = agent.compute_group_advantage(rewards_tensor.unsqueeze(0))  # Group normalization

    # Convert collected data to tensors
    states_tensor = torch.stack(all_states).to(dtype=torch.float32)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=agent.device)
    rewards_tensor = advantages.squeeze(0).to(device=agent.device)  # Use normalized advantages
    old_log_probs = torch.stack(all_log_probs).to(dtype=torch.float32).detach()

    # Perform GRPO update
    clip_range = 0.2
    _, policy_loss, kl_div = agent.update(states_tensor, actions_tensor, rewards_tensor, old_log_probs, clip_range)
    # Update stats file with training metrics
    for eps in range(data_collection_eps):
        agent.saver.stats_file['train'][episode_no][eps]['loss_policy'].append(policy_loss)
        agent.saver.stats_file['train'][episode_no][eps]['kl_divergence'].append(kl_div)
        agent.saver.validate_stats(episode_no, 'train', eps)




def train(agent, env, episodes, seed, output_path, threshold):
    """Training loop"""
    # threshold_crossed = 0
    for e in range(episodes):
        
        # one_episode(e, env, agent, episodes)
        one_episode_mod(e, env, agent, episodes)
        
        if e % 20 == 0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_model.pt")
            torch.save(agent.optimizer.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_optim.pt")
        # if env.error <= 0.0016:
            # threshold_crossed += 1
            # np.save(f'threshold_crossed', threshold_crossed)


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproduction')
    parser.add_argument('--config', type=str, default='h_s_2', help='Name of configuration file')
    parser.add_argument('--experiment_name', type=str, default='lower_bound_energy/', help='Name of experiment')
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    # parser.add_argument('--wandb_group', type=str, default='test/', help='Group of experiment run for wandb')
    # parser.add_argument('--wandb_name', type=str, default='test/', help='Name of experiment run for wandb')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':

    args = get_args(sys.argv[1:])


    results_path ="results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}")
    # device = torch.device(f"cpu:{0}")
    
    
    conf = get_config(args.experiment_name, f'{args.config}.cfg')

    loss_dict, scores_dict, test_scores_dict, actions_dict = dict(), dict(), dict(), dict()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wandb_project = 'Bench RL-QAS'
# 
    # wandb.login()
    # run = wandb.init(project=wandb_project,
                    # config=conf,
                    # group=args.wandb_group,
                    # name=args.wandb_name)
    

    actions_test = []
    action_test_dict = dict()
    error_test_dict = dict()
    error_noiseless_test_dict=dict()

    
    """ Environment and Agent initialization"""
    environment = CircuitEnv(conf, device=device)
    agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](conf, environment.action_size, environment.state_size, device)
    agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)

    if conf['agent']['init_net']: 
        PATH = f"{results_path}{conf['agent']['init_net']}{args.seed}"
        agent.policy_net.load_state_dict(torch.load(PATH+f"_model.pt"))
        agent.target_net.load_state_dict(torch.load(PATH+f"_model.pt"))
        agent.optim.load_state_dict(torch.load(PATH+f"_optim.pt"))
        agent.policy_net.eval()
        agent.target_net.eval()

        replay_buffer_load = torch.load(f"{PATH}_replay_buffer.pt")
        for i in replay_buffer_load.keys():
            agent.remember(**replay_buffer_load[i])

        if not conf['agent']['epsilon_restart']:
            agent.epsilon = agent.epsilon_min

    # train(agent, environment, conf['general']['episodes'], args.seed, f"{results_path}{args.experiment_name}{args.config}",conf['env']['accept_err'])
    # agent.saver.save_file()

    train(agent, environment, conf['general']['episodes'], args.seed, f"{results_path}{args.experiment_name}{args.config}", conf['env']['accept_err'])
    agent.saver.save_file()
            
    # Save both policy and value networks
    torch.save(agent.policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_model.pt")
    torch.save(agent.optim_policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_optim.pt")

    # wandb.finish()