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
from environment_VQC import CircuitEnv
import agents
import time
import json

class Saver:
    def __init__(self, results_path, experiment_seed):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode, episode_no):
        if mode == 'train':
            self.stats_file[mode][episode_no] = {'loss_policy': [],
                                                 'loss_value': [],
                                                 'actions': [],
                                                 'errors': [],
                                                 'errors_test':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [], 
                                                 'opt_ang': [],
                                                 'time' : [],
                                                 'rewards' : []
                                                 }
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
                                                 'errors': [],
                                                 'errors_test':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [],
                                                 'opt_ang': [],
                                                 'time' : [],
                                                 'rewards' : []
                                                 }

    def save_file(self):
        # file_name = f'{self.rpath}/summary_{self.exp_seed}.p'
        # with open(file_name, 'wb') as handle:
        #     pickle.dump(self.stats_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(self.stats_file['train'][0])
        with open(f"{self.rpath}/summary_{self.exp_seed}.json", "w") as outfile:
            json.dump(self.stats_file, outfile)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

    
def modify_state(state,env):
    
        
    if conf['agent']['en_state']:
        
        state = torch.cat((state, torch.tensor(env.prev_energy,dtype=torch.float,device=device).view(1)))
        
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        state = torch.cat((state, torch.tensor(env.done_threshold,dtype=torch.float,device=device).view(1)))
         
    return state


def agent_test(env, agent, episode_no, seed, output_path,threshold):
    """ Testing function of the trained agent. """    
    """NOT YET IMPLEMENTED FOR PPO ALGORITHM"""
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env)
    # current_epsilon = copy.copy(agent.epsilon)
    agent.policy_net.eval()

    for t in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        agent.epsilon = 0
        with torch.no_grad():
            action, _ = agent.act(state, ill_action_from_env)
            assert type(action) == int
            agent.saver.stats_file['test'][episode_no]['actions'].append(action)
        next_state, reward, done = env.step(agent.translate[action],train_flag=False)
        next_state = modify_state(next_state, env)
        state = next_state.clone()
        assert type(env.error) == float
        agent.saver.stats_file['train'][episode_no]['actions'].append(action.item())
        agent.saver.stats_file['test'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['test'][episode_no]['errors_test'].append(env.error_test)
        agent.saver.stats_file['test'][episode_no]['opt_ang'].append(env.opt_ang_save)
        agent.saver.stats_file['test'][episode_no]['rewards'].append(env.current_reward)
        
        if done:
            
            agent.saver.stats_file['test'][episode_no]['done_threshold'] = env.done_threshold
            errors_current_bond = [val['errors'][-1] for val in agent.saver.stats_file['test'].values()
                                   if val['done_threshold'] == env.done_threshold]
            if len(errors_current_bond) > 0 and min(errors_current_bond) > env.error:
                torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_model.pt")
                torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_optim.pt")
            # agent.epsilon = current_epsilon
            agent.saver.validate_stats(episode_no, 'test')
            # print("Hello")
            return reward, t
        

def one_episode(episode_no, env, agent, episodes):
    """Function performing full training episode with PPO."""
    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold
    
    state = modify_state(state, env)
    agent.policy.train()
    agent.value.train()
    
    # Lists to store episode data
    states, actions, rewards = [], [], []
    done = False
    
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        state_tensor = state.clone().detach().to(dtype=torch.double, device=agent.device)
        
        # Get action probabilities from policy network
        with torch.no_grad():
            action_probs = torch.softmax(agent.policy(state_tensor), dim=-1)
            # Mask illegal actions
            if ill_action_from_env is not None:
                action_probs[ill_action_from_env] = 0
                action_probs = action_probs / action_probs.sum()
            
            action = torch.multinomial(action_probs, 1).item()

        # Store state and action
        states.append(state_tensor)
        actions.append(action)
        
        # Take action in environment
        next_state, reward, done = env.step(agent.translate[action])
        rewards.append(reward)
        
        next_state = modify_state(next_state, env)
        state = next_state.clone()

        # Add action to stats file
        agent.saver.stats_file['train'][episode_no]['actions'].append(action)
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_test'].append(env.error_test)
        agent.saver.stats_file['train'][episode_no]['rewards'].append(env.current_reward)
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time()-t0)



        if done:
            # print('time:', time.time()-t0)
            if episode_no%20==0:
                print("episode: {}/{}, score: {}, rwd: {} \n"
                        .format(episode_no, episodes, itr, reward),flush=True)
            break

    # Stack states and convert actions and rewards to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long, device=agent.device)
    rewards = torch.tensor(rewards, dtype=torch.double, device=agent.device)
    
    # Compute returns
    returns = []
    R = 0
    for r in reversed(rewards.tolist()):
        R = r + agent.final_gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, dtype=torch.double, device=agent.device)
    
    # Get value predictions - without detaching
    values = agent.value(states).squeeze()
    
    # Compute advantages
    advantages = returns - values.detach()  # Detach values when computing advantages
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update
    entropy_coef = 0.01
    clip_range = 0.2
    value_coef = 0.5
    
    # Get action probabilities
    action_probs = agent.policy(states)
    log_probs = torch.log_softmax(action_probs, dim=-1)
    log_probs = log_probs[torch.arange(len(actions)), actions]
    
    # Compute losses
    total_loss, policy_loss, value_loss, entropy_loss = agent.compute_loss(
        log_probs, returns, values, advantages, entropy_coef, clip_range, value_coef
    )

    agent.saver.stats_file['train'][episode_no]['loss_policy'].append(policy_loss.item())
    agent.saver.stats_file['train'][episode_no]['loss_value'].append(value_loss.item())
    agent.saver.validate_stats(episode_no, 'train')
    
    # Optimize policy network
    agent.optim_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
    agent.optim_policy.step()
    
    # Optimize value network separately
    agent.optim_value.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.value.parameters(), max_norm=0.5)
    agent.optim_value.step()
            

def train(agent, env, episodes, seed, output_path,threshold):
    """Training loop"""
    threshold_crossed = 0
    for e in range(episodes):
        
        one_episode(e, env, agent, episodes)
        
        if e %20==0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_model.pt")
            torch.save(agent.value.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_value_model.pt")
            torch.save(agent.optim_policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_optim.pt")
            torch.save(agent.optim_value.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_value_optim.pt")
        if env.error <= 0.0016:
            threshold_crossed += 1
            #np.save( f'threshold_crossed', threshold_crossed )

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproduction')
    parser.add_argument('--config', type=str, default='h_s_2', help='Name of configuration file')
    parser.add_argument('--experiment_name', type=str, default='lower_bound_energy/', help='Name of experiment')
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    parser.add_argument('--wandb_group', type=str, default='test/', help='Group of experiment run for wandb')
    parser.add_argument('--wandb_name', type=str, default='test/', help='Name of experiment run for wandb')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':

    args = get_args(sys.argv[1:])


    results_path ="VQC/results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)
    # device = torch.device(f"cuda:{args.gpu_id}")
    device = torch.device(f"cpu:{0}")
    
    
    conf = get_config(args.experiment_name, f'{args.config}.cfg')

    loss_dict, scores_dict, test_scores_dict, actions_dict = dict(), dict(), dict(), dict()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


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
    torch.save(agent.value.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_value_model.pt")
    torch.save(agent.optim_policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_optim.pt")
    torch.save(agent.optim_value.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_value_optim.pt")
