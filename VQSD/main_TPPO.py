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
import wandb
os.environ['WANDB_DISABLED'] = 'false'
torch.set_num_threads(1)
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
                                                 'errors_noiseless': [],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [], 
                                                 'opt_ang': [],
                                                 'time': [],
                                                 'rewards': []}
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless': [],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [],
                                                 'opt_ang': [],
                                                 'time': [],
                                                 'rewards': []}

    def save_file(self):
        # print(self.stats_file['train'][0])
        with open(f"{self.rpath}/summary_{self.exp_seed}.json", "w") as outfile:
            json.dump(self.stats_file, outfile)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

def modify_state(state, env, conf, device):
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.double, device=device)
    if conf['agent']['en_state']:
        # print(f"Debug: env.prev_energy type={type(env.prev_energy)}, value={env.prev_energy}")
        if env.prev_energy is None:
            raise ValueError("env.prev_energy is None; ensure CircuitEnv.step() sets it correctly")
        prev_energy = torch.tensor(float(env.prev_energy), dtype=torch.double, device=device).unsqueeze(0)
        state = torch.cat((state, prev_energy))
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        done_threshold = torch.tensor([env.done_threshold], dtype=torch.double, device=device)
        state = torch.cat((state, done_threshold))
    return state

def agent_test(env, agent, episode_no, seed, output_path, threshold):
    """Testing function of the trained agent."""
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env, conf, agent.device)
    agent.policy.eval()

    for t in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        with torch.no_grad():
            action, _ = agent.act(state)
            assert type(action) == int
            agent.saver.stats_file['test'][episode_no]['actions'].append(action)
        next_state, reward, done = env.step(agent.translate[action], train_flag=False)
        next_state = modify_state(next_state, env, conf, agent.device)
        state = next_state.clone()
        assert type(env.error) == float 
        agent.saver.stats_file['test'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['test'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['test'][episode_no]['opt_ang'].append(env.opt_ang_save)
        agent.saver.stats_file['test'][episode_no]['rewards'].append(env.current_reward)
        
        if done:
            agent.saver.stats_file['test'][episode_no]['done_threshold'] = env.done_threshold
            agent.saver.stats_file['test'][episode_no]['bond_distance'] = env.current_bond_distance
            errors_current_bond = [val['errors'][-1] for val in agent.saver.stats_file['test'].values()
                                   if val['done_threshold'] == env.done_threshold]
            if len(errors_current_bond) > 0 and min(errors_current_bond) > env.error:
                torch.save(agent.policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_model.pt")
                torch.save(agent.optim_policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_optim.pt")
            agent.saver.validate_stats(episode_no, 'test')
            # print("Hello")
            return reward, t

def one_episode(episode_no, env, agent, episodes):
    """Function performing full training episode with TPPO."""
    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()
    # agent.saver.stats_file['train'][episode_no]['bond_distance'] = env.current_bond_distance
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold
    
    state = modify_state(state, env, conf, agent.device)
    
    done = False
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        action, log_prob = agent.act(state)
        next_state, reward, done = env.step(agent.translate[action])
        # Convert reward to CPU scalar before storing
        agent.rewards.append(reward.item() if torch.is_tensor(reward) else reward)
        
        agent.saver.stats_file['train'][episode_no]['actions'].append(action)
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['train'][episode_no]['rewards'].append(env.current_reward)
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time() - t0)

        next_state = modify_state(next_state, env, conf, agent.device)
        state = next_state

        # wandb.log({
        #     "train_by_step/step_no": itr,
        #     "train_by_step/episode_no": episode_no,
        #     "train_by_step/errors": env.error,
        #     "train_by_step/errors_noiseless": env.error_noiseless,
        #     "train_by_step/rewards": reward.item() if torch.is_tensor(reward) else reward,
        # })

        if done:
            break

    agent.update()
    agent.saver.validate_stats(episode_no, 'train')
    # wandb.log({
    #     "train_final/episode_len": itr,
    #     "train_final/errors": env.error,
    #     "train_final/errors_noiseless": env.error_noiseless,
    #     "train_final/done_threshold": env.done_threshold,
    #     "train_final/bond_distance": env.current_bond_distance,
    #     "train_final/episode_no": episode_no,
    #     "train_final/current_number_of_cnots": env.current_number_of_cnots
    # })

def train(agent, env, episodes, seed, output_path, threshold):
    """Training loop"""
    threshold_crossed = 0
    for e in range(episodes):
        one_episode(e, env, agent, episodes)
        if e % 20 == 0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_model.pt")
            torch.save(agent.value.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_value_model.pt")
            torch.save(agent.optim_policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_optim.pt")
            torch.save(agent.optim_value.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_value_optim.pt")
        if env.error <= 0.0016:
            threshold_crossed += 1
            #np.save(f'threshold_crossed', threshold_crossed)

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
    results_path ="VQSD/results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)
    # device = torch.device(f"cuda:{args.gpu_id}")
    device = torch.device("cpu")  # Uncomment to force CPU if needed

    conf = get_config(args.experiment_name, f'{args.config}.cfg')

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wandb_project = 'Bench RL-QAS'
    # wandb.login()
    # run = wandb.init(project=wandb_project, config=conf, group=args.wandb_group, name=args.wandb_name)

    """ Environment and Agent initialization """
    environment = CircuitEnv(conf, device=device)

    # Calculate effective state_size based on actual state output
    initial_state = environment.reset()
    base_state_size = initial_state.shape[0]  # Use actual size from reset
    effective_state_size = base_state_size
    if conf['agent']['en_state']:
        effective_state_size += 1  # For prev_energy
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        effective_state_size += 1  # For done_threshold

    # Debugging: Verify state size consistency
    modified_state = modify_state(initial_state, environment, conf, device)
    actual_state_size = modified_state.shape[0]
    # print(f"Environment state_size: {environment.state_size}")
    # print(f"Base state size (from reset): {base_state_size}, Effective state size: {effective_state_size}")
    # print(f"Modified state shape: {modified_state.shape}, Expected size: {effective_state_size}")
    if actual_state_size != effective_state_size:
        raise ValueError(f"State size mismatch: expected {effective_state_size}, got {actual_state_size}")
    if base_state_size != environment.state_size:
        print(f"Warning: environment.state_size ({environment.state_size}) differs from actual state size ({base_state_size})")

    # Initialize agent with the corrected effective state size
    agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](
        conf, environment.action_size, effective_state_size, device
    )
    agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)
    
    # Debug: Verify device placement
    # print(f"Agent policy device: {next(agent.policy.parameters()).device}")
    # print(f"Agent value device: {next(agent.value.parameters()).device}")
    # print(f"Agent action size: {agent.action_size}, Translate dict length: {len(agent.translate)}")

    train(agent, environment, conf['general']['episodes'], args.seed, 
          f"{results_path}{args.experiment_name}{args.config}", conf['env']['accept_err'])
    agent.saver.save_file()
    
    torch.save(agent.policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_model.pt")
    torch.save(agent.value.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_value_model.pt")
    torch.save(agent.optim_policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_optim.pt")
    torch.save(agent.optim_value.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_value_optim.pt")
