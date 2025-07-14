import numpy as np
import random
import torch
import sys
import os
import argparse
import pathlib
import copy
from utils import get_config
from environment import CircuitEnv
import agents
import time
import wandb
# os.environ['WANDB_DISABLED'] = 'true'
torch.set_num_threads(1)
import json
import math
from collections import defaultdict
import torch.multiprocessing as mp
from torch.multiprocessing import Array

# SCRIPT TO RUN: 
# python main_A3C.py --seed 1 --config vanilla_cobyla_H24q0p742_noiseless --experiment_name "A3C/"

UPDATE_GLOBAL_ITER = 5

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

# Shared Optimizer
class SharedAdam(torch.optim.Optimizer):
    def __init__(
            self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, 
            weight_decay=0, amsgrad=False
    ):
        defaults = defaultdict(
            lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['max_exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()
    
    def step(self, closure=None):
        """
        Performs a single optimization step
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                amsgrad = group['amsgrad']

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1-beta2)
                step_t = state['step'].item()
                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t

                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(
                                group['eps']
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(
                        group['eps']
                    )
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
            
            return loss


class Saver:
    def __init__(self, results_path, experiment_seed):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode, episode_no):
        if mode == 'train':
            self.stats_file[mode][episode_no] = {'loss_actor': [],
                                                'loss_critic': [],
                                                'actions': [],
                                                'errors': [],
                                                'errors_noiseless':[],
                                                'done_threshold': 0,
                                                'bond_distance': 0,
                                                'nfev': [], 
                                                'opt_ang': [],
                                                'time' : [],
                                                # 'rewards' : [],
                                                # Additional data saving based on the reference code:
                                                'save_circ':[],
                                                'ent_neg': [],
                                                'grad': []
                                                }
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
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
        # print("\nSaving Files . . .")
        # file_name = f'{self.rpath}/summary_{self.exp_seed}.p'
        # with open(file_name, 'wb') as handle:
        #     pickle.dump(self.stats_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(type(self.stats_file['train'][0]))

        with open(f"{self.rpath}/summary_{self.exp_seed}.json", "w") as outfile:
            json.dump(self.stats_file, outfile)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

  

class A3CWorker(mp.Process):
    def __init__(self, global_net, global_opt, 
                global_ep, global_data_dict, 
                global_data_dict_lock, name, args, conf, device):
        super(A3CWorker, self).__init__()

        # breakpoint()
                
        # == Global Model & Optimizer ==
        self.global_net = global_net
        self.global_opt = global_opt

        # == Global Iter Variable ==
        self.global_ep = global_ep
        # self.global_ep_data = global_ep_data
        # self.global_history_queue = global_history_queue
        self.global_data_dict = global_data_dict
        self.global_data_dict_lock = global_data_dict_lock

        # == Config Parameters ==
        self.name = 'w%02i' % name
        self.args = args
        self.conf = conf
        self.seed = args.seed * name
        self.device = device
        self.episodes = conf['general']['episodes']
        self.threshold = conf['env']['accept_err']
        self.output_path = f"results/{args.experiment_name}{args.config}"

        # == Local Model ==
        self.env = CircuitEnv(conf, device=device)
        self.local_net = agents.__dict__[
            conf['agent']['agent_type']].__dict__[
                conf['agent']['agent_class']](
                    conf, self.env.action_size, self.env.state_size, device)
        
        # print(f"Created Local Neural Network Named {self.name} (seed={self.seed*name})")

        
    def run(self):
        
        self.total_step = 0
        threshold_crossed = 0

        # for eps in range(self.episodes):
        while self.global_ep.value < self.episodes:
            
            t0 = time.time()
            state = self.env.reset()

            state = self.modify_state(state, self.env).double()
            states, actions, rewards, error_list, time_list = [], [], [], [], []

            episode_reward = 0
            for itr in range(self.env.num_layers + 1):

                # Get Illegal Action
                ill_action_from_env = self.env.illegal_action_new()

                # Actor selects action
                state_tensor = state.clone().detach().to(dtype=torch.double, device=self.local_net.device)
                action_logits = self.local_net.actor(state_tensor)
                action_logits[ill_action_from_env] = float('-inf') # mask with illegal action
                action = torch.distributions.Categorical(logits=action_logits).sample()

                # Take action and observe next state and reward
                next_state, reward, done = self.env.step(self.local_net.translate[action.item()])
                next_state = self.modify_state(next_state, self.env)

                # print(f'\tWorker_{self.name} | Reward={reward:.3} | Iter={itr}')

                episode_reward += reward

                # Store state, reward, action for each episode
                states.append(state_tensor)
                rewards.append(reward.item())
                actions.append(action.item())
                error_list.append(self.env.error)
                time_list.append(time.time() - t0)

                if self.total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    total_loss = self.push_and_pull(self.global_opt, self.local_net, 
                                    self.global_net, done, next_state,
                                    states, actions, rewards, self.local_net.gamma)
                    
                    # states, actions, rewards = [], [], []
                    # print("Total Step:", self.total_step, "| Global Episode:", self.global_ep.value, "| Local Iter:", itr, "| Worker",self.name[2])
                    # print("Rewards", rewards)
                    # print("Actions", actions)
                    # # print("States", states)
                    # print("Errors", self.env.error)
                    # print("Time", time.time()-t0)
                    # print("Loss", total_loss.item())

                    if done:
                        
                        if self.global_ep.value >= self.episodes:
                            break

                        print(f'DONE: Worker-{self.name} | Global Episode={self.global_ep.value+1} | Total Step {self.name} = {self.total_step+1} | Local Eps Iter={itr+1} | Time Len={len(time_list)} | Error Len={len(error_list)} | Reward Len={len(rewards)} | Actions Len={len(actions)} | States Len={len(states)}')
                        nfev = 0 # TO BE CHANGED

                        with self.global_data_dict_lock:

                            ep_id = str(self.global_ep.value)
                            
                            self.global_ep.value += 1

                            self.global_data_dict[ep_id] = {
                                'rewards': rewards,
                                'actions': actions,
                                'errors': error_list,
                                'time': time_list,
                                'loss': total_loss.item(),
                                'nfev': nfev,
                                'worker': self.name
                            }

                        # print(f"Worker_{self.name[2]} | Global Episode {ep_id}: {self.global_data_dict[ep_id]}")

                        states, actions, rewards, error_list, time_list = [], [], [], [], []
                        break

                # Update states
                state = next_state.clone()
                self.total_step += 1

            # if eps % 20 == 0 and eps > 0:
            #     self.global_net.saver.save_file()
            #     # torch.save(global_net.state_dict(), f"{self.output_path}/thresh_{self.threshold}_{self.seed}_global_net.pt")
            #     # torch.save(global_optimizer.state_dict(), f"{self.output_path}/thresh_{self.threshold}_{self.seed}_global_opt.pt")
            
            if self.env.error <= 0.0016:
                threshold_crossed += 1
                np.save(f'threshold_crossed', threshold_crossed)

    def v_wrap(self, np_array, dtype=np.float64):
        """Helper Function"""
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)
    
    def push_and_pull(self, global_opt, local_net, global_net, done, 
                    next_state, states, actions, rewards, gamma):
        
        # Value from Critic at State s
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=local_net.device)
        rewards = torch.tensor(rewards, dtype=torch.double, device=local_net.device)

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards.tolist()):
            R = r + local_net.final_gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.double, device=local_net.device)

        # Get values from critic network
        values = local_net.critic(states).squeeze()

        # Calculate and Normalize advantages
        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = torch.zeros_like(advantages)

        # Calculate Log Probs
        action_probs = local_net.actor(states)
        log_probs = torch.log_softmax(action_probs, dim=-1)
        log_probs = log_probs[torch.arange(len(actions)), actions]

        # Compute Loss
        total_loss, actor_loss, critic_loss, entropy_loss = local_net.compute_loss(
            log_probs, returns, values, advantages
        )

        # Calculate Local Gradients
        global_opt.zero_grad()
        total_loss.backward()

        # Push Local Parameters to Global
        for local_parameters, global_parameters in zip(local_net.parameters(), global_net.parameters()):
            global_parameters.__grad = local_parameters.grad
        
        global_opt.step()

        # Pull Global Parameters
        local_net.load_state_dict(global_net.state_dict())

        return total_loss


    def modify_state(self, state, env):
        args = get_args(sys.argv[1:])
        conf = get_config(args.experiment_name, f'{args.config}.cfg')   
        if conf['agent']['en_state']:
            state = torch.cat((
                state, 
                torch.tensor(env.prev_energy, dtype=state.dtype, device=self.device).view(1)
            ))
        
        if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
            state = torch.cat((
                state, 
                torch.tensor(env.done_threshold, dtype=state.dtype, device=self.device).view(1)
            ))
        return state



if __name__ == '__main__':

    # == Configs, Seeds, etc ==
    args = get_args(sys.argv[1:])
    results_path ="results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)   

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    # == Environment ==
    conf = get_config(args.experiment_name, f'{args.config}.cfg')
    # device = torch.device(f'cpu:{0}')
    device = torch.device(f"cuda:{args.gpu_id}")
    env = CircuitEnv(conf, device=device)


    # == Number of Parallel Training Processes ==
    # num_processes = mp.cpu_count() # Set based on the Number of CPUs
    num_processes =  conf['env']['num_workers'] # Number of Workers set from Hyperparameter
    mp.set_start_method('spawn')
    print(f"\nNumber of Parallel Processes (Workers): {num_processes}\n")


    # == Define Global Shared Model and Optimizer ==
    global_net = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](conf, env.action_size, env.state_size, device)
    global_net.share_memory()
    global_net.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)

    global_optimizer = SharedAdam(
        global_net.parameters(), 
        lr=conf['agent']['learning_rate'], betas=(0.92, 0.99)
    )


    with mp.Manager() as manager:
        # == Global Iteration Variable ==
        global_data_dict = manager.dict() # start empty dictionary
        global_ep = manager.Value('i', 0)
        global_data_dict_lock = manager.Lock()

        workers = []

        for i in range(num_processes):
            worker = A3CWorker(global_net, global_optimizer,
                                global_ep, global_data_dict,
                                global_data_dict_lock, i, args, conf, device)
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        global_data_dict_file = dict(global_data_dict.items())
        # print("Global Data Dict", global_data_dict_file)

    filename = f'{results_path}{args.experiment_name}{args.config}.json'

    with open(filename, 'w') as json_file:
        json.dump(global_data_dict_file, json_file)
    
    print(f'\nData saved to {filename}')

    # === Data Type Results Format: ===
    #
    # data_type = {
    #     '0': {                        -> global episode
    #         'rewards':[],             -> rewards in episode-0
    #         'actions':[],             -> actions in episode-0
    #         'errors':[],
    #         'time':[],
    #         'loss':float,
    #         'nfev':,
    #         'worker':string,
    #     },
    #     '1':{
    #         . . .
    #     },
    #       . . . 
    #     "N_EPS-1":{
    #   }
    # }