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
# import wandb
# os.environ['WANDB_DISABLED'] = 'true'
torch.set_num_threads(1)
import json

# SCRIPT TO RUN: 
# python main_A2C.py --seed 1 --config vanilla_cobyla_H24q0p742_noiseless --experiment_name "A2C/"

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
                                                 'rewards' : []
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
        # print("\nSAVE FILE")
        # file_name = f'{self.rpath}/summary_{self.exp_seed}.p'
        # with open(file_name, 'wb') as handle:
        #     pickle.dump(self.stats_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(self.stats_file['train'][0])
        # print(type(self.stats_file['train'][0]))

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
    """NOT YET IMPLEMENTED FOR ACTOR CRITIC ALGORITHM"""
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env)
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
                torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_model.pt")
                torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_optim.pt")
            # agent.epsilon = current_epsilon
            agent.saver.validate_stats(episode_no, 'test')
            
            return reward, t
        

def one_episode(episode_no, env, agent, episodes):
    """ Function preforming full trainig episode."""
    # print("\n # Training Episode", episode_no, "\n")


    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()

    agent.saver.stats_file['train'][episode_no]['bond_distance'] = env.current_bond_distance
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold

    state = modify_state(state, env).double()
    agent.actor.train()
    agent.critic.train()

    states, actions, rewards = [], [], []

    for itr in range(env.num_layers + 1):
        
        # Get Illegal Action
        ill_action_from_env = env.illegal_action_new()

        # Actor selects action
        state_tensor = state.clone().detach().to(dtype=torch.double, device=agent.device)
        action_logits = agent.actor(state_tensor)
        action_logits[ill_action_from_env] = float('-inf') # mask with illegal action
        action = torch.distributions.Categorical(logits=action_logits).sample()

        # Take action and observe next state and reward
        next_state, reward, done = env.step(agent.translate[action.item()])
        next_state = modify_state(next_state, env)

        # Store state, reward, action for each episode
        states.append(state_tensor)
        rewards.append(reward)
        actions.append(action)

        # Update states
        state = next_state.clone()

        assert type(env.error) == float
        agent.saver.stats_file['train'][episode_no]['actions'].append(action.item())
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['train'][episode_no]['rewards'].append(reward.item())
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time()-t0)

        # wandb.log(
        # {"train_by_step/step_no": itr,
        # "train_by_step/episode_no": episode_no,
        # "train_by_step/errors": env.error,
        # "train_by_step/errors_noiseless": env.error_noiseless,
        # "train_by_step/rewards": reward,
        # })
  
        if done:
            if episode_no%20==0:
                print("episode: {}/{}, score: {}, rwd: {} \n"
                        .format(episode_no, episodes, itr, reward),flush=True)
            break 
    
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

    # Get values from critic network
    values = agent.critic(states).squeeze()

    # Calculate and Normalize advantages
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Calculate Log Probs
    action_probs = agent.actor(states)
    log_probs = torch.log_softmax(action_probs, dim=-1)
    log_probs = log_probs[torch.arange(len(actions)), actions]

    # Compute Loss
    total_loss, actor_loss, critic_loss, entropy_loss = agent.compute_loss(
        log_probs, returns, values, advantages
    )

    agent.saver.stats_file['train'][episode_no]['loss_actor'].append(actor_loss.item())
    agent.saver.stats_file['train'][episode_no]['loss_critic'].append(critic_loss.item())
    agent.saver.validate_stats(episode_no, 'train')
    # wandb.log({"train_by_step/loss_actor":actor_loss.item()})
    # wandb.log({"train_by_step/loss_critic":critic_loss.item()})

    
    agent.optim_actor.zero_grad()
    actor_loss.backward()
    agent.optim_actor.step()

    agent.optim_critic.zero_grad()
    critic_loss.backward()
    agent.optim_critic.step()
    
    # wandb.log({
    #     "train_final/episode_len": itr,
    #     "train_final/errors": env.error,
    #     "train_final/errors_noiseless": env.error_noiseless,
    #     "train_final/done_threshold": env.done_threshold,
    #     "train_final/bond_distance": env.current_bond_distance,
    #     "train_final/episode_no": episode_no,
    #     "train_final/current_number_of_cnots": env.current_number_of_cnots,
    #     "train_final/return": sum(returns).item(),
    #     "train_final/actor_loss": actor_loss.item(),
    #     "train_final/critic_loss": critic_loss.item(),
    #     "train_final/entropy_loss": entropy_loss.item(),
    # })

            

def train(agent, env, episodes, seed, output_path,threshold):
    """Training loop"""
    # print("\n . . . --- Training Loop (Total", episodes, "episodes) --- . . .")
    threshold_crossed = 0
    for e in range(episodes):
        
        one_episode(e, env, agent, episodes)
        
        if e % 20==0 and e > 0:
            agent.saver.save_file()
            # torch.save(agent.actor.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_actor_model.pt")
            # torch.save(agent.critic.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_critic_optim.pt")
            # torch.save(agent.optim_actor.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_actor_optim.pt")
            # torch.save(agent.optim_critic.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_critic_optim.pt")

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

    results_path ="VQE/results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)   
    # device = torch.device(f"cuda:{args.gpu_id}")
    device = torch.device(f"cpu:{0}")
    
    # print("Args:", args.experiment_name, args.config)
    conf = get_config(args.experiment_name, f'{args.config}.cfg')
    # print("CONFIG:", conf)
    # print("\nAgent Type:", conf['agent']['agent_type'])
    # print("\nAgent Class:", conf['agent']['agent_class'])

    loss_dict, scores_dict, test_scores_dict, actions_dict = dict(), dict(), dict(), dict()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wandb_project = 'Bench RL-QAS'

    # wandb.login()
    # run = wandb.init(project=wandb_project,
    #                 config=conf,
    #                 group=args.wandb_group,
    #                 name=args.wandb_name,
    #                 settings=wandb.Settings(silent=True))
    

    actions_test = []
    action_test_dict = dict()
    error_test_dict = dict()
    error_noiseless_test_dict=dict()

    
    """ Environment and Agent initialization"""
    environment = CircuitEnv(conf, device=device)
    agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](conf, environment.action_size, environment.state_size, device)
    # Notes: agents -> folder ; A2C -> .py ; agent_class -> Python Class
    agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)

    if conf['agent']['init_net']: 
        # print("\n\tIf conf['agent']['init_net'] == True")
        PATH = f"{results_path}{conf['agent']['init_net']}{args.seed}"
        agent.policy_net.load_state_dict(torch.load(PATH+f"_model.pt"))
        agent.target_net.load_state_dict(torch.load(PATH+f"_model.pt"))
        agent.optim.load_state_dict(torch.load(PATH+f"_optim.pt"))
        agent.policy_net.eval()
        agent.target_net.eval()

        replay_buffer_load = torch.load(f"{PATH}_replay_buffer.pt")
        for i in replay_buffer_load.keys():
            agent.remember(**replay_buffer_load[i])


    train(agent, environment, conf['general']['episodes'], 
          args.seed, f"{results_path}{args.experiment_name}{args.config}",
          conf['env']['accept_err'])
    
    agent.saver.save_file()
            
    # torch.save(agent.actor.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_actor_model.pt")
    # torch.save(agent.critic.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_critic_model.pt")
    # torch.save(agent.optim_actor.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_actor_optim.pt")
    # torch.save(agent.optim_critic.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_critic_optim.pt")

    # wandb.finish()