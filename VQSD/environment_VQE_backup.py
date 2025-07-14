

import torch
from qulacs.gate import CNOT, RX, RY, RZ
from utils import *
from sys import stdout
import scipy
# import VQE as vc
import VQSD as vc
import os
import numpy as np
import copy
import curricula
from collections import Counter
try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from qulacs import ParametricQuantumCircuit

import copy
# import time

class CircuitEnv():

    def __init__(self, conf, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        self.random_halt = int(conf['env']['rand_halt'])
        
        
        self.n_shots =   conf['env']['n_shots'] 
        noise_models = ['depolarizing', 'two_depolarizing', 'amplitude_damping']
        noise_values = conf['env']['noise_values'] 
        
        self.mol = conf['problem']['ham_type']

        if noise_values !=0:
            indx = conf['env']['noise_values'].index(',')
            self.noise_values = [float(noise_values[1:indx]), float(noise_values[indx+1:-1])]
        else:
            self.noise_values = []
        self.noise_models = noise_models[0:len(self.noise_values)]
        
        if len(self.noise_models) > 0:
            self.phys_noise = True
        else:
            self.phys_noise = False

        self.err_mitig = conf['env']['err_mitig']
        

        self.ham_mapping = conf['problem']['mapping']
        self.geometry = conf['problem']['geometry'].replace(" ", "_")

        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type']
        
        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.
        
        
        self.noise_flag = True
        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0
        

        self.current_reward = 0
        
        self.curriculum_dict = {}
        __ham = np.load(f"mol_data/{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        # print(f"mol_data/{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        
        
        _, _, eigvals, energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']

        min_eig = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else min(eigvals) + energy_shift
        
        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals)+self.energy_shift

        # print("np.sum(weights)+energy shift ->", np.sum(self.weights)+self.energy_shift)
        # exit()

        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=min_eig)
     
        self.device = device
        self.ket = QuantumState(self.num_qubits)
        self.done_threshold = conf['env']['accept_err']
        

        stdout.flush()
        self.state_size = self.num_layers*self.num_qubits*(self.num_qubits+3+3)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0]*self.num_qubits
        self.illegal_actions = [[]]*self.num_qubits
        self.energy = 0
        self.opt_alg_save = 0

        self.action_size = (self.num_qubits*(self.num_qubits+2))
        self.previous_action = [0, 0, 0, 0]   

        if 'non_local_opt' in conf.keys():
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]
            self.optim_alg = conf['non_local_opt']['optim_alg']
            

            if 'a' in conf['non_local_opt'].keys():
                self.options = {'a': conf['non_local_opt']["a"], 'alpha': conf['non_local_opt']["alpha"],
                            'c': conf['non_local_opt']["c"], 'gamma': conf['non_local_opt']["gamma"],
                            'beta_1': conf['non_local_opt']["beta_1"],
                            'beta_2': conf['non_local_opt']["beta_2"]}

            if 'lamda' in conf['non_local_opt'].keys():
                self.options['lamda'] = conf['non_local_opt']["lamda"]

            if 'maxfev' in conf['non_local_opt'].keys():
                self.maxfev = {}
                self.maxfev['maxfev'] = int(conf['non_local_opt']["maxfev"])

            if 'maxfev1' in conf['non_local_opt'].keys():
                self.maxfevs = {}
                self.maxfevs['maxfev1'] = int( conf['non_local_opt']["maxfev1"] )
                self.maxfevs['maxfev2'] = int( conf['non_local_opt']["maxfev2"] )
                self.maxfevs['maxfev3'] = int( conf['non_local_opt']["maxfev3"] )
                
        else:
            self.global_iters = 0
            self.optim_method = None
        
            



    def step(self, action, train_flag = True) :
        
        """
        Action is performed on the first empty layer.
        
        
        Variable 'step_counter' points last non-empty layer.
        """  
        
        next_state = self.state.clone()
        
        self.step_counter += 1

        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """
        
        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis = action[3]
        
        
        

        
        self.action = action

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[ rot_qubit ]
        elif ctrl < self.num_qubits:
            gate_tensor = max( self.moments[ctrl], self.moments[targ] )

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1

        if rot_qubit < self.num_qubits:
            self.moments[ rot_qubit ] += 1
        elif ctrl < self.num_qubits:
            max_of_two_moments = max( self.moments[ctrl], self.moments[targ] )
            self.moments[ctrl] = max_of_two_moments +1
            self.moments[targ] = max_of_two_moments +1
            
        
        self.current_action = action
        self.illegal_action_new()

        if self.optim_method in ["scipy_each_step"]:
            thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits+3+j,:] = thetas[i][j,:]
        
            

        self.state = next_state.clone()
        
        energy,energy_noiseless = self.get_energy()

        if self.noise_flag == False:
            energy = energy_noiseless

        self.energy = energy
        
        
        
        
        
        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)
    
        self.error = float(abs(self.min_eig-energy))
        self.error_noiseless = float(abs(self.min_eig-energy_noiseless))

        
        rwd = self.reward_fn(energy)
        self.current_reward = float(rwd)
        self.prev_energy = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)

        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)
        
        self.previous_action = copy.deepcopy(action)
        

        if self.random_halt:
            if self.step_counter == self.halting_step:
                done = 1
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)
        
        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits+3]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        
        state = torch.zeros((self.num_layers, self.num_qubits+3+3, self.num_qubits))
        self.state = state
        
        
        if self.random_halt:
            statistics_generated = np.clip(np.random.negative_binomial(n=70,p=0.573, size=1),25,70)[0]
            
            self.halting_step = statistics_generated

        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*4
        self.illegal_actions = [[]]*self.num_qubits

        self.make_circuit(state)
        self.step_counter = -1

        
        self.moments = [0]*self.num_qubits
        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.geometry = self.geometry[:-3] + str(self.current_bond_distance)
        __ham = np.load(f"mol_data/{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        self.hamiltonian, self.weights,eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals)+self.energy_shift
        self.prev_energy = self.get_energy(state)[1]

        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        else:
            state = state[:, :self.num_qubits+3]
            return state.reshape(-1).to(self.device)

    def make_circuit(self, thetas=None):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits+3:]
        
        circuit = ParametricQuantumCircuit(self.num_qubits)
        
        for i in range(self.num_layers):
            
            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.add_gate(CNOT(ctrl[r], targ[r]))
                    
            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits+3] == 1)
            
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.add_parametric_RX_gate(rot_qubit, thetas[i][0][rot_qubit]) 
                    elif r == 1:
                        circuit.add_parametric_RY_gate(rot_qubit, thetas[i][1][rot_qubit])
                    elif r == 2:
                        circuit.add_parametric_RZ_gate(rot_qubit, thetas[i][2][rot_qubit])
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        
                        assert r >2
        
        return circuit

    def R_gate(self, qubit, axis, angle):
        if axis == 'X' or axis == 'x' or axis == 1:
            return RX(qubit, angle)
        elif axis == 'Y' or axis == 'y' or axis == 2:
            return RY(qubit, angle)
        elif axis == 'Z' or axis == 'z' or axis == 3:
            return RZ(qubit, angle)
        else:
            print("Wrong gate")
            return 1


    def get_energy(self, thetas=None):
        
        circ = self.make_circuit(thetas)
        
        qulacs_inst = vc.Parametric_Circuit(n_qubits = self.num_qubits, noise_models = self.noise_models, noise_values = self.noise_values)
        
        noisy_circ = qulacs_inst.construct_ansatz(self.state)
         
        expval_noisy = vc.get_exp_val(self.num_qubits,noisy_circ,self.hamiltonian,self.phys_noise,self.err_mitig)
        
        expval_noiseless = vc.get_exp_val(self.num_qubits,circ,self.hamiltonian)
        
        

        shot_noise = vc.get_shot_noise(self.weights, self.n_shots)        
            
        energy = expval_noisy + shot_noise + self.energy_shift
        
        energy_noiseless = expval_noiseless  + self.energy_shift
        
        return energy, energy_noiseless
        
    def scipy_optim(self, method, which_angles = [] ):
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:,self.num_qubits: self.num_qubits+3] == 1).nonzero( as_tuple = True )
        angles = thetas[rot_pos]

        qulacs_inst = vc.Parametric_Circuit(n_qubits=self.num_qubits,noise_models=self.noise_models,noise_values = self.noise_values)
        qulacs_circuit = qulacs_inst.construct_ansatz(state)
        
        x0 = np.asarray(angles.cpu().detach())

        def cost(x):
            return vc.get_energy_qulacs(x, observable = self.hamiltonian,
            weights = self.weights, circuit = qulacs_circuit,
            n_qubits = self.num_qubits, energy_shift = self.energy_shift,
            n_shots = int(self.n_shots), phys_noise = self.phys_noise,
                      which_angles=[])

        if list(which_angles):
            result_min_qulacs = scipy.optimize.minimize(cost, x0 = x0[which_angles], method = method, options = {'maxiter':self.global_iters})
            x0[which_angles] = result_min_qulacs['x']
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(x0, dtype=torch.float)
        else:
            result_min_qulacs = scipy.optimize.minimize(cost, x0 = x0, method = method, options = {'maxiter':self.global_iters})
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(result_min_qulacs['x'], dtype=torch.float)

        return thetas, result_min_qulacs['nfev'], result_min_qulacs['x']

    def reward_fn(self, energy):
        
        
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
        elif self.fn_type == "two_step_end":

            max_depth = self.step_counter == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold))/1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1.*(self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy)/abs(self.prev_energy - self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
        elif self.fn_type == "nive_fives":

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd
        
        elif self.fn_type == "incremental_with_fixed_ends":
            

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            # print(self.num_layers-1, self.step_counter, max_depth, self.error, rwd)
            return rwd
        
        elif self.fn_type == "log":
            return -np.log(1-(energy/self.min_eig))
        
        elif self.fn_type == "log_to_ground":
            
            return -np.log(self.error)
        
        elif self.fn_type == "log_to_threshold":
            if self.error < self.done_threshold + 1e-5:
                rwd = 11
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_0_end":
            rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_50_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 50
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_100_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 100
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_500_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 100
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
	
        elif self.fn_type == "log_to_threshold_500_end":
                if self.error < self.done_threshold + 1e-5:
                    rwd = 500
                else:
                    rwd = -np.log(abs(self.error - self.done_threshold))
                return rwd
        
        elif self.fn_type == "log_to_threshold_1000_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 1000
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_non_repeat_energy":
            if self.error < self.done_threshold + 1e-5:
                rwd = 30
            elif np.abs(self.energy-self.prev_energy) <= 1e-3:
                rwd = -30
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_to_threshold_bigger_end_no_repeat_actions":
            if self.current_action == self.previous_action:
                return -1 
            elif self.error < self.done_threshold + 1e-5:
                rwd = 20
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd
        
        elif self.fn_type == "log_neg_punish":
            return -np.log(1-(energy/self.min_eig)) - 5
        
        elif self.fn_type == "end_energy":

            max_depth = self.step_counter == (self.num_layers - 1)
            
            if ((self.error < self.done_threshold) or max_depth):
                rwd = (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            else:
                rwd = 0.0

        elif self.fn_type == "hybrid_reward":
            path = 'threshold_crossed.npy'
            if os.path.exists(path):
                
                threshold_pass_info = np.load(path)
                if threshold_pass_info > 8:

                    max_depth = self.step_counter == (self.num_layers - 1)
                    if (self.error < self.done_threshold):
                        rwd = 5.
                    elif max_depth:
                        rwd = -5.
                    else:
                        rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
                    return rwd
                else:
                    if self.error < self.done_threshold + 1e-5:
                        rwd = 11
                    else:
                        rwd = -np.log(abs(self.error - self.done_threshold))
                    return rwd
            else:
                np.save('threshold_crossed.npy', 0)
        
        elif self.fn_type == 'negative_above_chem_acc':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 1000*(self.done_threshold/self.error)
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_non_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = self.done_threshold/self.error
            return rwd
        
        elif self.fn_type == 'negative_above_chem_acc_slight_increment':
            if self.error > self.done_threshold:
                rwd = - (self.error/self.done_threshold)
            elif self.error == self.done_threshold:
                rwd = (self.error/self.done_threshold)
            else:
                rwd = 100*(self.done_threshold/self.error)
            return rwd


        elif self.fn_type == "cnot_reduce":

            max_depth = self.step_counter == (self.num_layers - 1)
            
            
            if (self.error < self.done_threshold):
                rwd = self.num_layers - self.cnot_rwd_weight*self.current_number_of_cnots
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy)/abs(self.prev_energy - self.min_eig),-1,1)
            return 

        
    def illegal_action_new(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]

        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action

                            
        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[0] == self.num_qubits:
                            
                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                                        
                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break 
            else:
                illegal_action[0] = action
        
        for indx in range(self.num_qubits):
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode




if __name__ == "__main__":
    pass