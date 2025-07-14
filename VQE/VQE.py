from qulacs import ParametricQuantumCircuit, QuantumState, DensityMatrix
from qulacs.gate import CNOT
from qulacs.gate import *

import numpy as np
from typing import List, Callable, Optional, Dict

from scipy.optimize import OptimizeResult





class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.noise_models = noise_models
        self.noise_values = noise_values
        self.ansatz = ParametricQuantumCircuit(n_qubits)

        

    def construct_ansatz(self, state):
        
        if len(self.noise_models) == 1:
            
            channels_1 = get_noise_channels(self.noise_models[0], self.n_qubits, self.noise_values[0])
        elif len(self.noise_models) == 2:
            channels_1 = get_noise_channels(self.noise_models[0],self.n_qubits,self.noise_values[0])
        elif len(self.noise_models) == 3:
            channels_1 = get_noise_channels(self.noise_models[0],self.n_qubits,self.noise_values[0])
            channels_3 = get_noise_channels(self.noise_models[2],self.n_qubits,self.noise_values[2])
        

        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.add_gate(CNOT(ctrl[r], targ[r]))

                    if len(self.noise_models) >= 2:
                        
                        self.ansatz.add_gate(TwoQubitDepolarizingNoise(ctrl[r], targ[r], self.noise_values[1]) )
            
            rot_direction_list = rot_pos[0]
            rot_qubit_list = rot_pos[1]


            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    
                    if r == 0:
                        self.ansatz.add_parametric_RX_gate(rot_qubit, thetas[0][rot_qubit])
                    elif r == 1:
                        self.ansatz.add_parametric_RY_gate(rot_qubit, thetas[1][rot_qubit])
                    elif r == 2:
                        self.ansatz.add_parametric_RZ_gate(rot_qubit,  thetas[2][rot_qubit])
                    else:
                        print('baler angle! lagabo na!')
                    
                    if len(self.noise_values) >= 1 and len(self.noise_values) <3 :
                        self.ansatz.add_gate(channels_1[rot_qubit])
                    elif len(self.noise_values) > 2:
                        self.ansatz.add_gate(channels_1[rot_qubit])
                        self.ansatz.add_gate(channels_3[rot_qubit])
                        

        
        return self.ansatz

        

def get_energy_qulacs_(angles, observable, 
                      weights,circuit, n_qubits, 
                      energy_shift, n_shots,
                      noise_value,
                      M=1000,
                      which_angles=[]):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    noise_value           [float]      : Physical error probability of the quantum channel; e.g, depolarizing etc
    n_shots               [int]        : Statistical noise, number of samples taken from QC
    M                     [int]        : Physical noise, number of repetitions during expectation value calculation (Q.trajectories)
    
    Output:
    expval [float] : expectation value 
    
    """
        
    parameter_count_qulacs = circuit.get_parameter_count()
    
    if not list(which_angles):
            which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])
        



    

    if noise_value == 0:
        M=1        

    expval = get_exp_val(n_qubits,circuit,observable,M)
    
    shot_noise = 0
    
    if n_shots > 0:
        weights1, weights2 = weights[np.abs(weights) > 0.05], weights[np.abs(weights) <= 0.05]
        mu,sigma1,sigma2=0,(10*n_shots)**(-0.5), (n_shots)**(-0.5)
        
        shot_noise+=(np.array(weights1).real).T@np.random.normal(mu,sigma1,len(weights1))
        shot_noise+=(np.array(weights2).real).T@np.random.normal(mu,sigma2,len(weights2))

    return expval + shot_noise + energy_shift


def get_energy_qulacs(angles, observable, 
                      weights,circuit, n_qubits, 
                      energy_shift, n_shots,
                      phys_noise = False,
                      which_angles=[]):
    """"
    Function for Qiskit energy minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    observable            [Observable] : Qulacs observable (Hamiltonian)
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    energy_shift          [float]      : energy shift for Qiskit Hamiltonian after freezing+removing orbitals
    n_shots               [int]        : Statistical noise, number of samples taken from QC
    phys_noise            [bool]       : Whether quantum error channels are available (DM simulation) 
    
    Output:
    expval [float] : expectation value 
    
    """
        
    parameter_count_qulacs = circuit.get_parameter_count()
    
    if not list(which_angles):
            which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])
          

    expval = get_exp_val(n_qubits,circuit,observable, phys_noise)
    
    shot_noise = get_shot_noise(weights, n_shots) 

    return expval + shot_noise + energy_shift

def get_shot_noise(weights, n_shots):
    
    shot_noise = 0
    
    if n_shots > 0:
        weights1, weights2 = weights[np.abs(weights) > 0.05], weights[np.abs(weights) <= 0.05]
        mu,sigma1,sigma2 =0,(10*n_shots)**(-0.5), (n_shots)**(-0.5)
        
        shot_noise +=(np.array(weights1).real).T@np.random.normal(mu,sigma1,len(weights1))
        shot_noise +=(np.array(weights2).real).T@np.random.normal(mu,sigma2,len(weights2))
        
    return shot_noise










        


def get_exp_val(n_qubits,circuit,op, phys_noise = False, err_mitig = 0):
    
    expval = 0
    if phys_noise == False:
        state = QuantumState(n_qubits)
        circuit.update_quantum_state(state)
        psi = state.get_vector()
        expval += (np.conj(psi).T @ op @ psi).real
    else:
        dm = DensityMatrix(n_qubits)
        circuit.update_quantum_state(dm)
        rho = dm.get_matrix()
        if err_mitig == 0:
            expval += np.real( np.trace(op @ rho) )
        else:
            expval += np.real( np.trace(op @ rho @ rho) / np.trace(rho @ rho))
        
    return expval

def get_noise_channels(model_name, n_qubits, error_prob):
    if model_name == "depolarizing":
        noise_model = DepolarizingNoise
    elif model_name == 'bitflip':
        noise_model = BitFlipNoise
    elif model_name == 'XZ':
        noise_model = IndependentXZNoise
    elif model_name =='dephasing':
        noise_model = DephasingNoise
    elif model_name == 'amplitude_damping':
        noise_model = AmplitudeDampingNoise
    elif model_name == 'two_depolarizing':
        noise_model = TwoQubitDepolarizingNoise
        
    fun = lambda x: noise_model(x,error_prob)

    channels = list(map(fun,range(n_qubits)))
    return channels




def min_spsa(
    fun: Callable,
    x0: List[float],
    maxfev: int = 10000,
    maxiter: Optional[int] = None,
    a: float = 1.0,
    alpha: float = 0.602,
    c: float = 1.0,
    gamma: float = 0.101
    )-> OptimizeResult:
    
    
    current_params = np.asarray(x0)
    
    n_params = len(current_params)
    
    A = 0.05 * maxfev
    
    if maxiter is None:
        maxiter = int(np.ceil(maxfev / 2))
        
    n_fevals = 0
    
    best_params = current_params 
    
    best_feval = fun(current_params)
    
    FE_best = 0
    
    
    
    
    
    
    for epoch in range(maxiter):
        
        ak = spsa_lr_dec(epoch, a, A, alpha)
        ck = spsa_grad_dec(epoch, c, gamma)
        
        grad = spsa_grad(fun, current_params, n_params, ck)
        
        n_fevals += 2 
        
        current_params -= ak * grad

        current_feval = fun(current_params)
        
        

        if current_feval < best_feval:
            best_feval = current_feval
            best_params = np.array(current_params)
            FE_best = n_fevals 
        
    return OptimizeResult(fun=best_feval,
                              x=best_params,
                              FE_best=FE_best,
                              nit=epoch,
                              nfev=n_fevals)


def min_adam_spsa(
    fun: Callable,
    x0: List[float],
    maxfev: int = 10000,
    maxiter: Optional[int] = None,
    a: float = 1.0,
    alpha: float = 0.602,
    c: float = 1.0,
    gamma: float = 0.101,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8
    )-> OptimizeResult:
    
    
    current_params = np.asarray(x0)
    
    n_params = len(current_params)
    
    A = 0.05 * maxfev
    
    if maxiter is None:
        maxiter = int(np.ceil(maxfev / 2))
        
    n_fevals = 0
    
    best_params = current_params 
    
    best_feval = fun(current_params)
    
    FE_best = 0
    
    m = 0
    
    v = 0
    
    
    
    
    
    for epoch in range(maxiter):
        
        ak = spsa_lr_dec(epoch, a, A, alpha)
        ck = spsa_grad_dec(epoch, c, gamma)
        
        grad = spsa_grad(fun, current_params, n_params, ck)
        
        a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1, beta_2, epsilon)
        
        
        
        
        n_fevals += 2 
        
        current_params -= ak * a_grad

        current_feval = fun(current_params)
        
        
        
        
        

        if current_feval < best_feval:
            best_feval = current_feval
            best_params = np.array(current_params)
            FE_best = n_fevals 
        
    return OptimizeResult(fun=best_feval,
                              x=best_params,
                              FE_best=FE_best,
                              nit=epoch,
                              nfev=n_fevals)

def min_adam_spsa3(
    fun1: Callable,
    fun2: Callable,
    fun3: Callable,
    x0: List[float],
    maxfev1: int = 3000,
    maxfev2: int = 2000,
    maxfev3: int = 1000,
    a: float = 1.0,
    alpha: float = 0.602,
    c: float = 1.0,
    gamma: float = 0.101,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8
    )-> OptimizeResult:
    
    
    current_params = np.asarray(x0)
    
    n_params = len(current_params)
    
    maxiter1 = int(np.ceil(maxfev1 / 2))
    
    maxiter2 = int(np.ceil(maxfev2 / 2))
    
    maxiter3 = int(np.ceil(maxfev3 / 2))
    
    maxiter = maxiter1 + maxiter2 + maxiter3
    
    A = 0.01 * maxiter
        
    n_fevals = 0
    
    best_params = current_params 
    
    best_feval = fun1(current_params)
    
    FE_best = 0
    
    m = 0
    
    v = 0
    
    
    
    
    
    for epoch in range(maxiter):
        
        ak = spsa_lr_dec(epoch, a, A, alpha)
        ck = spsa_grad_dec(epoch, c, gamma)
        
        if epoch < maxiter1:
            fun = fun1
        elif epoch >= maxiter1 and epoch < (maxiter1 + maxiter2):
            fun = fun2
        elif epoch >= (maxiter1 + maxiter2) and epoch < maxiter:
            fun = fun3
        
        grad = spsa_grad(fun, current_params, n_params, ck)
        
        
        
        
        
        
        n_fevals += 2 
        
        if epoch < maxiter - 20:
            a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1, beta_2, epsilon)
            
            current_params -= ak * a_grad
            
        else:
            current_params -= ak * grad

        current_feval = fun(current_params)
        
        
        
        
        

        if current_feval < best_feval:
            best_feval = current_feval
            best_params = np.array(current_params)
            FE_best = n_fevals 
        
    return OptimizeResult(fun=best_feval,
                              x=best_params,
                              FE_best=FE_best,
                              nit=epoch,
                              nfev=n_fevals)
        
def spsa_lr_dec(epoch, a, A, alpha):
    
    ak = a / (epoch + 1.0 + A) ** alpha

    return ak

def spsa_grad_dec(epoch, c, gamma):
    ck = c / (epoch + 1.0) ** gamma
    return ck

def spsa_grad(fun, current_params, n_params, ck):
    
    n_params = len(current_params)
    
    
    
    
    Deltak = np.random.choice([-1, 1], size=n_params)
    
    grad = ((fun(current_params + ck * Deltak) -
                     fun(current_params - ck * Deltak)) /
                    (2 * ck * Deltak))
    
    return grad


def adam_grad(epoch, grad, m, v, beta_1, beta_2, epsilon):
    
    m = beta_1 * m + (1 - beta_1) * grad
    v = beta_2 * v + (1 - beta_2) * np.power(grad, 2)
    m_hat = m / (1 - np.power(beta_1, epoch + 1))
    v_hat = v / (1 - np.power(beta_2, epoch + 1))
    
    return m_hat / (np.sqrt(v_hat) + epsilon), m, v

def min_spsa3_v2(
    fun1: Callable,
    fun2: Callable,
    fun3: Callable,
    x0: List[float],
    maxfev1: int = 2383,
    maxfev2: int = 715,
    maxfev3: int = 238,
    a: float = 1.0,
    alpha: float = 0.602,
    c: float = 1.0,
    gamma: float = 0.101,
    beta_1: float = 0.999,
    beta_2: float = 0.999,
    lamda: float = 0.4,
    epsilon: float = 1e-8,
        adam: bool = True,
    rglr: bool = False
    )-> OptimizeResult:
    
    
    current_params = np.asarray(x0)
    
    n_params = len(current_params)
    

    
    maxiter1 = int(np.ceil(maxfev1 / 2))
    
    maxiter2 = int(np.ceil(maxfev2 / 2))
    
    maxiter3 = int(np.ceil(maxfev3 / 2))
    
    maxiter = maxiter1 + maxiter2 + maxiter3
    
    A = 0.01 * maxiter
        
    n_fevals = 0
    
    best_params = current_params 
    
    best_feval = fun1(current_params)
    
    FE_best = 0
    
    m = 0
    
    v = 0
    
    
    
    
    
    epoch_ctr = 0
    
    for epoch in range(maxiter):
        
        ak = spsa_lr_dec_new(epoch_ctr, a, alpha)
        ck = spsa_grad_dec(epoch_ctr, c, gamma)
        
        
        
        
        if epoch < maxiter1:
            fun = fun1
        elif epoch >= maxiter1 and epoch < (maxiter1 + maxiter2):
            fun = fun2
            
            if rglr:
                epoch_ctr = 0
                m = 0
                v = 0
                
        elif epoch >= (maxiter1 + maxiter2) and epoch < maxiter:
            fun = fun3
                        
            if rglr:
                epoch_ctr = 0
                m = 0
                v = 0
        
        grad = spsa_grad(fun, current_params, n_params, ck)
        
        if adam:
            beta_1t = beta_1_t(epoch_ctr, beta_1, lamda)
            a_grad, m, v = adam_grad(epoch_ctr, grad, m, v, beta_1t, beta_2, epsilon)            
            
            
        else:
            a_grad = grad
            
            
        current_params -= ak * a_grad
        
        
        
        
        
        n_fevals += 2 
        


            

            



        current_feval = fun(current_params)
        
        
        
        
        

        if current_feval < best_feval:
            best_feval = current_feval
            best_params = np.array(current_params)
            FE_best = n_fevals
            
        epoch_ctr += 1
        
    return OptimizeResult(fun=best_feval,
                              x=best_params,
                              FE_best=FE_best,
                              nit=epoch,
                              nfev=n_fevals)


def min_spsa_v2(
    fun: Callable,
    x0: List[float],
    maxfev: int = 10000,
    maxiter: Optional[int] = None,
    a: float = 1.0,
    alpha: float = 0.602,
    c: float = 1.0,
    gamma: float = 0.101,
    lamda: float = 0.4,
    beta_1: float = 0.999,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    adam: bool = True)-> OptimizeResult:
    
    
    current_params = np.asarray(x0)
    
    n_params = len(current_params)
    
    
    
    if maxiter is None:
        maxiter = int(np.ceil(maxfev / 2))
        
    n_fevals = 0
    
    best_params = current_params 
    
    best_feval = fun(current_params)
    
    FE_best = 0
    
    m = 0
    
    v = 0
    
    
    
    
    
    for epoch in range(maxiter):
        
        ak = spsa_lr_dec_new(epoch, a, alpha)
        ck = spsa_grad_dec(epoch, c, gamma)
        
        grad = spsa_grad(fun, current_params, n_params, ck)
        
        if adam:
            if epoch > 0:
                beta_1t = beta_1_t(epoch, beta_1, lamda)
                a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1t, beta_2, epsilon)
            else:
                a_grad = grad
                
        else:
            a_grad = grad
        
        
        
        
        n_fevals += 2 
        
        current_params -= ak * a_grad

        current_feval = fun(current_params)
        
        
        
        
        

        if current_feval < best_feval:
            best_feval = current_feval
            best_params = np.array(current_params)
            FE_best = n_fevals 
        
    return OptimizeResult(fun=best_feval,
                              x=best_params,
                              FE_best=FE_best,
                              nit=epoch,
                              nfev=n_fevals)

def spsa_lr_dec_new(epoch, a, alpha = 0.602):
    ak = a / (epoch + 1.0 ) ** alpha
    return ak
    
def spsa_grad_dec_new(epoch, c, gamma = 0.101):
    ck = c / (epoch + 1.0) ** gamma
    return ck
    
def beta_1_t(epoch, beta_1_0, lamda):
    beta_1_t = beta_1_0 / (epoch + 1)**lamda
    return beta_1_t

def min_spsa_n_v2(
    fun: Callable,
    x0: List[float],
    maxfev: int = 10000,
    maxiter: Optional[int] = None,
    a: float = 1.0,
    alpha: float = 0.602,
    c: float = 1.0,
    gamma: float = 0.101,
    lamda: float = 0.4,
    beta_1: float = 0.999,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    adam: bool = True)-> OptimizeResult:
    
    
    current_params = np.asarray(x0)
    
    n_params = len(current_params)
    
    
    
    if maxiter is None:
        maxiter = int(np.ceil(maxfev / 2))
        
    n_fevals = 0
    
    best_params = current_params 
    
    best_feval = fun(current_params)
    
    FE_best = 0
    
    m = 0
    
    v = 0
    
    
    
    
    
    for epoch in range(maxiter):
        
        ak = spsa_lr_dec_new(epoch, a, alpha)
        ck = spsa_grad_dec(epoch, c, gamma)
        
        grad = spsa_grad(fun, current_params, n_params, ck)
        
        if adam:
            if epoch > 0:
                beta_1t = beta_1_t(epoch, beta_1, lamda)
                a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1t, beta_2, epsilon)
            else:
                a_grad = grad
                
        else:
            a_grad = grad
        
        
        
        
        n_fevals += 2 
        
        current_params -= ak * a_grad

        current_feval = fun(current_params)
        
        
        
        
        

        if current_feval < best_feval:
            best_feval = current_feval
            best_params = np.array(current_params)
            FE_best = n_fevals 
        
    return OptimizeResult(fun=best_feval,
                              x=best_params,
                              FE_best=FE_best,
                              nit=epoch,
                              nfev=n_fevals)



if __name__ == "__main__":
    pass


















