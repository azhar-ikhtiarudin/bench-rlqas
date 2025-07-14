import numpy as np
from qulacs import ParametricQuantumCircuit, QuantumState, Observable, DensityMatrix, QuantumCircuit
from qulacs.gate import CNOT, RX, RY, RZ, DepolarizingNoise, BitFlipNoise, IndependentXZNoise, DephasingNoise, AmplitudeDampingNoise, TwoQubitDepolarizingNoise
from typing import List, Callable, Optional, Dict
from scipy.optimize import OptimizeResult
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class ParametricVQC:
    def __init__(self, num_qubits, num_samples, noise_models=[], noise_values=[]):
        self.num_qubits = num_qubits
        self.noise_models = noise_models
        self.noise_values = noise_values
        self.ansatz = ParametricQuantumCircuit(num_qubits)
        self.num_samples = num_samples
        
        # Generate synthetic data
        X, y = make_classification(n_samples=self.num_samples, n_features=2, n_informative=2, 
                                  n_redundant=0, n_clusters_per_class=1, random_state=42)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, 2*y-1, test_size=0.2, random_state=42)  # Convert labels to {-1, +1}

    def construct_ansatz(self, state):
        if len(self.noise_models) == 1:
            channels_1 = get_noise_channels(self.noise_models[0], self.num_qubits, self.noise_values[0])
        elif len(self.noise_models) == 2:
            channels_1 = get_noise_channels(self.noise_models[0], self.num_qubits, self.noise_values[0])
        elif len(self.noise_models) == 3:
            channels_1 = get_noise_channels(self.noise_models[0], self.num_qubits, self.noise_values[0])
            channels_3 = get_noise_channels(self.noise_models[2], self.num_qubits, self.noise_values[2])

        for _, local_state in enumerate(state):
            thetas = local_state[self.num_qubits+3:]
            rot_pos = (local_state[self.num_qubits: self.num_qubits+3] == 1).nonzero(as_tuple=True)
            cnot_pos = (local_state[:self.num_qubits] == 1).nonzero(as_tuple=True)
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.add_gate(CNOT(ctrl[r], targ[r]))
                    if len(self.noise_models) >= 2:
                        self.ansatz.add_gate(TwoQubitDepolarizingNoise(ctrl[r], targ[r], self.noise_values[1]))

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
                        self.ansatz.add_parametric_RZ_gate(rot_qubit, thetas[2][rot_qubit])
                    
                    if len(self.noise_values) >= 1 and len(self.noise_values) < 3:
                        self.ansatz.add_gate(channels_1[rot_qubit])
                    elif len(self.noise_values) > 2:
                        self.ansatz.add_gate(channels_1[rot_qubit])
                        self.ansatz.add_gate(channels_3[rot_qubit])

        return self.ansatz
#make loss funcn without angles
def loss_wo_angles(self, circuit, n_shots=0):
    """Calculate classification loss for the current circuit parameters"""
    # Calculate predictions
    observable = Observable(self.num_qubits)
    observable.add_operator(1.0, "Z 0")
    #print("Obs:", observable)
    #exit()
    predictions = []
    for x in self.X_train:
        # Encode features
        feature_circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            feature_circuit.add_gate(RY(i, x[i % len(x)]))
        
        # Combine with variational circuit
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.merge_circuit(feature_circuit)
        full_circuit.merge_circuit(circuit)
        state = QuantumState(self.num_qubits)
        full_circuit.update_quantum_state(state)
        predictions.append(observable.get_expectation_value(state))
    
    # print(len(self.X_train))
    predictions = np.array(predictions)
    loss = np.mean((predictions - self.y_train)**2)
    if n_shots > 0:
        shot_noise = get_shot_noise(np.ones_like(predictions), n_shots)
        loss += shot_noise
    
    predictions_test = []
    for x in self.X_test:
        # Encode features
        feature_circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            feature_circuit.add_gate(RY(i, x[i % len(x)]))
        
        # Combine with variational circuit
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.merge_circuit(feature_circuit)
        full_circuit.merge_circuit(circuit)
        
        if len(self.noise_models) > 0:
            dm = DensityMatrix(self.num_qubits)
            full_circuit.update_quantum_state(dm)
            predictions_test.append(np.real(np.trace(observable.get_matrix() @ dm.get_matrix())))
        else:
            state = QuantumState(self.num_qubits)
            full_circuit.update_quantum_state(state)
            predictions_test.append(observable.get_expectation_value(state))
    
    predictions_test = np.array(predictions_test)
    loss_test = np.mean((predictions_test - self.y_test)**2)
    if n_shots > 0:
        shot_noise = get_shot_noise(np.ones_like(predictions_test), n_shots)  # Using your shot noise function
        loss_test += shot_noise

    return loss, loss_test

def get_classification_loss(self, angles, circuit, n_shots=0, which_angles=[]):
    """Calculate classification loss for given parameters"""
    parameter_count = circuit.get_parameter_count()
    if not list(which_angles):
        which_angles = np.arange(parameter_count)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])

    # Calculate predictions
    observable = Observable(self.num_qubits)
    observable.add_operator(1.0, "Z 0")
    
    predictions = []
    for x in self.X_train:
        # Encode features
        feature_circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            feature_circuit.add_gate(RY(i, x[i % len(x)]))
        
        # Combine with variational circuit
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.merge_circuit(feature_circuit)
        full_circuit.merge_circuit(circuit)
        
        if len(self.noise_models) > 0:
            dm = DensityMatrix(self.num_qubits)
            full_circuit.update_quantum_state(dm)
            predictions.append(np.real(np.trace(observable.get_matrix() @ dm.get_matrix())))
        else:
            state = QuantumState(self.num_qubits)
            full_circuit.update_quantum_state(state)
            predictions.append(observable.get_expectation_value(state))
    
    predictions = np.array(predictions)
    loss = np.mean((predictions - self.y_train)**2)
    if n_shots > 0:
        shot_noise = get_shot_noise(np.ones_like(predictions), n_shots)  # Using your shot noise function
        loss += shot_noise
    
    return loss

def get_accuracy(self, angles, circuit, X, y, which_angles=[]):
    """Calculate accuracy for given parameters"""
    parameter_count = circuit.get_parameter_count()
    if not list(which_angles):
        which_angles = np.arange(parameter_count)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])

    observable = Observable(self.num_qubits)
    observable.add_operator(1.0, "Z 0")
    
    predictions = []
    for x in X:
        # Encode features
        feature_circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            feature_circuit.add_gate(RY(i, x[i % len(x)]))
        
        # Combine with variational circuit
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.merge_circuit(feature_circuit)
        full_circuit.merge_circuit(circuit)
        
        # Get expectation value
        if len(self.noise_models) > 0:
            dm = DensityMatrix(self.num_qubits)
            full_circuit.update_quantum_state(dm)
            pred = np.real(np.trace(observable.get_matrix() @ dm.get_matrix()))
        else:
            state = QuantumState(self.num_qubits)
            full_circuit.update_quantum_state(state)
            pred = observable.get_expectation_value(state)
        predictions.append(np.sign(pred))
    
    return np.mean(np.array(predictions) == y)

def run(self, initial_params=None, optimizer=None, maxiter=100):
    """Run the VQC optimization"""
    if initial_params is None:
        initial_params = np.random.uniform(0, 2*np.pi, self.ansatz.get_parameter_count())
    
    if optimizer is None:
        optimizer = lambda fun, x0: min_spsa(fun, x0, maxiter=maxiter)
    
    # Create loss function with bound circuit
    def loss_fn(params):
        return self.get_classification_loss(params, self.ansatz)
    
    # Run optimization
    result = optimizer(loss_fn, initial_params)
    
    # Get final metrics
    train_acc = self.get_accuracy(result.x, self.ansatz, self.X_train, self.y_train)
    test_acc = self.get_accuracy(result.x, self.ansatz, self.X_test, self.y_test)
    
    return OptimizeResult(
        x=result.x,
        fun=result.fun,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        nit=result.nit,
        nfev=result.nfev
    )
    
def get_shot_noise(weights, n_shots):
    shot_noise = 0
    if n_shots > 0:
        weights1, weights2 = weights[np.abs(weights) > 0.05], weights[np.abs(weights) <= 0.05]
        mu, sigma1, sigma2 = 0, (10*n_shots)**(-0.5), (n_shots)**(-0.5)
        shot_noise += (np.array(weights1).real).T @ np.random.normal(mu, sigma1, len(weights1))
        shot_noise += (np.array(weights2).real).T @ np.random.normal(mu, sigma2, len(weights2))
    return shot_noise

#Optimisation methods
def get_noise_channels(model_name, num_qubits, error_prob):
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

    channels = list(map(fun,range(num_qubits)))
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
    m= 0
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