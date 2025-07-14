import numpy as np

# Qulacs
from qulacs import ParametricQuantumCircuit, QuantumState, DensityMatrix, circuit
from qulacs.gate import CNOT
from qulacs.gate import *
from qulacs.state import partial_trace
# from qulacsvis import circuit_drawer

from collections import Counter
import json
import copy


# -----------------------------------------------------------------------------

class Parametric_Circuit:
    def __init__(self,n_qubits,noise_models = [],noise_values = []):
        self.n_qubits = n_qubits
        self.noise_models = noise_models
        self.noise_values = noise_values
        self.ansatz = ParametricQuantumCircuit(n_qubits)

    def construct_ansatz(self, state, epsilon=0.01): 
        
        for _, local_state in enumerate(state):
            
            thetas = local_state[self.n_qubits+3:]
            rot_pos = (local_state[self.n_qubits: self.n_qubits+3] == 1).nonzero( as_tuple = True )
            cnot_pos = (local_state[:self.n_qubits] == 1).nonzero( as_tuple = True )
            
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    self.ansatz.add_gate(CNOT(ctrl[r], targ[r]))
            
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
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2     
        
        return self.ansatz


def cost_function_qulacs(angles, circuit, dephasing_circuit, n_qubits, shots, prob_typ, state, purity_before_diag,  which_angles = []):
    
    """"
    Function for Qiskit cost function minimization using Qulacs
    
    Input:
    angles                [array]      : list of trial angles for ansatz
    circuit               [circuit]    : ansatz circuit
    n_qubits              [int]        : number of qubits
    
    Output:
    expval [float] : cost function 
    
    """

    parameter_count_qulacs = circuit.get_parameter_count()
    
    if not list(which_angles):
            which_angles = np.arange(parameter_count_qulacs)
    
    for i, j in enumerate(which_angles):
        circuit.set_parameter(j, angles[i])
    
    if prob_typ == 'pure_state':
        qc_cost = ParametricQuantumCircuit(2*n_qubits)

        for qu in range(n_qubits):
            qc_cost.add_gate(CNOT(qu+n_qubits, qu))

        qc_cost.merge_circuit(circuit) 

        qc = shift_and_merge_circuits(circuit, qc_cost) # custom function that will will first shift qubit range, then merge circuit
        state_for_sampling = QuantumState(qc.get_qubit_count())
        qc.update_quantum_state(state_for_sampling)
        samples = state_for_sampling.sampling(sampling_count=shots)
        counts = from_samples_to_counts(samples) # Custom function to convert sampling data results from qulacs to standard count in qiskit

        for co in counts.keys():
            if co == '0'*2*n_qubits:
                purity_after_diag = counts[co]/shots
        else:
            purity_after_diag = 0


    elif prob_typ in ['mixed_state', 'reduce_heisen_model']:
        
        circuit.update_quantum_state(state)

        state_2n = np.kron(state.get_matrix(), state.get_matrix())

        rho_state_2n = DensityMatrix(2*n_qubits)
        rho_state_2n.load(state_2n) 

        dephasing_circuit.update_quantum_state(rho_state_2n) # Evolve with dephasing circuit  

        par_trace = partial_trace(rho_state_2n, list(range(n_qubits, 2*n_qubits))) # => already tested also, the partial_trace effect of qubit order is the same between qiskit and qulacs

        purity_after_diag = np.trace(par_trace.get_matrix() @ par_trace.get_matrix()).real

    # print("\n\tPurity before diag val:", purity_before_diag)
    # print("\tPurity after diag val:", purity_after_diag)
    # print("\tError:", purity_before_diag - purity_after_diag)

    return purity_before_diag - purity_after_diag

def get_cost(n_qubits, circuit, dephasing_circuit, prob_typ, state, purity_before_diag, shots):

    if prob_typ == 'pure_state':

        qc_cost = ParametricQuantumCircuit(2*n_qubits)
        qc_cost.merge_circuit(circuit) 
        qc = shift_and_merge_circuits(circuit, qc_cost) # custom function that will will first shift qubit range, then merge circuit
        for qu in range(n_qubits):
            qc.add_gate(CNOT(qu+n_qubits, qu))
        
        state_for_sampling = QuantumState(qc.get_qubit_count())
        
        qc.update_quantum_state(state_for_sampling)
        samples = state.sampling(sampling_count=shots)
        counts = from_samples_to_counts(samples) # Custom function to convert sampling data results from qulacs to standard count in qiskit

        for co in counts.keys():
            if co == '0'*2*n_qubits:
                purity_after_diag = counts[co]/shots
        else:
            purity_after_diag = 0

    elif prob_typ in ['mixed_state', 'reduce_heisen_model']:
        
        circuit.update_quantum_state(state)

        state_2n = np.kron(state.get_matrix(), state.get_matrix())

        rho_state_2n = DensityMatrix(2*n_qubits)

        rho_state_2n.load(state_2n)
        dephasing_circuit.update_quantum_state(rho_state_2n)

        par_trace = partial_trace(rho_state_2n, list(range(n_qubits, 2*n_qubits)))
        purity_after_diag = np.trace(par_trace.get_matrix() @ par_trace.get_matrix()).real

    # print("\n\tPurity before diag val:", purity_before_diag)
    # print("\tPurity after diag val:", purity_after_diag)
    # print("\tError:", purity_before_diag - purity_after_diag)


    return purity_before_diag - purity_after_diag


# Custom utility function for Qulacs
def from_samples_to_counts(samples, num_bits=None):
    if num_bits is None:
        num_bits = max(samples).bit_length()
    
    bitstrings = [format(sample, f'0{num_bits}b') for sample in samples]
    counts = dict(Counter(bitstrings))
    
    return counts

def shift_and_merge_circuits(qc1, qc2):
    # qc1: n qubits, qc2: 2n qubits

    # Get Q1 and Q2 JSON
    qc1_json = json.loads(qc1.to_json())
    qc2_json = json.loads(qc2.to_json())
    
    # Copy Q1 for shifting
    qc1_json_shifted = copy.deepcopy(qc1_json)
    
    # Shift qubit n
    n_qubits = int(qc1_json['qubit_count'])
    for i in range(len(qc1_json['gate_list'])):
        shifted_qubit = int(qc1_json['gate_list'][i]['target_qubit']) + n_qubits
        qc1_json_shifted['gate_list'][i]['target_qubit'] = str(shifted_qubit)

    # merge shifted qubit to q2 json
    qc2_json['gate_list'] =  qc2_json['gate_list'] + qc1_json_shifted['gate_list']

    qc2_json_str = json.dumps(qc2_json)
    qc2_merged = circuit.from_json(qc2_json_str)
    
    return qc2_merged



if __name__ == "__main__":
    pass