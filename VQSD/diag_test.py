import pickle, json
import numpy as np
from qulacs import circuit, ParametricQuantumCircuit

with open("state_output.pickle", "rb") as f:
    state = pickle.load(f)

with open("circuit_output.json", "r") as f:
    circuit_json = json.load(f)


print("Initial State:", state)
# print("Circuit JSON Type:", type(circuit_json))
# print(circuit_json)

state_h = (state.get_matrix()+state.get_matrix().conj().T) / 2
print("Eigen value:", np.linalg.eigvals(state_h).real)

# print("State type:", type(state))
# print("State Shape:", state.shape)
# print("Circuit type:", type(circuit_var))
# print("Circuit Shape:", len(circuit_var))

circuit_var = circuit.from_json(circuit_json)
circuit_var.update_quantum_state(state)

state_h = (state.get_matrix()+state.get_matrix().conj().T) / 2

print("Final State:", state)
print("Eigen value:", np.linalg.eigvals(state_h).real)