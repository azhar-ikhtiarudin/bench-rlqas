import networkx as nx
import random
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT, SWAP

n_qubits = 5
n_seeds = 10
# Number of edges per new node in BA graph
m = 2 

def generate_ba_graphs(n, m, seeds):
    graphs = []
    for seed in seeds:
        random.seed(seed)
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        graphs.append(G)
    return graphs

seeds = range(n_seeds)
ba_graphs = generate_ba_graphs(n_qubits, m, seeds)

def is_allowed_cnot(control, target, coupling_map):
    return (control, target) in coupling_map or (target, control) in coupling_map

def adjust_circuit_for_coupling(coupling_map, gates):
    circuit = QuantumCircuit(n_qubits)
    current_layout = list(range(n_qubits))

    for q in range(n_qubits):
        circuit.add_gate(H(q))

    for control, target in gates:
        mapped_control = current_layout[control]
        mapped_target = current_layout[target]
        if not is_allowed_cnot(mapped_control, mapped_target, coupling_map):
            G = nx.Graph(coupling_map)
            path = nx.shortest_path(G, mapped_control, mapped_target)
            for i in range(len(path) - 2):
                q1, q2 = path[i], path[i + 1]
                circuit.add_gate(SWAP(q1, q2))
                current_layout[current_layout.index(q1)], current_layout[current_layout.index(q2)] = q2, q1

            mapped_control = current_layout[control]
            mapped_target = current_layout[target]

        circuit.add_gate(CNOT(mapped_control, mapped_target))
    
    return circuit

cnot_gates = [(0, 1), (1, 2), (2, 3), (3, 4)]

print("\nBarabasi-Albert (m = 2) Results: ")
for i, G in enumerate(ba_graphs):
    coupling_map = list(G.edges())
    coupling_map += [(edge[1], edge[0]) for edge in coupling_map]
    print(f"Seed {i} Coupling Map: {list(G.edges())}")

    circuit = adjust_circuit_for_coupling(coupling_map, cnot_gates)

    state = QuantumState(n_qubits)
    circuit.update_quantum_state(state)

    probs = [abs(state.get_vector()[i])**2 for i in range(2**n_qubits)]
    counts = {}
    for idx, prob in enumerate(probs):
        if prob > 1e-10:
            bin_str = format(idx, f"0{n_qubits}b")
            counts[bin_str] = int(prob * 101)