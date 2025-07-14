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
    current_layout = list(range(n_qubits))  # Initial qubit mapping
    applied_gates = []  # Track gates for diagram

    # Add Hadamards to all qubits
    for q in range(n_qubits):
        circuit.add_gate(H(q))
        applied_gates.append(("H", q))

    # Process each CNOT
    for control, target in gates:
        mapped_control = current_layout[control]
        mapped_target = current_layout[target]
        
        if not is_allowed_cnot(mapped_control, mapped_target, coupling_map):
            G = nx.Graph(coupling_map)
            path = nx.shortest_path(G, mapped_control, mapped_target)
            for i in range(len(path) - 2):
                q1, q2 = path[i], path[i + 1]
                circuit.add_gate(SWAP(q1, q2))
                applied_gates.append(("SWAP", q1, q2))
                current_layout[current_layout.index(q1)], current_layout[current_layout.index(q2)] = q2, q1
            mapped_control = current_layout[control]
            mapped_target = current_layout[target]
        
        circuit.add_gate(CNOT(mapped_control, mapped_target))
        applied_gates.append(("CNOT", mapped_control, mapped_target))
    
    return circuit, applied_gates

def draw_circuit_diagram(applied_gates, n_qubits):
    lines = [[] for _ in range(n_qubits)]
    max_len = 0
    
    for gate_type, *qubits in applied_gates:
        if gate_type == "H":
            q = qubits[0]
            lines[q].append("H")
        elif gate_type == "CNOT":
            control, target = qubits
            lines[control].append("C")
            lines[target].append("T")
        elif gate_type == "SWAP":
            q1, q2 = qubits
            lines[q1].append("S")
            lines[q2].append("S")
        max_len = max(max_len, max(len(line) for line in lines))
    
    # Pad lines to equal length
    for line in lines:
        line.extend(["-" for _ in range(max_len - len(line))])
    
    # Print diagram
    diagram = "\n".join(f"q_{i}: {' '.join(lines[i])}" for i in range(n_qubits))
    return diagram

cnot_gates = [(0, 1), (1, 2), (2, 3), (3, 4)]

print("\nBarabási–Albert (m=2) Results:")
for i, G in enumerate(ba_graphs):
    coupling_map = list(G.edges())
    coupling_map += [(edge[1], edge[0]) for edge in coupling_map]
    print(f"\nSeed {i} Coupling Map: {list(G.edges())}")
    
    # Adjust circuit and get gates
    circuit, applied_gates = adjust_circuit_for_coupling(coupling_map, cnot_gates)
    
    # Print circuit diagram
    print(f"Circuit Diagram (Seed {i}):")
    print(draw_circuit_diagram(applied_gates, n_qubits))
    
    # Simulate
    state = QuantumState(n_qubits)
    circuit.update_quantum_state(state)
    
    # Get probabilities and counts
    probs = [abs(state.get_vector()[j])**2 for j in range(2**n_qubits)]
    counts = {}
    for idx, prob in enumerate(probs):
        if prob > 1e-10:
            bin_str = format(idx, f'0{n_qubits}b')
            counts[bin_str] = int(prob * 1024)
    
    print(f"Simulation Counts (Seed {i}): {counts}")