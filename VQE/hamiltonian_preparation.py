import pennylane as qml
# from pennylane import qchem 
# from pennylane import numpy as np
import scipy.linalg as la
import numpy as nnp


"""
Initialize:
    1. Which molecule?
    2. What is the bond disctance?
    3. Set up the geometry
    4. Which Fermion to qubit mapping?
"""
mol_name = "H2" # CHANGING THE MOLECULE
bondlength = 0.742 
geometry = f"H_.0_.0_.0;_H_.0_.0_{bondlength}" # CHANGEING THE BOND DISTANCE
mapping = "jordan_wigner" # CHANGING THE MAPPING

"""
Generating the molecule (can be kept as it is, for now)
"""
a = qml.data.load("qchem", molname=mol_name, bondlength=bondlength, basis="STO-3G", attributes=["hamiltonian", "vqe_energy" , "fci_energy"])

"""
Extracting the number of qubits
"""
qubits = int(nnp.log2(a[0].hamiltonian.matrix().shape[0]))


"""
Saving:
    1. The Hamiltonian matrix,
    2. The list of all eigenvalues,
    3. The weights of the Hamiltonian Pauli coefficient
"""
__ham = dict()
__ham['hamiltonian'], __ham['eigvals'], = a[0].hamiltonian.matrix(), la.eig(a[0].hamiltonian.matrix())[0].real
__ham['weights'], __ham['energy_shift'] = a[0].hamiltonian.coeffs, 0
nnp.savez(f"mol_data/{mol_name}_{qubits}q_geom_{geometry}_{mapping}",**__ham)
