import json
import numpy as np
import matplotlib.pyplot as plt



RL_ALGO = "DDQN"
seed = 1
with open(f'results/{RL_ALGO}/vanilla_cobyla_H24q0p742_noiseless/summary_{seed}.json', 'r') as openfile:
    data = json.load(openfile)


episodes = 30

for e in range(episodes):
    errors_list = data['train'][f'{e}']['errors']
    if np.min(errors_list) <= 0.0016:
        plt.semilogy(errors_list, '--.')
plt.plot([0.0016]*40, 'k-', label = 'Chem. accu')
plt.ylabel('Error')
plt.xlabel('Number of gates')
plt.legend()
plt.show()