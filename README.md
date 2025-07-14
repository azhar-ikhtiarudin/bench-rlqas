# Benchmarking RL algorithms for quantum architecture search on near-term quantum devices

Welcome to the official repository for BenchRL-QAS, a benchmarking framework designed to evaluate reinforcement learning (RL) algorithms in the context of quantum architecture search (QAS). This project is based on the paper:

# BenchRL-QAS: Benchmarking RL Algorithms for Quantum Architecture Search

Welcome to the official implementation of **BenchRL-QAS**, a framework designed to evaluate **reinforcement learning (RL)** algorithms in **quantum architecture search (QAS)**. This repository is based on the paper:

> **BenchRL-QAS: Benchmarking Reinforcement Learning Algorithms for Quantum Architecture Search**  
> ArXiv:

## 📖 Overview

**BenchRL-QAS** is a unified benchmarking platform for evaluating RL-based quantum circuit design techniques. It provides:

- Automated discovery of efficient parameterized quantum circuits
- Tasks spanning **4 key quantum algorithms** under **noiseless** and **noisy** settings
- A consistent and fair **weighted ranking metric**
- Evaluation across **9 RL agents**, covering both value-based and policy-gradient methods


## 🧠 Reinforcement Learning Algorithms

BenchRL-QAS benchmarks the following RL agents:

| Category           | Algorithm            | Description                                     |
|--------------------|----------------------|-------------------------------------------------|
| **Value-based**    | `DQN`                | Deep Q-Network                                  |
|                    | `DQN PER`            | Prioritized Experience Replay                   |
|                    | `DQN Rank`           | Rank-based action prioritization                |
|                    | `Dueling DQN`        | Dueling value and advantage streams             |
|                    | `DDQN`               | Double DQN to reduce overestimation             |
| **Policy-gradient**| `A2C`                | Advantage Actor-Critic                          |
|                    | `A3C`                | Asynchronous Advantage Actor-Critic             |
|                    | `PPO`                | Proximal Policy Optimization                    |
|                    | `TPPO`               | Truly Proximal Policy Optimization              |

Each algorithm is evaluated over multiple trials to ensure statistical significance.



## ⚛️ Quantum Problems

BenchRL-QAS supports four variational quantum algorithm (VQA) tasks:

| Task                        | Description                                                           | Action Type        |
|-----------------------------|-----------------------------------------------------------------------|--------------------|
| **State Preparation (GHZ)** | Prepare maximally entangled states                                    | Non-parameterized  |
| **VQSD**                    | Variational Quantum State Diagonalization                             | Parameterized      |
| **VQE**                     | Variational Quantum Eigensolver for ground-state estimation            | Parameterized      |
| **VQC**                     | Variational Quantum Classifier for supervised learning                 | Parameterized      |

Tasks are evaluated on 2–8 qubit systems in both noiseless and noisy settings.




## 🚀 Getting Started

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/bench-rl-qas/benchrl-qas.git
cd benchrl-qas
pip install -r requirements.txt
```

### Run Experiments

Run a specific agent (e.g., PPO) on a specific quantum task (e.g., VQE):

```bash
python main.py --task VQE --agent PPO
```

Benchmark **all agents** across **all tasks**:

```bash
python main.py --task all --agent all
```

Options:
- `--task`: one of `VQE`, `VQSD`, `VQC`, `stateprep`, or `all`
- `--agent`: one of `DQN`, `DDQN`, `PPO`, etc., or `all`




## 🤝 Contributing

We welcome contributions from the community! You can help by:

- Adding new RL algorithms or training strategies
- Extending support to new quantum tasks (e.g., QAOA, VQLS)
- Improving visualizations or logging tools
- Enhancing the benchmarking or noise models

Feel free to submit a pull request or open an issue!# bench-rlqas
