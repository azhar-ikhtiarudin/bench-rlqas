# BenchRL-QAS: Benchmarking RL Algorithms for Quantum Architecture Search

Welcome to the official implementation of **BenchRL-QAS**, a framework designed to evaluate **reinforcement learning (RL)** algorithms in **quantum architecture search (QAS)**. This repository is based on the paper:

> **BenchRL-QAS: Benchmarking Reinforcement Learning Algorithms for Quantum Architecture Search**  
> ArXiv:


## ❓ Why BenchRL-QAS?

Despite the rising use of RL in quantum circuit design, previous efforts:
- Often benchmark only a few RL agents in isolation,
- Lack standardized metrics or reproducibility,
- And rarely account for task complexity.

BenchRL-QAS addresses this gap by:
- Offering **the most extensive benchmark** for RL-QAS to date,
- Evaluating 9 RL algorithms (e.g., DQN, PPO, A3C, DDQN, TPPO),
- Including **both noiseless and noisy** experiments,
- Providing **task-specific insights**, e.g., which algorithms perform best for VQE vs VQC,
- Demonstrating the **“No Free Lunch” theorem** in practice, no RL agent dominates across all tasks,
- Enabling **reproducibility** through open-source code and datasets.

By using this benchmark, researchers and practitioners can:
- Select suitable RL agents tailored to their quantum tasks,
- Better understand trade-offs between circuit efficiency and accuracy,
- Accelerate the development of scalable, noise-resilient quantum algorithms.

## 📈 Highlights

- Outperforms recent QAS methods (e.g., TF-QAS by [He et al., 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29135)) in VQE (BeH₂) with **3 orders of magnitude better error** and **>50% fewer gates**.
- RL-VQC with DQN-rank achieves **>99.99% training and test accuracy**, surpassing hardware-efficient ansatz.


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
git clone git@github.com:azhar-ikhtiarudin/bench-rlqas.git
cd bench-rlqas
pip install -r requirements.txt
```

### Run Experiments

Run a specific agent (e.g., PPO) on a specific quantum task (e.g., VQE):

```bash
python main.py --task VQE --agent PPO
```

Benchmark **all agents** across **all tasks**:

```bash
python main.py --task ALL --agent ALL
```

Options:
- `--task`: one of `VQE`, `VQSD`, `VQC`, `Srate_Prep`, or `ALL`
- `--agent`: one of `A2C`, `A3C`, `DQN`, `DQN_PER`, `DQN_rank`, `Dueling_DQN`, `DDQN`, `PPO`, `TPPO`, or `ALL`



## 🤝 Contributing

We welcome contributions from the community! You can help by:

- Adding new RL algorithms or training strategies
- Extending support to new quantum tasks (e.g., QAOA, VQLS)
- Improving visualizations or logging tools
- Enhancing the benchmarking or noise models

Feel free to submit a pull request or open an issue!
