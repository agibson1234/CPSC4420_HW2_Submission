# CPSC 4420 / 6420 – Assignment 2: Tabular Q-Learning

A reproducible Q-learning implementation and hyperparameter study on the 4 × 12 CliffWalking-v1 environment. Includes the optional controlled-action-noise experiment.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python q_learning.py
```

The program uses Gymnasium's CliffWalking-v1 environment when Gymnasium is installed. If Gymnasium is unavailable, it uses a self-contained implementation of the standard deterministic 4 × 12 CliffWalking transition and reward rules.

## Contents

- `q_learning.py`: Tabular Q-learning with zero-initialized Q-values, epsilon-greedy exploration, Bellman updates, terminal-state handling, hyperparameter experiments, policy evaluation, and MP4 rendering.
- `results/metrics.csv`: Baseline and hyperparameter experiment results.
- `results/noise_metrics.csv`: Stochastic training and policy robustness results.
- `results/*.png`: Training reward curves and comparison plots.
- `results/*_q.npy`: Saved Q-tables.
- `results/trained_agent.mp4`: Video demonstration of the trained agent.
- `HW2_Report.pdf`: Report covering assignment sections A–H.

## Experiment Design

All standard experiments use the following baseline configuration:

- Training episodes: 3,000
- Maximum steps per episode: 200
- Random seed: 42
- Learning rate (α): 0.5
- Discount factor (γ): 0.99
- Initial epsilon (ε): 1.0
- Epsilon decay: 0.997
- Minimum epsilon: 0.05
- Evaluation episodes: 100
- Evaluation epsilon: 0 (no exploration)

The following hyperparameters are tested independently:

Learning rate: 0.1, 0.5, 0.9

Discount factor: 0.5, 0.9, 0.99

Epsilon decay: 0.99, 0.997, 0.999

Stochastic action noise (extra credit): 0, 0.10, 0.25

For the stochastic experiment, an intended action is replaced with a randomly selected different valid action according to the specified noise probability.

## Running the Program

From the project directory, execute:

```bash
python3 q_learning.py
```

The program trains the Q-learning agents, evaluates their performance, generates comparison plots, saves experiment results, and creates an MP4 demonstration of the learned policy.
