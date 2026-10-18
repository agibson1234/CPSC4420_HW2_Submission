# CPSC 4420 / 6420 – Assignment 2: Tabular Q-Learning

A reproducible Q-learning implementation and hyperparameter study on the 4 × 12 **CliffWalking-v0** environment. Includes the optional controlled-action-noise experiment.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python q_learning.py
```

The program runs the environment through **Gymnasium** when it is installed. If it is not installed, it uses a self-contained local implementation of the standard deterministic 4 × 12 CliffWalking transition/reward rules. The included report/figures were produced with this local implementation because Gymnasium was not available in the original execution environment. **Run with Gymnasium installed before submitting if you need the official environment results.**

## Contents

- `q_learning.py`: tabular Q-learning with zero-initialized Q-values, epsilon-greedy exploration, Bellman updates, termination handling, hyperparameter experiments, evaluation and MP4 renderer.
- `results/metrics.csv`: baseline and hyperparameter test measurements.
- `results/noise_metrics.csv`: noisy training and original-policy noise transfer.
- `results/*.png`: reward learning curves.
- `results/*_q.npy`: saved trained Q-tables.
- `results/trained_agent.mp4`: rendered demonstration of the trained, greedy agent.
- `HW2_Report.pdf`: report covering parts A–H and limitations.

## Experiment design

All experiments use 3,000 episodes (maximum 200 steps), seed 42, alpha 0.5, gamma 0.99, epsilon starting at 1.0 and decaying by 0.997 per episode, with minimum epsilon 0.05, unless the named parameter varies. Policy evaluations use 100 episodes, with epsilon = 0. Learning rates: 0.1/0.5/0.9. Discount factors: 0.5/0.9/0.99. Decay rates: 0.99/0.997/0.999. Extra-credit noise probabilities: 0/0.10/0.25. Under noise, a corrupted action is replaced uniformly by one of the other three actions.

## Public GitHub repository requirement

Create a new **public** repository, upload all the contents of this folder, and add its real URL to the report (the placeholder is intentionally not a fabricated link). Verify the repository opens in a private browsing window, and ensure the MP4 is present and playable. Do not submit individual Python files in Canvas; upload the PDF report.
