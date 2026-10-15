# BioAI — Drone Hierarchical Federated Learning

A research project for **hierarchical federated learning (HierFL) in drone networks**, with **Proximal Policy Optimization (PPO)** to tune local training and edge aggregation.

The project combines PyTorch models, Flower federated learning simulations, and a reinforcement learning environment that balances classification accuracy, estimated training time, and energy consumption.

## Overview

The learning hierarchy has three levels:

1. **Drone clients** train a neural network on local data partitions.
2. **Edge servers** aggregate updates within client clusters.
3. **Global aggregation** combines cluster models and evaluates the resulting model.

The PPO agent selects two parameters for each environment step:

| Parameter | Meaning | Range |
| --- | --- | --- |
| `k1` | Local mini-batch training iterations per client update | 5–25 |
| `k2` | Edge aggregation rounds within each global round | 2–10 |

Each step runs a complete HierFL experiment with the selected parameters, or generates synthetic metrics when `--simulate` is enabled. An episode contains multiple such steps.

## Features

- Hierarchical model aggregation using Flower and Ray.
- Drone computation and communication energy estimates.
- MNIST and CIFAR-10 dataset loading.
- IID, Dirichlet non-IID, and pathological non-IID partitioning.
- PPO training, checkpoint loading, retraining, and evaluation.
- Comparison of a trained policy against a random-action baseline.
- CSV logs, experiment summaries, and comparison plots.

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/DomenicoVillari3/bioai.git
cd bioai
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The repository includes two dependency snapshots:

| File | Contents |
| --- | --- |
| `requirements.txt` | A formatted package/version table; it is not directly usable with `pip install -r`. |
| `requirements2.txt` | Pinned pip requirements, including PyTorch, Flower, Ray, and dataset packages. |

For a compatible Linux/CUDA environment, the pip-formatted snapshot can be used as a starting point:

```bash
python -m pip install -r requirements2.txt
python -m pip install stable-baselines3 gym shimmy
```

The additional packages support the RL workflow and compatibility with the environment's legacy Gym API. They are not pinned in the repository. For CPU-only systems or other platforms, adapt the NVIDIA/CUDA-specific entries and use an appropriate PyTorch build.

**Dependency compatibility must be checked in your environment.** The repository does not define a single validated Python version or complete, pinned RL environment.

## Quick Start: Synthetic Metrics

Start with synthetic mode to exercise the PPO workflow without running federated model training:

```bash
python main.py --simulate --experiment_id 1
```

This mode generates synthetic accuracy, time, and energy values. It is useful for checking the workflow; its results do not measure actual federated learning performance. The federated learning dependencies are still imported.

## Run Federated Learning with PPO

Run PPO optimization using actual HierFL training:

```bash
python main.py --experiment_id 2
```

The current `OptHierFLEnv.run_HierFL()` configuration uses **MNIST**, **30 clients**, and **IID partitions**. PyTorch selects CUDA when available and otherwise uses the CPU.

Each RL step launches a new HierFL run. Training can therefore be computationally expensive.

### Retrain an Existing Policy

```bash
python main.py --retrain --experiment_id 2
```

When the experiment's policy checkpoint exists, this command loads it and saves a backup before further training.

### Test a Policy

```bash
python main.py --test --experiment_id 2
```

Testing uses the best checkpoint when available and falls back to the regular checkpoint. A trained checkpoint is required.

### Compare PPO with a Random Agent

```bash
python main.py --evaluate --experiment_id 2
```

Evaluation requires the best checkpoint and compares PPO with random actions over 25 episodes per agent. It produces a CSV summary and a comparison plot.

Use `--simulate` consistently when training and evaluating a policy intended for synthetic mode.

## Configuration

Configuration is currently defined in Python dictionaries rather than an external configuration file.

| Location | Configuration |
| --- | --- |
| `main.py` → `conf` | PPO settings, episode budget, episode length, history length, evaluation batches, and early stopping |
| `OptHierFLEnv.py` → `run_HierFL()` | Dataset, client count, global rounds, partitioning, batch size, and federated training settings |
| `OptHierFLEnv.py` → `compute_reward()` | Accuracy, time, energy, and improvement weights |
| `HierFL.py` → `cluster_configurations` | Client cluster layouts across training phases |
| `drone_manager.py` | Battery capacity, per-sample computation costs, channel capacity, and transmitter power |

The default PPO configuration includes `n_steps=128`, `batch_size=32`, a learning rate of `3e-4`, three environment steps per episode, and three global HierFL rounds per step.

The observation contains normalized accuracy, estimated training time, and total energy. When history is enabled, observations are concatenated across the configured history window. The reward favors accuracy and penalizes time and energy, with bonuses for positive improvements when history is available.

### Dataset and Model Selection

`load_datasets.py` supports `mnist` and `cifar10`, with `IID` or `NON-IID` partitioning. Non-IID options include `dirichlet`, `pathological-ordered`, and `pathological-balanced`.

The active `Net` in `model.py` is configured for **MNIST**. A CIFAR-10 architecture is included as commented code. Before switching datasets, select a compatible model architecture.

For standalone fixed-parameter experiments, `main_adrian.py` calls HierFL directly. Its default dataset is CIFAR-10, so align its dataset setting with the active model before running:

```bash
python main_adrian.py
```

## Outputs

PPO artifacts are saved under `output/Experiment_<id>/`.

| File | Purpose |
| --- | --- |
| `ppo_HierFL.zip` | Latest saved PPO policy |
| `ppo_HierFL_best.zip` | Best policy according to evaluation reward |
| `training_results.csv` | Evaluation reward mean and standard deviation during training |
| `conf.txt` | Saved PPO configuration |
| `test_episodes.csv` | Episode rewards from test mode |
| `res_test.csv` | Aggregate test results |
| `trained_vs_random.csv` | PPO and random-agent evaluation metrics |
| `comparison.png` | Evaluation comparison plot |

Additional logs include `step.csv`, `time.csv`, and detailed HierFL experiment summaries under `Experiments/`. These paths are separate from the PPO experiment directory; repeated runs can append to shared CSV logs.

## Repository Structure

| Path | Role |
| --- | --- |
| `main.py` | PPO training, retraining, testing, and evaluation entry point |
| `OptHierFLEnv.py` | RL environment, action decoding, reward, and synthetic mode |
| `HierFL.py` | Hierarchical federated learning orchestration |
| `client.py` | Flower client implementation |
| `strategy.py` | Custom federated averaging strategy |
| `model.py` | Neural network, local training, evaluation, and parameter utilities |
| `load_datasets.py` | Dataset loading and client partitioning |
| `drone_manager.py` | Drone battery, computation, and communication estimates |
| `evaluate_model.py` | PPO versus random-agent comparison |
| `main_adrian.py` | Standalone HierFL experiment runner |
| `reinforcement2.py` | Alternative experimental RL implementation |
| `plot.ipynb`, `reinforcement.ipynb` | Analysis and experimentation notebooks |
| `Experiments/`, `output/`, `results/`, `runs/` | Existing experiment artifacts and logs |

## Current Implementation Notes

- **Energy accounting:** the returned `final_results['total_energy']` in `HierFL.py` currently adds computation energy to itself. For computation plus communication energy, correct this expression before interpreting energy-based rewards or comparisons.
- **Metric interpretation:** `training_time` is an estimate based on per-sample costs, not elapsed execution time. Wall-clock timings are logged separately in `time.csv`.
- **Accuracy normalization:** the environment divides final accuracy by the number of global rounds; the CSV field `accuracy_avg` uses the same division. These values are not averages of per-round accuracies.
- **Cluster assumptions:** the hard-coded cluster layouts each total 30 clients. Update the layouts when changing the client count, and provide enough phase configurations for the requested global rounds.
- **CLI behavior:** `--output_dir` and `--model` are parsed but are not used to select output paths or checkpoints. Output selection currently uses `--experiment_id`. The available `--env_name` values use the same environment implementation.
- **Experimental alternatives:** `reinforcement2.py` contains placeholder CSV fields and is not the primary documented workflow.

## License

No license file is included in the repository. Contact the repository owner for reuse and distribution terms.

