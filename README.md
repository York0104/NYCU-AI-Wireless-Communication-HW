# Exercise 1.12: PPO for MountainCar-v0

This project implements Proximal Policy Optimization (PPO) for the `MountainCar-v0` task from OpenAI Gym.

The repository contains two versions of the program:

- `PPO_MountainCar-v0.py`: the runnable experiment version with training logs, reward plots, moving-average plots, and periodic model checkpoint saving.
- `PPO_MountainCar-v0_submit.py`: the cleaner submission version, kept close to the original exercise skeleton and focused on filling the PPO update block.

## Task Description

In `MountainCar-v0`, the agent controls an underpowered car that must reach the goal at the top of the mountain. Because the engine is too weak to drive directly uphill, the agent must move back and forth to build momentum.

- State space:
  - Car position
  - Car velocity
- Action space:
  - Push left
  - Push right
  - No push

The goal is to reach the flag within the episode step limit.

## Project Files

| File | Purpose |
|------|---------|
| `PPO_MountainCar-v0.py` | Extended runnable version for testing, logging, saving models, and plotting rewards. |
| `PPO_MountainCar-v0_submit.py` | Cleaner homework submission version based on the original starter code. |
| `requirements.txt` | Required Python packages and tested versions for this project. |
| `param/` | Output folder used by the extended version for checkpoints and reward figures. |
| `param_submit/` | Output folder used by the submission version if saving/logging is enabled. |

## Environment

Tested environment:

- Python `3.11.5`
- Windows PowerShell


## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
```

If you want to install packages manually into a clean environment:

```powershell
pip install -r requirements.txt
```

## Requirements

This project was tested with:

```text
torch==2.2.2+cpu
gym==0.26.2
tensorboardX==2.6.4
numpy==1.24.3
matplotlib==3.7.2
```

## How To Run

### 1. Extended runnable version

Short smoke test:

```powershell
python PPO_MountainCar-v0.py --epochs 3
```

Longer training example:

```powershell
python PPO_MountainCar-v0.py --epochs 500 --save-interval 50
```

This version will:

- print per-episode reward and moving average
- save model checkpoints to `param/net_param/`
- save reward figures to `param/img/`

Generated outputs include:

- `param/net_param/actor_net_*.pkl`
- `param/net_param/critic_net_*.pkl`
- `param/img/episode_rewards.png`
- `param/img/moving_average_rewards.png`

### 2. Submission version

Run the homework-style version:

```powershell
python PPO_MountainCar-v0_submit.py
```

This version is intentionally closer to the original exercise starter code and mainly focuses on the PPO update implementation.
