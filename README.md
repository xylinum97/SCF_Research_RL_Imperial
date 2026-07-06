# SCF Research — RL for a Solar Collector Field (Imperial)

Reinforcement-learning control of the **outlet temperature** of a solar thermal
collector field, built around an **anti-windup PI** demonstrator. The project runs a
full **offline → online** pipeline — clone the expert (BC), learn a conservative
value function offline (CQL), then adapt online (DDPG / TD3) — and evaluates how each
piece transfers to **new field data** ("Juan's" June-2026 days).

All experiments share one library, [`main_script/`](main_script), and each experiment
folder carries only its own `config.py`.

```
For GitHub/
├── main_script/                 shared library (env, actors, critics, agents, rollouts)
├── Behavioral Cloning Actor/    BC of the anti-windup expert          (Regular + CIRL)
├── CQL Offline Actor/           offline Conservative Q-Learning        (Regular + CIRL)
├── Online ActorCritic Finetune/ online DDPG/TD3, BC actor + CQL critic (2 approaches)
└── Online Tuning/               BC-vs-CQL quick-refinement comparison
```

## The shared design: `main_script` + per-folder `config.py`

`main_script` holds all the *behaviour* (the plant model, `FlowActor`/`GainActor`,
`SingleCritic`/`TwinCritic`, `DDPG`/`TD3`/`CQL`, `ReplayBuffer`, rollouts/metrics).
Each folder's `config.py` holds only the *constants* that differ between experiments
(gain box, observation box, state/action dims, hyper-parameters, dataset lists).

A notebook or script wires them together in four lines:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))  # reach main_script
from main_script import *
import config as cfg
from config import *
configure(cfg)          # inject this folder's constants into the shared classes
```

After `configure(cfg)`, factories like `Actor()` / `Critic()` and classes like
`SolarFieldEnv`, `DDPG`, `CQL` build themselves from the active config — so the same
library serves the direct-flow (`ACTOR_KIND='flow'`, 10-D state) and gain
(`ACTOR_KIND='gain'`, 9-D state) variants without any code duplication.

## The control problem

Positional PI with back-calculation anti-windup (`Kw = Ki/Kp`), expert gains
`Kp=-0.5`, `Ti=300 s`, `Kd=0`. Setpoints: **sunny 80 °C**, **cloudy 65 °C**
(auto-detected from filename). Flow `q ∈ [0, 40] L/min`.

Two actor parameterisations recur throughout:
- **Regular** — actor outputs the flow `q` directly (10-D state incl. `q_prev`).
- **CIRL** — actor outputs PI **gains** (9-D state); `[Kp,Ki,Kd]` for the BC/CQL line
  or `[Kp,Ki,Kw]` for the online line.

## The four experiments

| Folder | Method | Headline result (mean over evaluation days) |
|---|---|---|
| **Behavioral Cloning Actor** | Supervised clone of the anti-windup expert; 15 dataset-combo policies per variant | closed-loop MAE ≈ 1–2 °C |
| **CQL Offline Actor** | Offline CQL-H from an expert-PI + noise buffer; 15 combos per variant | CIRL ≈ 1.4 °C MAE |
| **Online ActorCritic Finetune** | Online DDPG/TD3 on Juan's days; **BC actor + CQL-derived critic**; Approach 1 (gain search → critic-exploit) & Approach 2 (critic-driven) | **MAE ≈ 0.11 °C** (Expert PI 1.22) — ~11× better |
| **Online Tuning** | Light head-to-head: BC-init vs CQL-init, each given a *quick* DDPG refine on Juan's days | CQL-refined **0.18** vs BC-refined **0.98** (single-seed) |

Each folder has its own README with the full method, layout, and per-day numbers.

## Datasets

Closed-loop `.xlsx` logs with columns `T_sc, Tin, Ta, I, theta, q` (and optional
`T_ref`). The original campaign is 4 sunny (Oct 21–24 2025) + 1 cloudy (Oct 20 2025);
the online experiments adapt to 4 new "Juan" sunny days (Jun 16–19 2026). Each package
ships the data it needs under `data/`.

## Usage

```bash
git clone https://github.com/stallyargha97-png/SCF_Research_RL_Imperial.git
cd SCF_Research_RL_Imperial
pip install -r requirements.txt
```

- **Train**: run the `train/*.py` scripts (BC, CQL) or the training notebooks (online).
- **Evaluate**: open the `evaluate/*.ipynb` notebooks and Run All.

Notebooks resolve `main_script` and their `config.py` via a relative path, so they run
from inside their folder without installation. Requires Python 3.10+, PyTorch
(CPU is fine), NumPy, pandas, matplotlib, openpyxl.

## Requirements

```
pip install -r requirements.txt
```
