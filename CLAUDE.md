# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Direct Industrial Implementation of Reinforcement Learning based SCF Control:
Leveraging Behavioral Cloning for Actor-Critic Policy Initialization**

- Author: Argha Pradipta (CID: 06058496)
- Supervisor: Antonio Del Rio Chanona
- Institution: Department of Chemical Engineering, Imperial College London
- Repository: https://github.com/xylinum97/SCF_Research_RL_Imperial

This repo is the **code and experiments** side of the project: a solar-collector-field
(SCF) plant model, an expert anti-windup PI controller, and a four-phase learning
pipeline (BC → CQL → BC-vs-CQL comparison → online actor-critic fine-tuning) built
around a shared library plus per-experiment configs. The **written report** (LaTeX)
lives in the sibling repository `Research-Project-Report---Reinforcement-Learning`,
checked out locally as `../Report` relative to this repo's root — the `report-*`
skills in `.claude/skills/` operate on that sibling checkout by default.

## Repository layout

```
SCF_Research_RL_Imperial/
├── main_script/                  shared library: env, actors, critics, agents, rollouts
│   ├── env.py                    SolarFieldEnv (POMDP wrapper around the plant)
│   ├── physics.py                collector field plant model
│   ├── networks.py               FlowActor/GainActor, SingleCritic/TwinCritic
│   ├── agents.py                 DDPG, TD3, CQL
│   ├── replay.py                 ReplayBuffer
│   ├── rollout.py                closed-loop rollout + metrics
│   └── util.py, _ctx.py          config injection (configure(cfg)), helpers
├── Behavioral Cloning Actor/     Phase 1: BC (Regular + CIRL variants)
├── CQL Offline Actor/            Phase 2: offline CQL-H (Regular + CIRL variants)
├── BC vs CQL Comparison/         Phase 3: BC-vs-CQL quick-refinement comparison
├── Online ActorCritic Finetune/  Phase 4: online DDPG/TD3, BC actor + CQL critic
├── requirements.txt
└── README.md
```

Each experiment folder is self-contained: its own `config_*.py` (constants only —
gain box, observation box, dims, hyperparameters, dataset lists), `data/`, `train/`,
`evaluate/`, `policies/`, `charts/`. All *behaviour* lives once in `main_script/`; a
notebook or script wires a folder's config into the shared library with:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from main_script import *
import config as cfg
from config import *
configure(cfg)
```

## The control problem

- Plant: solar thermal collector field; manipulate pump flow `q` to hold outlet
  temperature `T_out` at setpoint `T_ref` under irradiance disturbance.
- Expert baseline: anti-windup PI, `Kp = -0.5`, `Ti = 300 s` (so `Ki = Kp/Ti`),
  back-calculation anti-windup `Kw = Ki/Kp`.
- Flow bound: `q ∈ [0, 40] L/min`.
- Two actor parameterisations recur everywhere:
  - **Regular** — actor outputs flow `q` directly (10-D state, includes `q_prev`).
  - **CIRL** — actor outputs PI gains `[Kp, Ki, Kw]` (9-D state); `Kw` is pinned to
    `Ki/Kp` offline, learned online.
- Datasets: offline training on s20 (cloudy, 65 °C) + s21–s24 (sunny, 80 °C);
  online adaptation/evaluation on four unseen sunny days, 16–19 June 2026.

## The four-phase pipeline and headline numbers

| Phase | Method | Headline result |
|---|---|---|
| 1 | Behavioural Cloning (BC) of the expert PI | closed-loop MAE ≈ 1.36 °C (matches, can't beat, the expert) |
| 2 | Offline Conservative Q-Learning (CQL) | zero-shot MAE on unseen days = **0.245 °C**, ~5× more accurate than the BC/expert start (1.224 °C) |
| 3 | BC-vs-CQL, same light critic-driven refinement | CQL-refined start stays ~5× ahead of the BC-refined start |
| 4 | Online DDPG/TD3 fine-tuning (BC actor + CQL critic) | best result: DDPG MAE = **0.109 °C**, a 91% reduction vs. the expert PI |

Treat these numbers (and any you compute) as **provisional until re-derived from the
current code and data** — CLAUDE.md is not the source of truth for results, the
`evaluate/*.ipynb` notebooks and their output tables are.

## Working conventions

- Prefer editing existing files; don't scaffold new experiment folders unless asked.
- `main_script/` changes affect every experiment — check all four folders' configs
  are still compatible before committing a change there.
- Config files hold constants only; don't let behaviour leak into `config_*.py`.
- Don't commit large generated artefacts (trained policy weights, chart PNGs) unless
  the user asks — check `.gitignore` and prefer to keep the repo diff reviewable.
- No comments in code beyond what's needed to explain non-obvious constants (e.g.
  why a gain box has a particular bound).
