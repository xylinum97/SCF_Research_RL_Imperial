# Online Actor-Critic Fine-tune — BC actor + CQL critic (Anti-Windup)

Online DDPG / TD3 fine-tuning of a solar-thermal outlet-temperature controller that
**combines two pretrained pieces**:

- **Actor ← Behavioral Cloning** — warm-started from `bc_cirl_setpoint_anti_windup_best.pt`
  (BC trunk; the output head is re-centred to the expert PI gains).
- **Critic ← CQL** — the "first critic" is obtained by loading the trained CQL policy
  (`cql_cirl_actor_best.pt`), rolling it out on the data, and fitting the critic to
  evaluate that policy (TD policy-evaluation). The online agent therefore *starts*
  with the CQL Q-function as its critic.

Everything runs in the anti-windup **v2 action space**: the actor outputs `[Kp, Ki, Kw]`
(positional PI + back-calculation anti-windup), same gain box as `DDPG/TD3_CIRL_Anti_Windup_v2`.

**Trained *and* evaluated on Juan's 4 new days** (`16–19_06_2026`, all Sunny, T_ref = 80 °C) —
an adaptation-to-new-data test. Three days are used for fine-tuning; the 4th
(`19_06_2026`) is held out for the validation curve; the final table reports all 4.

## The 4 notebooks (2 DDPG + 2 TD3, two approaches)

| Notebook | Algo | Approach |
|---|---|---|
| `DDPG_Approach1_SearchThenCriticExploit.ipynb` | DDPG | **1** |
| `TD3_Approach1_SearchThenCriticExploit.ipynb`  | TD3  | **1** |
| `DDPG_Approach2_CriticDriven.ipynb`            | DDPG | **2** |
| `TD3_Approach2_CriticDriven.ipynb`             | TD3  | **2** |

**Approach 1 — SEARCH → FREEZE → CRITIC-EXPLOIT** (same method as the v2 notebooks):
first a (1+1)-ES search *discovers* the optimal constant gains (ranked by reward −
overshoot), then those gains are frozen as the BC anchor, and finally the critic
(**initialised from CQL**) fine-tunes the actor to lower overshoot at fixed MAE.

**Approach 2 — critic-driven** — after warm-starting (actor ← BC, critic ← CQL), the
actor is trained *iteratively purely on the critic's feedback* (deterministic policy
gradient, no gain-search phase, no BC anchor pull).

## Layout

```
Online ActorCritic Finetune/
├── data/       Juan's 4 new days (16–19_06_2026)
├── policies/   warm-start inputs: bc_cirl_..._best.pt (actor), cql_cirl_actor_best.pt (critic src)
│               → each notebook also writes its fine-tuned actor here (*_best.pt)
├── charts/     fine-tune curves + per-day tracking (written on run)
└── *.ipynb     the 4 training notebooks
```

## Usage

Paths are relative (`BASE_DIR = os.getcwd()`), so open a notebook from this folder and
Run All. Each notebook: warm-starts the actor, pretrains the critic from CQL, fine-tunes,
prints a per-Juan-day MAE/RMSE/overshoot table (vs Expert PI), and saves
`policies/<algo>_online_ac_approach<n>_best.pt` + charts.

Key knobs at the top of the training cells: `EPISODES` (default 200), `WARMUP_STEPS`,
`CQL_PRETRAIN_STEPS` (4000), `BC_GUIDE_WEIGHT` / `RL_FINE_ALPHA` (Approach 1),
`CQL_PRETRAIN_NOISE`.

## Requirements

```
pip install -r requirements.txt
```
Python 3.10+, PyTorch (CPU fine), NumPy, pandas, matplotlib, openpyxl.
