# BC vs CQL Comparison — BC vs CQL with a quick refinement (Juan's new data)

A light-weight head-to-head: which **offline-pretrained starting policy** adapts better to
Juan's new days when given only a *quick* online refinement — **Behavioral Cloning** or
**blind Conservative Q-Learning (CQL)**?

```
Method 1:  BC (existing data + expert PI)  ─┐
                                            ├─ quick online DDPG refine on Juan ─► trained policy
Method 2:  Blind CQL (existing data)       ─┘
```

Both warm-starts are refined with the **same short DDPG** (~50 episodes, critic-driven, **no**
gain-search / freeze / exploit) in the **original `[Kp,Ki,Kw]` gain box** — deliberately
lighter and less fine than the [DDPG/TD3 gain-search package](../Online%20ActorCritic%20Finetune).
Train and evaluate both on Juan's 4 new days (`16–19_06_2026`).

- **Method 1 (BC)** — actor = BC feature trunk + expert-PI head (`bc_cirl_setpoint_anti_windup_best.pt`);
  its raw start is therefore the Expert PI.
- **Method 2 (CQL)** — actor = the trained CQL policy (`cql_cirl_actor_best.pt`).

## Result (mean over Juan's 4 days)

| Controller | MAE °C | RMSE | Overshoot °C |
|---|---|---|---|
| Expert PI | 1.224 | 1.731 | 5.92 |
| BC raw (Method 1 start) | 1.224 | 1.731 | 5.92 |
| CQL raw (Method 2 start) | 0.243 | 0.501 | 3.67 |
| Method 1: BC → refined | 0.978 | 1.414 | 5.49 |
| **Method 2: CQL → refined** | **0.180** | 0.419 | 3.27 |

**BC raw ≡ Expert PI** (1.224): the warm-start zeroes the actor's output head and sets it to the
expert gains, so before refinement the BC actor is a *constant* expert-gain controller; the BC
feature trunk only contributes once the head un-freezes during refinement (1.224 → 0.978).

**Takeaway:** the offline CQL policy transfers to new data ~5× better out-of-the-box than the
expert/BC start, and after the same quick refinement it stays ~5× ahead (0.180 vs 0.978).
Blind-CQL is a far more sample-efficient warm-start for adapting to new data with a light nudge;
the BC+expert start needs the heavier gain-search fine-tuning to catch up.

## Layout

```
BC vs CQL Comparison/
├── data/       Juan's 4 new days (16–19_06_2026)
├── policies/   warm-start inputs (bc_..._best.pt, cql_cirl_actor_best.pt)
│               + refined outputs (online_tuning_bc_refined.pt, online_tuning_cql_refined.pt)
├── charts/     comparison bar chart + per-day tracking overlays
└── BC_vs_CQL_Online_Tuning.ipynb
```

## Usage

Open `BC_vs_CQL_Online_Tuning.ipynb` from this folder and Run All (paths are relative). Knobs at
the top of the quick-refine cell: `QUICK_EPISODES` (50), `WARMUP_STEPS`, `UPDATE_EVERY`.

## Requirements

```
pip install -r requirements.txt
```
Python 3.10+, PyTorch (CPU fine), NumPy, pandas, matplotlib, openpyxl.
