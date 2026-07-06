# CQL Offline Actor — Solar Thermal Collector (Anti-Windup)

Offline **Conservative Q-Learning (CQL-H)** actors for outlet-temperature control
of a solar thermal collector field. Two actor variants, mirroring the
[Behavioral Cloning Actor](../Behavioral%20Cloning%20Actor) package:

| Variant     | Actor output                       | Training script                        |
|-------------|------------------------------------|----------------------------------------|
| **Regular** | flow `q ∈ [0, 40]` directly        | `train/CQL_Regular_Anti_Windup.py`     |
| **CIRL**    | PI gains `[Kp, Ki, Kd]` (then PI)  | `train/CQL_CIRL_Setpoint_Anti_Windup.py` |

The actor networks are architecturally **identical** to the BC networks, so the
policies drop straight into the evaluation notebooks.

## How it's offline

CQL never interacts with the plant online. For each dataset combo the replay
buffer is filled **once** by rolling out the anti-windup expert PI
(`Kp=-0.5`, `Ti=300 s`, back-calc `Kw=Ki/Kp`) with a small Gaussian action noise
(`PREFILL_NOISE=0.05`) for state-action coverage. The agent then trains purely by
sampling that fixed buffer:

```
critic loss = TD_loss + α · ( logmeanexp_a Q1(s,a) − E_data[Q1(s,a)] )   # CQL(H)
actor  loss = − E[ Q1(s, π(s)) ]                                          # delayed, TD3-style
```

Twin critic (`h=256`), target smoothing, `α = CQL_ALPHA = 1.0`. Reward is
tracking-based: `r = −log(err²+ε) − λ·log(Δq²+ε)`.

## Replay-buffer variation (same as BC)

Each variant trains **15 policies** — every non-empty subset of the 4 sunny days
`C(4,1..4)`, each combined with the cloudy day — and copies the lowest
closed-loop MAE model to `*_best.pt`. Setpoints: sunny 80 °C, cloudy 65 °C
(auto-detected from filename).

## Layout

```
CQL Offline Actor/
├── data/        5 closed-loop datasets — 1 cloudy (s20) + 4 sunny (s21–s24)
├── train/       CQL training scripts (.py)   — produce the policies + charts
├── evaluate/    policy-evaluation notebooks (.ipynb) — closed-loop rollouts
├── policies/    trained actors (.pt) — 15 combos + best, per variant
└── charts/      training + evaluation figures (.png)
```

## Usage

Paths resolve **relative to the repo**, so it runs as-is after cloning.

**Train** (writes `policies/*.pt` and `charts/*_comparison.png`):
```bash
python train/CQL_Regular_Anti_Windup.py
python train/CQL_CIRL_Setpoint_Anti_Windup.py
```
Budget knobs at the top of each script: `EPOCHS`, `UPDATES_PER_EPOCH`,
`TRAIN_NFILES` (which combo sizes to train), `CQL_ALPHA`, `PREFILL_NOISE`.

**Evaluate** — open the notebooks in `evaluate/` and Run All. They load every
combo from `policies/`, run closed-loop rollouts on each dataset, and write
figures to `charts/`. Set `EVAL_DATASET` at the top of each notebook.

> **Note on shipped policies:** the `.pt` files included here were trained at a
> **reduced budget** (`EPOCHS=25`, `UPDATES_PER_EPOCH=200`) so the package is
> populated and the notebooks are runnable out of the box. Re-run the training
> scripts at their default budget (`EPOCHS=100`, `UPDATES_PER_EPOCH=500`) for
> final results — the Regular variant in particular keeps improving with more
> updates since it learns the flow map without an imitation anchor.

## Requirements

```
pip install -r requirements.txt
```
Python 3.10+, PyTorch (CPU fine), NumPy, pandas, matplotlib, openpyxl.
