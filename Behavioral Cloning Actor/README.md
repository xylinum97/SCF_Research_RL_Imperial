# Behavioral Cloning Actor — Solar Thermal Collector (Anti-Windup)

Behavioral Cloning (BC) of an **anti-windup PI expert** for outlet-temperature control
of a solar thermal collector field. Two actor variants are provided:

| Variant     | Actor output                       | Training script                     |
|-------------|------------------------------------|-------------------------------------|
| **Regular** | flow `q ∈ [0, 40]` directly        | `train/BC_Regular_Anti_Windup.py`   |
| **CIRL**    | PI gains `[Kp, Ki, Kw]` (then PI)  | `train/BC_CIRL_Setpoint_Anti_Windup.py` |

The expert is a positional PI controller with back-calculation anti-windup
(`Kw = Ki/Kp = 1/Ti`), gains from `Main_aggresive.m` (`Kp=-0.5`, `Ti=300 s`).
In the CIRL actor the third gain `Kw` is pinned to that expert constant
(zero-width box), so offline only `Kp` and `Ki` are shaped.
Setpoints: **sunny = 80 °C**, **cloudy = 65 °C** (auto-detected from filename).

## Layout

```
Behavioral Cloning Actor/
├── data/        5 closed-loop datasets — 1 cloudy (s20) + 4 sunny (s21–s24)
├── train/       BC training scripts (.py)      — produce the policies + charts
├── evaluate/    policy-evaluation notebooks (.ipynb) — closed-loop rollouts
├── policies/    trained actors (.pt) — 15 dataset combos + best, per variant
└── charts/      training + evaluation figures (.png)
```

Each variant trains **15 policies** — every non-empty subset of the 4 sunny days
(`C(4,1..4)`), each combined with the cloudy day — and copies the lowest closed-loop
MAE model to `*_best.pt`.

## Usage

Paths are resolved **relative to the repo**, so it runs as-is after cloning.

**Train** (writes `policies/*.pt` and `charts/*_train_comparison.png`):
```bash
python train/BC_Regular_Anti_Windup.py
python train/BC_CIRL_Setpoint_Anti_Windup.py
```

**Evaluate** — open the notebooks in `evaluate/` and Run All. They load every combo
from `policies/`, run closed-loop rollouts on each dataset, and write figures to
`charts/`. Set `EVAL_DATASET` at the top of each notebook to pick the rollout day.

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+, PyTorch (CPU is fine), NumPy, pandas, matplotlib, openpyxl.
