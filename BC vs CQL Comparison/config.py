"""Config for the BC vs CQL Comparison (BC vs CQL quick refinement) experiment.
Only constants live here; all behaviour comes from ../main_script."""
import os
import numpy as np

# ── Paths (relative to this folder) ───────────────────────────────────────────
BASE_DIR  = os.getcwd()
DATA_DIR  = os.path.join(BASE_DIR, 'data')
SAVE_DIR  = os.path.join(BASE_DIR, 'policies'); os.makedirs(SAVE_DIR,  exist_ok=True)
CHART_DIR = os.path.join(BASE_DIR, 'charts');   os.makedirs(CHART_DIR, exist_ok=True)

SEED = 42

# ── Anti-windup PI constants ──────────────────────────────────────────────────
KP_EXPERT = -0.50
KI_EXPERT = -0.50 / 300.0
KD_EXPERT =  0.00
Q_MIN, Q_MAX = 0.0, 40.0
TS = 1.0
INT_E_CLIP = 40000.0

EPS_LOG          = 1e-4
LAMBDA_SMOOTH    = 0.0
OVERSHOOT_WEIGHT = 0.15

TREF_SUNNY, TREF_CLOUDY = 80.0, 65.0

# ── State + action spaces ─────────────────────────────────────────────────────
STATE_DIM  = 9
ACTION_DIM = 3
ACTOR_KIND  = 'gain'     # actor outputs normalised gains
CRITIC_KIND = 'single'   # DDPG single critic

OBS_LOW  = np.array([ 30.0, -40.0, -40000.0, -5.0,    0.0,  0.0, 10.0,   0.0, 55.0], dtype=np.float32)
OBS_HIGH = np.array([100.0,  55.0,  40000.0,  5.0, 1200.0, 50.0, 90.0, 180.0, 85.0], dtype=np.float32)

# ORIGINAL [Kp, Ki, Kw] gain box — native to the trained CQL policy.
GAIN_LOW  = np.array([-3.5, -0.060, -0.35], dtype=np.float32)
GAIN_HIGH = np.array([-0.1, -0.0002, 0.05], dtype=np.float32)

START_GAIN      = np.array([KP_EXPERT, KI_EXPERT, KI_EXPERT / KP_EXPERT], dtype=np.float32)
BC_GUIDE_WEIGHT = 0.10
RL_FINE_ALPHA   = 0.2

# ── DDPG hyper-parameters ─────────────────────────────────────────────────────
GAMMA = 0.99
TAU   = 0.01
LR_A  = 1e-4
LR_C  = 1e-3

# ── Quick-refinement knobs ────────────────────────────────────────────────────
QUICK_EPISODES    = 50
SHORT_EP_STEPS    = 1000
BATCH_SIZE        = 256
UPDATE_EVERY      = 4
UPDATES_PER_CYCLE = 1
WARMUP_STEPS      = 500
EVAL_WIN          = 2000

# ── Datasets + warm-start checkpoints ─────────────────────────────────────────
JUAN_FILES = [
    "16_06_2026__Sunny_Closed_Loop.xlsx",
    "17_06_2026__Sunny_Closed_Loop.xlsx",
    "18_06_2026__Sunny_Closed_Loop.xlsx",
    "19_06_2026__Sunny_Closed_Loop.xlsx",
]
BC_CKPT  = 'bc_cirl_setpoint_anti_windup_best.pt'
CQL_CKPT = 'cql_cirl_actor_best.pt'
