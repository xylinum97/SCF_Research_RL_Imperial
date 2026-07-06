"""Config for the Online ActorCritic Finetune experiment (DDPG/TD3, gain search).
Only constants live here; behaviour comes from ../main_script. Per-notebook
overrides (e.g. Approach-2's EPISODES/UPDATES_PER_CYCLE) are set in each notebook."""
import os
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
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
ACTOR_KIND = 'gain'

OBS_LOW  = np.array([ 30.0, -40.0, -40000.0, -5.0,    0.0,  0.0, 10.0,   0.0, 55.0], dtype=np.float32)
OBS_HIGH = np.array([100.0,  55.0,  40000.0,  5.0, 1200.0, 50.0, 90.0, 180.0, 85.0], dtype=np.float32)

# WIDENED [Kp, Ki, Kw] gain box (tuned).
GAIN_LOW  = np.array([-6.0, -0.100, -0.60], dtype=np.float32)
GAIN_HIGH = np.array([-0.05, -0.0001, 0.10], dtype=np.float32)

START_GAIN      = np.array([KP_EXPERT, KI_EXPERT, KI_EXPERT / KP_EXPERT], dtype=np.float32)
BC_GUIDE_WEIGHT = 0.10
RL_FINE_ALPHA   = 0.3

# ── Agent hyper-parameters ────────────────────────────────────────────────────
GAMMA = 0.99
TAU   = 0.01
LR_A  = 1e-4
LR_C  = 1e-3
POLICY_DELAY = 2      # TD3
TARGET_NOISE = 0.1    # TD3
NOISE_CLIP   = 0.25   # TD3

# ── Training knobs (defaults; Approach-2 raises EPISODES/UPDATES_PER_CYCLE) ────
EPISODES          = 300
SHORT_EP_STEPS    = 1000
BATCH_SIZE        = 256
EVAL_EVERY        = 10
UPDATE_EVERY      = 4
UPDATES_PER_CYCLE = 1
WARMUP_STEPS      = 800
EVAL_WIN          = 2000

# ── CQL critic pre-train ──────────────────────────────────────────────────────
CQL_PRETRAIN_STEPS = 8000
CQL_PRETRAIN_NOISE = 0.03

# ── Datasets + warm-start checkpoints ─────────────────────────────────────────
JUAN_FILES = [
    "16_06_2026__Sunny_Closed_Loop.xlsx",
    "17_06_2026__Sunny_Closed_Loop.xlsx",
    "18_06_2026__Sunny_Closed_Loop.xlsx",
    "19_06_2026__Sunny_Closed_Loop.xlsx",
]
BC_CKPT  = 'bc_cirl_setpoint_anti_windup_best.pt'
CQL_CKPT = 'cql_cirl_actor_best.pt'
