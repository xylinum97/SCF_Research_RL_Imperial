"""Common CQL constants shared by config_regular.py and config_cirl.py.
Behaviour lives in ../main_script; only constants live here."""
import os
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # robust for scripts + notebooks
DATA_DIR  = os.path.join(BASE_DIR, 'data')
SAVE_DIR  = os.path.join(BASE_DIR, 'policies'); os.makedirs(SAVE_DIR,  exist_ok=True)
CHART_DIR = os.path.join(BASE_DIR, 'charts');   os.makedirs(CHART_DIR, exist_ok=True)

SEED = 42

# Anti-windup PI expert (behavioural policy for the offline buffer)
KP_EXPERT = -0.50
KI_EXPERT = -0.50 / 300.0
KD_EXPERT =  0.00
KW        = KI_EXPERT / KP_EXPERT
Q_MIN, Q_MAX = 0.0, 40.0
TS = 1.0
INT_E_CLIP = 40000.0

EPS_LOG       = 1e-4
LAMBDA_SMOOTH = 0.30
PREFILL_NOISE = 0.05

TREF_SUNNY, TREF_CLOUDY = 80.0, 65.0

# CQL(H) hyper-parameters
GAMMA         = 0.99
TAU           = 0.005
LR_A          = 1e-4
LR_C          = 3e-4
BATCH_SIZE    = 256
POLICY_DELAY  = 2
CQL_ALPHA     = 1.0
CQL_N_ACTIONS = 10
EPOCHS            = 100
UPDATES_PER_EPOCH = 500
EVAL_EVERY        = 5

# Datasets (full paths)
_SUNNY = ["21_10_2025__Sunny_Closed_Loop.xlsx", "22_10_2025__Sunny_Closed_Loop.xlsx",
          "23_10_2025__Sunny_Closed_Loop.xlsx", "24_10_2025__Sunny_Closed_Loop.xlsx"]
SUNNY_FILES    = [os.path.join(DATA_DIR, n) for n in _SUNNY]
CLOUDY_FILE    = os.path.join(DATA_DIR, "20_10_2025__Cloudy_Closed_Loop.xlsx")
INCLUDE_CLOUDY = True
DAY_LABELS     = ["s21", "s22", "s23", "s24"]
TRAIN_NFILES   = [1, 2, 3, 4]
