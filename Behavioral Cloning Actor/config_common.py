"""Common BC constants shared by config_regular.py and config_cirl.py.
Behaviour lives in ../main_script; only constants live here."""
import os
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
SAVE_DIR  = os.path.join(BASE_DIR, 'policies'); os.makedirs(SAVE_DIR,  exist_ok=True)
CHART_DIR = os.path.join(BASE_DIR, 'charts');   os.makedirs(CHART_DIR, exist_ok=True)

SEED = 42

# Anti-windup PI expert (the demonstrator being cloned)
KP_EXPERT = -0.50
KI_EXPERT = -0.50 / 300.0
KD_EXPERT =  0.00
KW        = KI_EXPERT / KP_EXPERT
Q_MIN, Q_MAX = 0.0, 40.0
TS = 1.0
INT_E_CLIP = 40000.0
EPS_LOG    = 1e-4

TREF_SUNNY, TREF_CLOUDY = 80.0, 65.0

# Reward weights (imitation-dominant)
W_IMIT  = 3
W_TRACK = 0.5

# Supervised training hyper-parameters
LEARNING_RATE = 5e-4
BATCH_SIZE    = 256
EPOCHS        = 390
TEST_SPLIT    = 0.20
VAL_SPLIT     = 0.15

# Datasets
_SUNNY = ["21_10_2025__Sunny_Closed_Loop.xlsx", "22_10_2025__Sunny_Closed_Loop.xlsx",
          "23_10_2025__Sunny_Closed_Loop.xlsx", "24_10_2025__Sunny_Closed_Loop.xlsx"]
SUNNY_FILES    = [os.path.join(DATA_DIR, n) for n in _SUNNY]
CLOUDY_FILE    = os.path.join(DATA_DIR, "20_10_2025__Cloudy_Closed_Loop.xlsx")
INCLUDE_CLOUDY = True
DAY_LABELS     = ["s21", "s22", "s23", "s24"]
TRAIN_NFILES   = [1, 2, 3, 4]
