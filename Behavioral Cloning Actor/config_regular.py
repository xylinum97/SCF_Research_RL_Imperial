"""BC Regular config — actor clones flow q directly (10-D state, action_dim=1)."""
import numpy as np
from config_common import *

STATE_DIM  = 10
ACTION_DIM = 1
ACTOR_KIND = 'flow'
OBS_LOW  = np.array([ 30.0, -40.0, -40000.0, -5.0,    0.0,  0.0, 10.0,   0.0, Q_MIN, 55.0], dtype=np.float32)
OBS_HIGH = np.array([100.0,  55.0,  40000.0,  5.0, 1200.0, 50.0, 90.0, 180.0, Q_MAX, 85.0], dtype=np.float32)
GAIN_LOW  = np.array([0.0], dtype=np.float32)     # flow actor: no gain box
GAIN_HIGH = np.array([1.0], dtype=np.float32)
START_GAIN = np.array([0.0], dtype=np.float32)
LAMBDA_MIN = 0.05
LAMBDA_MAX = 0.30
