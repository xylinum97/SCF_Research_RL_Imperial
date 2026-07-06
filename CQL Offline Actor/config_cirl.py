"""CQL CIRL config — actor outputs [Kp,Ki,Kd] (9-D state, action_dim=3)."""
import numpy as np
from config_common import *

STATE_DIM  = 9
ACTION_DIM = 3
ACTOR_KIND = 'gain'
OBS_LOW  = np.array([ 30.0, -40.0, -40000.0, -5.0,    0.0,  0.0, 10.0,   0.0, 55.0], dtype=np.float32)
OBS_HIGH = np.array([100.0,  55.0,  40000.0,  5.0, 1200.0, 50.0, 90.0, 180.0, 85.0], dtype=np.float32)
# [Kp, Ki, Kd] box (Kd pinned to 0)
GAIN_LOW  = np.array([-0.625, -0.625 / 300, 0], dtype=np.float32)
GAIN_HIGH = np.array([-0.375, -0.125 / 300, 0], dtype=np.float32)
START_GAIN = np.array([KP_EXPERT, KI_EXPERT, KD_EXPERT], dtype=np.float32)
