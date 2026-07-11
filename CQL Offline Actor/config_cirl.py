"""CQL CIRL config — actor outputs [Kp,Ki,Kw] (9-D state, action_dim=3).
Kw is the back-calculation anti-windup gain, pinned to the expert constant
KW = Ki/Kp (zero-width box), so the actor only shapes Kp and Ki offline."""
import numpy as np
from config_common import *

STATE_DIM  = 9
ACTION_DIM = 3
ACTOR_KIND = 'gain'
OBS_LOW  = np.array([ 30.0, -40.0, -40000.0, -5.0,    0.0,  0.0, 10.0,   0.0, 55.0], dtype=np.float32)
OBS_HIGH = np.array([100.0,  55.0,  40000.0,  5.0, 1200.0, 50.0, 90.0, 180.0, 85.0], dtype=np.float32)
# [Kp, Ki, Kw] box (Kw pinned to the expert constant KW = Ki/Kp)
GAIN_LOW  = np.array([-0.625, -0.625 / 300, KW], dtype=np.float32)
GAIN_HIGH = np.array([-0.375, -0.125 / 300, KW], dtype=np.float32)
START_GAIN = np.array([KP_EXPERT, KI_EXPERT, KW], dtype=np.float32)
