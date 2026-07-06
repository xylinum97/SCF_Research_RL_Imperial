"""Anti-windup PI environment (config-driven).

The agent proposes normalised gains [Kp, Ki, Kw]; the env runs positional PI with
back-calculation anti-windup and returns a logarithmic tracking reward. Constants
(gain box, obs box, Q limits, reward weights) come from the active config.
"""
import numpy as np

from ._ctx import get_cfg
from .physics import normalize_obs, denormalize_gains, dataset_tref, solar_model


class SolarFieldEnv:
    """Positional PI + back-calc anti-windup; bumpless preload uses the ACTUAL gains."""
    def __init__(self, data, tref=None, lambda_smooth=None):
        self.c = get_cfg()
        self.d = data
        self.tref_seq = data.get('tref_seq', None)
        self.tref = (dataset_tref(data['name']) if tref is None else tref)
        self.lambda_smooth = self.c.LAMBDA_SMOOTH if lambda_smooth is None else lambda_smooth

    def _tref_now(self):
        if self.tref_seq is not None:
            return float(self.tref_seq[min(self.t, len(self.tref_seq) - 1)])
        return self.tref

    def reset(self):
        d = self.d
        self.t = 0
        self.tout = float(d['T_sc'][0]); self.tout_prev = self.tout
        self.q_prev = float(d['q'][0])
        self.int_e = 0.0; self._preloaded = False
        return self._obs()

    def _obs(self):
        tref = self._tref_now()
        e = tref - self.tout
        de = -(self.tout - self.tout_prev) / self.c.TS
        raw = np.array([self.tout, e, self.int_e, de, self.d['I_sol'][self.t],
                        self.d['Ta'][self.t], self.d['Tin'][self.t],
                        self.d['theta'][self.t], tref], dtype=np.float32)
        return normalize_obs(raw)

    def step(self, gains_norm):
        c = self.c; d = self.d; t = self.t
        TS = c.TS; Q_MIN = c.Q_MIN; Q_MAX = c.Q_MAX; INT = c.INT_E_CLIP; EPS = c.EPS_LOG
        tref = self._tref_now()
        e = tref - self.tout
        de = -(self.tout - self.tout_prev) / TS
        Kp, Ki, Kw = denormalize_gains(np.asarray(gains_norm, dtype=np.float32))
        if not self._preloaded:
            self.int_e = float(np.clip((self.q_prev - Kp * e) / Ki, -INT, INT))
            self._preloaded = True
        q_raw = Kp * e + Ki * self.int_e
        q = float(np.clip(q_raw, Q_MIN, Q_MAX))
        tout_next = solar_model(q, d['Tin'][t], d['I_sol'][t], d['Ta'][t], d['theta'][t], self.tout)
        err = tout_next - tref
        r_track = -float(np.log(err ** 2 + EPS))
        r_smooth = -self.lambda_smooth * float(np.log((q - self.q_prev) ** 2 + EPS))
        reward = r_track + r_smooth
        dq = q - self.q_prev
        self.int_e = float(np.clip(self.int_e + (e + Kw * (q - q_raw)) * TS, -INT, INT))
        self.tout_prev = self.tout; self.tout = tout_next; self.q_prev = q
        self.t += 1
        done = self.t >= d['N'] - 1
        info = {'Tout': tout_next, 'q': q, 'dq': dq, 'e': err}
        return (self._obs() if not done else np.zeros(c.STATE_DIM, np.float32)), reward, done, info
