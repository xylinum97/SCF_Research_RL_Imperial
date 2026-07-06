"""Solar-field plant model + normalisation helpers (config-driven).

The plant coefficients are universal across every experiment, so they live here.
Everything that varies between experiments (obs/gain boxes) comes from the active
config via ``_ctx.get_cfg()``.
"""
import numpy as np
from ._ctx import get_cfg

# ── Universal solar-field coefficients ────────────────────────────────────────
_V = 0.0023; _Ac = 2.37; _Ae = 2.00; _Cp = 4200; _rho = 997
_beta1 = 0.6500; _beta2 = 0.0163; _h1 = 2.4109; _h2 = 0.0011
_K1 = _beta1 / (_V * _rho * _Cp)
_K2 = _h1 * _Ac / (_V * _rho * _Cp)
_K3 = _h2 * _Ac / (_V * _rho * _Cp)
_K4 = 1.0 / (_V * 23 * 60 * 1000)


def solar_model(q, Tin, Ig, Ta, theta, Tout_m):
    """Numpy one-step outlet-temperature update."""
    Tmed = (Tin + Tout_m) / 2.0
    cos_t = np.cos(np.deg2rad(180.0 - theta))
    dTout = (_K1 * Ig * (_Ac + _beta2 * _Ae * cos_t) - _K2 * (Tmed - Ta)
             - _K3 * (Tmed - Ta) ** 2 + _K4 * q * (Tin - Tout_m))
    return Tout_m + dTout


def solar_model_torch(q, Tin, Ig, Ta, theta, Tout_m):
    """Torch one-step outlet-temperature update (differentiable)."""
    import torch
    Tmed = (Tin + Tout_m) / 2.0
    cos_t = torch.cos(torch.deg2rad(180.0 - theta))
    dTout = (_K1 * Ig * (_Ac + _beta2 * _Ae * cos_t) - _K2 * (Tmed - Ta)
             - _K3 * (Tmed - Ta) ** 2 + _K4 * q * (Tin - Tout_m))
    return Tout_m + dTout


# alias used by some scripts
solarfield_model_np = solar_model
solarfield_model_torch = solar_model_torch


# ── Normalisation (config-driven boxes) ───────────────────────────────────────
def normalize_obs(s):
    c = get_cfg()
    return (s - c.OBS_LOW) / (c.OBS_HIGH - c.OBS_LOW + 1e-8)


normalize_states = normalize_obs   # alias used by the BC/CQL scripts


def denormalize_gains(g):
    c = get_cfg()
    return g * (c.GAIN_HIGH - c.GAIN_LOW + 1e-8) + c.GAIN_LOW


def normalize_gains(gain):
    c = get_cfg()
    return np.clip((gain - c.GAIN_LOW) / (c.GAIN_HIGH - c.GAIN_LOW + 1e-8), 1e-4, 1 - 1e-4)


def denormalize_gains_torch(gains_norm):
    """Differentiable [0,1] -> physical gain mapping (for BC/CIRL training)."""
    import torch
    c = get_cfg()
    low  = torch.tensor(c.GAIN_LOW,  dtype=torch.float32)
    high = torch.tensor(c.GAIN_HIGH, dtype=torch.float32)
    return gains_norm * (high - low + 1e-8) + low


def dataset_tref(name):
    c = get_cfg()
    return c.TREF_CLOUDY if "Cloudy" in str(name) else c.TREF_SUNNY
