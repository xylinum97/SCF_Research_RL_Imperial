"""Dataset loading, closed-loop rollouts, and metrics (config-driven).

Supports three control laws via ``control``:
  * 'pid_kw' — actor outputs [Kp,Ki,Kw]; q = Kp*e + Ki*int_e, anti-windup with Kw   (online)
  * 'pid_kd' — actor outputs [Kp,Ki,Kd]; q = Kp*e + Ki*int_e + Kd*de, Kw = Ki/Kp     (CIRL BC/CQL)
  * 'flow'   — actor outputs q directly; anti-windup Kw = Ki/Kp of the expert         (Regular BC/CQL)
"""
import os
import numpy as np
import pandas as pd
import torch

from ._ctx import get_cfg
from .physics import (normalize_obs, denormalize_gains, dataset_tref, solar_model)
from .env import SolarFieldEnv
from .util import DEVICE


def load_dataset(filename):
    c = get_cfg()
    base = getattr(c, "DATA_DIR", getattr(c, "BASE_DIR", "."))
    df = pd.read_excel(os.path.join(base, filename))
    d = dict(
        T_sc=df['T_sc'].to_numpy(float), Tin=df['Tin'].to_numpy(float),
        Ta=df['Ta'].to_numpy(float),     I_sol=df['I'].to_numpy(float),
        theta=df['theta'].to_numpy(float), q=df['q'].to_numpy(float),
        N=len(df), name=filename,
    )
    if 'T_ref' in df.columns:
        d['tref_seq'] = df['T_ref'].to_numpy(float)
    return d


def mae_rmse(Tout, tref):
    e = Tout - tref
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def peak_overshoot(Tout, tref):
    return float(np.max(Tout - tref))


def rollout_expert(data, tref=None):
    """Fixed anti-windup expert PI (Kp,Ki,Kd from config)."""
    c = get_cfg()
    seq = data.get('tref_seq', None)
    scalar_tref = dataset_tref(data['name']) if tref is None else tref
    d = data; tout = float(d['T_sc'][0]); tout_prev = tout; q_prev = float(d['q'][0])
    tr0 = float(seq[0]) if seq is not None else scalar_tref
    int_e = float(np.clip((q_prev - c.KP_EXPERT * (tr0 - tout)) / c.KI_EXPERT, -c.INT_E_CLIP, c.INT_E_CLIP))
    Kw = c.KI_EXPERT / c.KP_EXPERT
    Tout_l = []; q_l = []
    for t in range(d['N'] - 1):
        tr_t = float(seq[min(t, len(seq) - 1)]) if seq is not None else scalar_tref
        e = tr_t - tout; de = -(tout - tout_prev) / c.TS
        q_raw = c.KP_EXPERT * e + c.KI_EXPERT * int_e + c.KD_EXPERT * de
        q = float(np.clip(q_raw, c.Q_MIN, c.Q_MAX))
        tn = solar_model(q, d['Tin'][t], d['I_sol'][t], d['Ta'][t], d['theta'][t], tout)
        Tout_l.append(tn); q_l.append(q)
        int_e = float(np.clip(int_e + (e + Kw * (q - q_raw)) * c.TS, -c.INT_E_CLIP, c.INT_E_CLIP))
        tout_prev = tout; tout = tn; q_prev = q
    return np.array(Tout_l), np.array(q_l)


def rollout_policy(actor, data, tref=None):
    """Rollout a [Kp,Ki,Kw] gain actor through the SolarFieldEnv."""
    env = SolarFieldEnv(data, tref); obs = env.reset(); done = False
    Tout_l = []; q_l = []; dq_l = []
    actor.eval()
    with torch.no_grad():
        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = actor(s).cpu().numpy()[0]
            obs, r, done, info = env.step(a)
            Tout_l.append(info['Tout']); q_l.append(info['q']); dq_l.append(info['dq'])
    return np.array(Tout_l), np.array(q_l), np.array(dq_l)


def rollout_full(actor, data, tref=None):
    """Like rollout_policy but also returns the per-step denormalised gains."""
    env = SolarFieldEnv(data, tref); obs = env.reset(); done = False
    T = []; Q = []; G = []; actor.eval()
    with torch.no_grad():
        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a = actor(s).cpu().numpy()[0]; G.append(denormalize_gains(a))
            obs, r, done, info = env.step(a); T.append(info['Tout']); Q.append(info['q'])
    return np.array(T), np.array(Q), np.array(G)


def window(data, start, length):
    keys = ['T_sc', 'Tin', 'Ta', 'I_sol', 'theta', 'q']
    w = {k: data[k][start:start + length] for k in keys}
    w['N'] = len(w['T_sc']); w['name'] = data['name']
    if 'tref_seq' in data:
        w['tref_seq'] = data['tref_seq'][start:start + length]
    return w


def eval_gain(gn, days, win=None):
    """Mean (MAE, max-overshoot, mean-reward) of CONSTANT normalised gains gn
    rolled through the env over each day's start window (used by the gain search)."""
    c = get_cfg()
    win = getattr(c, 'EVAL_WIN', 2000) if win is None else win
    gn = np.asarray(gn, dtype=np.float32); maes = []; ovrs = []; rews = []
    for dv in days:
        wd = window(dv, 0, min(win, dv['N'])); env = SolarFieldEnv(wd, dataset_tref(dv['name']))
        env.reset(); done = False; T = []; rsum = 0.0; n = 0
        while not done:
            _, r, done, info = env.step(gn); T.append(info['Tout']); rsum += r; n += 1
        tr = dataset_tref(dv['name']); T = np.array(T)
        maes.append(mae_rmse(T, tr)[0]); ovrs.append(peak_overshoot(T, tr)); rews.append(rsum / max(n, 1))
    return float(np.mean(maes)), float(np.max(ovrs)), float(np.mean(rews))


def eval_actor(actor, days, win=None):
    """Mean (MAE, reward, overshoot) of a gain actor over each day's start window."""
    c = get_cfg()
    win = getattr(c, 'EVAL_WIN', 2000) if win is None else win
    maes = []; rews = []; ovrs = []; actor.eval()
    for dv in days:
        wd = window(dv, 0, min(win, dv['N'])); env = SolarFieldEnv(wd, dataset_tref(dv['name']))
        obs = env.reset(); done = False; T = []; rsum = 0.0; n = 0
        with torch.no_grad():
            while not done:
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                obs, r, done, info = env.step(actor(s).cpu().numpy()[0]); T.append(info['Tout']); rsum += r; n += 1
        tr = dataset_tref(dv['name']); T = np.array(T)
        maes.append(mae_rmse(T, tr)[0]); rews.append(rsum / max(n, 1)); ovrs.append(peak_overshoot(T, tr))
    return float(np.mean(maes)), float(np.mean(rews)), float(np.mean(ovrs))
