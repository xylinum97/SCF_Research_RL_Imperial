"""Actor / Critic networks + warm-start helpers (config-driven).

Two actor heads:
  * FlowActor  — state -> q in [Q_MIN, Q_MAX]         (Regular BC / CQL; action_dim = 1)
  * GainActor  — state -> gains in [0, 1]^ACTION_DIM  (CIRL / online;   action_dim = 3)

Two critics:
  * SingleCritic — DDPG (one Q)
  * TwinCritic   — TD3 / CQL (two Q, with Q1())

Factories ``Actor()`` / ``Critic()`` pick the right class from the active config
(``cfg.ACTOR_KIND`` / ``cfg.CRITIC_KIND``), so downstream code just calls ``Actor()``.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._ctx import get_cfg
from .physics import normalize_gains
from .util import DEVICE


class FlowActor(nn.Module):
    """state (in_dim) -> q in [q_min, q_max] via affine-sigmoid."""
    def __init__(self, in_dim=10, h=128, q_min=0.0, q_max=40.0):
        super().__init__()
        self.q_min = q_min; self.q_max = q_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h),      nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, 64),     nn.ReLU(),
            nn.Linear(64, 1),     nn.Sigmoid()
        )

    def forward(self, x):
        return self.q_min + (self.q_max - self.q_min) * self.net(x).squeeze(-1)


class GainActor(nn.Module):
    """state (in_dim) -> normalised gains in [0, 1]^out_dim."""
    def __init__(self, in_dim=9, h=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h),      nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, 64),     nn.ReLU(),
            nn.Linear(64, out_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SingleCritic(nn.Module):
    """DDPG single Q(s, a)."""
    def __init__(self, state_dim=9, action_dim=3, h=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.l1(x)); x = F.relu(self.l2(x))
        return self.l3(x)


class TwinCritic(nn.Module):
    """TD3 / CQL twin Q, with Q1()."""
    def __init__(self, state_dim=9, action_dim=3, h=256):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, h)
        self.l2 = nn.Linear(h, h); self.l3 = nn.Linear(h, 1)
        self.l4 = nn.Linear(state_dim + action_dim, h)
        self.l5 = nn.Linear(h, h); self.l6 = nn.Linear(h, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(x)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(x)))))
        return q1, q2

    def Q1(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))


# ── Factories driven by the active config ─────────────────────────────────────
def Actor():
    c = get_cfg()
    if getattr(c, "ACTOR_KIND", "gain") == "flow":
        return FlowActor(in_dim=c.STATE_DIM, q_min=c.Q_MIN, q_max=c.Q_MAX).to(DEVICE)
    return GainActor(in_dim=c.STATE_DIM, out_dim=c.ACTION_DIM).to(DEVICE)


def Critic():
    c = get_cfg()
    if getattr(c, "CRITIC_KIND", "single") == "twin":
        return TwinCritic(state_dim=c.STATE_DIM, action_dim=c.ACTION_DIM).to(DEVICE)
    return SingleCritic(state_dim=c.STATE_DIM, action_dim=c.ACTION_DIM).to(DEVICE)


# ── Warm-start helpers (gain actors) ──────────────────────────────────────────
def warm_start_actor(actor, ckpt=None):
    """Load the BC trunk if present, then re-centre the output head so the actor
    starts as a constant controller at cfg.START_GAIN (the expert)."""
    c = get_cfg()
    ckpt = ckpt or getattr(c, "BC_CKPT", "bc_cirl_setpoint_anti_windup_best.pt")
    path = os.path.join(c.SAVE_DIR, ckpt)
    if os.path.exists(path):
        try:
            actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            print(f"[warm-start] BC trunk loaded from {ckpt}")
        except Exception as ex:
            print(f"[warm-start] could not load {ckpt} ({ex}); random trunk")
    else:
        print("[warm-start] no BC checkpoint; random trunk")
    norm = normalize_gains(c.START_GAIN)
    logit = np.log(norm / (1.0 - norm)).astype(np.float32)
    with torch.no_grad():
        actor.net[-2].weight.zero_()
        actor.net[-2].bias.copy_(torch.tensor(logit))
    print(f"[warm-start] head re-centred to START_GAIN Kp={c.START_GAIN[0]} Ki={c.START_GAIN[1]:.5f}")
    return actor


def load_actor_raw(path):
    """Load a raw actor state-dict into a fresh Actor (no head re-centre)."""
    a = Actor()
    a.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    a.eval()
    print(f"[load] raw actor from {os.path.basename(path)}")
    return a
