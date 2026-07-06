"""Actor-critic agents (config-driven): DDPG, TD3, CQL(H).

Hyper-parameters (gamma, tau, lrs, delays, cql alpha…) are read from the active
config with sensible defaults, so a folder only overrides what it needs.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from ._ctx import get_cfg
from .networks import Actor, SingleCritic, TwinCritic, warm_start_actor
from .util import DEVICE


def _single_critic():
    c = get_cfg()
    return SingleCritic(state_dim=c.STATE_DIM, action_dim=c.ACTION_DIM).to(DEVICE)


def _twin_critic():
    c = get_cfg()
    return TwinCritic(state_dim=c.STATE_DIM, action_dim=c.ACTION_DIM).to(DEVICE)


def _h(name, default):
    return getattr(get_cfg(), name, default)


class DDPG:
    def __init__(self, gamma=None, tau=None, lr_a=None, lr_c=None, warm_start=True):
        gamma = _h('GAMMA', 0.99) if gamma is None else gamma
        tau   = _h('TAU', 0.01)  if tau   is None else tau
        lr_a  = _h('LR_A', 1e-4) if lr_a  is None else lr_a
        lr_c  = _h('LR_C', 1e-3) if lr_c  is None else lr_c
        self.actor = Actor()
        if warm_start:
            warm_start_actor(self.actor)
        self.actor_target = copy.deepcopy(self.actor); self.actor_target.eval()
        self.critic = _single_critic()
        self.critic_target = copy.deepcopy(self.critic); self.critic_target.eval()
        self.opt_a = torch.optim.Adam(self.actor.parameters(),  lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma = gamma; self.tau = tau
        self.bc_weight = 0.0; self.bc_target = None; self.bc_alpha = 2.5

    @torch.no_grad()
    def act(self, obs, noise=None):
        self.actor.eval()
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if noise is not None:
            a = np.clip(a + noise.sample(), 0.0, 1.0)
        return a.astype(np.float32)

    def update(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return None
        self.actor.eval()
        s, a, r, s2, d = buffer.sample(batch_size)
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = r + self.gamma * (1 - d) * self.critic_target(s2, a2)
        q = self.critic(s, a)
        loss_c = F.mse_loss(q, q_target)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        a_pred = self.actor(s)
        Q = self.critic(s, a_pred)
        if self.bc_weight > 0.0 and self.bc_target is not None:
            lmbda = self.bc_alpha / (Q.abs().mean().detach() + 1e-6)
            loss_a = -lmbda * Q.mean() + self.bc_weight * F.mse_loss(a_pred, self.bc_target.expand_as(a_pred))
        else:
            loss_a = -Q.mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
        for tp, p in zip(self.actor_target.parameters(),  self.actor.parameters()):
            tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
        return float(loss_c.item()), float(loss_a.item())


class TD3:
    def __init__(self, gamma=None, tau=None, lr_a=None, lr_c=None,
                 policy_delay=None, target_noise=None, noise_clip=None, warm_start=True):
        gamma = _h('GAMMA', 0.99) if gamma is None else gamma
        tau   = _h('TAU', 0.01)  if tau   is None else tau
        lr_a  = _h('LR_A', 1e-4) if lr_a  is None else lr_a
        lr_c  = _h('LR_C', 1e-3) if lr_c  is None else lr_c
        policy_delay = _h('POLICY_DELAY', 2)   if policy_delay is None else policy_delay
        target_noise = _h('TARGET_NOISE', 0.1) if target_noise is None else target_noise
        noise_clip   = _h('NOISE_CLIP', 0.25)  if noise_clip   is None else noise_clip
        self.actor = Actor()
        if warm_start:
            warm_start_actor(self.actor)
        self.actor_target = copy.deepcopy(self.actor); self.actor_target.eval()
        self.critic = _twin_critic()
        self.critic_target = copy.deepcopy(self.critic); self.critic_target.eval()
        self.opt_a = torch.optim.Adam(self.actor.parameters(),  lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma = gamma; self.tau = tau
        self.policy_delay = policy_delay; self.target_noise = target_noise
        self.noise_clip = noise_clip; self.total_it = 0
        self.bc_weight = 0.0; self.bc_target = None; self.bc_alpha = 2.5

    @torch.no_grad()
    def act(self, obs, noise=None):
        self.actor.eval()
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if noise is not None:
            a = np.clip(a + noise.sample(), 0.0, 1.0)
        return a.astype(np.float32)

    def update(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return None
        self.total_it += 1
        self.actor.eval()
        s, a, r, s2, d = buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(0.0, 1.0)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)
        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        loss_a = None
        if self.total_it % self.policy_delay == 0:
            a_pred = self.actor(s)
            Q = self.critic.Q1(s, a_pred)
            if self.bc_weight > 0.0 and self.bc_target is not None:
                lmbda = self.bc_alpha / (Q.abs().mean().detach() + 1e-6)
                loss_a = -lmbda * Q.mean() + self.bc_weight * F.mse_loss(a_pred, self.bc_target.expand_as(a_pred))
            else:
                loss_a = -Q.mean()
            self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
            for tp, p in zip(self.actor_target.parameters(),  self.actor.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
            loss_a = float(loss_a.item())
        return float(loss_c.item()), loss_a


class CQL:
    """Blind Conservative Q-Learning (CQL-H), offline. Twin critic + conservative
    regulariser. Works for both flow (action_dim=1) and gain (action_dim=3) actors;
    the flow action is normalised to [0,1] for the critic via ``_norm_act``."""
    def __init__(self, gamma=None, tau=None, lr_a=None, lr_c=None,
                 policy_delay=None, cql_alpha=None, cql_n_actions=None):
        c = get_cfg()
        self.gamma = _h('GAMMA', 0.99) if gamma is None else gamma
        self.tau   = _h('TAU', 0.005)  if tau   is None else tau
        lr_a  = _h('LR_A', 1e-4) if lr_a is None else lr_a
        lr_c  = _h('LR_C', 3e-4) if lr_c is None else lr_c
        self.policy_delay  = _h('POLICY_DELAY', 2)    if policy_delay  is None else policy_delay
        self.cql_alpha     = _h('CQL_ALPHA', 1.0)     if cql_alpha     is None else cql_alpha
        self.cql_n_actions = _h('CQL_N_ACTIONS', 10)  if cql_n_actions is None else cql_n_actions
        self.action_dim = c.ACTION_DIM
        self._flow = getattr(c, "ACTOR_KIND", "gain") == "flow"
        self._q_min = c.Q_MIN; self._q_max = c.Q_MAX
        self.actor = Actor()
        self.actor_target = copy.deepcopy(self.actor); self.actor_target.eval()
        self.critic = _twin_critic()
        self.critic_target = copy.deepcopy(self.critic); self.critic_target.eval()
        self.opt_a = torch.optim.Adam(self.actor.parameters(),  lr=lr_a)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self._ctr = 0

    def _norm_act(self, x):
        """Actor output -> normalised action in [0,1] for the critic."""
        if self._flow:
            return ((x - self._q_min) / (self._q_max - self._q_min)).unsqueeze(1)
        return x

    def update(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return None
        self.actor.eval()
        self._ctr += 1
        s, a, r, s2, d = buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(a) * 0.02).clamp(-0.05, 0.05)
            a2 = (self._norm_act(self.actor_target(s2)) + noise).clamp(0.0, 1.0)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)
        q1, q2 = self.critic(s, a)
        td_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        B, n = s.shape[0], self.cql_n_actions
        s_rep = s.unsqueeze(1).repeat(1, n, 1).view(B * n, -1)
        rand_a = torch.rand(B * n, self.action_dim, device=DEVICE)
        with torch.no_grad():
            curr_a = self._norm_act(self.actor(s_rep))
        q1_rand, _ = self.critic(s_rep, rand_a)
        q1_curr, _ = self.critic(s_rep, curr_a)
        cat = torch.cat([q1_rand.view(B, n), q1_curr.view(B, n)], dim=1)
        logmeanexp = torch.logsumexp(cat, dim=1, keepdim=True) - np.log(cat.shape[1])
        cql_gap = (logmeanexp - q1).mean()
        loss_c = td_loss + self.cql_alpha * cql_gap
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        loss_a = None
        if self._ctr % self.policy_delay == 0:
            loss_a = -self.critic.Q1(s, self._norm_act(self.actor(s))).mean()
            self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
            for tp, p in zip(self.actor_target.parameters(),  self.actor.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
            loss_a = float(loss_a.item())
        return float(loss_c.item()), float(cql_gap.item()), loss_a
