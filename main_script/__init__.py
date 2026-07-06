"""main_script — shared library for the solar-thermal actor/critic experiments.

Usage in a notebook or script:

    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))  # reach main_script
    from main_script import *
    import config as cfg
    from config import *          # bring the folder's constants into scope
    configure(cfg)                # inject config into the shared classes

Everything below is then available with its usual name (Actor, Critic, DDPG, TD3,
CQL, SolarFieldEnv, ReplayBuffer, OUNoise, solar_model, normalize_obs, …).
"""
from ._ctx import configure, get_cfg
from .util import DEVICE, set_seed
from .physics import (solar_model, solar_model_torch, solarfield_model_np, solarfield_model_torch,
                      normalize_obs, normalize_states, denormalize_gains, denormalize_gains_torch, normalize_gains, dataset_tref)
from .networks import (FlowActor, GainActor, SingleCritic, TwinCritic,
                       Actor, Critic, warm_start_actor, load_actor_raw)
from .replay import ReplayBuffer, OUNoise
from .env import SolarFieldEnv
from .agents import DDPG, TD3, CQL
from .rollout import (load_dataset, mae_rmse, peak_overshoot, rollout_expert,
                      rollout_policy, rollout_full, window, eval_actor, eval_gain)

__all__ = [
    "configure", "get_cfg", "DEVICE", "set_seed",
    "solar_model", "solar_model_torch", "solarfield_model_np", "solarfield_model_torch",
    "normalize_obs", "normalize_states", "denormalize_gains", "denormalize_gains_torch", "normalize_gains", "dataset_tref",
    "FlowActor", "GainActor", "SingleCritic", "TwinCritic", "Actor", "Critic",
    "warm_start_actor", "load_actor_raw",
    "ReplayBuffer", "OUNoise", "SolarFieldEnv", "DDPG", "TD3", "CQL",
    "load_dataset", "mae_rmse", "peak_overshoot", "rollout_expert",
    "rollout_policy", "rollout_full", "window", "eval_actor", "eval_gain",
]
