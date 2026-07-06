"""Config context: a single active config object injected by ``configure(cfg)``.

The shared classes/functions read constants from this object, so an experiment
folder only has to define its own ``config.py`` and call ``configure(config)``.
"""

CFG = None


def configure(cfg):
    """Register the active config (usually the folder's ``config`` module)."""
    global CFG
    CFG = cfg
    return cfg


def get_cfg():
    if CFG is None:
        raise RuntimeError("main_script is not configured — call configure(cfg) first "
                           "(e.g. `import config as cfg; configure(cfg)`).")
    return CFG
