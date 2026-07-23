"""Multi-seed OFFLINE comparison: BC vs CQL, faithful to the single-seed pipelines.

Seed 42 replicates the original BC-vs-CQL offline comparison
(`BC_vs_CQL_Offline_Policy_Comparison.ipynb`: Expert 1.224 / BC 1.224 / CQL 0.243),
and every other seed reruns the IDENTICAL pipelines so the comparison is
apples-to-apples with seed 42:

  * BC  — the BC package trainer at its FULL budget (EPOCHS=390), [Kp,Ki,Kw] box
          (Kw pinned to the expert constant Ki/Kp),
          trained on the 4 original sunny days + the cloudy day.
  * CQL — the CQL_CIRL.ipynb pipeline in the ORIGINAL [Kp,Ki,Kw] box: offline
          buffer pre-filled from that seed's raw BC actor (noise 0.05) over the 5
          original days, then 100 epochs x 242 CQL-H updates (the realised budget
          of the original run), best checkpoint by the cloudy-x3 proxy MAE.

Each seed's BC and CQL actors are saved under multiseed/policies/. Evaluation is
zero-shot on Juan's 4 new days with the SAME rollout code as the comparison
notebook (BC and CQL both via 'pid_kw', each in its own box). Outputs:

  * results.json                                  — per-seed per-day MAEs
  * traces.npz                                    — Tout traces for the notebook
  * charts/multiseed_seed{S}_tracking_4days.png   — 4 charts x 4 sub-charts (one per Juan day)
  * charts/multiseed_offline_bc_vs_cql.png        — mean +/- std summary bars

    python run_multiseed.py            # orchestrate: train missing seeds, then report
    python run_multiseed.py bc 42      # (internal) one BC training worker
    python run_multiseed.py cql 42     # (internal) one CQL training worker
    python run_multiseed.py report     # evaluation + charts only (needs saved policies)
"""
import sys, os, json, subprocess
import numpy as np

# Windows consoles default to cp1252 and crash on unicode (λ, °C, —). Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)                      # BC vs CQL Comparison
FG   = os.path.dirname(PKG)                       # For GitHub
POL  = os.path.join(HERE, "policies"); os.makedirs(POL, exist_ok=True)

JUAN = ["16_06_2026__Sunny_Closed_Loop.xlsx", "17_06_2026__Sunny_Closed_Loop.xlsx",
        "18_06_2026__Sunny_Closed_Loop.xlsx", "19_06_2026__Sunny_Closed_Loop.xlsx"]
ORIG = ["21_10_2025__Sunny_Closed_Loop.xlsx", "22_10_2025__Sunny_Closed_Loop.xlsx",
        "23_10_2025__Sunny_Closed_Loop.xlsx", "24_10_2025__Sunny_Closed_Loop.xlsx",
        "20_10_2025__Cloudy_Closed_Loop.xlsx"]
SEEDS = [int(s) for s in os.environ.get("MULTISEED_SEEDS", "42,36,27,21,14,23,39,94,15,88,26,83,51,18,68,86,33,47,99,57,52,71,78,89,49,84,13,30,81,62").split(",")]

# ── frozen CQL pipeline knobs (== the original CQL_CIRL.ipynb run) ────────────
CQL_EPOCHS        = 100
CQL_UPD_PER_EPOCH = 242          # realised auto-budget of the original run (24,200 updates)
CQL_EVAL_EVERY    = 10
BATCH_SIZE        = 256
PREFILL_NOISE     = 0.05
CLOUDY_WEIGHT     = 3.0
PROXY_WIN         = 2000

# The two gain boxes (identical to BC_vs_CQL_Offline_Policy_Comparison.ipynb)
BC_LOW  = np.array([-0.625, -0.625 / 300, 1.0 / 300], np.float32)   # [Kp,Ki,Kw] — Kw pinned to expert Ki/Kp
BC_HIGH = np.array([-0.375, -0.125 / 300, 1.0 / 300], np.float32)

SMOKE = os.environ.get("MULTISEED_SMOKE") == "1"   # tiny budgets for a pipeline test


def bc_path(seed):  return os.path.join(POL, f"bc_seed{seed}.pt")
def cql_path(seed): return os.path.join(POL, f"cql_seed{seed}.pt")


# ═══════════════════════════════ workers ═══════════════════════════════════════
def worker_bc(seed):
    """Retrain the BC CIRL policy with the package trainer at FULL budget."""
    import importlib.util
    path = os.path.join(FG, "Behavioral Cloning Actor", "train", "BC_CIRL_Setpoint_Anti_Windup.py")
    spec = importlib.util.spec_from_file_location("bctr", path)
    T = importlib.util.module_from_spec(spec); sys.modules["bctr"] = T
    spec.loader.exec_module(T)
    T.SEED = seed
    if SMOKE:
        T.EPOCHS = 2
    import torch
    files = list(T.SUNNY_FILES) + [T.CLOUDY_FILE]
    print(f"[bc seed={seed}] training on {len(files)} files, EPOCHS={T.EPOCHS}", flush=True)
    out = T.load_data(files)
    _, actor, _, _ = T.train(*out)                # save_path=None: never touch package policies
    torch.save(actor.state_dict(), bc_path(seed))
    print(f"[bc seed={seed}] saved -> {bc_path(seed)}", flush=True)


def worker_cql(seed):
    """Replicate the CQL_CIRL.ipynb offline pipeline ([Kp,Ki,Kw] box) at this seed."""
    os.chdir(PKG)                                  # config.py resolves BASE_DIR from cwd
    sys.path.insert(0, PKG); sys.path.insert(0, FG)
    import torch
    from main_script import (configure, set_seed, DEVICE, CQL, ReplayBuffer, SolarFieldEnv,
                             GainActor, load_dataset, window, rollout_policy, mae_rmse,
                             dataset_tref)
    import config as cfg
    configure(cfg); set_seed(seed)

    epochs, upd = (2, 20) if SMOKE else (CQL_EPOCHS, CQL_UPD_PER_EPOCH)

    data_dir = os.path.join(FG, "CQL Offline Actor", "data")
    train_days = [load_dataset(os.path.join(data_dir, f)) for f in ORIG]

    # offline dataset = THIS seed's raw BC actor rolled out over the 5 original days
    bc = GainActor(in_dim=cfg.STATE_DIM, out_dim=cfg.ACTION_DIM).to(DEVICE)
    bc.load_state_dict(torch.load(bc_path(seed), map_location=DEVICE, weights_only=True))
    bc.eval()
    buffer = ReplayBuffer()
    for ds in train_days:
        env = SolarFieldEnv(ds, dataset_tref(ds["name"]))
        obs = env.reset(); done = False
        with torch.no_grad():
            while not done:
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a = bc(s).cpu().numpy()[0]
                a = np.clip(a + np.random.randn(3).astype(np.float32) * PREFILL_NOISE, 0.0, 1.0)
                obs2, r, done, info = env.step(a)
                buffer.add(obs, a, r, obs2, done)
                obs = obs2
    print(f"[cql seed={seed}] offline buffer filled from BC rollouts: {len(buffer)} transitions",
          flush=True)

    def quick_avg_mae(actor):
        """Cloudy-x3 weighted MAE over the first PROXY_WIN steps of each original day."""
        maes = []; ws = []
        for ds in train_days:
            wd = window(ds, 0, min(PROXY_WIN, ds["N"])); tr = dataset_tref(ds["name"])
            T, _, _ = rollout_policy(actor, wd, tr)
            maes.append(mae_rmse(T, tr)[0])
            ws.append(CLOUDY_WEIGHT if "Cloudy" in ds["name"] else 1.0)
        maes = np.array(maes); ws = np.array(ws)
        return float((maes * ws).sum() / ws.sum())

    agent = CQL(gamma=0.99, tau=0.005, lr_a=1e-4, lr_c=3e-4,
                policy_delay=2, cql_alpha=1.0, cql_n_actions=10)
    best = float("inf"); save = cql_path(seed)
    print(f"[cql seed={seed}] proxy MAE before training = {quick_avg_mae(agent.actor):.3f} | "
          f"{epochs} epochs x {upd} updates", flush=True)
    for epoch in range(1, epochs + 1):
        lc = []; cg = []
        for _ in range(upd):
            out = agent.update(buffer, BATCH_SIZE)
            if out is not None:
                lc.append(out[0]); cg.append(out[1])
        if epoch % CQL_EVAL_EVERY == 0 or epoch in (1, epochs):
            mae = quick_avg_mae(agent.actor); tag = ""
            if mae < best:
                best = mae; torch.save(agent.actor.state_dict(), save); tag = "  <- saved best"
            print(f"[cql seed={seed}] epoch {epoch:3d}/{epochs} | loss_c={np.mean(lc):8.4f} "
                  f"cql_gap={np.mean(cg):+.4f} | proxy MAE={mae:.3f} (best {best:.3f}){tag}",
                  flush=True)
    if not os.path.exists(save):
        torch.save(agent.actor.state_dict(), save)
    print(f"[cql seed={seed}] saved -> {save}", flush=True)


# ═══════════════════════════ evaluation + charts ══════════════════════════════
def report():
    """Zero-shot evaluation of every seed's BC + CQL actor on Juan's 4 days, with
    the SAME rollout code as BC_vs_CQL_Offline_Policy_Comparison.ipynb. Produces
    results.json, traces.npz, one 4-subplot tracking chart per seed, and the
    summary bar chart."""
    os.chdir(PKG)
    sys.path.insert(0, PKG); sys.path.insert(0, FG)
    import torch
    from main_script import (configure, DEVICE, GainActor, load_dataset, normalize_obs,
                             solar_model, mae_rmse, rollout_expert, dataset_tref)
    import config as cfg
    configure(cfg)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    def load_actor(path):
        a = GainActor(in_dim=cfg.STATE_DIM, out_dim=cfg.ACTION_DIM).to(DEVICE)
        a.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        a.eval(); return a

    def rollout_gains(actor, data, glow, ghigh, mode):
        """== the comparison notebook: mode='pid_kd' (BC box) or 'pid_kw' (CQL box)."""
        tref = dataset_tref(data["name"])
        Tin = data["Tin"]; I = data["I_sol"]; Ta = data["Ta"]; th = data["theta"]
        tout = float(data["T_sc"][0]); tout_prev = tout; q_prev = float(data["q"][0]); int_e = None
        T = []; A = {"Kp": [], "Ki": [], "Kw": [], "q": []}
        with torch.no_grad():
            for t in range(data["N"] - 1):
                e = tref - tout; de = -(tout - tout_prev) / cfg.TS
                ie = 0.0 if int_e is None else int_e
                state = np.array([tout, e, ie, de, I[t], Ta[t], Tin[t], th[t], tref], np.float32)
                s = torch.from_numpy(normalize_obs(state).reshape(1, -1)).to(DEVICE)
                g = np.clip(actor(s).cpu().numpy()[0], 0, 1) * (ghigh - glow + 1e-8) + glow
                Kp, Ki, g3 = g
                if int_e is None:      # bumpless preload with the POLICY's own gains
                    int_e = float(np.clip((q_prev - Kp * e) / Ki, -cfg.INT_E_CLIP, cfg.INT_E_CLIP))
                if mode == "pid_kd":
                    q_raw = Kp * e + Ki * int_e + g3 * de; Kw = Ki / Kp
                else:
                    q_raw = Kp * e + Ki * int_e; Kw = g3
                q = float(np.clip(q_raw, cfg.Q_MIN, cfg.Q_MAX))
                tn = solar_model(q, Tin[t], I[t], Ta[t], th[t], tout)
                T.append(tn)
                A["Kp"].append(Kp); A["Ki"].append(Ki); A["Kw"].append(Kw); A["q"].append(q)
                int_e = float(np.clip(int_e + (e + Kw * (q - q_raw)) * cfg.TS,
                                      -cfg.INT_E_CLIP, cfg.INT_E_CLIP))
                tout_prev = tout; tout = tn; q_prev = q
        return np.array(T), {k: np.array(v, np.float32) for k, v in A.items()}

    days = [load_dataset(f) for f in JUAN]
    trefs = [dataset_tref(d["name"]) for d in days]
    traces = {}; results = {"schema": 2, "days": [d["name"] for d in days],
                            "expert": {}, "bc": {}, "cql": {}}

    print("[report] expert rollouts ...", flush=True)
    exp_maes = []
    for i, d in enumerate(days):
        Te, Qe = rollout_expert(d)
        traces[f"expert_{i}"] = Te.astype(np.float32)
        traces[f"expert_{i}_q"] = Qe.astype(np.float32)
        exp_maes.append(mae_rmse(Te, trefs[i])[0])
    results["expert"] = {"per_day": exp_maes, "mean": float(np.mean(exp_maes))}

    # only seeds with BOTH policies are comparable apples-to-apples
    seeds_done = [s for s in SEEDS if os.path.exists(bc_path(s)) and os.path.exists(cql_path(s))]
    skipped = [s for s in SEEDS if s not in seeds_done]
    if skipped:
        print(f"[report] seeds missing a BC or CQL policy (skipped): {skipped}", flush=True)

    for seed in seeds_done:
        for method, path, box, mode in [("bc",  bc_path(seed),  (BC_LOW, BC_HIGH), "pid_kw"),
                                        ("cql", cql_path(seed), (np.asarray(cfg.GAIN_LOW),
                                                                 np.asarray(cfg.GAIN_HIGH)), "pid_kw")]:
            actor = load_actor(path); maes = []
            for i, d in enumerate(days):
                T, A = rollout_gains(actor, d, box[0], box[1], mode)
                traces[f"{method}_{seed}_{i}"] = T.astype(np.float32)
                for k, v in A.items():
                    traces[f"{method}_{seed}_{i}_{k}"] = v
                maes.append(mae_rmse(T, trefs[i])[0])
            results[method][str(seed)] = {"per_day": maes, "mean": float(np.mean(maes))}
            print(f"[report] {method.upper():3s} seed={seed}  per-day MAE="
                  f"{[round(m, 3) for m in maes]}  mean={np.mean(maes):.3f}", flush=True)

    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(HERE, "traces.npz"), **traces)

    # ── 4 tracking charts: one per seed, 4 sub-charts (one per Juan day) ──────
    chart_dir = cfg.CHART_DIR
    for seed in seeds_done:
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for i, d in enumerate(days):
            ax = axes[i // 2, i % 2]; tr = trefs[i]
            Te = traces[f"expert_{i}"]; Tb = traces[f"bc_{seed}_{i}"]; Tc = traces[f"cql_{seed}_{i}"]
            t = np.arange(len(Tb))
            ax.axhline(tr, ls="--", c="k", lw=1, label=f"T_ref = {tr:.0f} C")
            ax.plot(t, Te[:len(t)], c="gray",    ls="-.", lw=1.2, label="Expert PI")
            ax.plot(t, Tb,          c="#2ca02c", lw=1.3,  label="BC")
            ax.plot(t, Tc,          c="#d62728", lw=1.3,  label="CQL")
            mb = results["bc"][str(seed)]["per_day"][i]; mc = results["cql"][str(seed)]["per_day"][i]
            ax.set_title(f"{d['name'][:11]}   |   BC MAE={mb:.3f}   CQL MAE={mc:.3f}", fontsize=10)
            ax.set_ylabel("Outlet temperature [C]"); ax.set_xlabel("time step"); ax.grid(alpha=.3)
            if i == 0:
                ax.legend(fontsize=9)
        fig.suptitle(f"Seed {seed} — Expert vs BC vs CQL on Juan's 4 new days (offline, zero-shot)",
                     fontsize=12)
        fig.tight_layout()
        out = os.path.join(chart_dir, f"multiseed_seed{seed}_tracking_4days.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[report] chart -> charts/{os.path.basename(out)}", flush=True)

    # ── action charts: per seed, actions [Kp, Ki, Kw] + applied flow q vs time ─
    ACT_ROWS = [("Kp", "Kp  [-]"), ("Ki", "Ki  [-]"), ("Kw", "Kw  (anti-windup)  [-]"),
                ("q",  "flow q  [L/min]")]
    KP_E, KI_E = cfg.KP_EXPERT, cfg.KI_EXPERT
    EXP_G = {"Kp": KP_E, "Ki": KI_E, "Kw": KI_E / KP_E}
    for seed in seeds_done:
        fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex="col")
        for j, d in enumerate(days):
            for r, (key, ylab) in enumerate(ACT_ROWS):
                ax = axes[r, j]
                Ab = traces[f"bc_{seed}_{j}_{key}"]; Ac = traces[f"cql_{seed}_{j}_{key}"]
                t = np.arange(len(Ab))
                if key == "q":
                    Qe = traces[f"expert_{j}_q"]
                    ax.plot(t, Qe[:len(t)], c="gray", ls="-.", lw=1.0, label="Expert PI")
                else:
                    ax.axhline(EXP_G[key], ls="-.", c="gray", lw=1.0,
                               label=f"Expert ({EXP_G[key]:.4f})")
                ax.plot(t, Ab, c="#2ca02c", lw=1.0, label="BC")
                ax.plot(t, Ac, c="#d62728", lw=1.0, label="CQL")
                if r == 0:
                    ax.set_title(d["name"][:11], fontsize=10)
                if j == 0:
                    ax.set_ylabel(ylab)
                if r == len(ACT_ROWS) - 1:
                    ax.set_xlabel("time step")
                ax.grid(alpha=.3)
                if r == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"Seed {seed} — actions applied at each time step on Juan's 4 new days "
                     f"(BC [Kp,Ki,Kw pinned to Ki/Kp] vs CQL [Kp,Ki,Kw]; q = resulting flow)", fontsize=12)
        fig.tight_layout()
        out = os.path.join(chart_dir, f"multiseed_seed{seed}_actions_4days.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"[report] chart -> charts/{os.path.basename(out)}", flush=True)

    # ── summary bar chart (mean +/- std over seeds) ───────────────────────────
    bc_v  = np.array([results["bc"][str(s)]["mean"]  for s in seeds_done])
    cq_v  = np.array([results["cql"][str(s)]["mean"] for s in seeds_done])
    if len(bc_v) and len(cq_v):
        exp_m = results["expert"]["mean"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar([0, 1], [bc_v.mean(), cq_v.mean()], yerr=[bc_v.std(), cq_v.std()],
               capsize=8, color=["#2ca02c", "#d62728"], alpha=.75, width=.55)
        ax.scatter([0] * len(bc_v), bc_v, color="k", zorder=3, s=25)
        ax.scatter([1] * len(cq_v), cq_v, color="k", zorder=3, s=25)
        ax.axhline(exp_m, ls="--", c="gray", label=f"Expert PI ({exp_m:.3f})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["BC", "CQL"])
        ax.set_ylabel("Closed-loop MAE on Juan's days [C]")
        ax.set_title(f"Offline BC vs CQL over {len(bc_v)} seeds (mean +/- std)")
        ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(chart_dir, "multiseed_offline_bc_vs_cql.png"), dpi=150)
        plt.close(fig)
        print("\n" + "=" * 52 + "\nOFFLINE BC vs CQL — mean +/- std over seeds (Juan days)\n" + "=" * 52)
        print(f"Expert  MAE = {exp_m:.3f}")
        print(f"BC      MAE = {bc_v.mean():.3f} +/- {bc_v.std():.3f}   vals={[round(float(x), 3) for x in bc_v]}")
        print(f"CQL     MAE = {cq_v.mean():.3f} +/- {cq_v.std():.3f}   vals={[round(float(x), 3) for x in cq_v]}")
    print("MULTISEED DONE", flush=True)


# ═══════════════════════════════ orchestrator ══════════════════════════════════
def orchestrate():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # avoid Windows cp1252 crash on λ/°C
    for method, path_fn in [("bc", bc_path), ("cql", cql_path)]:
        for seed in SEEDS:
            if os.path.exists(path_fn(seed)):
                print(f"[skip] {method} seed={seed} (policy exists)", flush=True); continue
            print(f"[run] {method} seed={seed} ...", flush=True)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), method, str(seed)],
                               env=env)
            if r.returncode != 0 or not os.path.exists(path_fn(seed)):
                print(f"   FAIL {method} seed={seed} (exit {r.returncode}) — continuing", flush=True)
    report()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        {"bc": worker_bc, "cql": worker_cql}[sys.argv[1]](int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "report":
        report()
    else:
        orchestrate()
