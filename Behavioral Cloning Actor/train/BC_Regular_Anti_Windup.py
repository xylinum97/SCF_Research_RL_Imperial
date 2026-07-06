"""
BC Regular Anti-Windup — Behavioral Cloning of the anti-windup expert (flow actor).
Refactored to import shared components (physics, networks) from ../../main_script;
only the reward, supervised data loading, training loop, and evaluation are kept
here. Constants live in ../config_regular.py.
"""
import itertools, shutil, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parent.parent          # package root
sys.path.insert(0, str(ROOT))                           # for config_regular
sys.path.insert(0, str(ROOT.parent))                    # For GitHub -> main_script
from main_script import *
import config_regular as cfg
from config_regular import *
configure(cfg); set_seed(SEED)

SAVE_DIR  = Path(cfg.SAVE_DIR)
CHART_DIR = Path(cfg.CHART_DIR)
solarfield_model_np    = solar_model
solarfield_model_torch = solar_model_torch
def RegularBCNet(in_dim=10): return FlowActor(in_dim=in_dim, q_min=Q_MIN, q_max=Q_MAX)


def compute_reward(q_pred: torch.Tensor,
                   q_expert: torch.Tensor,
                   Tout: torch.Tensor,
                   q_prev: torch.Tensor,
                   t_ref: torch.Tensor,
                   lambda_smooth: float = 1.0) -> torch.Tensor:
    r_imit   = -torch.log((q_pred - q_expert) ** 2 + EPS_LOG)
    r_track  = -torch.log((Tout   - t_ref)    ** 2 + EPS_LOG)
    r_smooth = -torch.log((q_pred - q_prev)   ** 2 + EPS_LOG)
    return W_IMIT * r_imit + W_TRACK * r_track + lambda_smooth * r_smooth


def load_data(excel_files: list) -> tuple:
    """
    Run the expert PI on each file with irradiance-derived T_ref at every step.

    T_ref(t) = get_tref(Ig(t))  — varies continuously within each episode.

    Returns
    -------
    states    : (N, 10) [Tout_m, e, int_e, de, Ig, Ta, Tin, theta, q_prev, T_ref]
    raw_states: (N, 6)  [Tout_m, Tin, Ta, Ig, theta, q_prev]
    dq_expert : (N, 1)  positional q[t+1] from the anti-windup expert PI
    t_ref_arr : (N,)    per-sample T_ref (from irradiance)
    """
    all_states = []
    all_raw    = []
    all_dq     = []
    all_tref   = []

    for path in excel_files:
        path = Path(path)
        if not path.exists():
            print(f"  [SKIP] not found: {path.name}")
            continue

        df = pd.read_excel(path)
        required = {"I", "T_sc", "Ta", "Tin", "q", "theta"}
        missing  = required - set(df.columns)
        if missing:
            print(f"  [SKIP] {path.name} missing columns: {missing}")
            continue

        I_arr   = df["I"].to_numpy(dtype=np.float64)
        T_sc    = df["T_sc"].to_numpy(dtype=np.float64)
        Ta_arr  = df["Ta"].to_numpy(dtype=np.float64)
        Tin_arr = df["Tin"].to_numpy(dtype=np.float64)
        q_meas  = df["q"].to_numpy(dtype=np.float64)
        th_arr  = df["theta"].to_numpy(dtype=np.float64)

        N = len(df)

        # Constant T_ref for entire episode — determined by dataset type
        tref_full = np.full(N, dataset_tref(path), dtype=np.float64)

        Tout_m    = np.zeros(N)
        q         = np.zeros(N)
        int_track = np.zeros(N)
        de_track  = np.zeros(N)
        Tout_m[0] = T_sc[0]
        q[0]      = q_meas[0]
        # Bumpless start: q_raw(0) = q[0]  (Main_aggresive.m)
        e0    = tref_full[0] - Tout_m[0]
        int_e = (q[0] - KP_EXPERT * e0) / KI_EXPERT

        # ── Expert PI: positional control + back-calculation anti-windup ──
        for i in range(1, N):
            e   = tref_full[i - 1] - Tout_m[i - 1]
            de  = -(Tout_m[i - 1] - Tout_m[max(0, i - 2)]) / TS
            int_track[i - 1] = int_e
            de_track[i - 1]  = de
            u     = KP_EXPERT * e + KI_EXPERT * int_e + KD_EXPERT * de
            q_raw = u                                 # positional control output
            q[i]  = np.clip(q_raw, Q_MIN, Q_MAX)
            int_e = float(np.clip(int_e + (e + KW * (q[i] - q_raw)) * TS,
                                  -INT_E_CLIP, INT_E_CLIP))
            Tout_m[i] = solarfield_model_np(
                q[i], Tin_arr[i-1], I_arr[i-1], Ta_arr[i-1], th_arr[i-1], Tout_m[i-1]
            )

        # ── Build state arrays (int_e / de are those actually applied) ──
        e_arr   = tref_full - Tout_m
        int_arr = int_track
        de_arr  = de_track

        # State at step t; T_ref appended as last feature
        states_f = np.column_stack([
            Tout_m[:-1], e_arr[:-1], int_arr[:-1], de_arr[:-1],
            I_arr[:-1],  Ta_arr[:-1], Tin_arr[:-1], th_arr[:-1], q[:-1],
            tref_full[:-1]
        ])
        raw_f  = np.column_stack([
            Tout_m[:-1], Tin_arr[:-1], Ta_arr[:-1],
            I_arr[:-1],  th_arr[:-1],  q[:-1]
        ])
        # Positional expert action to clone = the flow the expert applied next step.
        dq_exp = q[1:].reshape(-1, 1)
        tref_f = tref_full[:-1].astype(np.float32)

        valid = ~(np.isnan(states_f).any(1) | np.isnan(raw_f).any(1) | np.isnan(dq_exp).any(1))

        all_states.append(states_f[valid].astype(np.float32))
        all_raw.append(raw_f[valid].astype(np.float32))
        all_dq.append(dq_exp[valid].astype(np.float32))
        all_tref.append(tref_f[valid])

        tref_min = tref_full.min(); tref_max = tref_full.max(); tref_mean = tref_full.mean()
        print(f"  [OK] {path.name:45s}  "
              f"T_ref: {tref_min:.1f}–{tref_max:.1f}°C (mean={tref_mean:.1f})  "
              f"{int(valid.sum()):5d} samples")

    states     = np.concatenate(all_states)
    raw_states = np.concatenate(all_raw)
    dq_expert  = np.concatenate(all_dq)
    t_ref_arr  = np.concatenate(all_tref)
    return states, raw_states, dq_expert, t_ref_arr


def train(states, raw_states, dq_expert, t_ref_arr, save_path=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    S_norm  = torch.from_numpy(normalize_states(states))
    S_raw   = torch.from_numpy(raw_states)
    DQ_exp  = torch.from_numpy(dq_expert)
    T_ref_t = torch.from_numpy(t_ref_arr)

    full_ds = TensorDataset(S_norm, S_raw, DQ_exp, T_ref_t)
    N = len(full_ds)

    n_test     = int(N * TEST_SPLIT)
    n_trainval = N - n_test
    trainval_ds, test_ds = random_split(
        full_ds, [n_trainval, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    n_val   = int(n_trainval * VAL_SPLIT)
    n_train = n_trainval - n_val
    train_ds, val_ds = random_split(
        trainval_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED + 1)
    )

    print(f"  Split  —  train: {n_train:,}  |  val: {n_val:,}  |  test: {n_test:,}  "
          f"(total: {N:,})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = RegularBCNet(in_dim=states.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )

    history         = {"train_reward": [], "val_reward": [], "lambda_smooth": []}
    best_val_reward = -1e9
    best_w          = None

    for epoch in range(1, EPOCHS + 1):
        # sigmoid lambda schedule: completes by epoch EPOCHS//2, then holds at LAMBDA_MAX
        half     = EPOCHS / 4
        t_capped = min(epoch, half)
        x        = (t_capped / half - 0.5) * 10
        lambda_t = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) / (1 + np.exp(-x))

        model.train()
        tr_rwds = []
        for s_n, s_r, q_e, t_r in train_loader:
            q_pred  = torch.clamp(model(s_n), Q_MIN, Q_MAX)
            Tout    = solarfield_model_torch(
                q_pred, s_r[:, 1], s_r[:, 3], s_r[:, 2], s_r[:, 4], s_r[:, 0]
            )
            reward = compute_reward(q_pred, q_e.squeeze(1), Tout, s_r[:, 5], t_r, lambda_t)
            loss   = -reward.mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_rwds.append(reward.mean().item())

        model.eval()
        va_rwds = []
        with torch.no_grad():
            for s_n, s_r, q_e, t_r in val_loader:
                q_pred  = torch.clamp(model(s_n), Q_MIN, Q_MAX)
                Tout    = solarfield_model_torch(
                    q_pred, s_r[:, 1], s_r[:, 3], s_r[:, 2], s_r[:, 4], s_r[:, 0]
                )
                va_rwds.append(compute_reward(q_pred, q_e.squeeze(1), Tout, s_r[:, 5], t_r, lambda_t).mean().item())

        train_rwd = float(np.mean(tr_rwds))
        val_rwd   = float(np.mean(va_rwds))
        history["train_reward"].append(train_rwd)
        history["val_reward"].append(val_rwd)
        history["lambda_smooth"].append(lambda_t)
        scheduler.step(val_rwd)

        if val_rwd > best_val_reward:
            best_val_reward = val_rwd
            best_w          = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  "
                  f"train_rwd={train_rwd:+.5f}  val_rwd={val_rwd:+.5f}  "
                  f"λ={lambda_t:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_w)
    model.eval()

    if save_path is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_w, save_path)

    print(f"\n  Best val reward : {best_val_reward:+.5f}")
    if save_path is not None:
        print(f"  Weights saved   : {save_path}")
    print(f"  (closed-loop MAE reported in step [3] below)")

    return history, model, best_val_reward


def evaluate(model, states, raw_states, t_ref_arr):
    model.eval()
    S_norm = torch.from_numpy(normalize_states(states))
    S_raw  = torch.from_numpy(raw_states)

    with torch.no_grad():
        q_pred  = torch.clamp(model(S_norm), Q_MIN, Q_MAX)
        Tout    = solarfield_model_torch(
            q_pred, S_raw[:, 1], S_raw[:, 3], S_raw[:, 2], S_raw[:, 4], S_raw[:, 0]
        )

    Tout_np = Tout.numpy(); q_np = q_pred.numpy()
    err     = Tout_np - t_ref_arr
    mae     = float(np.mean(np.abs(err)))
    rmse    = float(np.sqrt(np.mean(err ** 2)))
    print(f"  Tout vs T_ref(Ig)  MAE={mae:.4f} °C  RMSE={rmse:.4f} °C")
    return {'Tout': Tout_np, 'q': q_np, 'error': err,
            'mae': mae, 'rmse': rmse, 'mse': float(np.mean(err**2)), 'n': len(err)}


def evaluate_closed_loop(model, excel_files: list) -> dict:
    """
    Run a full closed-loop rollout on each file and compute MAE/RMSE.
    Identical to the notebook rollout — errors compound step to step.
    """
    model.eval()
    all_errors = []

    for path in excel_files:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_excel(path)
        I_arr   = df["I"].to_numpy(dtype=np.float64)
        T_sc    = df["T_sc"].to_numpy(dtype=np.float64)
        Ta_arr  = df["Ta"].to_numpy(dtype=np.float64)
        Tin_arr = df["Tin"].to_numpy(dtype=np.float64)
        q_meas  = df["q"].to_numpy(dtype=np.float64)
        th_arr  = df["theta"].to_numpy(dtype=np.float64)
        N    = len(df)
        tref = float(dataset_tref(path))

        tout      = float(T_sc[0])
        tout_prev = tout
        q_prev    = float(q_meas[0])
        e0        = tref - tout
        int_e     = (q_prev - KP_EXPERT * e0) / KI_EXPERT   # bumpless start
        Kw        = KI_EXPERT / KP_EXPERT
        Tout_list = []

        with torch.no_grad():
            for i in range(N - 1):
                e  = tref - tout
                de = -(tout - tout_prev) / TS
                state = np.array(
                    [tout, e, int_e, de, I_arr[i], Ta_arr[i], Tin_arr[i], th_arr[i], q_prev, tref],
                    dtype=np.float32)
                s_norm = torch.from_numpy(normalize_states(state.reshape(1, -1)))
                q_raw  = float(model(s_norm).squeeze().item())
                q      = float(np.clip(q_raw, Q_MIN, Q_MAX))
                tout_next = solarfield_model_np(q, Tin_arr[i], I_arr[i], Ta_arr[i], th_arr[i], tout)
                Tout_list.append(tout_next)
                int_e = float(np.clip(int_e + (e + Kw * (q - q_raw)) * TS, -INT_E_CLIP, INT_E_CLIP))
                tout_prev = tout; tout = tout_next; q_prev = q

        errors = np.array(Tout_list) - tref
        all_errors.append(errors)
        mae_f = float(np.mean(np.abs(errors)))
        print(f"    {path.name:45s}  CL-MAE={mae_f:.4f}°C")

    all_err = np.concatenate(all_errors)
    mae  = float(np.mean(np.abs(all_err)))
    rmse = float(np.sqrt(np.mean(all_err ** 2)))
    print(f"  Closed-loop  MAE={mae:.4f}°C  RMSE={rmse:.4f}°C  ({len(all_err):,} steps)")
    return {'mae': mae, 'rmse': rmse, 'n': len(all_err)}


def plot_results(history, save_dir, combo_name=""):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f" [{combo_name}]" if combo_name else ""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_reward"], label="Train reward", linewidth=1.5, color="steelblue")
    ax.plot(history["val_reward"],   label="Val reward",   linewidth=1.5, color="coral")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward  (r_imit + r_track + λ·r_smooth)")
    ax.set_title(f"BC Regular Anti-Windup — Reward Curves{tag}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "bc_regular_anti_windup_loss.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to: {save_dir}")


def main():
    """
    Train 15 BC Regular Anti-Windup policies with flexible T_ref.
    Each combo = C(4,k) sunny subset + cloudy file.
    T_ref(t) = get_tref(Ig(t)) every timestep in both training and rollout.
    Models saved as bc_regular_anti_windup_{combo_name}.pt.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = ROOT / "charts"

    all_combos = []
    for r in range(1, len(SUNNY_FILES) + 1):
        if r not in TRAIN_NFILES:          # skip combo sizes not selected this run
            continue
        for combo in itertools.combinations(range(len(SUNNY_FILES)), r):
            all_combos.append(combo)

    summary = []
    total   = len(all_combos)
    print(f"Training combo sizes {sorted(TRAIN_NFILES)} → {total} policies this run\n")

    for idx, combo in enumerate(all_combos, start=1):
        sunny_subset = [SUNNY_FILES[i] for i in combo]
        labels       = [DAY_LABELS[i]  for i in combo]
        combo_name   = "_".join(labels)
        save_path    = SAVE_DIR / f"bc_regular_anti_windup_{combo_name}.pt"

        # Each combo includes the cloudy file for low-Ig coverage
        files_subset = sunny_subset + ([CLOUDY_FILE] if INCLUDE_CLOUDY else [])

        print("\n" + "=" * 70)
        cloudy_tag = " + cloudy" if INCLUDE_CLOUDY else " (sunny only)"
        print(f"POLICY {idx:>2}/{total}  —  {len(combo)} sunny{cloudy_tag}  [{combo_name}]")
        for f in files_subset:
            print(f"   {Path(f).name}")
        print(f"   T_ref: sunny={TREF_SUNNY}°C  cloudy={TREF_CLOUDY}°C  (auto-detected from filename)")
        print("=" * 70)

        print(f"\n[1] Loading & simulating expert PI with dynamic T_ref(Ig)...")
        states, raw_states, dq_expert, t_ref_arr = load_data(files_subset)
        print(f"  Total samples : {len(states):,}  "
              f"T_ref: {t_ref_arr.min():.1f}–{t_ref_arr.max():.1f}°C "
              f"(mean={t_ref_arr.mean():.1f}°C)")

        print(f"\n[2] Training (log-reward, delta_q, dynamic T_ref in state, 80/20 holdout)...")
        history, model, best_val = train(
            states, raw_states, dq_expert, t_ref_arr, save_path=save_path
        )

        print(f"\n[3] Closed-loop evaluation (matches notebook rollout)...")
        cl_metrics = evaluate_closed_loop(model, files_subset)

        print(f"\n[4] Plotting reward curves...")
        plot_results(history, plot_dir / f"policy_{combo_name}", combo_name=combo_name)

        summary.append({
            "idx": idx, "combo": combo, "combo_name": combo_name,
            "n_files": len(combo), "n_samples": len(states),
            "n_eval": cl_metrics['n'], "best_val": best_val,
            "test_mae": cl_metrics['mae'], "test_rmse": cl_metrics['rmse'],
            "test_mse": cl_metrics['mae'] ** 2, "save_path": save_path,
        })

    best      = min(summary, key=lambda x: x["test_mae"])
    best_dest = SAVE_DIR / "bc_regular_anti_windup_best.pt"
    shutil.copy(best["save_path"], best_dest)

    print("\n" + "=" * 100)
    print(f"SUMMARY — all {total} BC Regular Anti-Windup policies  "
          f"(closed-loop MAE | T_ref: sunny={TREF_SUNNY}°C cloudy={TREF_CLOUDY}°C)")
    print(f"{'#':>4}  {'Combo':>20}  {'N':>1}  {'Total':>7}  {'EvalN':>6}  "
          f"{'BestRwd':>10}  {'CL-MAE':>8}  {'CL-RMSE':>9}")
    print("-" * 100)
    for r in summary:
        marker = " <- best" if r is best else ""
        print(f"  {r['idx']:>2}  {r['combo_name']:>20}  {r['n_files']:>1}  "
              f"{r['n_samples']:>7,}  {r['n_eval']:>6,}  "
              f"{r['best_val']:>10.6f}  "
              f"{r['test_mae']:>8.4f}  {r['test_rmse']:>9.4f}{marker}")
    print("=" * 100)
    print(f"\nBest policy: {best['combo_name']}  →  {best_dest}")
    print(f"  T_ref: sunny={TREF_SUNNY}°C  cloudy={TREF_CLOUDY}°C  (detected from filename)")

    colors  = {1: "steelblue", 2: "coral", 3: "seagreen", 4: "mediumpurple"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for n_f in range(1, 5):
        sub = [r for r in summary if r["n_files"] == n_f]
        x   = [r["idx"] for r in sub]
        axes[0].plot(x, [r["test_mae"]  for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=f"{n_f} dataset(s)")
        axes[1].plot(x, [r["test_rmse"] for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=f"{n_f} dataset(s)")
    axes[0].set_ylabel("MAE  [°C]")
    axes[0].set_title("Test MAE vs T_ref(Ig)\n(dynamic setpoint, sunny+cloudy)")
    axes[1].set_ylabel("RMSE  [°C]")
    axes[1].set_title("Test RMSE vs T_ref(Ig)\n(dynamic setpoint, sunny+cloudy)")
    for ax in axes:
        ax.set_xlabel("Policy index (1–15)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "bc_regular_anti_windup_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
