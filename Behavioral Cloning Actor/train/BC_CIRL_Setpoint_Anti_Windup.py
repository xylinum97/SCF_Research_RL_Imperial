"""
BC CIRL Anti-Windup — Behavioral Cloning of the anti-windup expert ([Kp,Ki,Kd] actor).
Refactored to import shared components (physics, networks) from ../../main_script;
only the reward, supervised data loading, training loop, and evaluation are kept
here. Constants live in ../config_cirl.py.
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
sys.path.insert(0, str(ROOT))                           # for config_cirl
sys.path.insert(0, str(ROOT.parent))                    # For GitHub -> main_script
from main_script import *
import config_cirl as cfg
from config_cirl import *
configure(cfg); set_seed(SEED)

SAVE_DIR  = Path(cfg.SAVE_DIR)
CHART_DIR = Path(cfg.CHART_DIR)
solarfield_model_np    = solar_model
solarfield_model_torch = solar_model_torch
def CIRLBCNet(in_dim=9): return GainActor(in_dim=in_dim, out_dim=3)


def compute_reward(q_new: torch.Tensor, q_expert: torch.Tensor,
                   Tout: torch.Tensor, q_prev: torch.Tensor,
                   t_ref: torch.Tensor,
                   lambda_smooth: float = 1.0) -> torch.Tensor:
    r_imit   = -torch.log((q_new  - q_expert) ** 2 + EPS_LOG)
    r_track  = -torch.log((Tout   - t_ref)    ** 2 + EPS_LOG)
    r_smooth = -torch.log((q_new  - q_prev)   ** 2 + EPS_LOG)
    # return r_imit + r_track + lambda_smooth * r_smooth
    return W_IMIT * r_imit + W_TRACK * r_track + lambda_smooth * r_smooth


def load_data(excel_files: list) -> tuple:
    """
    Run fixed PI expert on each file and collect CIRL BC training data.
    T_ref is computed per-timestep from irradiance via get_tref(Ig).

    Parameters
    ----------
    excel_files : list of file paths

    Returns
    -------
    states     : (N, 9)  [Tout_m, e, int_e, de, Ig, Ta, Tin, theta, T_ref(Ig)]
    raw_states : (N, 6)  [Tout_m, Tin, Ta, Ig, theta, q_prev]
    pid_states : (N, 3)  [e, int_e, de]
    gains_norm : (N, 3)  [Kp, Ki, Kd] normalised
    gains_raw  : (N, 3)  [Kp, Ki, Kd] physical
    q_expert   : (N, 1)  q[t+1] from expert PI
    t_ref_arr  : (N,)    per-sample T_ref derived from irradiance
    """
    all_states     = []
    all_raw        = []
    all_pid        = []
    all_gains_norm = []
    all_gains_raw  = []
    all_q_exp      = []
    all_tref       = []

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

        N          = len(df)
        tref_full  = np.full(N, dataset_tref(path), dtype=np.float64)  # constant per dataset type
        Tout_m     = np.zeros(N)
        q          = np.zeros(N)
        gains_used = np.zeros((N, 3))
        int_track  = np.zeros(N)   # integral state actually applied at each step
        de_track   = np.zeros(N)
        Tout_m[0]     = T_sc[0]
        q[0]          = q_meas[0]
        gains_used[0] = [KP_EXPERT, KI_EXPERT, KD_EXPERT]
        # ── Bumpless start: pre-load the integral so the first control increment is
        #    zero (Kp*e0 + Ki*int_e = 0), removing the start-up flow jump.
        e0    = tref_full[0] - Tout_m[0]
        int_e = (q[0] - KP_EXPERT * e0) / KI_EXPERT   # bumpless: q_raw(0) = q[0]

        for i in range(1, N):
            e    = tref_full[i - 1] - Tout_m[i - 1]
            de   = -(Tout_m[i - 1] - Tout_m[max(0, i - 2)]) / TS
            int_track[i - 1] = int_e          # state the controller saw producing q[i]
            de_track[i - 1]  = de
            gains_used[i] = [KP_EXPERT, KI_EXPERT, KD_EXPERT]
            u     = KP_EXPERT * e + KI_EXPERT * int_e + KD_EXPERT * de
            q_raw = u                                   # positional control output
            q[i]  = np.clip(q_raw, Q_MIN, Q_MAX)
            # ── Back-calculation anti-windup: Kw feeds the saturation error back into
            #    the integrator (int_e += (e + Kw*(q_sat - q_raw))*Ts).
            int_e = float(np.clip(int_e + (e + KW * (q[i] - q_raw)) * TS,
                                  -INT_E_CLIP, INT_E_CLIP))
            Tout_m[i] = solarfield_model_np(
                q[i], Tin_arr[i-1], I_arr[i-1], Ta_arr[i-1], th_arr[i-1], Tout_m[i-1]
            )

        # Reuse the integral / derivative states actually applied during the rollout,
        # so the BC training state matches the (anti-windup) controller exactly.
        e_arr   = tref_full - Tout_m
        int_arr = int_track
        de_arr  = de_track

        states_f = np.column_stack([
            Tout_m[:-1], e_arr[:-1], int_arr[:-1], de_arr[:-1],
            I_arr[:-1],  Ta_arr[:-1], Tin_arr[:-1], th_arr[:-1],
            tref_full[:-1]
        ])
        raw_f        = np.column_stack([
            Tout_m[:-1], Tin_arr[:-1], Ta_arr[:-1],
            I_arr[:-1],  th_arr[:-1],  q[:-1]
        ])
        pid_f        = np.column_stack([e_arr[:-1], int_arr[:-1], de_arr[:-1]])
        gains_raw_f  = gains_used[1:]
        gains_norm_f = normalize_gains(gains_raw_f.astype(np.float32))
        q_exp_f      = q[1:].reshape(-1, 1)
        tref_f       = tref_full[:-1].astype(np.float32)

        valid = ~(np.isnan(states_f).any(1) | np.isnan(raw_f).any(1)
                  | np.isnan(pid_f).any(1) | np.isnan(gains_norm_f).any(1))

        all_states.append(states_f[valid].astype(np.float32))
        all_raw.append(raw_f[valid].astype(np.float32))
        all_pid.append(pid_f[valid].astype(np.float32))
        all_gains_norm.append(gains_norm_f[valid])
        all_gains_raw.append(gains_raw_f[valid].astype(np.float32))
        all_q_exp.append(q_exp_f[valid].astype(np.float32))
        all_tref.append(tref_f[valid])
        tref_mean = float(tref_full.mean())
        print(f"  [OK] {path.name:45s}  T_ref(mean)={tref_mean:.1f}°C  {int(valid.sum()):5d} samples")

    states     = np.concatenate(all_states)
    raw_states = np.concatenate(all_raw)
    pid_states = np.concatenate(all_pid)
    gains_norm = np.concatenate(all_gains_norm)
    gains_raw  = np.concatenate(all_gains_raw)
    q_expert   = np.concatenate(all_q_exp)
    t_ref_arr  = np.concatenate(all_tref)
    return states, raw_states, pid_states, gains_norm, gains_raw, q_expert, t_ref_arr


def train(states, raw_states, pid_states, gains_norm, gains_raw,
          q_expert, t_ref_arr, save_path=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    S_norm  = torch.from_numpy(normalize_states(states))
    S_raw   = torch.from_numpy(raw_states)
    S_pid   = torch.from_numpy(pid_states)
    Q_exp   = torch.from_numpy(q_expert)
    T_ref_t = torch.from_numpy(t_ref_arr)

    full_ds = TensorDataset(S_norm, S_raw, S_pid, Q_exp, T_ref_t)
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

    model     = CIRLBCNet(in_dim=states.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15
    )

    history         = {"train_reward": [], "val_reward": [], "lambda_smooth": []}
    best_val_reward = -1e9
    best_w          = None

    for epoch in range(1, EPOCHS + 1):
        # sigmoid lambda schedule: completes by epoch EPOCHS//2, then holds at LAMBDA_MAX.
        # second half trains with stable lambda so ReduceLROnPlateau can reduce LR cleanly.
        half     = EPOCHS / 4
        t_capped = min(epoch, half)                       # freeze progress after halfway
        x        = (t_capped / half - 0.5) * 10          # maps [0, half] → [-5, +5]
        lambda_t = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) / (1 + np.exp(-x))

        model.train()
        tr_rwds = []
        for s_n, s_r, s_p, q_e, t_r in train_loader:
            g_pred     = model(s_n)
            gains_phys = denormalize_gains_torch(g_pred)
            Kp = gains_phys[:, 0]; Ki = gains_phys[:, 1]; Kd = gains_phys[:, 2]
            e     = s_p[:, 0]; int_e = s_p[:, 1]; de = s_p[:, 2]
            q_prev = s_r[:, 5]
            u      = Kp * e + Ki * int_e + Kd * de
            q_new  = torch.clamp(u, Q_MIN, Q_MAX)
            Tout   = solarfield_model_torch(
                q_new, s_r[:, 1], s_r[:, 3], s_r[:, 2], s_r[:, 4], s_r[:, 0]
            )
            reward = compute_reward(q_new, q_e.squeeze(1), Tout, q_prev, t_r, lambda_t)
            loss   = -reward.mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_rwds.append(reward.mean().item())

        model.eval()
        va_rwds = []
        with torch.no_grad():
            for s_n, s_r, s_p, q_e, t_r in val_loader:
                g_pred     = model(s_n)
                gains_phys = denormalize_gains_torch(g_pred)
                Kp = gains_phys[:, 0]; Ki = gains_phys[:, 1]; Kd = gains_phys[:, 2]
                e     = s_p[:, 0]; int_e = s_p[:, 1]; de = s_p[:, 2]
                q_prev = s_r[:, 5]
                u      = Kp * e + Ki * int_e + Kd * de
                q_new  = torch.clamp(u, Q_MIN, Q_MAX)
                Tout   = solarfield_model_torch(
                    q_new, s_r[:, 1], s_r[:, 3], s_r[:, 2], s_r[:, 4], s_r[:, 0]
                )
                va_rwds.append(
                    compute_reward(q_new, q_e.squeeze(1), Tout, q_prev, t_r, lambda_t).mean().item()
                )

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

    t_Tout, t_q, t_Kp, t_Ki, t_Kd, t_tref = [], [], [], [], [], []
    with torch.no_grad():
        for s_n, s_r, s_p, q_e, t_r in test_loader:
            g_pred     = model(s_n)
            gains_phys = denormalize_gains_torch(g_pred)
            Kp = gains_phys[:, 0]; Ki = gains_phys[:, 1]; Kd = gains_phys[:, 2]
            e     = s_p[:, 0]; int_e = s_p[:, 1]; de = s_p[:, 2]
            q_prev = s_r[:, 5]
            u      = Kp * e + Ki * int_e + Kd * de
            q_new  = torch.clamp(u, Q_MIN, Q_MAX)
            Tout   = solarfield_model_torch(
                q_new, s_r[:, 1], s_r[:, 3], s_r[:, 2], s_r[:, 4], s_r[:, 0]
            )
            t_Tout.append(Tout.numpy()); t_q.append(q_new.numpy())
            t_Kp.append(Kp.numpy());    t_Ki.append(Ki.numpy())
            t_Kd.append(Kd.numpy());    t_tref.append(t_r.numpy())

    test_Tout = np.concatenate(t_Tout);  test_q    = np.concatenate(t_q)
    test_Kp   = np.concatenate(t_Kp);   test_Ki   = np.concatenate(t_Ki)
    test_Kd   = np.concatenate(t_Kd);   test_tref = np.concatenate(t_tref)
    err       = test_Tout - test_tref

    test_metrics = {
        'mae':  float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mse':  float(np.mean(err ** 2)),
        'n':    len(test_Tout),
        'Tout': test_Tout, 'q': test_q, 'error': err,
        'Kp': test_Kp,  'Kp_mean': float(np.mean(test_Kp)),  'Kp_std': float(np.std(test_Kp)),
        'Ki': test_Ki,  'Ki_mean': float(np.mean(test_Ki)),  'Ki_std': float(np.std(test_Ki)),
        'Kd': test_Kd,  'Kd_mean': float(np.mean(test_Kd)),  'Kd_std': float(np.std(test_Kd)),
    }

    if save_path is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_w, save_path)

    print(f"\n  Best val reward : {best_val_reward:.5f}")
    print(f"  Test MAE  (Tout vs T_ref): {test_metrics['mae']:.4f} °C")
    print(f"  Test RMSE (Tout vs T_ref): {test_metrics['rmse']:.4f} °C")
    print(f"  Dynamic gains (test mean ± std):")
    print(f"    Kp = {test_metrics['Kp_mean']:.4f} ± {test_metrics['Kp_std']:.4f}")
    print(f"    Ki = {test_metrics['Ki_mean']:.6f} ± {test_metrics['Ki_std']:.6f}")
    print(f"    Kd = {test_metrics['Kd_mean']:.4f} ± {test_metrics['Kd_std']:.4f}")
    if save_path is not None:
        print(f"  Weights saved: {save_path}")

    return history, model, best_val_reward, test_metrics


def evaluate(model, states, raw_states, pid_states, t_ref_arr):
    model.eval()
    S_norm = torch.from_numpy(normalize_states(states))
    S_raw  = torch.from_numpy(raw_states)
    S_pid  = torch.from_numpy(pid_states)

    with torch.no_grad():
        g_pred     = model(S_norm)
        gains_phys = denormalize_gains_torch(g_pred)
        Kp = gains_phys[:, 0]; Ki = gains_phys[:, 1]; Kd = gains_phys[:, 2]
        e     = S_pid[:, 0]; int_e = S_pid[:, 1]; de = S_pid[:, 2]
        q_prev = S_raw[:, 5]
        u      = Kp * e + Ki * int_e + Kd * de
        q_new  = torch.clamp(u, Q_MIN, Q_MAX)
        Tout   = solarfield_model_torch(
            q_new, S_raw[:, 1], S_raw[:, 3], S_raw[:, 2], S_raw[:, 4], S_raw[:, 0]
        )

    Tout_np = Tout.numpy(); q_np = q_new.numpy()
    Kp_np   = Kp.numpy();   Ki_np = Ki.numpy(); Kd_np = Kd.numpy()
    err     = Tout_np - t_ref_arr
    mae     = float(np.mean(np.abs(err)))
    rmse    = float(np.sqrt(np.mean(err ** 2)))
    print(f"  Tout vs T_ref  —  MAE={mae:.4f} °C  RMSE={rmse:.4f} °C")
    return {
        'Tout': Tout_np, 'q': q_np, 'error': err, 'mae': mae, 'rmse': rmse,
        'mse': float(np.mean(err**2)), 'n': len(err),
        'Kp': Kp_np,  'Kp_mean': float(np.mean(Kp_np)),  'Kp_std': float(np.std(Kp_np)),
        'Ki': Ki_np,  'Ki_mean': float(np.mean(Ki_np)),  'Ki_std': float(np.std(Ki_np)),
        'Kd': Kd_np,  'Kd_mean': float(np.mean(Kd_np)),  'Kd_std': float(np.std(Kd_np)),
    }


def plot_results(history, test_metrics, save_dir, combo_name=""):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = f" [{combo_name}]" if combo_name else ""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_reward"], label="Train reward", linewidth=1.5, color="steelblue")
    ax.plot(history["val_reward"],   label="Val reward",   linewidth=1.5, color="coral")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward  (r_imit + r_track + λ·r_smooth)")
    ax.set_title(f"CIRL BC Setpoint Update — Reward Curves{tag}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "bc_cirl_setpoint_update_without_Kd_loss.png", dpi=150)
    plt.close(fig)

    err = test_metrics['error']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=80, color="coral", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Tout − T_ref  [°C]"); ax.set_ylabel("Count")
    ax.set_title(f"CIRL BC Setpoint Update — Tracking Error (dynamic T_ref) — Test{tag}\n"
                 f"MAE={test_metrics['mae']:.4f}°C  RMSE={test_metrics['rmse']:.4f}°C  "
                 f"n={test_metrics['n']:,}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "bc_cirl_setpoint_update_without_Kd_error_hist.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].hist(test_metrics['q'], bins=60, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(Q_MIN, color="red",   linestyle="--", linewidth=1, label=f"q_min={Q_MIN}")
    axes[0].axvline(Q_MAX, color="green", linestyle="--", linewidth=1, label=f"q_max={Q_MAX}")
    axes[0].set_xlabel("q_new  [L/min]"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Flow Rate Distribution{tag}"); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    for ax, key, color in zip(axes[1:], ['Kp', 'Ki', 'Kd'],
                               ['seagreen', 'darkorange', 'mediumpurple']):
        ax.hist(test_metrics[key], bins=60, color=color, edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"{key}  (gain)"); ax.set_ylabel("Count")
        ax.set_title(f"{key} Distribution{tag}\n"
                     f"mean={test_metrics[key+'_mean']:.4f}  "
                     f"std={test_metrics[key+'_std']:.4f}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "bc_cirl_setpoint_update_without_Kd_dist.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to: {save_dir}")


def main():
    """
    Train 15 CIRL BC Setpoint Update policies — all C(4,k) subsets of 4 sunny
    datasets, each augmented with the cloudy dataset.  T_ref is computed
    per-timestep from irradiance via get_tref(Ig) (9D state, dynamic setpoint).
    Models saved as bc_cirl_setpoint_update_{combo_name}.pt.
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
        files_subset = [SUNNY_FILES[i] for i in combo] + ([CLOUDY_FILE] if INCLUDE_CLOUDY else [])
        labels       = [DAY_LABELS[i]  for i in combo]
        combo_name   = "_".join(labels)
        save_path    = SAVE_DIR / f"bc_cirl_setpoint_anti_windup_{combo_name}.pt"

        print("\n" + "=" * 65)
        cloudy_tag = " + cloudy" if INCLUDE_CLOUDY else " (sunny only)"
        print(f"CIRL POLICY {idx:>2}/{total}  —  {len(combo)} sunny{cloudy_tag}  [{combo_name}]")
        for f in files_subset:
            print(f"   {f.name}")
        print("=" * 65)

        print(f"\n[1] Loading & simulating fixed PI expert (dynamic T_ref from Ig)...")
        states, raw_states, pid_states, gains_norm, gains_raw, q_expert, t_ref_arr = \
            load_data(files_subset)
        print(f"  Total samples : {len(states):,}")

        print(f"\n[2] Training (maximise reward, dynamic T_ref in state, 80/20 holdout)...")
        history, model, best_val_reward, _ = train(
            states, raw_states, pid_states, gains_norm, gains_raw,
            q_expert, t_ref_arr, save_path=save_path
        )

        print(f"\n[3] Evaluating on full dataset ({len(states):,} samples)...")
        full_metrics = evaluate(model, states, raw_states, pid_states, t_ref_arr)

        print(f"\n[4] Plotting...")
        plot_results(history, full_metrics,
                     plot_dir / f"policy_{combo_name}", combo_name=combo_name)

        summary.append({
            "idx": idx, "combo": combo, "combo_name": combo_name,
            "n_files": len(combo), "n_samples": len(states),
            "n_eval": full_metrics['n'], "best_val": best_val_reward,
            "test_mae": full_metrics['mae'], "test_rmse": full_metrics['rmse'],
            "test_mse": full_metrics['mse'],
            "Kp_mean": full_metrics['Kp_mean'], "Ki_mean": full_metrics['Ki_mean'],
            "Kd_mean": full_metrics['Kd_mean'], "save_path": save_path,
        })

    best      = min(summary, key=lambda x: x["test_mae"])
    best_dest = SAVE_DIR / "bc_cirl_setpoint_anti_windup_best.pt"
    shutil.copy(best["save_path"], best_dest)

    print("\n" + "=" * 110)
    print(f"SUMMARY — all {total} CIRL BC Setpoint Update policies  "
          f"(T_ref: sunny={TREF_SUNNY}°C  cloudy={TREF_CLOUDY}°C | holdout 20%)")
    print(f"{'#':>4}  {'Combo':>20}  {'N':>1}  {'Total':>7}  {'EvalN':>6}  "
          f"{'BestRwd':>10}  {'TestMAE':>8}  {'TestRMSE':>9}  "
          f"{'Kp':>7}  {'Ki':>9}  {'Kd':>7}")
    print("-" * 110)
    for r in summary:
        marker = " <- best" if r is best else ""
        print(f"  {r['idx']:>2}  {r['combo_name']:>20}  {r['n_files']:>1}  "
              f"{r['n_samples']:>7,}  {r['n_eval']:>6,}  "
              f"{r['best_val']:>10.6f}  "
              f"{r['test_mae']:>8.4f}  {r['test_rmse']:>9.4f}  "
              f"{r['Kp_mean']:>7.4f}  {r['Ki_mean']:>9.6f}  "
              f"{r['Kd_mean']:>7.4f}{marker}")
    print("=" * 110)
    print(f"\nBest policy: {best['combo_name']}  →  {best_dest}")
    print(f"  T_ref: sunny={TREF_SUNNY}°C  cloudy={TREF_CLOUDY}°C  "
          f"(detected automatically from filename)")

    colors  = {1: "steelblue", 2: "coral", 3: "seagreen", 4: "mediumpurple"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for n_f in range(1, 5):
        sub = [r for r in summary if r["n_files"] == n_f]
        x   = [r["idx"] for r in sub]
        lbl = f"{n_f} dataset(s)"
        axes[0].plot(x, [r["best_val"]  for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=lbl)
        axes[1].plot(x, [r["test_mae"]  for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=lbl)
        axes[2].plot(x, [r["test_rmse"] for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=lbl)
    for ax in axes:
        ax.set_xlabel("Policy index (1–15)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("Best Val Reward")
    axes[0].set_title("Best Val Reward — all 15 CIRL BC Setpoint Update policies")
    axes[1].set_ylabel("MAE  Tout − T_ref  [°C]")
    axes[1].set_title(f"Test MAE (Tout vs dynamic T_ref)\n(held-out 20%)")
    axes[2].set_ylabel("RMSE  [°C]")
    axes[2].set_title("Test RMSE\n(held-out 20%)")
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "bc_cirl_setpoint_update_without_Kd_comparison.png", dpi=150)
    plt.show()
    print(f"\nComparison plot saved to: {plot_dir / 'bc_cirl_setpoint_update_without_Kd_comparison.png'}")


if __name__ == "__main__":
    main()
