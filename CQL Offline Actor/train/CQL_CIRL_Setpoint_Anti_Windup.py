"""
CQL CIRL Anti-Windup — Conservative Q-Learning (offline), gain actor.
Refactored to import shared components (physics, networks, ReplayBuffer, CQL) from
../../main_script; only the offline-buffer collection, closed-loop evaluation, and
the combo training loop are variant-specific and kept here. Constants live in
../config_cirl.py.
"""
import itertools, shutil, sys, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent          # package root (CQL Offline Actor)
sys.path.insert(0, str(ROOT))                           # for config_cirl
sys.path.insert(0, str(ROOT.parent))                    # For GitHub -> main_script
from main_script import *
import config_cirl as cfg
from config_cirl import *
configure(cfg); set_seed(SEED)

SAVE_DIR  = Path(cfg.SAVE_DIR)
CHART_DIR = Path(cfg.CHART_DIR)
solarfield_model_np = solar_model            # alias used below


def collect_offline_buffer(excel_files, buffer):
    """Roll out the anti-windup expert PI on each file; the stored action is the
    (noised) normalised expert gain vector, giving coverage around the expert
    point for the CQL conservatism to be meaningful."""
    added = 0
    a_expert = normalize_gains(np.array([KP_EXPERT, KI_EXPERT, KD_EXPERT], dtype=np.float32))
    for path in excel_files:
        path = Path(path)
        if not path.exists():
            print(f"  [SKIP] not found: {path.name}"); continue
        df = pd.read_excel(path)
        I_arr   = df["I"].to_numpy(float);   T_sc = df["T_sc"].to_numpy(float)
        Ta_arr  = df["Ta"].to_numpy(float);  Tin_arr = df["Tin"].to_numpy(float)
        q_meas  = df["q"].to_numpy(float);   th_arr  = df["theta"].to_numpy(float)
        N    = len(df); tref = float(dataset_tref(path))

        def obs(tout, tout_prev, int_e, i):
            e  = tref - tout
            de = -(tout - tout_prev) / TS
            raw = np.array([tout, e, int_e, de, I_arr[i], Ta_arr[i],
                            Tin_arr[i], th_arr[i], tref], dtype=np.float32)
            return normalize_states(raw)

        tout = float(T_sc[0]); tout_prev = tout; q_prev = float(q_meas[0])
        int_e = (q_prev - KP_EXPERT * (tref - tout)) / KI_EXPERT   # bumpless start

        for i in range(N - 1):
            s  = obs(tout, tout_prev, int_e, i)
            e  = tref - tout
            de = -(tout - tout_prev) / TS
            # noised normalised gains -> physical gains actually applied
            a = np.clip(a_expert + np.random.randn(3).astype(np.float32) * PREFILL_NOISE, 0.0, 1.0)
            Kp, Ki, Kd = denormalize_gains(a)
            q_raw = Kp * e + Ki * int_e + Kd * de
            q     = float(np.clip(q_raw, Q_MIN, Q_MAX))
            Kw    = Ki / Kp                                    # anti-windup gain from applied gains
            tout_next = solarfield_model_np(q, Tin_arr[i], I_arr[i], Ta_arr[i], th_arr[i], tout)
            err = tout_next - tref
            r   = (-np.log(err ** 2 + EPS_LOG)
                   - LAMBDA_SMOOTH * np.log((q - q_prev) ** 2 + EPS_LOG))
            int_e_next = float(np.clip(int_e + (e + Kw * (q - q_raw)) * TS, -INT_E_CLIP, INT_E_CLIP))
            done = (i == N - 2)
            j    = min(i + 1, N - 1)
            s2   = obs(tout_next, tout, int_e_next, j) if not done else np.zeros(9, np.float32)
            buffer.add(s, a.astype(np.float32), float(r), s2, done)
            added += 1
            tout_prev = tout; tout = tout_next; q_prev = q; int_e = int_e_next
    return added


def evaluate_closed_loop(actor, excel_files, verbose=False):
    actor.eval()
    all_errors = []
    for path in excel_files:
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_excel(path)
        I_arr   = df["I"].to_numpy(float);   T_sc = df["T_sc"].to_numpy(float)
        Ta_arr  = df["Ta"].to_numpy(float);  Tin_arr = df["Tin"].to_numpy(float)
        q_meas  = df["q"].to_numpy(float);   th_arr  = df["theta"].to_numpy(float)
        N    = len(df); tref = float(dataset_tref(path))

        tout = float(T_sc[0]); tout_prev = tout; q_prev = float(q_meas[0])
        int_e = (q_prev - KP_EXPERT * (tref - tout)) / KI_EXPERT
        Tout_list = []
        with torch.no_grad():
            for i in range(N - 1):
                e  = tref - tout
                de = -(tout - tout_prev) / TS
                state = np.array([tout, e, int_e, de, I_arr[i], Ta_arr[i],
                                  Tin_arr[i], th_arr[i], tref], dtype=np.float32)
                s_norm = torch.from_numpy(normalize_states(state).reshape(1, -1))
                g_norm = np.clip(actor(s_norm).numpy()[0], 0.0, 1.0)
                Kp, Ki, Kd = denormalize_gains(g_norm)
                q_raw = Kp * e + Ki * int_e + Kd * de
                q     = float(np.clip(q_raw, Q_MIN, Q_MAX))
                Kw    = Ki / Kp
                tout_next = solarfield_model_np(q, Tin_arr[i], I_arr[i], Ta_arr[i], th_arr[i], tout)
                Tout_list.append(tout_next)
                int_e = float(np.clip(int_e + (e + Kw * (q - q_raw)) * TS, -INT_E_CLIP, INT_E_CLIP))
                tout_prev = tout; tout = tout_next; q_prev = q
        errors = np.array(Tout_list) - tref
        all_errors.append(errors)
        if verbose:
            print(f"    {path.name:45s}  CL-MAE={np.mean(np.abs(errors)):.4f} C")
    all_err = np.concatenate(all_errors)
    return {'mae':  float(np.mean(np.abs(all_err))),
            'rmse': float(np.sqrt(np.mean(all_err ** 2))),
            'n':    len(all_err)}


def plot_curves(history, save_dir, combo_name=""):
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tag = f" [{combo_name}]" if combo_name else ""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(history["epoch"], history["loss_c"], color="steelblue")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Critic loss")
    ax[0].set_title(f"CQL CIRL Anti-Windup — critic loss{tag}"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["epoch"], history["proxy_mae"], color="coral")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Closed-loop MAE [°C]")
    ax[1].set_title(f"CQL CIRL Anti-Windup — proxy MAE{tag}"); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "cql_cirl_setpoint_anti_windup_loss.png", dpi=150)
    plt.close(fig)


def train_combo(files_subset, save_path, combo_name):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    buffer = ReplayBuffer()
    n_add  = collect_offline_buffer(files_subset, buffer)
    print(f"  Offline buffer: {n_add:,} transitions (expert PI + noise={PREFILL_NOISE})")

    agent   = CQL()
    history = {"epoch": [], "loss_c": [], "cql_gap": [], "proxy_mae": []}
    best_mae = float('inf'); best_w = None

    for epoch in range(1, EPOCHS + 1):
        lc, cg = [], []
        for _ in range(UPDATES_PER_EPOCH):
            out = agent.update(buffer)
            if out is not None:
                lc.append(out[0]); cg.append(out[1])
        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            m = evaluate_closed_loop(agent.actor, files_subset)
            history["epoch"].append(epoch)
            history["loss_c"].append(float(np.mean(lc)) if lc else 0.0)
            history["cql_gap"].append(float(np.mean(cg)) if cg else 0.0)
            history["proxy_mae"].append(m['mae'])
            tag = ''
            if m['mae'] < best_mae:
                best_mae = m['mae']
                best_w   = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                tag = '  <- best'
            print(f"  Epoch {epoch:4d}/{EPOCHS}  loss_c={np.mean(lc) if lc else 0:8.4f}  "
                  f"cql_gap={np.mean(cg) if cg else 0:+.4f}  CL-MAE={m['mae']:.4f}"
                  f"  (best {best_mae:.4f}){tag}")

    if best_w is None:
        best_w = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    agent.actor.load_state_dict(best_w); agent.actor.eval()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_w, save_path)
    return history, agent.actor, best_mae


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    all_combos = []
    for r in range(1, len(SUNNY_FILES) + 1):
        if r not in TRAIN_NFILES:
            continue
        for combo in itertools.combinations(range(len(SUNNY_FILES)), r):
            all_combos.append(combo)

    summary = []
    total   = len(all_combos)
    print(f"CQL CIRL Anti-Windup — training combo sizes {sorted(TRAIN_NFILES)} "
          f"-> {total} policies\n")

    for idx, combo in enumerate(all_combos, start=1):
        labels     = [DAY_LABELS[i] for i in combo]
        combo_name = "_".join(labels)
        save_path  = SAVE_DIR / f"cql_cirl_setpoint_anti_windup_{combo_name}.pt"
        files_subset = [SUNNY_FILES[i] for i in combo] + ([CLOUDY_FILE] if INCLUDE_CLOUDY else [])

        print("=" * 70)
        print(f"POLICY {idx:>2}/{total}  [{combo_name}]  ({len(combo)} sunny + cloudy)")
        print("=" * 70)

        history, actor, best_mae = train_combo(files_subset, save_path, combo_name)
        cl = evaluate_closed_loop(actor, files_subset, verbose=True)
        plot_curves(history, CHART_DIR / f"policy_{combo_name}", combo_name=combo_name)

        summary.append({"idx": idx, "combo_name": combo_name, "n_files": len(combo),
                        "best_mae": best_mae, "test_mae": cl['mae'], "test_rmse": cl['rmse'],
                        "n_eval": cl['n'], "save_path": save_path})
        print(f"  saved -> {save_path.name}\n")

    best      = min(summary, key=lambda x: x["test_mae"])
    best_dest = SAVE_DIR / "cql_cirl_setpoint_anti_windup_best.pt"
    shutil.copy(best["save_path"], best_dest)

    print("=" * 90)
    print(f"SUMMARY — {total} CQL CIRL Anti-Windup policies (closed-loop MAE)")
    print(f"{'#':>3}  {'Combo':>20}  {'N':>1}  {'CL-MAE':>8}  {'CL-RMSE':>9}")
    print("-" * 90)
    for r in summary:
        marker = " <- best" if r is best else ""
        print(f"{r['idx']:>3}  {r['combo_name']:>20}  {r['n_files']:>1}  "
              f"{r['test_mae']:>8.4f}  {r['test_rmse']:>9.4f}{marker}")
    print("=" * 90)
    print(f"\nBest policy: {best['combo_name']}  ->  {best_dest}")

    colors  = {1: "steelblue", 2: "coral", 3: "seagreen", 4: "mediumpurple"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for n_f in range(1, 5):
        sub = [r for r in summary if r["n_files"] == n_f]
        if not sub:
            continue
        x = [r["idx"] for r in sub]
        axes[0].plot(x, [r["test_mae"]  for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=f"{n_f} dataset(s)")
        axes[1].plot(x, [r["test_rmse"] for r in sub], marker=markers[n_f],
                     color=colors[n_f], linewidth=1.5, markersize=7, label=f"{n_f} dataset(s)")
    axes[0].set_ylabel("MAE [°C]");  axes[0].set_title("CQL CIRL Anti-Windup — Closed-loop MAE")
    axes[1].set_ylabel("RMSE [°C]"); axes[1].set_title("CQL CIRL Anti-Windup — Closed-loop RMSE")
    for ax in axes:
        ax.set_xlabel("Policy index"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "cql_cirl_setpoint_anti_windup_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
