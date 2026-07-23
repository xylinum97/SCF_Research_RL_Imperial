# Multi-seed offline comparison (BC vs CQL)

Answers the supervisor's seed-robustness question by rerunning the **exact single-seed
pipelines** of `../BC_vs_CQL_Offline_Policy_Comparison.ipynb` across **4 seeds
(42, 36, 27, 21)**. Seed 42 replicates the original comparison (Expert 1.224 /
BC 1.224 / CQL 0.243), so the other seeds are apples-to-apples with it:

- **BC** — BC package trainer at full budget (390 epochs), `[Kp,Ki,Kw]` box
  (`Kw` pinned to the expert constant `Ki/Kp`),
  4 original sunny days + cloudy day.
- **CQL** — the `CQL_CIRL.ipynb` offline pipeline in the original `[Kp,Ki,Kw]` box:
  buffer pre-filled from that seed's raw BC actor (noise 0.05) over the 5 original
  days, then 100 × 242 CQL-H updates (the realised budget of the original run),
  best checkpoint by the cloudy-×3 proxy MAE.

Every seed's actors are evaluated **zero-shot on Juan's 4 new days** with the same
rollout code as the comparison notebook.

```bash
python run_multiseed.py          # several hours: 4 seeds x {BC, CQL}; resumes if interrupted
python run_multiseed.py report   # evaluation + charts only (needs policies/ already trained)
```

Outputs: `results.json` (per-seed per-day MAEs), `traces.npz` (Tout traces),
`policies/{bc,cql}_seed{S}.pt`, and in `../charts/`:
`multiseed_seed{S}_tracking_4days.png` (4 charts × 4 sub-charts, one per Juan day)
plus `multiseed_offline_bc_vs_cql.png` (mean ± std bars). Then open
`MultiSeed_Offline_BC_vs_CQL.ipynb` to view everything.

Each training runs in a fresh subprocess (the BC worker imports the BC package's
`config_cirl`, the CQL worker the BC vs CQL Comparison `config`, so one process would
cache the wrong constants).
