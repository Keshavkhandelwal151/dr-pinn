# dr-pinn — DR triage and retinal hemodynamic risk mapping (research prototype)

**Physics-Informed Neural Networks for Retinal Hemodynamic Stress Estimation and Early Risk Mapping in Diabetic Retinopathy** — B.Tech CSE research project.

> ⚠️ Research prototype, not a medical device. Model-derived risk estimates are experimental, depend on stated modelling assumptions, and require independent clinical validation.

## Status
| Milestone | State |
|---|---|
| M1 classical 1D vessel-network solver | ✅ done |
| M2 viscosity models, outlet resistance, geometry features, rank-sensitivity study | ✅ done |
| M3 vessel mask → vessel graph, sub-pixel width, DRIVE pipeline | ✅ done |
| M4 datasets: loaders, integrity, duplicates, leakage-free splits | ✅ done |
| M5 vessel + artery/vein U-Net, CV, test set, A/V graphs | ✅ built — train on a Colab T4 |
| M6 DR grade triage: quality check, MobileNetV3, calibration, uncertainty, Grad-CAM | ✅ built — train on a Colab T4 |
| M7 core thesis: Monte Carlo shear risk on artery/vein graphs vs IDRiD lesions (pre-registered) | ✅ built — run in Colab after M5 |
| M8 PINN / physics-informed GNN vs classical solver (done early: CPU only) | ✅ built |
| M9 API backend (FastAPI) and M10 dashboard | ✅ built — notebook step 16, or `scripts/m9_serve.py` |
| M11 results summary and thesis outline | ✅ built — notebook step 17 |

All 235 tests pass (synthetic data; real-data results come from the Colab runs).

## Run on Google Colab (recommended)
Open `notebooks/DR_PINN_Colab.ipynb` in Colab and choose **Runtime → Run all**. The notebook mounts Google Drive, installs the matching zip into `MyDrive/dr-pinn` (it checks the code version and asks for the right zip), and runs the tests, demo, benchmark and an interactive narrowing explorer. Updating to a newer milestone zip keeps your `data/`, `checkpoints/` and `reports/`.

## Local setup (Python 3.10+)
```bash
cd dr-pinn
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install torch          # from Milestone 5 (preinstalled in Colab)
```

## Commands
```bash
python -m pytest                          # all tests (~75 s; segmentation and PINN tests need PyTorch, skipped without it)
python scripts/m2_rank_sensitivity.py     # which unknowns reorder the WSS ranking (~15 s)
python scripts/m3_width_accuracy.py       # radius accuracy: distance transform vs area method (~20 s)
python scripts/m3_drive_graphs.py --drive-dir data/DRIVE   # graphs from DRIVE expert masks
python scripts/m4_prepare_datasets.py     # check all datasets in data/, find duplicates, write data/splits/manifest.csv
python scripts/m4_prepare_datasets.py --synthetic --data-dir /tmp/fake --out /tmp/fake_out   # software test only
python scripts/m5_train_segmentation.py --drive-dir data/DRIVE --rite-dir data/RITE   # U-Net CV + final model (GPU)
python scripts/m5_train_segmentation.py --synthetic /tmp/fake_m5 --preset tiny --out /tmp/m5 --ckpt-dir /tmp/m5ck   # software test only
python scripts/m8_pinn_vs_solver.py --drive-dir data/DRIVE   # PINNs vs classical solver (CPU, ~6 min; --quick ~2 min)
python scripts/m7_lesion_association.py --idrid-dir data/IDRiD --model checkpoints/m5_full/final.pt   # core experiment (CPU)
python scripts/m7_lesion_association.py --synthetic --scenario null   # software check: null / planted controls
python scripts/m6_triage.py --data-dir data --preset full    # DR triage model (GPU; resumes after a disconnect)
python scripts/m6_triage.py --synthetic --preset tiny        # software check, no data needed
python scripts/m9_serve.py --project .                      # API + dashboard at http://127.0.0.1:8000
python scripts/m9_check_disc.py --idrid-dir data/IDRiD      # how often the disc estimate hits the expert disc
python scripts/m11_collect_results.py --project .           # every result in one summary (reports/summary/)
python scripts/m12_evaluate.py --probs-dir reports/m5_full/probs/test --predictions reports/m6_full/m6_predictions.csv   # accuracy, precision, recall, F1, R², heat maps (CPU, no PyTorch)
python scripts/m12_evaluate.py --synthetic /tmp/fake_m12    # software check of the evaluation
python scripts/m5_train_segmentation.py --preset av --out reports/m5_av --ckpt-dir checkpoints/m5_av   # A/V-focused U-Net (GPU)
python scripts/demo_synthetic.py          # synthetic stenosis + blood-pressure sweep demo
python scripts/benchmark_solver.py        # latency on YOUR machine -> reports/benchmarks/solver_latency.json
```

## What exists now
- `src/drpinn/hemodynamics/units.py` — SI conversions; MAP and ocular perfusion pressure (labelled approximations); optic-disc pixel scale.
- `network.py` — vessel graph container with input validation and unit-mistake warnings.
- `solver.py` — steady Hagen–Poiseuille + Kirchhoff solver with pressure, inflow and terminal-resistance boundary conditions; `evaluate_pressure_field` scores any pressure field (used later to check the PINN).
- `synthetic.py` — single pipe, series chain, Murray tree, random trees with loops (software testing only).
- `rheology.py` — Newtonian, Fåhræus–Lindqvist (Pries 1992), Carreau–Yasuda; nonlinear solve with convergence reporting.
- `boundary.py` — Murray-scaled terminal resistance for vessel endpoints.
- `vessel_graph/features.py` — geometry-only features (the no-physics baseline).
- `uncertainty/ranking.py` — tie-aware ranks, percentiles, top-set overlap.
- `vessel_graph/extract.py` — vessel mask → graph (junctions, segments, sub-pixel width, crossing and border flags); `to_network(pixel_size_um)`.
- `vessel_graph/synthetic_masks.py`, `validation.py` — masks of known geometry; centreline recall/precision and radius agreement.
- `datasets/drive.py` — finds DRIVE images, expert masks and field-of-view masks in any folder layout.
- Method notes for M3: [`docs/vessel_graph.md`](docs/vessel_graph.md).
- `datasets/` — IDRiD, APTOS, RITE, DRIVE loaders; `signatures.py` (integrity + duplicates), `splits.py` (leakage-free splits), `synthetic.py` (fake datasets for testing). Dataset facts and terms: [`docs/datasets.md`](docs/datasets.md); setup: [`data/README.md`](data/README.md).
- `segmentation/` — U-Net (vessel, artery, vein), preprocessing, tiled prediction, training with resume, metrics, artery/vein graph separation. Method notes: [`docs/segmentation.md`](docs/segmentation.md).
- `pinn/` — physics residual on graphs, nodal and coordinate PINNs, physics-informed GNN, Jacobi/CG baselines. Method and results: [`docs/pinn.md`](docs/pinn.md).
- `uncertainty/montecarlo.py`, `regional/` — Monte Carlo WSS risk per segment, patch aggregation, image-clustered statistics, M7 pipeline. Pre-registered plan: [`docs/m7_analysis_plan.md`](docs/m7_analysis_plan.md).
- `triage/` — image quality checks, MobileNetV3 grading, temperature scaling, referral threshold, uncertainty, Grad-CAM, triage card. Plan and method: [`docs/m6_triage.md`](docs/m6_triage.md).
- `api/` — analysis service, FastAPI app, optic-disc estimate, and the dashboard page (`api/static/index.html`). See [`docs/api.md`](docs/api.md).
- `evaluation.py`, `scripts/m12_evaluate.py` — accuracy, precision, recall, F1, R² (radius, expected grade), confusion heat maps, per-photo heat map, spatial error maps. `segmentation/av.py` `av_label_map(method="vote")` — one artery/vein decision per vessel segment, crossings kept in both trees.
- Thesis structure, claims to avoid and the consolidated limitations: [`docs/thesis_outline.md`](docs/thesis_outline.md).
- Method notes: [`docs/hemodynamics_solver.md`](docs/hemodynamics_solver.md).

## Docs
- [`docs/PLAN.md`](docs/PLAN.md) — architecture, use-case mapping, untested assumptions, roadmap, checklist.
