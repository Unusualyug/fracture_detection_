# Revision Runbook — leak-free retraining & paper v4

Status key: ✅ implemented in code · ⏳ waiting on data/human input · ⛔ blocked

| # | Requirement | Implementation | Status |
|---|---|---|---|
| 1 | No leakage: source-stem grouping, perceptual-hash dedup across all sources, independent train/val/test | `scripts/make_split.py` (MD5 + pHash Hamming ≤ `--phash-distance` (default 4) + `.rf.` stem union-find; GroupShuffleSplit 70/15/15, seed 42; integrity assertion that no group spans partitions) | ✅ code, ⏳ real data |
| 2 | Verify annotations: class IDs, spot-check labels + boxes, clinician if possible | `scripts/verify_annotations.py` + `datasets/class_mapping.template.json`; outputs audit JSON, issues CSV, and a stratified **review sample CSV to be completed by a qualified reviewer** | ✅ code, ⛔ clinician review (no clinician in this environment; do not claim verification until the CSV returns signed) |
| 3 | Document datasets; per-source performance | `datasets/dataset_registry.csv` (one row per source: name/URL/licence/anatomy/counts), `datasets/data_inventory.template.csv`, `source_dataset` carried through split manifest → prediction CSVs → per-source metrics in calibration/confounding reports | ✅ templates, ⏳ fill from download records |
| 4 | Held-out test + external dataset | `scripts/train.py`: checkpoint selected on validation F1 only, test touched once, optional `--external-dir` (untouched ImageFolder) evaluated with the same frozen checkpoint; `scripts/aggregate.py` for multi-seed mean±SD | ✅ code, ⏳ external dataset choice/download |
| 5 | Calibration + target-sensitivity threshold | `scripts/calibrate_threshold.py`: temperature scaling fit **on validation**, threshold chosen on calibrated validation for target sensitivity (default 0.95), frozen for test/external; ECE/Brier + group-cluster bootstrap CIs | ✅ code |
| 6 | Grad-CAM vs boxes + source-confounding | `scripts/gradcam_eval.py` (pointing game, energy-in-box vs chance, IoU, randomised-weight sanity check), `scripts/source_confounding.py` (per-source spread, leave-one-source-out, source predictability from embeddings, border/marker ablation + McNemar) | ✅ code, ⏳ needs verified boxes + heat maps |
| 7 | Revise paper | `revision/paper_v4_draft.md` — revised Methods/Results/Discussion/Limitations/Conclusion with `[PENDING: …]` placeholders; numbers are filled **only** from files under `results/` | ✅ draft, ⏳ results |

## Exact commands (once data is in place)

```bash
python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt numpy Pillow scikit-learn
# 0. Fill datasets/dataset_registry.csv, build datasets/data_inventory.csv, copy/edit class mapping
cp datasets/class_mapping.template.json datasets/class_mapping.json
# 1. Audit annotations, send results/annotation_audit/annotation_review_sample.csv to a clinician
.venv/bin/python scripts/verify_annotations.py --labels-dir raw_data/labels
# 2. Leak-free split
.venv/bin/python scripts/make_split.py --images-dir raw_data/images --labels-dir raw_data/labels \
    --manifest datasets/data_inventory.csv --out-dir dataset_grouped --results-dir results
# 3. Retrain (repeat seeds 0,1,2; optionally other backbones)
.venv/bin/python scripts/train.py --model resnet50 --seed 0 --data-dir dataset_grouped \
    --external-dir external_data     # writes checkpoint + val/test/external prediction CSVs
# 4. Aggregate
.venv/bin/python scripts/aggregate.py
# 5. Calibration + threshold (target sensitivity pre-declared: 0.95)
.venv/bin/python scripts/calibrate_threshold.py \
    --val results/runs/resnet50_seed0_val_predictions.csv \
    --test results/runs/resnet50_seed0_test_predictions.csv \
    --external results/runs/resnet50_seed0_external_predictions.csv --target-sensitivity 0.95
# 6a. Grad-CAM vs verified boxes (heat maps: run backend GradCAM per test image, save .npy per stem)
.venv/bin/python scripts/gradcam_eval.py --boxes results/boxes_test.csv --heatmap-dir results/heatmaps/test
# 6b. Source confounding
.venv/bin/python scripts/source_confounding.py --predictions results/runs/resnet50_seed0_test_predictions.csv
# 7. Smoke-test the plumbing any time (synthetic data; numbers are meaningless)
.venv/bin/python scripts/smoke_test.py   # requires PyTorch
```

## Hard blockers before any number may enter the paper
1. `raw_data/` (images + `.txt` annotations) is **not in this repo** (git-ignored, absent).
2. Clinician/reviewer spot-check (Req. 2) is a human task — the sample sheet is generated, not completed.
3. An external dataset must be chosen, downloaded under its licence, and recorded in the registry.
4. PyTorch is not installed in this sandbox and training 12k images is CPU-scale work from the v3 run; expect hours per run or use a GPU machine.
