"""Calibration + target-sensitivity thresholding (revision requirement 5).

Protocol (no test-set peeking):

  1. Fit a single temperature T on the **validation** predictions by
     minimising negative log-likelihood of the temperature-scaled logits.
  2. On calibrated validation probabilities, choose the lowest threshold
     whose sensitivity is >= --target-sensitivity (default 0.95); among
     thresholds meeting the target, keep the one with highest specificity.
  3. Freeze (T, threshold) and apply them unchanged to the held-out test
     predictions and, if given, to an external-dataset predictions file.
  4. Report accuracy/sensitivity/specificity/PPV/NPV/F1/AUC, ECE (15 bins),
     Brier score -- overall and per source_dataset -- plus group-cluster
     bootstrap 95% CIs (resampling groups, not images).

Inputs are prediction CSVs with columns:
    y_true in {fracture,no_fracture} or {0,1} where 1 (fracture) is the
    positive class, p_fracture in [0,1];
    optional: image, source_dataset, group_id.

    python scripts/calibrate_threshold.py \
        --val results/runs/resnet50_seed0_val_predictions.csv \
        --test results/runs/resnet50_seed0_test_predictions.csv \
        --external results/external_predictions.csv \
        --target-sensitivity 0.95 --out-dir results/calibration
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EPS = 1e-6


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    y = np.array([1 if str(r["y_true"]).strip().lower() in ("1", "fracture", "positive", "true") else 0 for r in rows])
    p = np.clip(np.array([float(r["p_fracture"]) for r in rows]), EPS, 1 - EPS)
    src = np.array([r.get("source_dataset") or "unknown" for r in rows])
    grp = np.array([r.get("group_id") or r.get("image") or str(i) for i, r in enumerate(rows)])
    return y, p, src, grp


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    def div(a, b): return float(a / b) if b else float("nan")
    order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p))
    s = p[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    npos, nneg = y.sum(), len(y) - y.sum()
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")
    bins = np.linspace(0, 1, 16); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum(): ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return {"n": int(len(y)), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "accuracy": div(tp + tn, len(y)), "sensitivity": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "auc": auc,
            "ece_15bin": float(ece), "brier": float(np.mean((p - y) ** 2))}


def fit_temperature(y, p):
    """Golden-section search for T minimising NLL of scaled logits."""
    logit = np.log(p / (1 - p))
    def nll(t):
        pc = 1 / (1 + np.exp(-logit / t))
        pc = np.clip(pc, EPS, 1 - EPS)
        return -float(np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    lo, hi, gr = 0.05, 20.0, (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(200):
        if nll(c) < nll(d): hi = d
        else: lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    t = (lo + hi) / 2
    return float(t), 1 / (1 + np.exp(-logit / t))


def choose_threshold(y, p, target):
    candidates = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best = None
    for thr in candidates:
        m = metrics(y, p, float(thr))
        if m["sensitivity"] >= target and (best is None or m["specificity"] > best[1]["specificity"]):
            best = (float(thr), m)
    if best is None:
        raise SystemExit(f"target sensitivity {target} unreachable on validation (max={metrics(y,p,0)['sensitivity']:.4f})")
    return best


def bootstrap_ci(y, p, thr, grp, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.unique(grp); by = {g: np.nonzero(grp == g)[0] for g in groups}
    keys = ["accuracy", "sensitivity", "specificity", "f1", "auc"]
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in rng.choice(groups, size=len(groups), replace=True)])
        m = metrics(y[idx], p[idx], thr)
        for k in keys: vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def report(name, y, p, src, grp, thr, do_boot):
    out = {"overall": metrics(y, p, thr), "per_source": {}}
    for s in sorted(set(src)):
        m = src == s
        if y[m].sum() and (1 - y[m]).sum():
            out["per_source"][str(s)] = metrics(y[m], p[m], thr)
    if do_boot:
        out["bootstrap_95ci_group_cluster"] = bootstrap_ci(y, p, thr, grp)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--external", type=Path)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=Path("results/calibration"))
    args = ap.parse_args()

    yv, pv, sv, gv = load(args.val)
    temp, pv_cal = fit_temperature(yv, pv)
    thr, val_at_thr = choose_threshold(yv, pv_cal, args.target_sensitivity)
    result = {"temperature_fitted_on_validation": temp,
              "threshold_selected_on_calibrated_validation": thr,
              "target_sensitivity": args.target_sensitivity,
              "validation_at_threshold": val_at_thr,
              "validation_ece_before_calibration": metrics(yv, pv, 0.5)["ece_15bin"],
              "validation_ece_after_calibration": metrics(yv, pv_cal, 0.5)["ece_15bin"]}
    print(f"Temperature T={temp:.4f} | threshold={thr:.4f} | val sensitivity={val_at_thr['sensitivity']:.4f} "
          f"specificity={val_at_thr['specificity']:.4f} | ECE {result['validation_ece_before_calibration']:.4f}"
          f" -> {result['validation_ece_after_calibration']:.4f}")
    for label, path in (("test", args.test), ("external", args.external)):
        if path is None: continue
        y, p, src, grp = load(path)
        p_cal = 1 / (1 + np.exp(-np.log(p / (1 - p)) / temp))
        result[label] = report(label, y, p_cal, src, grp, thr, args.bootstrap > 0)
        m = result[label]["overall"]
        print(f"{label:>8}: acc={m['accuracy']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f} "
              f"auc={m['auc']:.4f} ece={m['ece_15bin']:.4f} (n={m['n']}, threshold frozen from validation)")
        for s, sm in result[label]["per_source"].items():
            print(f"           source={s}: acc={sm['accuracy']:.4f} sens={sm['sensitivity']:.4f} n={sm['n']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration_threshold.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out_dir}/calibration_threshold.json")


if __name__ == "__main__":
    main()
