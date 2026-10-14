"""Source-confounding tests (revision requirement 6, second half).

Three complementary checks, all driven by prediction/feature CSVs so they
can be re-run without retraining:

  1. Per-source performance spread on a predictions CSV: if sensitivity
     collapses on one source while the pooled number looks fine, the model
     may be exploiting source-specific cues (borders, text markers,
     acquisition protocol) rather than fracture signal.
  2. Leave-one-source-out table: metrics computed on each source alone
     (i.e. the estimate you would get had that source been external).
  3. Source predictability: given an embeddings CSV (one row per image,
     feature columns f0..fN, plus source_dataset), a cross-validated
     logistic regression predicts the *source dataset* from image
     features.  Accuracy far above chance means source identity is
     linearly readable from the representation -- a shortcut risk
     (Zech et al.; DeGrave et al.).  Also accepts --ablation-predictions:
     predictions on border-cropped / marker-masked copies of the same
     images; reports the paired accuracy change and McNemar's test.

    python scripts/source_confounding.py \
        --predictions results/runs/resnet50_seed0_test_predictions.csv \
        --ablation-predictions results/ablation_predictions.csv \
        --embeddings results/embeddings_test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_predictions(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    y = np.array([1 if str(r["y_true"]).strip().lower() in ("1", "fracture") else 0 for r in rows])
    p = np.array([float(r["p_fracture"]) for r in rows])
    src = np.array([r.get("source_dataset") or "unknown" for r in rows])
    img = np.array([r.get("image") or str(i) for i, r in enumerate(rows)])
    thr = float(rows[0].get("threshold") or 0.5) if rows and rows[0].get("threshold") else 0.5
    return y, p, src, img, thr


def sens_spec(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"n": int(len(y)), "accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--ablation-predictions", type=Path, help="predictions on border/marker-ablated copies, same image ids")
    ap.add_argument("--embeddings", type=Path, help="CSV: image,source_dataset,f0,f1,...")
    ap.add_argument("--threshold", type=float, help="override threshold (default: from CSV or 0.5)")
    ap.add_argument("--out", type=Path, default=Path("results/source_confounding.json"))
    args = ap.parse_args()
    out: dict = {}

    if args.predictions:
        y, p, src, img, thr_file = load_predictions(args.predictions)
        thr = args.threshold if args.threshold is not None else thr_file
        pred = (p >= thr).astype(int)
        out["threshold"] = thr
        out["per_source"] = {s: sens_spec(y[src == s], pred[src == s]) for s in sorted(set(src))}
        out["leave_one_source_out"] = {
            s: {**sens_spec(y[src == s], pred[src == s]),
                "note": "performance on this source alone (external-style estimate)"}
            for s in sorted(set(src))}
        sens = [v["sensitivity"] for v in out["per_source"].values() if v["sensitivity"] == v["sensitivity"]]
        out["sensitivity_spread_across_sources"] = (
            {"min": min(sens), "max": max(sens), "range": max(sens) - min(sens)} if sens else {})
        if args.ablation_predictions:
            y2, p2, _, img2, _ = load_predictions(args.ablation_predictions)
            order = {name: i for i, name in enumerate(img2)}
            pairs = [(i, order[name]) for i, name in enumerate(img) if name in order]
            if pairs:
                a = np.array([i for i, _ in pairs]); b = np.array([j for _, j in pairs])
                before = (p[a] >= thr).astype(int) == y[a]
                after = (p2[b] >= thr).astype(int) == y2[b]
                b01 = int((before & ~after).sum()); b10 = int((~before & after).sum())
                # McNemar chi-square with continuity correction
                chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) else 0.0
                out["ablation_border_marker"] = {
                    "n_paired": len(pairs), "accuracy_before": float(before.mean()),
                    "accuracy_after_ablation": float(after.mean()),
                    "delta": float(after.mean() - before.mean()),
                    "mcnemar_discordant": {"correct_to_wrong": b01, "wrong_to_correct": b10},
                    "mcnemar_chi2_cc": float(chi2),
                    "interpretation": "|delta| large => predictions depend on the ablated (non-anatomical) regions"}

    if args.embeddings:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise SystemExit("scikit-learn required for the embeddings test: pip install scikit-learn")
        with args.embeddings.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
            feats = [k for k in rows[0] if k.startswith("f") and k[1:].isdigit()]
        X = np.array([[float(r[k]) for k in feats] for r in rows])
        labels = np.array([r["source_dataset"] for r in rows])
        counts = Counter(labels)
        chance = max(counts.values()) / len(labels)
        if len(counts) >= 2 and min(counts.values()) >= 5:
            scores = cross_val_score(LogisticRegression(max_iter=1000),
                                     X, labels, cv=5)
            out["source_predictability_from_embeddings"] = {
                "n": len(labels), "sources": dict(counts), "cv_accuracy_mean": float(scores.mean()),
                "majority_class_chance": float(chance),
                "interpretation": "CV accuracy >> chance => source identity is readable from the "
                                  "representation (shortcut/confounding risk)"}
        else:
            out["source_predictability_from_embeddings"] = {
                "skipped": "need >=2 sources with >=5 images each", "sources": dict(counts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
