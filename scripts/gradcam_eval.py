"""Quantitative Grad-CAM validation against bounding boxes (requirement 6).

For every fracture image with a verified box, compute:

  * pointing game: is the CAM maximum inside the (union of) box(es)?
  * energy ratio: fraction of total CAM mass inside the box(es)
    (chance level = box area fraction, also reported);
  * IoU of the binarised CAM (top ``--cam-percentile``% of mass / values
    >= ``--cam-threshold`` of the max) against the box mask;
  * Spearman correlation between the original CAM and a CAM from a
    randomised-weight model, if --randomised-dir is given (sanity check of
    Adebayo et al.: a faithful explanation must change when weights are
    randomised; high correlation = fail).

Heat maps are .npy files (float array, any size; resized by nearest
neighbour to the box grid) named after the image stem.

Input CSV columns: image, heatmap (path) OR use --heatmap-dir, plus box
columns x_min,y_min,x_max,y_max in normalised coordinates (repeat rows
for multiple boxes per image). Optional: source_dataset.

    python scripts/gradcam_eval.py --boxes results/boxes_test.csv \
        --heatmap-dir results/heatmaps/test --out-dir results/gradcam_eval
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel())).astype(float)
    rb = np.argsort(np.argsort(b.ravel())).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def resize(cam: np.ndarray, shape) -> np.ndarray:
    if cam.shape == shape:
        return cam
    yi = (np.linspace(0, cam.shape[0] - 1, shape[0])).round().astype(int)
    xi = (np.linspace(0, cam.shape[1] - 1, shape[1])).round().astype(int)
    return cam[np.ix_(yi, xi)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--heatmap-dir", type=Path)
    ap.add_argument("--randomised-dir", type=Path, help="heat maps from a weight-randomised model (sanity check)")
    ap.add_argument("--grid", type=int, default=224)
    ap.add_argument("--cam-threshold", type=float, default=0.5, help="binarise CAM at this fraction of its max")
    ap.add_argument("--out-dir", type=Path, default=Path("results/gradcam_eval"))
    args = ap.parse_args()

    with args.boxes.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    boxes: dict[str, list] = defaultdict(list)
    source: dict[str, str] = {}
    for r in rows:
        boxes[r["image"]].append([float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")])
        source[r["image"]] = r.get("source_dataset") or "unknown"

    per_image, missing = [], 0
    for image, bxs in sorted(boxes.items()):
        hpath = Path(args.heatmap_dir) / (Path(image).stem + ".npy") if args.heatmap_dir else None
        if hpath is None or not hpath.exists():
            missing += 1
            continue
        cam = resize(np.load(hpath).astype(float), (args.grid, args.grid))
        cam = cam / cam.max() if cam.max() > 0 else cam
        mask = np.zeros_like(cam, dtype=bool)
        for x0, y0, x1, y1 in bxs:
            mask[int(y0 * args.grid):max(int(y1 * args.grid), int(y0 * args.grid) + 1),
                 int(x0 * args.grid):max(int(x1 * args.grid), int(x0 * args.grid) + 1)] = True
        peak = np.unravel_index(np.argmax(cam), cam.shape)
        binary = cam >= args.cam_threshold
        inter = (binary & mask).sum(); union = (binary | mask).sum()
        rec = {"image": image, "source_dataset": source[image],
               "pointing_game_hit": bool(mask[peak]),
               "energy_inside_box": float(cam[mask].sum() / cam.sum()) if cam.sum() else 0.0,
               "box_area_fraction": float(mask.mean()),
               "iou_at_threshold": float(inter / union) if union else 0.0}
        if args.randomised_dir:
            rp = args.randomised_dir / (Path(image).stem + ".npy")
            if rp.exists():
                rec["spearman_vs_randomised"] = spearman(cam, resize(np.load(rp).astype(float), cam.shape))
        per_image.append(rec)

    def agg(recs):
        if not recs: return {"n": 0}
        return {"n": len(recs),
                "pointing_game_accuracy": float(np.mean([r["pointing_game_hit"] for r in recs])),
                "mean_energy_inside_box": float(np.mean([r["energy_inside_box"] for r in recs])),
                "mean_box_area_fraction_chance": float(np.mean([r["box_area_fraction"] for r in recs])),
                "mean_iou": float(np.mean([r["iou_at_threshold"] for r in recs])),
                **({"mean_spearman_vs_randomised": float(np.nanmean([r["spearman_vs_randomised"] for r in recs]))}
                   if "spearman_vs_randomised" in recs[0] else {})}
    summary = {"overall": agg(per_image), "missing_heatmaps": missing,
               "per_source": {s: agg([r for r in per_image if r["source_dataset"] == s])
                              for s in sorted({r["source_dataset"] for r in per_image})}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "gradcam_per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        if per_image:
            w = csv.DictWriter(fh, fieldnames=list(per_image[0])); w.writeheader(); w.writerows(per_image)
    (args.out_dir / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}/gradcam_summary.json (+ per-image CSV); missing heatmaps: {missing}")


if __name__ == "__main__":
    main()
