"""Verify YOLO-style annotations and build a clinician spot-check sample.

Requirement 2 of the revision.  The v3 paper labelled an image ``fracture``
whenever its annotation file was non-blank, without checking the class ID
inside the file.  This script makes that check explicit and auditable:

  * every non-empty line must be ``<class_id> <x> <y> <w> <h>`` (YOLO,
    normalised centre + size);
  * the class ID must exist in a class-mapping JSON, e.g.
    ``{"0": "fracture"}`` -- IDs mapping to anything else, or not in the
    mapping at all, are reported, never silently treated as fracture;
  * box coordinates must satisfy 0<=x,y<=1, 0<w,h<=1 and the box must lie
    inside the image;
  * a deterministic, class-stratified random sample of annotated images is
    written out for independent (ideally clinician) review.  The reviewer
    fills in ``review_label`` / ``bbox_ok`` / ``reviewer``; nothing in the
    paper may claim verification until that file comes back completed.

Usage
-----
    python scripts/verify_annotations.py \
        --labels-dir raw_data/labels \
        --class-mapping datasets/class_mapping.json \
        --sample-frac 0.10 --seed 42 \
        --out-dir results/annotation_audit

Outputs: annotation_audit.json (counts, invalid files, class distribution),
annotation_review_sample.csv (spot-check sheet), annotation_issues.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path


def parse_file(path: Path, mapping: dict[str, str]):
    """Return (boxes, issues) for one annotation file."""
    boxes, issues = [], []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line {lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = str(int(float(parts[0])))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(f"line {lineno}: non-numeric field in {line!r}")
            continue
        if class_id not in mapping:
            issues.append(f"line {lineno}: class id {class_id} not in class mapping {sorted(mapping)}")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            issues.append(f"line {lineno}: box out of range x={x} y={y} w={w} h={h}")
            continue
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            issues.append(f"line {lineno}: box extends outside image x={x} y={y} w={w} h={h}")
            continue
        boxes.append({"class_id": class_id, "label": mapping[class_id],
                      "x_center": x, "y_center": y, "width": w, "height": h,
                      "x_min": x - w / 2, "y_min": y - h / 2,
                      "x_max": x + w / 2, "y_max": y + h / 2})
    return boxes, issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, default=Path("raw_data/labels"))
    ap.add_argument("--class-mapping", type=Path, default=Path("datasets/class_mapping.json"))
    ap.add_argument("--sample-frac", type=float, default=0.10, help="fraction sampled for human review")
    ap.add_argument("--min-sample", type=int, default=30, help="minimum sampled per class label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("results/annotation_audit"))
    args = ap.parse_args()

    if not args.labels_dir.is_dir():
        sys.exit(f"labels directory not found: {args.labels_dir}")
    if not args.class_mapping.is_file():
        sys.exit(f"class mapping not found: {args.class_mapping} "
                 f"(create it, e.g. {{\"0\": \"fracture\"}} -- one entry per source-dataset class id)")
    mapping = {str(k): str(v) for k, v in json.loads(args.class_mapping.read_text()).items()}

    files = sorted(args.labels_dir.rglob("*.txt"))
    class_counts, label_counts = Counter(), Counter()
    issue_rows, valid = [], []  # valid: (file, boxes)
    n_empty = 0
    for path in files:
        boxes, issues = parse_file(path, mapping)
        if path.stat().st_size == 0 or not boxes and not issues:
            n_empty += 1
            label_counts["no_fracture (empty/missing annotation)"] += 1
        for box in boxes:
            class_counts[f"{box['class_id']}:{box['label']}"] += 1
        if boxes:
            labels = {b["label"] for b in boxes}
            label_counts["/".join(sorted(labels))] += 1
            valid.append((path, boxes))
        for issue in issues:
            issue_rows.append({"file": str(path), "issue": issue})

    # Stratified spot-check sample, deterministic under --seed.
    rng = random.Random(args.seed)
    by_label: dict[str, list] = {}
    for path, boxes in valid:
        by_label.setdefault(boxes[0]["label"], []).append((path, boxes))
    sample = []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        k = min(len(items), max(args.min_sample, round(len(items) * args.sample_frac)))
        sample.extend((label, p, b) for p, b in items[:k])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "annotation_review_sample.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_stem", "annotation_file", "auto_label", "n_boxes", "boxes_xyxy_normalised",
                    "review_label", "bbox_ok", "reviewer", "reviewer_role", "review_date", "notes"])
        for label, path, boxes in sorted(sample, key=lambda s: s[1].name):
            xyxy = ";".join(f"{b['x_min']:.4f},{b['y_min']:.4f},{b['x_max']:.4f},{b['y_max']:.4f}" for b in boxes)
            w.writerow([path.stem, str(path), label, len(boxes), xyxy, "", "", "", "", "", ""])
    with (args.out_dir / "annotation_issues.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "issue"]); w.writeheader(); w.writerows(issue_rows)

    summary = {
        "labels_dir": str(args.labels_dir), "class_mapping": mapping,
        "n_annotation_files": len(files), "n_empty_files": n_empty,
        "n_files_with_valid_boxes": len(valid), "n_issue_rows": len(issue_rows),
        "boxes_by_class": dict(class_counts), "files_by_label": dict(label_counts),
        "spot_check_sample": {"n": len(sample), "sample_frac": args.sample_frac,
                              "seed": args.seed,
                              "status": "PENDING HUMAN REVIEW - do not claim verification until "
                                        "annotation_review_sample.csv is completed and returned"},
    }
    (args.out_dir / "annotation_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir}/annotation_audit.json, annotation_issues.csv, "
          f"annotation_review_sample.csv ({len(sample)} images for human review)")


if __name__ == "__main__":
    main()
