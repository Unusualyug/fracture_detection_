"""Aggregate per-run results into Table VII (``results/summary_test.csv``).

Reads every ``results/runs/<model>_seed<k>.json`` written by
``scripts/train.py``, groups the runs by backbone and reports each metric as
mean +/- standard deviation across seeds.

Usage
-----
    python scripts/aggregate.py
    python scripts/aggregate.py --runs-dir results/runs --out-dir results

Outputs
-------
    results/summary_test.csv     Table VII in machine-readable form
    results/table_vii.md         the same table as a Markdown fragment
    results/table_vii.tex        the same table as IEEEtran rows

The standard deviation is taken across seeds on a single fixed split, so it
reflects training randomness only and not test-set sampling variability
(manuscript, Section IV-H).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("accuracy", "Accuracy"),
    ("sensitivity", "Sensitivity"),
    ("f1", "F1-score"),
    ("auc", "AUC"),
]

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet121",
}
MODEL_ORDER = ["resnet50", "efficientnet_b0", "densenet121"]


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] skipping unreadable {path}")
            continue
        if "test" not in payload:
            print(f"[warn] skipping {path}: no 'test' block")
            continue
        runs.append(payload)
    if not runs:
        raise SystemExit(
            f"no run files found in {runs_dir} -- run scripts/train.py first"
        )
    return runs


def fmt(value: float, percent: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}" if percent else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["model"]].append(run)

    order = [m for m in MODEL_ORDER if m in grouped] + [m for m in grouped if m not in MODEL_ORDER]

    # ---------------------------------------------------------------- CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "metric", "mean", "sd", "n_seeds", "per_seed"])
        for model in order:
            model_runs = sorted(grouped[model], key=lambda r: r["seed"])
            for key, _label in METRICS:
                values = [r["test"][key] for r in model_runs if r["test"].get(key) is not None]
                if not values:
                    continue
                mean = statistics.fmean(values)
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                writer.writerow([
                    model,
                    key,
                    f"{mean:.6f}",
                    f"{sd:.6f}",
                    len(values),
                    " ".join(f"{v:.6f}" for v in values),
                ])
    print(f"wrote {csv_path}")

    # ----------------------------------------------------------- Markdown
    lines = ["| Model | Accuracy | Sensitivity | F1-score | AUC |", "|---|---|---|---|---|"]
    for model in order:
        model_runs = sorted(grouped[model], key=lambda r: r["seed"])
        cells = []
        for key, _label in METRICS:
            values = [r["test"][key] for r in model_runs if r["test"].get(key) is not None]
            if not values:
                cells.append("n/a")
                continue
            mean = statistics.fmean(values) * 100
            sd = (statistics.stdev(values) if len(values) > 1 else 0.0) * 100
            cells.append(f"{mean:.2f} ± {sd:.2f}")
        lines.append(f"| {MODEL_LABELS.get(model, model)} | " + " | ".join(cells) + " |")
    markdown = "\n".join(lines) + "\n"
    (args.out_dir / "table_vii.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out_dir / 'table_vii.md'}")

    # -------------------------------------------------------------- LaTeX
    tex_lines = []
    for model in order:
        model_runs = sorted(grouped[model], key=lambda r: r["seed"])
        cells = []
        for key, _label in METRICS:
            values = [r["test"][key] for r in model_runs if r["test"].get(key) is not None]
            if not values:
                cells.append("n/a")
                continue
            mean = statistics.fmean(values) * 100
            sd = (statistics.stdev(values) if len(values) > 1 else 0.0) * 100
            cells.append(f"${mean:.2f} \\pm {sd:.2f}$")
        tex_lines.append(f"{MODEL_LABELS.get(model, model)} & " + " & ".join(cells) + r" \\")
    (args.out_dir / "table_vii.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / 'table_vii.tex'}")

    # ------------------------------------------------------------ console
    print()
    print("Held-out test results, mean ± SD over seeds")
    print("=" * 78)
    print(markdown)
    for model in order:
        model_runs = sorted(grouped[model], key=lambda r: r["seed"])
        seeds = ", ".join(str(r["seed"]) for r in model_runs)
        epochs = ", ".join(str(r.get("selected_epoch", "?")) for r in model_runs)
        minutes = ", ".join(f"{r.get('train_minutes', float('nan')):.0f}" for r in model_runs)
        print(f"  {MODEL_LABELS.get(model, model):<14} seeds [{seeds}]  "
              f"selected epochs [{epochs}]  train min [{minutes}]")


if __name__ == "__main__":
    main()
