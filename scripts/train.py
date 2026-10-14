"""Fine-tune one backbone on the source-grouped split (Section IV-H).

Trains a single (model, seed) configuration on ``dataset_grouped/train``,
selects the checkpoint with the highest validation F1-score, and evaluates
that checkpoint **once** on the held-out test partition.

Protocol reproduced from the manuscript
---------------------------------------
* backbones: ResNet50, EfficientNet-B0, DenseNet121 (torchvision, ImageNet
  pretrained), all layers trainable;
* inputs resized to 224 x 224 and normalised with ImageNet statistics;
* training augmentation: random horizontal flip and rotation up to 10 deg;
* Adam, lr 1e-4, cross-entropy loss, batch size 32, 5 epochs (CPU);
* checkpoint chosen on the epoch with the highest validation F1-score;
* Python / NumPy / PyTorch seeds fixed;
* the test partition is touched only after model selection.

One run writes one JSON file into ``results/runs/``; ``scripts/aggregate.py``
turns those files into Table VII.

Examples
--------
    python scripts/train.py --model resnet50       --seed 0
    python scripts/train.py --model efficientnet_b0 --seed 1
    python scripts/train.py --model densenet121    --seed 2

    # the full grid (3 backbones x 3 seeds), batch size reduced if RAM is tight
    for m in resnet50 efficientnet_b0 densenet121; do
      for s in 0 1 2; do python scripts/train.py --model $m --seed $s --batch-size 16; done
    done
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ["fracture", "no_fracture"]  # ImageFolder orders folders alphabetically
POSITIVE_INDEX = 0                          # "fracture" is the positive class

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch seeds so a run is repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name: str, num_classes: int = 2, init: str = "pretrained") -> nn.Module:
    """Backbone with a fresh classifier head.

    ``init="pretrained"`` reproduces the manuscript protocol (ImageNet
    weights).  ``init="random"`` builds the same architecture without
    downloading weights; it is intended only for offline smoke tests of this
    pipeline and must not be used for reported results.
    """
    weights = None
    if init == "pretrained":
        weights = {
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
            "densenet121": models.DenseNet121_Weights.DEFAULT,
        }[name]

    if name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return model


def make_loaders(data_dir: Path, batch_size: int, workers: int, seed: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image_datasets = {
        split: datasets.ImageFolder(str(data_dir / split), train_tf if split == "train" else eval_tf)
        for split in ("train", "val", "test")
    }
    for split, dataset in image_datasets.items():
        if dataset.classes != CLASS_NAMES:
            raise SystemExit(
                f"ERROR: {split} class folders are {dataset.classes}; expected {CLASS_NAMES}"
            )

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
            drop_last=False,
        ),
        "val": DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=workers),
        "test": DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers),
    }
    return image_datasets, loaders


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred, p_fracture) for a loader."""
    model.eval()
    truths, preds, probs = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)[:, POSITIVE_INDEX]
        preds.append(logits.argmax(dim=1).cpu().numpy())
        probs.append(probabilities.cpu().numpy())
        truths.append(labels.numpy())
    return (
        np.concatenate(truths),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def write_predictions(path: Path, dataset, y_true, p_fracture, manifest_map: dict) -> None:
    """Per-image predictions CSV consumed by calibrate_threshold.py /
    source_confounding.py. Class 0 is fracture (the positive class)."""
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["image", "y_true", "p_fracture", "source_dataset", "group_id"])
        for (sample_path, _), truth, prob in zip(dataset.samples, y_true, p_fracture):
            name = Path(sample_path).name
            meta = manifest_map.get(name, {})
            w.writerow([name, CLASS_NAMES[int(truth)], f"{float(prob):.6f}",
                        meta.get("source_dataset", "unknown"), meta.get("group_id", "")])
    print(f"  wrote {path}")


def load_manifest_map(manifest: Path | None) -> dict:
    """filename -> {source_dataset, group_id} from a split_manifest.csv."""
    if manifest is None:
        return {}
    import csv as _csv
    mapping = {}
    with Path(manifest).open(newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            mapping[Path(row["image"]).name] = {
                "source_dataset": row.get("source_dataset", "unknown"),
                "group_id": row.get("group_id", "")}
    return mapping


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_fracture: np.ndarray) -> dict:
    """Accuracy, sensitivity, F1 (fracture class) and AUC.

    Computed from the confusion matrix directly so the script has no hard
    dependency on scikit-learn; AUC uses the rank (Mann-Whitney) formula and
    therefore equals ``sklearn.metrics.roc_auc_score``.
    """
    positive = y_true == POSITIVE_INDEX
    tp = int(np.sum(positive & (y_pred == POSITIVE_INDEX)))
    fn = int(np.sum(positive & (y_pred != POSITIVE_INDEX)))
    fp = int(np.sum(~positive & (y_pred == POSITIVE_INDEX)))
    tn = int(np.sum(~positive & (y_pred != POSITIVE_INDEX)))

    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0

    # rank-based AUC (handles ties by average rank)
    order = np.argsort(p_fracture, kind="mergesort")
    ranks = np.empty(len(p_fracture), dtype=np.float64)
    sorted_scores = p_fracture[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(np.sum(positive)), int(np.sum(~positive))
    if n_pos and n_neg:
        auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "n": int(len(y_true)),
        "n_fracture": n_pos,
        "n_no_fracture": n_neg,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="resnet50",
                        choices=("resnet50", "efficientnet_b0", "densenet121"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_grouped"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers; keep 0 on a memory-constrained CPU host")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads; 0 leaves the PyTorch default")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--init", default="pretrained", choices=("pretrained", "random"),
                        help="'random' skips the ImageNet weight download; "
                             "for offline pipeline testing only, never for reported results")
    parser.add_argument("--resume", action="store_true",
                        help="skip the run if its result JSON already exists")
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"),
                        help="split_manifest.csv from make_split.py (source/group per image)")
    parser.add_argument("--external-dir", type=Path, default=None,
                        help="external dataset as ImageFolder (fracture/no_fracture); evaluated "
                             "once with the selected checkpoint, never used for selection")
    parser.add_argument("--class-weighted", action="store_true",
                        help="weight cross-entropy by inverse train class frequency")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="initialise weights from a checkpoint saved by an earlier run "
                             "(continued fine-tuning on the training partition only)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    result_path = args.out_dir / f"{args.model}_seed{args.seed}.json"
    if args.resume and result_path.exists():
        print(f"[skip] {result_path} already exists")
        return

    if not (args.data_dir / "train").is_dir():
        sys.exit(f"ERROR: {args.data_dir / 'train'} not found -- run scripts/make_split.py first")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )

    set_seed(args.seed)
    print(f"model={args.model}  seed={args.seed}  device={device}")
    print(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    image_datasets, loaders = make_loaders(args.data_dir, args.batch_size, args.workers, args.seed)
    for split, dataset in image_datasets.items():
        counts = {name: dataset.targets.count(i) for i, name in enumerate(CLASS_NAMES)}
        print(f"  {split:<5} {len(dataset):>6,} images  {counts}")

    set_seed(args.seed)  # re-seed: ImageFolder/DataLoader construction must not shift the seed
    model = build_model(args.model, init=args.init).to(device)
    if args.init_from is not None:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"  initialised from checkpoint {args.init_from}")
    if args.class_weighted:
        counts = np.array([image_datasets["train"].targets.count(i) for i in range(len(CLASS_NAMES))], dtype=float)
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
        print(f"  class-weighted loss: counts={counts.astype(int).tolist()} weights={weights.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    best_f1, best_state, best_epoch = -1.0, None, -1
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        running_loss, seen = 0.0, 0
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            seen += inputs.size(0)
        train_loss = running_loss / max(seen, 1)

        # ---- validate (model selection) ----
        y_val, p_val, prob_val = predict(model, loaders["val"], device)
        val_metrics = binary_metrics(y_val, p_val, prob_val)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": None,
            "val_accuracy": val_metrics["accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        # validation loss, for the training-curve figure
        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for inputs, labels in loaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                val_loss += criterion(model(inputs), labels).item() * inputs.size(0)
                val_seen += inputs.size(0)
        record["val_loss"] = val_loss / max(val_seen, 1)
        history.append(record)

        print(f"  epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={record['val_loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  val_auc={val_metrics['auc']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - started

    # ---- single evaluation of the selected checkpoint on the held-out test set
    if best_state is None:
        sys.exit("ERROR: no checkpoint was selected")
    model.load_state_dict(best_state)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / f"{args.model}_seed{args.seed}.pth"
    torch.save(best_state, checkpoint_path)
    print(f"  saved selected checkpoint -> {checkpoint_path}")
    manifest_map = load_manifest_map(args.manifest if args.manifest and args.manifest.exists() else None)

    y_val_best, _, prob_val_best = predict(model, loaders["val"], device)
    write_predictions(args.out_dir / f"{args.model}_seed{args.seed}_val_predictions.csv",
                      image_datasets["val"], y_val_best, prob_val_best, manifest_map)

    y_test, p_test, prob_test = predict(model, loaders["test"], device)
    test_metrics = binary_metrics(y_test, p_test, prob_test)
    write_predictions(args.out_dir / f"{args.model}_seed{args.seed}_test_predictions.csv",
                      image_datasets["test"], y_test, prob_test, manifest_map)

    external_metrics = None
    if args.external_dir is not None:
        if not args.external_dir.is_dir():
            sys.exit(f"ERROR: --external-dir not found: {args.external_dir}")
        eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        external_ds = datasets.ImageFolder(str(args.external_dir), eval_tf)
        if external_ds.classes != CLASS_NAMES:
            raise SystemExit(f"ERROR: external class folders are {external_ds.classes}; expected {CLASS_NAMES}")
        external_loader = DataLoader(external_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        y_ext, p_ext, prob_ext = predict(model, external_loader, device)
        external_metrics = binary_metrics(y_ext, p_ext, prob_ext)
        write_predictions(args.out_dir / f"{args.model}_seed{args.seed}_external_predictions.csv",
                          external_ds, y_ext, prob_ext,
                          {Path(s).name: {"source_dataset": "external", "group_id": ""} for s, _ in external_ds.samples})
        print(f"  EXTERNAL accuracy={external_metrics['accuracy']:.4f} "
              f"sensitivity={external_metrics['sensitivity']:.4f} auc={external_metrics['auc']:.4f}")

    print(f"\n  selected epoch {best_epoch} (val F1 = {best_f1:.4f})")
    print(f"  TEST  accuracy={test_metrics['accuracy']:.4f}  "
          f"sensitivity={test_metrics['sensitivity']:.4f}  f1={test_metrics['f1']:.4f}  "
          f"auc={test_metrics['auc']:.4f}")

    payload = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selected_epoch": best_epoch,
        "best_val_f1": best_f1,
        "device": str(device),
        "train_minutes": round(elapsed / 60.0, 2),
        "class_names": CLASS_NAMES,
        "positive_class": CLASS_NAMES[POSITIVE_INDEX],
        "history": history,
        "test": test_metrics,
        "external": external_metrics,
        "checkpoint": str(args.out_dir / f"{args.model}_seed{args.seed}.pth"),
        "test_predictions": {
            "y_true": y_test.tolist(),
            "y_pred": p_test.tolist(),
            "p_fracture": [round(float(v), 6) for v in prob_test],
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {result_path}  ({elapsed / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
