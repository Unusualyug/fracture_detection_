"""Group-aware re-split of the fracture image pool (Section IV-H of the paper).

This script builds the leak-free, source-grouped train/validation/test split
described in Section IV-H and Table VI of the manuscript.

Grouping rule (Section IV-H)
---------------------------
Two images belong to the same group if *any* of the following holds:

  1. they have the same MD5 hash (byte-identical files);
  2. they have the same perceptual hash (pHash) of the greyscale image
     (detects re-encoded / resized copies);
  3. they share the same file-name stem *before* the Roboflow ``.rf.<hash>``
     suffix (detects augmented copies exported from one source image).

Each connected set under these relations becomes one group.  Groups -- not
images -- are then split 70/15/15 with scikit-learn's ``GroupShuffleSplit``
using a fixed seed, so that no group can span two partitions.

Usage
-----
    python scripts/make_split.py \
        --images-dir raw_data/images \
        --labels-dir raw_data/labels \
        --out-dir    dataset_grouped

The expected layout is one folder of images plus a sibling folder of
per-image ``.txt`` annotation files with matching basenames.  An image whose
annotation file exists and is non-empty is labelled ``fracture``; an image
whose annotation file is empty or missing is labelled ``no_fracture``.

Outputs
-------
    dataset_grouped/{train,val,test}/{fracture,no_fracture}/...
    results/split_manifest.csv      one row per image (image,label,group,split)
    results/split_summary.json      group counts and partition sizes

Run ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from sklearn.model_selection import GroupShuffleSplit
except ImportError:  # pragma: no cover
    sys.exit("scikit-learn is required:  pip install scikit-learn")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Roboflow export pattern:  <source>.rf.<hex>.jpg
RF_SUFFIX_RE = re.compile(r"\.rf\.[0-9a-fA-F]+$")

DEFAULT_SEED = 42


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def sha_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 digest of a file, read in chunks to bound memory."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DCT_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of size n x n."""
    if n not in _DCT_CACHE:
        k = np.arange(n).reshape(-1, 1)
        x = np.arange(n).reshape(1, -1)
        _DCT_CACHE[n] = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    return _DCT_CACHE[n]


def phash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> str | None:
    """Perceptual hash (pHash) of an image, returned as a hex string.

    The image is converted to greyscale, resized to 32x32, transformed with a
    2-D DCT and reduced to the top-left 8x8 low-frequency block.  Bits are set
    where a coefficient exceeds the median of the block (excluding the DC
    term), matching the standard ``imagehash.phash`` construction.

    Returns ``None`` if the image cannot be decoded, so that unreadable files
    are grouped by their other keys rather than crashing the run.
    """
    size = hash_size * highfreq_factor
    try:
        with Image.open(path) as image:
            grey = image.convert("L").resize((size, size), Image.LANCZOS)
            pixels = np.asarray(grey, dtype=np.float64)
    except Exception:  # unreadable / truncated file
        return None

    basis = _dct_matrix(size)
    coeffs = basis @ pixels @ basis.T
    low = coeffs[:hash_size, :hash_size]
    median = np.median(low.reshape(-1)[1:])
    bits = (low > median).reshape(-1)
    return np.packbits(bits).tobytes().hex()


def source_stem(path: Path) -> str:
    """File-name stem with any Roboflow ``.rf.<hash>`` suffix removed.

    ``abc_mp4-1_jpg.rf.9f2c1ab.jpg`` -> ``abc_mp4-1_jpg``
    ``plain_name.png``               -> ``plain_name``
    """
    return RF_SUFFIX_RE.sub("", path.stem)


class UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------
def collect_images(images_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        sys.exit(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} found in {images_dir}")
    return files


def label_for(image_path: Path, images_dir: Path, labels_dir: Path) -> tuple[str, str]:
    """Return (label, label_file_relative_name) for one image."""
    rel = image_path.relative_to(images_dir)
    candidate = labels_dir / rel.with_suffix(".txt")
    if not candidate.exists():
        # fall back to a flat lookup by basename
        candidate = labels_dir / (image_path.stem + ".txt")
    if candidate.exists() and candidate.stat().st_size > 0:
        return "fracture", str(candidate.relative_to(labels_dir))
    return "no_fracture", ""


def build_groups(images: list[Path]) -> tuple[list[int], dict[str, int]]:
    """Union-find over the three grouping keys; return (group_id per image, stats)."""
    uf = UnionFind(len(images))
    first_seen: dict[tuple[str, str], int] = {}
    stats = Counter()

    for index, path in enumerate(images):
        keys = (
            ("md5", sha_md5(path)),
            ("stem", source_stem(path)),
        )
        phash_value = phash(path)
        keys = keys + (("phash", phash_value),) if phash_value else keys

        for kind, value in keys:
            if not value:
                continue
            slot = (kind, value)
            if slot in first_seen:
                if uf.find(first_seen[slot]) != uf.find(index):
                    stats[f"merged_by_{kind}"] += 1
                uf.union(first_seen[slot], index)
            else:
                first_seen[slot] = index

    roots: dict[int, int] = {}
    group_ids: list[int] = []
    for index in range(len(images)):
        root = uf.find(index)
        if root not in roots:
            roots[root] = len(roots)
        group_ids.append(roots[root])

    stats["n_images"] = len(images)
    stats["n_groups"] = len(roots)
    stats["n_unique_md5"] = sum(1 for k in first_seen if k[0] == "md5")
    stats["n_unique_phash"] = sum(1 for k in first_seen if k[0] == "phash")
    stats["n_unique_stem"] = sum(1 for k in first_seen if k[0] == "stem")
    return group_ids, dict(stats)


def split_groups(group_ids: list[int], seed: int, val_size: float, test_size: float):
    """70/15/15 group-wise split via two GroupShuffleSplit calls."""
    images = np.zeros(len(group_ids), dtype=np.int8)
    groups = np.asarray(group_ids)

    holdout = val_size + test_size
    first = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, rest_idx = next(first.split(images, groups=groups))

    share_of_rest = test_size / holdout
    second = GroupShuffleSplit(n_splits=1, test_size=share_of_rest, random_state=seed)
    val_rel, test_rel = next(
        second.split(images[rest_idx], groups=groups[rest_idx])
    )
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]
    return train_idx, val_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=Path("raw_data/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("raw_data/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset_grouped"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="how to materialise the split (symlink/hardlink save disk space)",
    )
    parser.add_argument(
        "--no-materialise",
        action="store_true",
        help="write the manifest and summary only, without building dataset folders",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        sys.exit(f"images directory not found: {args.images_dir}")
    if not args.labels_dir.is_dir():
        print(f"[warn] labels directory not found ({args.labels_dir}); "
              f"every image would be labelled no_fracture", file=sys.stderr)

    print(f"[1/5] scanning {args.images_dir} ...")
    images = collect_images(args.images_dir)
    print(f"      found {len(images):,} images")

    print("[2/5] labelling and hashing (MD5 + pHash) ...")
    labels: list[str] = []
    label_files: list[str] = []
    for i, path in enumerate(images, start=1):
        label, label_file = label_for(path, args.images_dir, args.labels_dir)
        labels.append(label)
        label_files.append(label_file)
        if i % 500 == 0 or i == len(images):
            print(f"      {i:,}/{len(images):,}", flush=True)

    group_ids, stats = build_groups(images)
    print(f"      {stats['n_groups']:,} groups from {stats['n_images']:,} images "
          f"({stats['n_images'] / max(stats['n_groups'], 1):.2f} files/group)")

    print("[3/5] splitting groups "
          f"({1 - args.val_size - args.test_size:.2f}/{args.val_size}/{args.test_size}, seed {args.seed}) ...")
    train_idx, val_idx, test_idx = split_groups(
        group_ids, args.seed, args.val_size, args.test_size
    )
    partitions = {"train": train_idx, "val": val_idx, "test": test_idx}

    # -- integrity check: no group may appear in more than one partition -----
    seen: dict[int, str] = {}
    clashes = 0
    for name, idx in partitions.items():
        for i in idx:
            gid = group_ids[i]
            if gid in seen and seen[gid] != name:
                clashes += 1
            seen[gid] = name
    if clashes:
        sys.exit(f"FATAL: {clashes} images belong to a group that spans two partitions")

    # -- report -------------------------------------------------------------
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "seed": args.seed,
        "images_dir": str(args.images_dir),
        "labels_dir": str(args.labels_dir),
        "group_stats": stats,
        "partitions": {},
        "group_overlap_check": "passed (no group spans two partitions)",
    }

    print("\n      partition   fracture  no_fracture    total   groups")
    print("      " + "-" * 52)
    for name, idx in partitions.items():
        counts = Counter(labels[i] for i in idx)
        n_groups = len({group_ids[i] for i in idx})
        summary["partitions"][name] = {
            "fracture": counts.get("fracture", 0),
            "no_fracture": counts.get("no_fracture", 0),
            "total": len(idx),
            "groups": n_groups,
        }
        print(f"      {name:<11} {counts.get('fracture', 0):>9,} {counts.get('no_fracture', 0):>12,} "
              f"{len(idx):>8,} {n_groups:>8,}")
        for i in idx:
            rows.append(
                {
                    "image": str(images[i].relative_to(args.images_dir)),
                    "label": labels[i],
                    "label_file": label_files[i],
                    "group_id": group_ids[i],
                    "split": name,
                }
            )
    total = sum(len(v) for v in partitions.values())
    print("      " + "-" * 52)
    print(f"      {'total':<11} {sum(1 for l in labels if l == 'fracture'):>9,} "
          f"{sum(1 for l in labels if l == 'no_fracture'):>12,} {total:>8,} "
          f"{stats['n_groups']:>8,}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.results_dir / "split_manifest.csv"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("image,label,label_file,group_id,split\n")
        for row in rows:
            handle.write(
                f"{row['image']},{row['label']},{row['label_file']},"
                f"{row['group_id']},{row['split']}\n"
            )
    (args.results_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n      manifest -> {manifest}")
    print(f"      summary  -> {args.results_dir / 'split_summary.json'}")

    # -- materialise --------------------------------------------------------
    if args.no_materialise:
        print("[5/5] --no-materialise set; dataset folders not written")
        return

    print(f"[4/5] writing dataset folders to {args.out_dir} ({args.copy_mode}) ...")
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for name, idx in partitions.items():
        for i in idx:
            destination = args.out_dir / name / labels[i] / images[i].name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = images[i]
            if args.copy_mode == "copy":
                shutil.copy2(source, destination)
            elif args.copy_mode == "symlink":
                destination.symlink_to(source.resolve())
            else:
                try:
                    destination.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)
    print("[5/5] done")


if __name__ == "__main__":
    main()
