"""Smoke test for the scripts/ pipeline, run on a synthetic mini-dataset.

Builds an image pool that mimics a real Roboflow/Kaggle pool -- several
augmented copies of one source image sharing a file-name stem, plus
byte-identical and re-encoded duplicates arriving under *different* names --
and then checks that the pipeline behaves as the manuscript describes.

Checks performed
----------------
  * ``phash()`` is stable across re-encoding and separates distinct images;
  * ``source_stem()`` removes the ``.rf.`` suffix;
  * ``make_split.py`` collapses all copies of one source into one group
    (30 sources -> 30 groups), and no group spans two partitions;
  * ``train.py`` runs end to end and writes a well-formed result JSON;
  * ``aggregate.py`` writes ``summary_test.csv`` with the expected shape.

The training step uses randomly initialised weights, so the *numbers* it
produces are meaningless -- this test validates the plumbing only, and would
take far longer than the real run without a GPU.  Use it before committing a
change to the pipeline, and never quote its output as a result.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "test_output" / "smoke"
N_SOURCES = 30


def load_make_split():
    """Import scripts/make_split.py as a module so its helpers can be tested."""
    path = ROOT / "scripts" / "make_split.py"
    spec = importlib.util.spec_from_file_location("make_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_image(path: Path, pattern: int, size=(224, 224)) -> None:
    """Draw a deterministic, pattern-dependent image (so pHash will differ)."""
    random.seed(pattern)
    image = Image.new("L", size, color=random.randint(0, 60))
    draw = ImageDraw.Draw(image)
    for _ in range(16):
        x0, y0 = random.randint(0, size[0] - 60), random.randint(0, size[1] - 60)
        draw.ellipse(
            [x0, y0, x0 + random.randint(20, 70), y0 + random.randint(20, 70)],
            fill=random.randint(90, 255),
            outline=random.randint(0, 255),
        )
    image.save(path)


def build_pool() -> tuple[Path, Path]:
    images, labels = WORK / "raw_data" / "images", WORK / "raw_data" / "labels"
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    for index in range(N_SOURCES):
        name = f"src{index:03d}_jpg"
        label = "fracture" if index % 2 == 0 else "no_fracture"
        h1, h2, h3 = (f"{index:04d}{tag}" for tag in ("aaaa1111", "bbbb2222", "cccc3333"))

        primary = images / f"{name}.rf.{h1}.jpg"
        make_image(primary, index)

        with Image.open(primary) as img:
            # augmented copies of the same source share the file-name stem
            img.rotate(4, expand=True).save(images / f"{name}.rf.{h2}.jpg")
            img.rotate(-6, expand=True).save(images / f"{name}.rf.{h3}.jpg")
            # re-encoded duplicate: same pixels, different bytes AND different
            # stem, so only the perceptual hash can link it to the source
            img.save(images / f"reencoded_{index:03d}.png")

        # byte-identical duplicate under a different stem, as happens when two
        # Kaggle datasets carry the same image: only MD5 links it
        shutil.copy2(primary, images / f"duplicate_{index:03d}.jpg")

        for stem in (
            f"{name}.rf.{h1}",
            f"{name}.rf.{h2}",
            f"{name}.rf.{h3}",
            f"reencoded_{index:03d}",
            f"duplicate_{index:03d}",
        ):
            text = "0 0.5 0.5 0.2 0.2" if label == "fracture" else ""
            (labels / f"{stem}.txt").write_text(text, encoding="utf-8")

    return images, labels


def run(*args: str) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(Path(a).name if '/' in a else a for a in args)}")
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip().splitlines()
    for line in output[-20:]:
        print("   " + line)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(args)}")
    return result


def check_hash_helpers(make_split) -> None:
    """phash() must survive re-encoding, and source_stem() must strip .rf."""
    scratch = WORK / "hash_check"
    scratch.mkdir(parents=True, exist_ok=True)

    original = scratch / "one.jpg"
    make_image(original, 101)
    with Image.open(original) as img:
        img.save(scratch / "one_reencoded.png")   # identical pixels, new bytes
        img.rotate(90, expand=True).save(scratch / "one_rotated.jpg")
    other = scratch / "two.jpg"
    make_image(other, 202)

    h_original = make_split.phash(original)
    h_reencoded = make_split.phash(scratch / "one_reencoded.png")
    h_rotated = make_split.phash(scratch / "one_rotated.jpg")
    h_other = make_split.phash(other)

    assert h_original == h_reencoded, "pHash changed after lossless re-encoding"
    assert h_original != h_other, "pHash collided for two distinct images"
    print(f"[OK] pHash stable across re-encoding ({h_original}); "
          f"distinct images differ; rotated copy differs ({h_rotated != h_original})")

    cases = {
        "abc_mp4-1_jpg.rf.9f2c1ab.jpg": "abc_mp4-1_jpg",
        "plain_name.png": "plain_name",
        "chest_xray.rf.0000ffff.jpeg": "chest_xray",
    }
    for filename, expected in cases.items():
        got = make_split.source_stem(Path(filename))
        assert got == expected, f"source_stem({filename}) = {got}, expected {expected}"
    print(f"[OK] source_stem strips the .rf. suffix ({len(cases)} cases)")


def main() -> None:
    python = sys.executable
    make_split = load_make_split()

    images, labels = build_pool()
    n_images = len(list(images.iterdir()))
    print(f"built synthetic pool: {n_images} images, {N_SOURCES} distinct sources "
          f"({N_SOURCES // 2} fracture / {N_SOURCES // 2} no_fracture)")

    check_hash_helpers(make_split)

    # ---------------------------------------------------------------- split
    run(python, "scripts/make_split.py",
        "--images-dir", str(images), "--labels-dir", str(labels),
        "--out-dir", str(WORK / "dataset_grouped"),
        "--results-dir", str(WORK / "results"),
        "--copy-mode", "copy")

    summary = json.loads((WORK / "results" / "split_summary.json").read_text())
    groups = summary["group_stats"]["n_groups"]
    assert groups == N_SOURCES, f"expected {N_SOURCES} groups (one per source), got {groups}"
    assert "passed" in summary["group_overlap_check"]
    print(f"\n[OK] grouping: {n_images} files collapsed to {groups} source groups "
          f"(md5 + pHash + stem rules all contributed)")
    print(f"[OK] overlap check: {summary['group_overlap_check']}")

    for name, counts in summary["partitions"].items():
        print(f"[OK] {name:<5} {counts['total']:>4} images in {counts['groups']:>2} groups  "
              f"({counts['fracture']} fracture / {counts['no_fracture']} no_fracture)")
        assert counts["total"] > 0, f"partition {name} is empty"

    assert summary["partitions"]["test"]["fracture"] > 0
    assert summary["partitions"]["test"]["no_fracture"] > 0, (
        "held-out test partition must contain both classes for the metrics to be defined"
    )

    manifest_rows = (WORK / "results" / "split_manifest.csv").read_text().splitlines()
    assert len(manifest_rows) == n_images + 1, "manifest row count mismatch"
    print(f"[OK] manifest has {len(manifest_rows) - 1} rows")

    # --------------------------------------------------------------- train
    run(python, "scripts/train.py",
        "--model", "resnet50", "--seed", "0",
        "--data-dir", str(WORK / "dataset_grouped"),
        "--out-dir", str(WORK / "results" / "runs"),
        "--epochs", "1", "--batch-size", "4", "--init", "random",
        "--threads", "2")

    result_file = WORK / "results" / "runs" / "resnet50_seed0.json"
    payload = json.loads(result_file.read_text())
    for key in ("accuracy", "sensitivity", "f1", "auc"):
        assert key in payload["test"], f"missing metric {key}"
        value = payload["test"][key]
        assert value == value, f"{key} is NaN"          # NaN != NaN
        assert 0.0 <= value <= 1.0, f"{key}={value} out of range"
    assert payload["test"]["n"] == summary["partitions"]["test"]["total"]
    print(f"\n[OK] train.py wrote {result_file.name} "
          f"(epochs={payload['epochs']}, selected_epoch={payload['selected_epoch']}, "
          f"test n={payload['test']['n']})")

    # ----------------------------------------------------------- aggregate
    run(python, "scripts/aggregate.py",
        "--runs-dir", str(WORK / "results" / "runs"),
        "--out-dir", str(WORK / "results"))

    csv_path = WORK / "results" / "summary_test.csv"
    rows = csv_path.read_text().strip().splitlines()
    assert rows[0] == "model,metric,mean,sd,n_seeds,per_seed", rows[0]
    assert len(rows) == 5, f"expected header + 4 metric rows, got {len(rows)}"
    print("\n" + csv_path.read_text())
    assert (WORK / "results" / "table_vii.md").exists()
    assert (WORK / "results" / "table_vii.tex").exists()
    print("[OK] aggregate.py wrote summary_test.csv, table_vii.md, table_vii.tex")

    print("=" * 72)
    print("SMOKE TEST PASSED - the pipeline runs end to end.")
    print("The numbers above come from randomly initialised weights on synthetic")
    print("images and are NOT results. Run scripts/train.py on the real data to")
    print("obtain the values for Table VII.")
    print("=" * 72)


if __name__ == "__main__":
    main()
