"""Export Grad-CAM heat maps (.npy) for images listed in a boxes CSV.

Uses the checkpoint saved by scripts/train.py and the same GradCAM class
as the deployed backend.  Also writes, with --randomised-out, heat maps
from a copy of the model whose layer4 + classifier weights have been
re-randomised (Adebayo et al. sanity check input for gradcam_eval.py).

    python scripts/export_gradcam.py \
        --checkpoint results/runs/resnet50_seed0.pth \
        --boxes results/boxes_test.csv --out-dir results/heatmaps/test \
        --randomised-out results/heatmaps/test_randomised
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.gradcam import GradCAM  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build(checkpoint: Path, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, required=True, help="CSV with an 'image' column (paths or stems)")
    ap.add_argument("--images-root", type=Path, required=True, help="root searched recursively for image files")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--randomised-out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")
    with args.boxes.open(newline="", encoding="utf-8") as fh:
        names = sorted({r["image"] for r in csv.DictReader(fh)})
    index = {p.stem: p for p in args.images_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def export(model, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        cam = GradCAM(model, model.layer4)
        done = 0
        for name in names:
            path = index.get(Path(name).stem)
            if path is None:
                continue
            with Image.open(path) as im:
                tensor = tf(im.convert("RGB")).unsqueeze(0).to(device)
            heat = cam.generate(tensor, class_idx=0)  # fracture class
            np.save(out_dir / (Path(name).stem + ".npy"), heat.astype(np.float32))
            done += 1
        print(f"wrote {done} heat maps -> {out_dir}")

    model = build(args.checkpoint, device)
    export(model, args.out_dir)
    if args.randomised_out:
        torch.manual_seed(args.seed)
        randomised = build(args.checkpoint, device)
        for layer in (randomised.layer4, randomised.fc):
            for module in layer.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.reset_parameters()
        randomised.eval()
        export(randomised, args.randomised_out)


if __name__ == "__main__":
    main()
