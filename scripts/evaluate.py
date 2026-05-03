from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "ultralytics_cfg" / "datasets" / "VisDrone.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CMIFES-YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "eval" / "metrics.json"))
    parser.add_argument("--fps-iters", type=int, default=200)
    parser.add_argument("--fps-warmup", type=int, default=50)
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def measure_fps(model: YOLO, imgsz: int, warmup: int, iters: int) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = model.model.to(device).eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            module(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            module(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return round(iters / elapsed, 3)


def main() -> None:
    args = parse_args()
    weights_path = resolve_path(args.weights)
    data_path = resolve_path(args.data)
    output_path = resolve_path(args.output)

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        plots=args.plots,
        save_json=True,
        verbose=True,
    )

    summary = {
        "weights": str(weights_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "mAP50": round(float(metrics.box.map50), 6),
        "mAP50_95": round(float(metrics.box.map), 6),
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "fps": measure_fps(model, args.imgsz, args.fps_warmup, args.fps_iters),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
