from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ultralytics_cfg" / "models" / "11"
DEFAULT_DATA = PROJECT_ROOT / "ultralytics_cfg" / "datasets" / "VisDrone.yaml"

EXPERIMENTS = {
    "baseline": MODEL_DIR / "cmife-abl-a_baseline.yaml",
    "p3_single": MODEL_DIR / "cmife-abl-b_p3_single.yaml",
    "p3_dual": MODEL_DIR / "cmife-abl-c_p3_dual.yaml",
    "p3_p4_dual": MODEL_DIR / "cmife-abl-d_p3_p4_dual.yaml",
    "p3_p4_p5_dual": MODEL_DIR / "cmife-abl-e_p3_p4_p5_dual.yaml",
    "full": MODEL_DIR / "cmife-abl-f_full.yaml",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CMIFES-YOLO ablation experiments.")
    parser.add_argument("--experiments", default="all", help="Comma-separated keys or `all`.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "ablation"))
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print selected experiments without training.")
    return parser.parse_args()


def select_experiments(selection: str) -> list[str]:
    if selection.lower() == "all":
        return list(EXPERIMENTS)

    selected = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = [item for item in selected if item not in EXPERIMENTS]
    if unknown:
        valid = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiment key(s): {unknown}. Valid keys: {valid}")
    return selected


def run_one(key: str, args: argparse.Namespace, data_path: Path, project_path: Path) -> dict:
    cfg = EXPERIMENTS[key]
    if not cfg.exists():
        raise FileNotFoundError(f"Model config not found: {cfg}")

    print(f"\n[RUN] {key}: {cfg}")
    start = time.perf_counter()
    model = YOLO(str(cfg))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(project_path),
        name=key,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cache=args.cache,
        exist_ok=True,
        verbose=True,
    )
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device, verbose=True)
    elapsed = time.perf_counter() - start

    return {
        "experiment": key,
        "model": str(cfg),
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "mAP50": round(float(metrics.box.map50), 6),
        "mAP50_95": round(float(metrics.box.map), 6),
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "elapsed_hours": round(elapsed / 3600, 4),
    }


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    project_path = resolve_path(args.project)
    selected = select_experiments(args.experiments)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    print("[INFO] Selected experiments:")
    for key in selected:
        print(f"  - {key}: {EXPERIMENTS[key]}")

    if args.dry_run:
        return

    project_path.mkdir(parents=True, exist_ok=True)
    summary_path = project_path / "ablation_summary.json"
    results = []
    for key in selected:
        result = run_one(key, args, data_path, project_path)
        results.append(result)
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[OK] Ablation summary written to {summary_path}")


if __name__ == "__main__":
    main()
