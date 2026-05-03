from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics_cfg" / "models" / "11" / "cmife-yolo.yaml"
DEFAULT_DATA = PROJECT_ROOT / "ultralytics_cfg" / "datasets" / "VisDrone.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CMIFES-YOLO with reproducible command-line arguments.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model YAML path.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows to avoid dataloader spawn issues.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train"))
    parser.add_argument("--name", default="cmifes_yolo")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--cache", action="store_true", help="Cache images during training.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when supported by Ultralytics.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    project_path = resolve_path(args.project)

    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(project_path),
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        cache=args.cache,
        pretrained=args.pretrained,
        exist_ok=args.exist_ok,
        verbose=True,
    )


if __name__ == "__main__":
    main()
