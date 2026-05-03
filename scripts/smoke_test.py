from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics_cfg" / "models" / "11" / "cmife-yolo.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CMIFES-YOLO model from YAML to verify patch installation.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")

    model = YOLO(str(model_path))
    model.info(verbose=True)
    print(f"[OK] Model YAML parsed successfully: {model_path}")


if __name__ == "__main__":
    main()
