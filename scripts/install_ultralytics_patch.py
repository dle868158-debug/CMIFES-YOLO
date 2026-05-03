from __future__ import annotations

import argparse
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = PROJECT_ROOT / "ultralytics_src"
PATCH_DIRS = ("models", "nn")


def locate_ultralytics() -> Path:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("Ultralytics is not installed. Run `python -m pip install -e .` first.")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def iter_patch_files():
    for dirname in PATCH_DIRS:
        root = PATCH_ROOT / dirname
        for path in root.rglob("*"):
            if path.is_file():
                yield path


def copy_with_backup(src: Path, dst: Path, backup_root: Path | None, dry_run: bool) -> None:
    rel = src.relative_to(PATCH_ROOT)
    if dry_run:
        print(f"[DRY-RUN] {rel} -> {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if backup_root is not None and dst.exists():
        backup_path = backup_root / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, backup_path)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install CMIFE/CMIFES patch files into the installed Ultralytics package.")
    parser.add_argument("--dry-run", action="store_true", help="Print copy operations without modifying site-packages.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up overwritten Ultralytics files.")
    args = parser.parse_args()

    if not PATCH_ROOT.exists():
        raise FileNotFoundError(f"Patch directory not found: {PATCH_ROOT}")

    package_root = locate_ultralytics()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = None if args.no_backup else PROJECT_ROOT / ".patch_backups" / f"ultralytics_{stamp}"

    files = list(iter_patch_files())
    if not files:
        raise RuntimeError(f"No patch files found under {PATCH_ROOT}")

    print(f"[INFO] Ultralytics package: {package_root}")
    print(f"[INFO] Patch source: {PATCH_ROOT}")
    if backup_root is not None:
        print(f"[INFO] Backup directory: {backup_root}")

    for src in files:
        rel = src.relative_to(PATCH_ROOT)
        copy_with_backup(src, package_root / rel, backup_root, args.dry_run)

    print(f"[OK] Processed {len(files)} patch files.")


if __name__ == "__main__":
    main()
