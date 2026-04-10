"""
VisDrone to YOLO Format Converter
Converts VisDrone2019 annotations to YOLO format for use with Ultralytics YOLO.

VisDrone format:
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
    - category: 0-9 (pedestrian=0, people=1, bicycle=2, car=3, van=4, truck=5, tricycle=6, awning-tricycle=7, bus=8, motor=9)

YOLO format:
    class_id x_center y_center width height (all normalized to 0-1)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_visdrone_to_yolo(visdrone_dir, output_dir):
    """
    Convert VisDrone annotations to YOLO format.

    Args:
        visdrone_dir: Root directory containing VisDrone2019-DET-* folders
        output_dir: Output directory for YOLO format dataset
    """
    visdrone_dir = Path(visdrone_dir)
    output_dir = Path(output_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    splits_map = {
        'VisDrone2019-DET-train': 'train',
        'VisDrone2019-DET-val': 'val',
        'VisDrone2019-DET-test-dev': 'test',
        'VisDrone2019-DET-test-challenge': 'test',
    }

    total_converted = 0

    for folder_name, split in splits_map.items():
        source_dir = visdrone_dir / folder_name
        if not source_dir.exists():
            print(f"[SKIP] {folder_name} not found, skipping...")
            continue

        images_dir = source_dir / 'images'
        annotations_dir = source_dir / 'annotations'

        if not images_dir.exists():
            print(f"[SKIP] {images_dir} not found, skipping...")
            continue
        if not annotations_dir.exists():
            print(f"[SKIP] {annotations_dir} not found, skipping...")
            continue

        print(f"\n[CONVERT] Processing {folder_name} -> {split}...")

        # Get all annotation files
        annotation_files = list(annotations_dir.glob('*.txt'))

        for ann_file in tqdm(annotation_files, desc=f"Converting {split}"):
            img_name = ann_file.stem + '.jpg'
            img_path = images_dir / img_name

            if not img_path.exists():
                img_path = images_dir / (ann_file.stem + '.png')
            if not img_path.exists():
                img_path = images_dir / (ann_file.stem + '.jpeg')

            if not img_path.exists():
                print(f"[WARN] Image not found for {ann_file.name}, skipping...")
                continue

            # Get image dimensions
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"[WARN] Failed to read image {img_path}: {e}")
                continue

            # Convert annotations
            yolo_lines = []
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(',')
                    if len(parts) < 7:
                        continue

                    try:
                        x, y, w, h = map(int, parts[:4])
                        score = int(parts[4])
                        category = int(parts[5])  # VisDrone: 0-9

                        # Skip ignored regions (score=0) and category=0 is pedestrian
                        # But actually in VisDrone, score=0 means ignored region
                        # Valid objects have score>=1
                        if score == 0:
                            continue

                        # Convert to YOLO format (normalized)
                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h

                        # Clip to valid range
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        norm_w = max(0.001, min(1, norm_w))
                        norm_h = max(0.001, min(1, norm_h))

                        # Note: VisDrone category is 0-9, YOLO should be 0-9
                        # But check if category starts from 1 (some versions)
                        # Based on the sample file, pedestrian seems to be 0
                        # Let's keep category as-is (0-9)
                        yolo_lines.append(f"{category} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                    except (ValueError, IndexError):
                        continue

            # Only copy and create label if there are valid annotations
            if yolo_lines:
                # Copy image
                shutil.copy(img_path, output_dir / 'images' / split / img_name)

                # Write YOLO format label
                with open(output_dir / 'labels' / split / (ann_file.stem + '.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))

                total_converted += 1

    print(f"\n[COMPLETE] Converted {total_converted} images to YOLO format")
    print(f"[OUTPUT] Dataset saved to: {output_dir}")

    return output_dir


def verify_dataset(dataset_dir):
    """Verify the converted dataset."""
    dataset_dir = Path(dataset_dir)

    for split in ['train', 'val', 'test']:
        img_dir = dataset_dir / 'images' / split
        lbl_dir = dataset_dir / 'labels' / split

        if img_dir.exists():
            n_images = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
            n_labels = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
            print(f"  {split}: {n_images} images, {n_labels} labels")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert VisDrone to YOLO format')
    parser.add_argument('--input', '-i', type=str,
                        default=r'C:\Users\Administrator\Desktop\D035.VisDrone2019',
                        help='VisDrone dataset root directory')
    parser.add_argument('--output', '-o', type=str,
                        default=r'C:\Users\Administrator\Desktop\D035.VisDrone2019_YOLO',
                        help='Output directory for YOLO format dataset')

    args = parser.parse_args()

    print("=" * 60)
    print("VisDrone to YOLO Format Converter")
    print("=" * 60)

    # Convert dataset
    dataset_path = convert_visdrone_to_yolo(args.input, args.output)

    # Verify
    print("\n[VERIFY] Dataset verification:")
    verify_dataset(dataset_path)

    print("\n[READY] Dataset is ready for training!")
    print(f"Use this path in your YAML config: {dataset_path}")
