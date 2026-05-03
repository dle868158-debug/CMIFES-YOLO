# Reproducibility Protocol

Use this page as the experiment log template for paper tables and reviewer
responses. Every result should be traceable to one commit, one dataset YAML,
one model YAML, and one checkpoint.

## Environment

Record:

```text
OS:
GPU:
CUDA:
Python:
PyTorch:
Ultralytics:
Repository commit:
Patch install command:
```

Recommended setup:

```bash
conda env create -f environment.yml
conda activate cmifes-yolo
python scripts/install_ultralytics_patch.py
```

## Dataset

Record:

```text
Dataset name:
Dataset source:
Dataset version/date:
Dataset YAML:
Train images:
Validation images:
Test images:
Class list:
Conversion script and command:
```

For VisDrone:

```bash
python scripts/convert_visdrone_to_yolo.py ^
  --input datasets/VisDrone2019 ^
  --output datasets/VisDrone-YOLO
```

## Training Record

Copy this block into an issue, PR, lab note, or paper artifact log.

```text
Experiment id:
Commit SHA:
Model YAML:
Dataset YAML:
Seed:
Image size:
Batch size:
Epochs:
Optimizer:
Initial LR:
Scheduler:
Workers:
Device:
Start time:
End time:
Run directory:
Best checkpoint:
Last checkpoint:
```

Canonical training command:

```bash
python scripts/train.py ^
  --model ultralytics_cfg/models/11/cmife-yolo.yaml ^
  --data ultralytics_cfg/datasets/VisDrone.yaml ^
  --epochs 300 ^
  --imgsz 640 ^
  --batch 8 ^
  --device 0 ^
  --workers 0 ^
  --seed 42 ^
  --name cmifes_yolo
```

## Evaluation Record

```text
Checkpoint:
Dataset YAML:
Split:
mAP50:
mAP50-95:
Precision:
Recall:
FPS:
Evaluation JSON:
```

Canonical evaluation command:

```bash
python scripts/evaluate.py ^
  --weights runs/train/cmifes_yolo/weights/best.pt ^
  --data ultralytics_cfg/datasets/VisDrone.yaml ^
  --imgsz 640 ^
  --device 0 ^
  --output runs/eval/cmifes_yolo.json
```

## Artifact Rules

Do not commit large artifacts. Save them outside Git and record URLs:

```text
Checkpoint URL:
TensorBoard/W&B URL:
Prediction samples URL:
Raw metrics archive URL:
```
