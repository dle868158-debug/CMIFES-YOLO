# CMIFES-YOLO
CMIFES-YOLO: A lightweight YOLO-based object detection model optimized for UAV small target detection on VisDrone dataset, with improved feature fusion and attention mechanisms for industrial inspection and aerial monitoring scenarios.
# CMIFES-YOLO

Cross-Modality Integration Feature Enhancement Strategy for YOLO-based Object Detection.

This repository contains the implementation of CMIFES-YOLO, an enhanced YOLO detector designed for improved small-object detection in drone imagery, built upon the Ultralytics YOLO framework.

## Project Structure

```
CMIFES-YOLO/
├── ultralytics_src/              # Modified Ultralytics source code
│   ├── __init__.py
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── yolo/                 # YOLO task models
│   │       ├── __init__.py
│   │       ├── detect/
│   │       ├── classify/
│   │       ├── segment/
│   │       ├── pose/
│   │       ├── obb/
│   │       ├── world/
│   │       └── yoloe/
│   └── nn/
│       ├── __init__.py
│       ├── tasks.py
│       └── modules/              # Custom modules (CMIFE, CMIFES)
│           ├── __init__.py
│           ├── CMIFE.py          # Cross-Modality Integration Feature Enhancement
│           ├── CMIFES.py         # Extended variant with enhanced skip connections
│           ├── conv.py
│           ├── block.py
│           ├── head.py
│           ├── activation.py
│           ├── transformer.py
│           └── utils.py
├── ultralytics_cfg/             # Model and dataset configurations
│   ├── models/11/
│   │   ├── cmife-yolo.yaml       # Main CMIFE-YOLO model config
│   │   ├── yolo11.yaml           # Baseline YOLO11 config
│   │   └── cmife-abl-*.yaml      # Ablation study configs (a–f)
│   └── datasets/
│       └── VisDrone.yaml         # VisDrone dataset configuration
├── scripts/                      # Training and evaluation scripts
│   ├── train-yolo11-CMIFE*.py    # Training scripts (multiple versions)
│   ├── train_cmife_v*.py         # Additional training variants
│   ├── batch_train.py            # Batch training utility
│   ├── convert_visdrone_to_yolo.py  # Dataset conversion tool
│   ├── eval_analysis.py          # Evaluation and analysis
│   ├── grad_cam_vis.py           # Grad-CAM visualization
│   ├── run_ablation.py           # Ablation study runner
│   ├── run_comparison.py         # Comparison with baselines
│   └── relitu.py                 # Relative illumination utility
├── visualization/                # Network architecture visualization
│   ├── 1_visualize_full_model.py
│   └── 2_visualize_cmifes_module.py
├── figures/                      # Figure generation scripts
│   ├── draw_cmifes.py
│   ├── draw_cmifes_yolo.py
│   └── make_fig_training_curves.py
├── configs/                      # Additional configuration files
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CMIFES-YOLO.git
cd CMIFES-YOLO

# Install dependencies
pip install ultralytics
# or install from pyproject.toml
pip install -e .
```

## Quick Start

### Training

```bash
# Train CMIFE-YOLO on VisDrone dataset
python scripts/train-yolo11-CMIFE.py --data ultralytics_cfg/datasets/VisDrone.yaml --cfg ultralytics_cfg/models/11/cmife-yolo.yaml --epochs 300 --imgsz 640
```

### Evaluation

```bash
# Run evaluation
python scripts/eval_analysis.py --weights runs/train/exp/weights/best.pt --data ultralytics_cfg/datasets/VisDrone.yaml
```

### Ablation Study

```bash
# Run ablation experiments
python scripts/run_ablation.py
```

### Comparison with Baselines

```bash
# Compare with baseline YOLO models
python scripts/run_comparison.py
```

## Dataset

The model is evaluated on the [VisDrone](https://github.com/VisDrone/VisDrone2018-DET-toolkit) dataset. Configure your dataset path in `ultralytics_cfg/datasets/VisDrone.yaml`.

## Key Modules

- **CMIFE.py**: Cross-Modality Integration Feature Enhancement module that enhances feature representation through multi-scale feature fusion.
- **CMIFES.py**: Extended CMIFE variant with enhanced skip connections for improved gradient flow and feature propagation.

## Citation

If you use this code in your research, please cite our paper.

## License

This project is based on the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework and inherits the AGPL-3.0 license.
