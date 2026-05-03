# CMIFES-YOLO

Reproducible research code for CMIFES-YOLO, a YOLO11-style detector with
Cross-scale Multi-level Information Fusion Enhancement modules for small-object
detection experiments.

This repository is organized for paper review and follow-up reproduction:
clone the repository, create the environment, install the Ultralytics patch,
prepare the dataset, train, evaluate, and record each experiment from a fixed
commit.

> Status: source release scaffold. Datasets, trained weights, and full paper
> result tables are intentionally not committed. Add weights through GitHub
> Releases, Hugging Face, or another artifact host.

## Repository Layout

```text
.
|-- configs/                         # Additional baseline/attention configs
|-- docs/                            # Dataset, experiment, and reproducibility notes
|-- figures/                         # Figure generation scripts
|-- scripts/
|   |-- install_ultralytics_patch.py  # Copies CMIFE/CMIFES patch files into installed Ultralytics
|   |-- train.py                     # Main reproducible training entry point
|   |-- evaluate.py                  # Main reproducible evaluation entry point
|   |-- run_ablation.py              # Batch ablation runner
|   |-- convert_visdrone_to_yolo.py  # VisDrone annotation conversion utility
|   `-- smoke_test.py                # Optional local model construction check
|-- ultralytics_cfg/
|   |-- datasets/                    # Dataset YAML files
|   `-- models/11/                   # CMIFES-YOLO and ablation model YAML files
|-- ultralytics_src/                 # Patch source for the installed Ultralytics package
|-- visualization/                   # Architecture visualization scripts
|-- environment.yml                  # Conda environment for CUDA 12.4 workflows
|-- requirements.txt                 # Pip dependency set
|-- pyproject.toml                   # Minimal project metadata and dependency install
|-- CITATION.cff
`-- LICENSE
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/dle868158-debug/CMIFES-YOLO.git
cd CMIFES-YOLO
```

### 2. Create the Environment

For the Windows RTX 4060 Laptop GPU workflow, use the conda file:

```bash
conda env create -f environment.yml
conda activate cmifes-yolo
```

For a pip-only environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### 3. Install the CMIFE/CMIFES Ultralytics Patch

The repository stores only the modified Ultralytics files under
`ultralytics_src/`. Install Ultralytics first, then copy the patch files into
the installed package:

```bash
python scripts/install_ultralytics_patch.py
```

Optional dry run:

```bash
python scripts/install_ultralytics_patch.py --dry-run
```

### 4. Prepare VisDrone in YOLO Format

Expected default dataset layout:

```text
datasets/VisDrone-YOLO/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

If you start from the official VisDrone folders, convert them:

```bash
python scripts/convert_visdrone_to_yolo.py ^
  --input datasets/VisDrone2019 ^
  --output datasets/VisDrone-YOLO
```

Then confirm `ultralytics_cfg/datasets/VisDrone.yaml` points to that output
directory. For a custom dataset, copy the YAML and change `path`, `train`,
`val`, `test`, and `names`.

### 5. Smoke Test

This checks that the patched Ultralytics package can parse the CMIFES-YOLO YAML:

```bash
python scripts/smoke_test.py --model ultralytics_cfg/models/11/cmife-yolo.yaml
```

### 6. Train

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

Outputs are written to `runs/train/cmifes_yolo/` by default.

### 7. Evaluate

```bash
python scripts/evaluate.py ^
  --weights runs/train/cmifes_yolo/weights/best.pt ^
  --data ultralytics_cfg/datasets/VisDrone.yaml ^
  --imgsz 640 ^
  --device 0 ^
  --output runs/eval/cmifes_yolo.json
```

### 8. Run Ablations

Run every predefined ablation:

```bash
python scripts/run_ablation.py ^
  --experiments all ^
  --data ultralytics_cfg/datasets/VisDrone.yaml ^
  --epochs 300 ^
  --batch 8 ^
  --device 0 ^
  --workers 0
```

Run selected experiments:

```bash
python scripts/run_ablation.py --experiments baseline,p3_dual,full
```

## Main Model and Ablation Configs

```text
ultralytics_cfg/models/11/cmife-yolo.yaml
ultralytics_cfg/models/11/cmife-abl-a_baseline.yaml
ultralytics_cfg/models/11/cmife-abl-b_p3_single.yaml
ultralytics_cfg/models/11/cmife-abl-c_p3_dual.yaml
ultralytics_cfg/models/11/cmife-abl-d_p3_p4_dual.yaml
ultralytics_cfg/models/11/cmife-abl-e_p3_p4_p5_dual.yaml
ultralytics_cfg/models/11/cmife-abl-f_full.yaml
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the intended experiment
matrix.

## Reproducibility Checklist

For every table row in the paper, record:

- repository commit SHA
- model YAML
- dataset YAML and dataset version
- train/val split
- random seed
- image size, batch size, epochs, optimizer, learning rate schedule
- GPU model and CUDA/PyTorch versions
- checkpoint path or artifact URL
- raw metrics JSON path

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for a complete template.

## Large Files

Do not commit datasets, `runs/`, or `*.pt` weights. Use:

- GitHub Releases for small public checkpoints
- Hugging Face Hub for model weights and reproducibility artifacts
- an institutional data repository for paper supplementary materials

## Citation

If this code is useful for your research, cite the repository with
[CITATION.cff](CITATION.cff). Replace the placeholder paper metadata after the
manuscript is accepted or posted as a preprint.

## License

This project builds on Ultralytics YOLO and keeps the AGPL-3.0 license. See
[LICENSE](LICENSE).
