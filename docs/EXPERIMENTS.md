# Experiment Matrix

This repository separates model definitions from training commands. Keep model
architecture changes in YAML files and keep training settings in command logs or
issue/PR descriptions.

## Primary Model

```text
ultralytics_cfg/models/11/cmife-yolo.yaml
```

## Ablation Set

| Key | Config | Purpose |
| --- | --- | --- |
| baseline | `cmife-abl-a_baseline.yaml` | YOLO11n baseline without CMIFE |
| p3_single | `cmife-abl-b_p3_single.yaml` | Add one CMIFE module at P3 |
| p3_dual | `cmife-abl-c_p3_dual.yaml` | Add cascaded CMIFE modules at P3 |
| p3_p4_dual | `cmife-abl-d_p3_p4_dual.yaml` | Add cascaded CMIFE modules at P3 and P4 |
| p3_p4_p5_dual | `cmife-abl-e_p3_p4_p5_dual.yaml` | Add cascaded CMIFE modules at P3, P4, and P5 |
| full | `cmife-abl-f_full.yaml` | Full CMIFE-YOLO with cross-scale fusion |

Run all:

```bash
python scripts/run_ablation.py --experiments all
```

Run selected experiments:

```bash
python scripts/run_ablation.py --experiments baseline,p3_dual,full
```

## Result Table Template

| Experiment | Params (M) | FLOPs (G) | mAP50 | mAP50-95 | Precision | Recall | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | | | | | | | |
| p3_single | | | | | | | |
| p3_dual | | | | | | | |
| p3_p4_dual | | | | | | | |
| p3_p4_p5_dual | | | | | | | |
| full | | | | | | | |

Each row should link to:

- commit SHA
- model YAML
- dataset YAML
- training command
- checkpoint artifact
- evaluation JSON
