# Dataset Preparation

The default configuration targets VisDrone converted to YOLO format. The code
also works with any Ultralytics-compatible detection dataset after editing the
dataset YAML.

## VisDrone Source Layout

Place the original VisDrone folders under:

```text
datasets/VisDrone2019/
|-- VisDrone2019-DET-train/
|   |-- images/
|   `-- annotations/
|-- VisDrone2019-DET-val/
|   |-- images/
|   `-- annotations/
`-- VisDrone2019-DET-test-dev/
    |-- images/
    `-- annotations/
```

Convert to YOLO format:

```bash
python scripts/convert_visdrone_to_yolo.py ^
  --input datasets/VisDrone2019 ^
  --output datasets/VisDrone-YOLO
```

Expected converted layout:

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

## Dataset YAML

The default YAML is:

```text
ultralytics_cfg/datasets/VisDrone.yaml
```

For a different local path, edit:

```yaml
path: datasets/VisDrone-YOLO
train: images/train
val: images/val
test: images/test
```

For a new dataset, copy the YAML and update `names` and `nc`.

## Paper Reporting

For a reviewable paper result, report:

- dataset source and download date
- conversion command
- train/validation/test image counts
- class list
- whether ignored regions or crowd annotations were filtered
- dataset YAML used in training and evaluation
