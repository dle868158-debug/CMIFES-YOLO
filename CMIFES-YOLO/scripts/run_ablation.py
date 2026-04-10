"""
消融实验脚本: 逐步验证各改进模块的有效性
用法: python run_ablation.py

消融实验设计 (对应论文表4):
  (a) YOLOv11n 基线
  (b) + P3 单CMIFE
  (c) + P3 双CMIFE (串联)
  (d) + P3、P4 双CMIFE
  (e) + P3、P4、P5 双CMIFE
  (f) + 跨尺度全局融合 (完整模型)

消融实验设计 (对应论文表6 - 串联次数):
  单次CMIFE (P3/P4/P5各1个)
  双次CMIFE (P3/P4/P5各2个) — 本文
  三次CMIFE (P3/P4/P5各3个)
"""
import os
import json
import time
from pathlib import Path
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(CURRENT_DIR, 'VisDrone.yaml')

TRAIN_ARGS = dict(
    data=DATA_YAML,
    epochs=300,
    imgsz=640,
    batch=8,
    device=0,
    patience=50,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=10,
    cos_lr=True,
    workers=0,
    verbose=True,
)


def create_ablation_yamls():
    """创建消融实验所需的YAML配置"""
    ablation_dir = os.path.join(CURRENT_DIR, "configs", "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # ===== (a) YOLOv11n 基线 =====
    yaml_a = """# (a) YOLOv11n Baseline
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[16, 19, 22], 1, Detect, [nc]]
"""

    # ===== (b) + P3 单CMIFE =====
    yaml_b = """# (b) + P3 Single CMIFE
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]    # 16
  # P3 单CMIFE
  - [[3, 16], 1, CMIFE, [256]]     # 17
  # PAN
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]    # 20
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]    # 23
  - [[17, 20, 23], 1, Detect, [nc]]
"""

    # ===== (c) + P3 双CMIFE =====
    yaml_c = """# (c) + P3 Dual CMIFE
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]    # 16
  # P3 双CMIFE
  - [[3, 16], 1, CMIFE, [256]]     # 17
  - [-1, 1, CMIFE, [256]]          # 18
  # PAN
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]    # 21
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]    # 24
  - [[18, 21, 24], 1, Detect, [nc]]
"""

    # ===== (d) + P3、P4 双CMIFE =====
    yaml_d = """# (d) + P3, P4 Dual CMIFE
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]     # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]     # 16
  # P3 双CMIFE
  - [[3, 16], 1, CMIFE, [256]]      # 17
  - [-1, 1, CMIFE, [256]]           # 18
  # PAN + P4 双CMIFE
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]     # 21
  - [[6, 21], 1, CMIFE, [512]]      # 22
  - [-1, 1, CMIFE, [512]]           # 23
  # PAN继续
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]     # 26
  - [[18, 23, 26], 1, Detect, [nc]]
"""

    # ===== (e) + P3、P4、P5 双CMIFE =====
    yaml_e = """# (e) + P3, P4, P5 Dual CMIFE (no global fusion)
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]     # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]     # 16
  # P3 双CMIFE
  - [[3, 16], 1, CMIFE, [256]]      # 17
  - [-1, 1, CMIFE, [256]]           # 18
  # P4 双CMIFE
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]     # 21
  - [[6, 21], 1, CMIFE, [512]]      # 22
  - [-1, 1, CMIFE, [512]]           # 23
  # P5 双CMIFE
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]     # 26
  - [[10, 26], 1, CMIFE, [1024]]    # 27
  - [-1, 1, CMIFE, [1024]]          # 28
  # 无跨尺度融合, 直接检测
  - [[18, 23, 28], 1, Detect, [nc]]
"""

    # ===== (f) 完整模型 → 直接用 yolov11n_CMIFE_v2.yaml =====

    # ===== 串联次数对比: 单次CMIFE =====
    yaml_single = """# Single CMIFE at each level
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]      # 16
  # P3 单CMIFE
  - [[3, 16], 1, CMIFE, [256]]       # 17
  # P4 单CMIFE
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]      # 20
  - [[6, 20], 1, CMIFE, [512]]       # 21
  # P5 单CMIFE
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]      # 24
  - [[10, 24], 1, CMIFE, [1024]]     # 25
  # 跨尺度融合
  - [[17, 21, 25], 1, CMIFE, [512]]  # 26
  - [[17, 21, 26], 1, Detect, [nc]]
"""

    # ===== 串联次数对比: 三次CMIFE =====
    yaml_triple = """# Triple CMIFE at each level
nc: 10
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]        # 16
  # P3 三CMIFE
  - [[3, 16], 1, CMIFE, [256]]         # 17
  - [-1, 1, CMIFE, [256]]              # 18
  - [-1, 1, CMIFE, [256]]              # 19
  # P4 三CMIFE
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]        # 22
  - [[6, 22], 1, CMIFE, [512]]         # 23
  - [-1, 1, CMIFE, [512]]              # 24
  - [-1, 1, CMIFE, [512]]              # 25
  # P5 三CMIFE
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]        # 28
  - [[10, 28], 1, CMIFE, [1024]]       # 29
  - [-1, 1, CMIFE, [1024]]             # 30
  - [-1, 1, CMIFE, [1024]]             # 31
  # 跨尺度融合
  - [[19, 25, 31], 1, CMIFE, [512]]    # 32
  - [[19, 25, 32], 1, Detect, [nc]]
"""

    configs = {
        "ablation_a_baseline.yaml": yaml_a,
        "ablation_b_p3_single.yaml": yaml_b,
        "ablation_c_p3_dual.yaml": yaml_c,
        "ablation_d_p3p4_dual.yaml": yaml_d,
        "ablation_e_p3p4p5_dual.yaml": yaml_e,
        "cascade_single.yaml": yaml_single,
        "cascade_triple.yaml": yaml_triple,
    }

    for name, content in configs.items():
        path = os.path.join(ablation_dir, name)
        with open(path, 'w') as f:
            f.write(content)
        print(f"  已创建: {name}")

    return ablation_dir


def run_ablation_experiment(name, yaml_path, results_dir):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"  消融实验: {name}")
    print(f"{'='*60}")

    try:
        model = YOLO(yaml_path)
        train_args = {**TRAIN_ARGS}
        train_args["project"] = str(results_dir)
        train_args["name"] = name

        start_time = time.time()
        model.train(**train_args)
        train_time = time.time() - start_time

        # 验证
        metrics = model.val(data=DATA_YAML, imgsz=640, device=0)

        result = {
            "config": name,
            "mAP50": round(float(metrics.box.map50) * 100, 1),
            "mAP50_95": round(float(metrics.box.map) * 100, 1),
            "precision": round(float(metrics.box.mp) * 100, 1),
            "recall": round(float(metrics.box.mr) * 100, 1),
            "train_time_hours": round(train_time / 3600, 2),
        }

        info = model.info(verbose=False)
        if info:
            result["params_M"] = round(info[1] / 1e6, 1) if len(info) > 1 else "N/A"
            result["flops_G"] = round(info[2] / 1e9, 1) if len(info) > 2 else "N/A"

        print(f"  {name}: mAP@0.5={result['mAP50']}%")
        return result

    except Exception as e:
        print(f"  {name} 失败: {e}")
        return {"config": name, "error": str(e)}


def main():
    results_dir = Path(CURRENT_DIR) / "runs" / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 创建消融实验YAML
    print("创建消融实验配置...")
    ablation_dir = create_ablation_yamls()

    # 定义实验顺序
    experiments = {
        # 表4: 模块有效性验证
        "(a) YOLOv11n基线": os.path.join(ablation_dir, "ablation_a_baseline.yaml"),
        "(b) +P3单CMIFE": os.path.join(ablation_dir, "ablation_b_p3_single.yaml"),
        "(c) +P3双CMIFE": os.path.join(ablation_dir, "ablation_c_p3_dual.yaml"),
        "(d) +P3P4双CMIFE": os.path.join(ablation_dir, "ablation_d_p3p4_dual.yaml"),
        "(e) +P3P4P5双CMIFE": os.path.join(ablation_dir, "ablation_e_p3p4p5_dual.yaml"),
        "(f) +跨尺度融合(完整)": os.path.join(CURRENT_DIR, "yolov11n_CMIFE_v2.yaml"),
        # 表6: 串联次数对比
        "单次CMIFE": os.path.join(ablation_dir, "cascade_single.yaml"),
        "三次CMIFE": os.path.join(ablation_dir, "cascade_triple.yaml"),
    }

    print("\n可用消融实验:")
    for i, name in enumerate(experiments.keys()):
        print(f"  [{i}] {name}")

    selection = input("\n请选择实验 (逗号分隔, 或 'all'): ").strip()
    if selection.lower() == 'all':
        selected = list(experiments.keys())
    else:
        indices = [int(x.strip()) for x in selection.split(',')]
        keys = list(experiments.keys())
        selected = [keys[i] for i in indices if i < len(keys)]

    # 运行
    all_results = []
    for name in selected:
        yaml_path = experiments[name]
        result = run_ablation_experiment(name, yaml_path, results_dir)
        all_results.append(result)

        with open(results_dir / "ablation_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 90)
    print("  消融实验结果汇总")
    print("=" * 90)
    print(f"{'配置':<30} {'mAP@0.5':>10} {'mAP@0.5:0.95':>15} {'Precision':>12} {'Recall':>10} {'ΔmAP':>8}")
    print("-" * 90)

    baseline_map = None
    for r in all_results:
        if "error" in r:
            print(f"  {r['config']:<30} ERROR: {r['error'][:40]}")
            continue
        map50 = r.get('mAP50', 0)
        if baseline_map is None:
            baseline_map = map50
        delta = round(map50 - baseline_map, 1) if baseline_map else "—"
        print(f"  {r['config']:<30} {map50:>10} {r.get('mAP50_95','N/A'):>15} "
              f"{r.get('precision','N/A'):>12} {r.get('recall','N/A'):>10} {delta:>8}")

    print(f"\n结果已保存到: {results_dir / 'ablation_results.json'}")


if __name__ == '__main__':
    main()
