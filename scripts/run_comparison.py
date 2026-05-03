"""
对比实验脚本: 在VisDrone数据集上对比多种目标检测方法
用法: python run_comparison.py

对比方法:
  1. YOLOv5n (基线)
  2. YOLOv8n (基线)
  3. YOLOv9t
  4. YOLOv10n
  5. YOLOv11n (主基线)
  6. YOLOv11n + SE
  7. YOLOv11n + CBAM
  8. YOLOv11n + ECA
  9. RT-DETR-l
  10. CMIFE-YOLO v2 (本文)
"""
from ultralytics import YOLO
import os
import json
import time
from pathlib import Path


# ======================== 配置 ========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(CURRENT_DIR, 'VisDrone.yaml')

# 统一训练参数
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

# 对比方法配置
EXPERIMENTS = {
    # ===== 不同YOLO版本对比 =====
    "YOLOv5n": {
        "model": "yolov5n.pt",  # 会自动下载
        "args": {},
    },
    "YOLOv8n": {
        "model": "yolov8n.pt",
        "args": {},
    },
    "YOLOv9t": {
        "model": "yolov9t.pt",
        "args": {},
    },
    "YOLOv10n": {
        "model": "yolov10n.pt",
        "args": {},
    },
    "YOLOv11n": {
        "model": "yolo11n.pt",
        "args": {},
    },
    "RT-DETR-l": {
        "model": "rtdetr-l.pt",
        "args": {"epochs": 100},  # RT-DETR收敛快，100轮足够
    },

    # ===== 注意力机制对比 (需要自定义YAML) =====
    # 注: SE/CBAM/ECA需要创建对应的YAML配置，下面提供模板
    "YOLOv11n_SE": {
        "model": os.path.join(CURRENT_DIR, "configs", "yolov11n_SE.yaml"),
        "args": {},
        "need_config": True,
    },
    "YOLOv11n_CBAM": {
        "model": os.path.join(CURRENT_DIR, "configs", "yolov11n_CBAM.yaml"),
        "args": {},
        "need_config": True,
    },
    "YOLOv11n_ECA": {
        "model": os.path.join(CURRENT_DIR, "configs", "yolov11n_ECA.yaml"),
        "args": {},
        "need_config": True,
    },

    # ===== 本文方法 =====
    "CMIFE_YOLO_v2": {
        "model": os.path.join(CURRENT_DIR, "yolov11n_CMIFE_v2.yaml"),
        "args": {},
    },
}


def create_attention_configs():
    """创建SE/CBAM/ECA的YAML配置文件"""
    config_dir = os.path.join(CURRENT_DIR, "configs")
    os.makedirs(config_dir, exist_ok=True)

    # ============ SE (Squeeze-and-Excitation) ============
    se_yaml = """# YOLOv11n + SE Attention
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
  # P3 + SE
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  # P4 + SE
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  # Detect
  - [[16, 19, 22], 1, Detect, [nc]]
"""

    # ============ CBAM ============
    cbam_yaml = """# YOLOv11n + CBAM Attention
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

    # ============ ECA ============
    eca_yaml = """# YOLOv11n + ECA Attention
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

    configs = {
        "yolov11n_SE.yaml": se_yaml,
        "yolov11n_CBAM.yaml": cbam_yaml,
        "yolov11n_ECA.yaml": eca_yaml,
    }

    for name, content in configs.items():
        path = os.path.join(config_dir, name)
        with open(path, 'w') as f:
            f.write(content)
        print(f"  已创建: {path}")


def run_single_experiment(name, config, results_dir):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  开始训练: {name}")
    print(f"{'='*60}")

    model_path = config["model"]
    extra_args = config.get("args", {})

    try:
        model = YOLO(model_path)

        # 合并训练参数
        train_args = {**TRAIN_ARGS, **extra_args}
        train_args["project"] = str(results_dir)
        train_args["name"] = name

        # 训练
        start_time = time.time()
        results = model.train(**train_args)
        train_time = time.time() - start_time

        # 验证
        metrics = model.val(data=DATA_YAML, imgsz=640, device=0)

        # 收集结果
        result = {
            "model": name,
            "mAP50": round(float(metrics.box.map50) * 100, 1),
            "mAP50_95": round(float(metrics.box.map) * 100, 1),
            "precision": round(float(metrics.box.mp) * 100, 1),
            "recall": round(float(metrics.box.mr) * 100, 1),
            "train_time_hours": round(train_time / 3600, 2),
        }

        # 模型信息
        info = model.info(verbose=False)
        if info:
            result["params_M"] = round(info[1] / 1e6, 1) if len(info) > 1 else "N/A"
            result["flops_G"] = round(info[2] / 1e9, 1) if len(info) > 2 else "N/A"

        print(f"\n  {name} 完成: mAP@0.5={result['mAP50']}%")
        return result

    except Exception as e:
        print(f"\n  {name} 训练失败: {e}")
        return {"model": name, "error": str(e)}


def main():
    results_dir = Path(CURRENT_DIR) / "runs" / "comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 创建注意力机制配置
    print("创建注意力机制配置文件...")
    create_attention_configs()

    # 选择要运行的实验
    print("\n可用实验:")
    for i, name in enumerate(EXPERIMENTS.keys()):
        print(f"  [{i}] {name}")

    selection = input("\n请输入要运行的实验编号 (逗号分隔, 或 'all' 运行全部): ").strip()

    if selection.lower() == 'all':
        selected = list(EXPERIMENTS.keys())
    else:
        indices = [int(x.strip()) for x in selection.split(',')]
        keys = list(EXPERIMENTS.keys())
        selected = [keys[i] for i in indices if i < len(keys)]

    print(f"\n将运行以下实验: {selected}")

    # 依次运行
    all_results = []
    for name in selected:
        config = EXPERIMENTS[name]
        result = run_single_experiment(name, config, results_dir)
        all_results.append(result)

        # 保存中间结果
        with open(results_dir / "comparison_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总表格
    print("\n" + "=" * 100)
    print("  对比实验结果汇总")
    print("=" * 100)
    header = f"{'模型':<25} {'mAP@0.5':>10} {'mAP@0.5:0.95':>15} {'Precision':>12} {'Recall':>10} {'Params(M)':>12} {'FLOPs(G)':>12}"
    print(header)
    print("-" * 100)

    for r in all_results:
        if "error" in r:
            print(f"  {r['model']:<25} {'ERROR':>10} {r['error'][:50]}")
        else:
            print(f"  {r['model']:<25} {r.get('mAP50','N/A'):>10} {r.get('mAP50_95','N/A'):>15} "
                  f"{r.get('precision','N/A'):>12} {r.get('recall','N/A'):>10} "
                  f"{r.get('params_M','N/A'):>12} {r.get('flops_G','N/A'):>12}")

    print(f"\n结果已保存到: {results_dir / 'comparison_results.json'}")


if __name__ == '__main__':
    main()
