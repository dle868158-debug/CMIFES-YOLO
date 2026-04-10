"""
评估与分析脚本: 模型性能综合评估
用法: python eval_analysis.py --model runs/CMIFE_v2/train/weights/best.pt

功能:
  1. 各类别AP统计 (论文表7)
  2. Confusion Matrix生成
  3. 模型参数量/FLOPs/FPS统计
  4. 检测结果可视化对比
  5. PR曲线绘制
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from ultralytics import YOLO

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(CURRENT_DIR, 'VisDrone.yaml')

CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

CLASS_NAMES_CN = [
    '行人', '人群', '自行车', '汽车', '货车',
    '卡车', '三轮车', '遮阳篷三轮车', '公交车', '摩托车'
]


def evaluate_model(model_path, output_dir):
    """全面评估单个模型"""
    print(f"\n评估模型: {model_path}")

    model = YOLO(model_path)

    # 1. 模型信息
    print("\n--- 模型信息 ---")
    info = model.info(verbose=True, detailed=True)

    # 2. 验证集评估
    print("\n--- 验证集评估 ---")
    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        device=0,
        plots=True,         # 自动生成PR曲线、confusion matrix等
        save_json=True,
        verbose=True,
    )

    # 3. 收集各类别AP
    print("\n--- 各类别AP ---")
    per_class_ap50 = metrics.box.ap50  # shape: (num_classes,)
    per_class_ap = metrics.box.ap      # shape: (num_classes,)

    class_results = []
    print(f"{'类别':<20} {'AP@0.5':>10} {'AP@0.5:0.95':>15}")
    print("-" * 50)
    for i, (name_cn, name_en) in enumerate(zip(CLASS_NAMES_CN, CLASS_NAMES)):
        if i < len(per_class_ap50):
            ap50 = round(float(per_class_ap50[i]) * 100, 1)
            ap = round(float(per_class_ap[i]) * 100, 1)
        else:
            ap50, ap = 0.0, 0.0
        print(f"  {name_cn}({name_en}){'':<5} {ap50:>10} {ap:>15}")
        class_results.append({
            "class_cn": name_cn,
            "class_en": name_en,
            "AP50": ap50,
            "AP50_95": ap,
        })

    # 4. FPS测试
    print("\n--- 推理速度测试 ---")
    fps = measure_fps(model, imgsz=640, num_warmup=50, num_test=200)

    # 5. 汇总结果
    summary = {
        "model_path": str(model_path),
        "mAP50": round(float(metrics.box.map50) * 100, 1),
        "mAP50_95": round(float(metrics.box.map) * 100, 1),
        "precision": round(float(metrics.box.mp) * 100, 1),
        "recall": round(float(metrics.box.mr) * 100, 1),
        "fps": fps,
        "per_class": class_results,
    }

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(model_path).stem
    result_path = os.path.join(output_dir, f"eval_{model_name}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存到: {result_path}")

    return summary


def measure_fps(model, imgsz=640, num_warmup=50, num_test=200):
    """测量推理FPS"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    m = model.model.to(device)
    m.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            m(dummy)

    # 测量
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_test):
            m(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    fps = round(num_test / (end - start), 1)
    print(f"  FPS: {fps} ({device}, imgsz={imgsz})")
    return fps


def compare_models(model_paths, output_dir):
    """对比多个模型的各类别AP"""
    all_results = {}

    for path in model_paths:
        name = Path(path).stem
        print(f"\n评估: {name}")
        model = YOLO(path)
        metrics = model.val(data=DATA_YAML, imgsz=640, device=0, verbose=False)
        all_results[name] = {
            "mAP50": round(float(metrics.box.map50) * 100, 1),
            "per_class_ap50": [round(float(x) * 100, 1) for x in metrics.box.ap50],
        }

    # 绘制各类别AP对比柱状图
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(CLASS_NAMES_CN))
    width = 0.8 / len(all_results)

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for i, (name, data) in enumerate(all_results.items()):
        ap_values = data["per_class_ap50"]
        # 确保长度一致
        while len(ap_values) < len(CLASS_NAMES_CN):
            ap_values.append(0)
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, ap_values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel('目标类别', fontsize=12)
    ax.set_ylabel('AP@0.5 (%)', fontsize=12)
    ax.set_title('各类别检测精度对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_CN, fontsize=10, rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "class_ap_comparison.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n各类别AP对比图已保存到: {output_path}")


def generate_detection_comparison(model_paths, image_path, output_dir):
    """生成检测结果对比图 (论文图7)"""
    import cv2

    models = {}
    for path in model_paths:
        name = Path(path).stem
        models[name] = YOLO(path)

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=300)
    if n == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        results = model(image_path, imgsz=640, conf=0.25)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated_rgb)
        axes[i].set_title(name, fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    img_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"detection_compare_{img_name}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"检测对比图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CMIFE-YOLO 评估与分析')
    parser.add_argument('--model', type=str, nargs='+', required=True,
                        help='模型权重路径 (支持多个，用空格分隔)')
    parser.add_argument('--output', type=str, default='runs/eval',
                        help='输出目录')
    parser.add_argument('--compare_image', type=str, default=None,
                        help='用于检测对比的图片路径')
    parser.add_argument('--mode', choices=['eval', 'compare', 'detect', 'all'],
                        default='all', help='运行模式')
    args = parser.parse_args()

    if args.mode in ('eval', 'all'):
        for model_path in args.model:
            evaluate_model(model_path, args.output)

    if args.mode in ('compare', 'all') and len(args.model) > 1:
        compare_models(args.model, args.output)

    if args.mode in ('detect', 'all') and args.compare_image:
        generate_detection_comparison(args.model, args.compare_image, args.output)


if __name__ == '__main__':
    main()
