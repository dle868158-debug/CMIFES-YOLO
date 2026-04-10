"""
CMIFE-YOLO v7 训练脚本
融合方案A(多尺度扩展P3/P4/P5) + 方案B(SE轻量化注意力)

使用方法:
    python train_cmife_v7.py

训练结果自动保存到: runs/comparison/CMIFE_YOLO_v7/
"""
from ultralytics import YOLO
import os

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # 加载模型
    model = YOLO("yolov11n_CMIFE_v7.yaml")

    # 开始训练
    results = model.train(
        data="VisDrone.yaml",
        epochs=300,
        patience=100,
        batch=8,
        imgsz=640,
        device="0",
        workers=0,
        project="runs/comparison",
        name="CMIFE_YOLO_v7",
        pretrained=False,
        optimizer="SGD",
        verbose=True,
        seed=0,
        deterministic=True,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("\n训练完成!")
    print(f"结果保存在: {results.save_dir}")
    print(f"最佳mAP50: {results.best_map50:.4f}")
