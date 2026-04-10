# CMIFE-YOLO v3 训练脚本
# width=0.25 恢复基线宽度，避免轻量化过度压缩导致性能下降

# 使用方法:
# 1. 确认 VisDrone.yaml 中的 path 指向正确的数据集路径
# 2. 运行: python train_cmife_v3.py

import sys
import os

# 确保使用本地修复后的 CMIFE 模块
sys.path.insert(0, r'C:\Users\Administrator\Desktop\CC\2-ultralytics-main - MDFA')

from ultralytics import YOLO

def main():
    # 使用 width=0.25 的新配置
    model_yaml = r'C:\Users\Administrator\Desktop\CC\2-ultralytics-main - MDFA\yolov11n_CMIFE_v3.yaml'
    data_yaml = r'C:\Users\Administrator\Desktop\CC\2-ultralytics-main - MDFA\VisDrone.yaml'

    print("=" * 60)
    print("CMIFE-YOLO v3 训练脚本")
    print("配置: width=0.25 (恢复基线宽度)")
    print("=" * 60)

    # 加载模型
    model = YOLO(model_yaml)

    # 训练
    # epochs=300 从头训练, resume=True 从上次中断处继续
    results = model.train(
        data=data_yaml,
        epochs=300,
        batch=8,
        imgsz=640,
        device=0,
        project=r'C:\Users\Administrator\Desktop\CC\2-ultralytics-main - MDFA\runs\detect',
        name='CMIFE_v3_w025',
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10,
        close_mosaic=10,
        cos_lr=True,
        amp=True,       # 启用混合精度训练 (验证修复)
        seed=0,
        deterministic=True,
        plots=True,
        save=True,
        save_period=10,
        val=True,
    )

    print("\n训练完成!")
    print(f"结果保存在: {results.save_dir}")

if __name__ == '__main__':
    main()
