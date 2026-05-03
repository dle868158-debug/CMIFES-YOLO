"""
CMIFE-YOLO v6 训练脚本
与官方YOLOv11n backbone完全一致 + CMIFE双级联增强P3特征

与v5的关键区别：
  - backbone L7使用Conv→1024 (而非v5的512), 使P5=1024ch与官方YOLOv11n一致
  - L24使用C3k2(c2=1024) 正确处理1280ch输入 (而非v5的c2=512导致bottleneck>1)

预期结果:
  - 参数量: ~3.2M (vs 基线2.59M, +24%)
  - mAP50目标: >35% (超越基线)
"""
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ultralytics import YOLO

def main():
    # 确认CMIFE.py存在
    cmife_path = os.path.join(project_root, 'ultralytics', 'nn', 'modules', 'CMIFE.py')
    if not os.path.exists(cmife_path):
        print(f"ERROR: CMIFE.py not found at {cmife_path}")
        return

    # 加载YAML配置
    model_path = os.path.join(project_root, 'yolov11n_CMIFE_v6.yaml')
    model = YOLO(model_path)

    # 训练
    results = model.train(
        data='VisDrone.yaml',
        epochs=300,
        imgsz=640,
        batch=8,
        device='0',
        workers=0,
        project='runs/comparison',
        name='CMIFE_YOLO_v6',
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=10,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        cos_lr=True,
        amp=True,
        plots=True,
        val=True,
        seed=0,
        deterministic=True,
    )

    # 打印最佳结果
    print("\n" + "="*60)
    print("CMIFE-YOLO v6 训练完成!")
    print("="*60)

if __name__ == '__main__':
    main()
