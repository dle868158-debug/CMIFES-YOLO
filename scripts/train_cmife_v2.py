"""
CMIFE-YOLO v2 训练脚本
用法: python train_cmife_v2.py
"""
from ultralytics import YOLO
import os


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 模型配置文件 (CMIFE_v2 增强版)
    model_yaml = os.path.join(current_dir, 'yolov11n_CMIFE_v2.yaml')

    # 数据集配置文件
    data_yaml = os.path.join(current_dir, 'VisDrone.yaml')

    # 加载模型 (从yaml构建，不加载预训练权重)
    model = YOLO(model_yaml)

    # 打印模型信息
    model.info(verbose=True)

    # 启动训练
    results = model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=8,
        amp=False,  # 关闭混合精度，解决自定义层类型报错
        device=0,
        patience=50,        # 早停: 50轮无提升则停止
        save_period=10,      # 每10轮保存一次
        optimizer='SGD',
        lr0=0.01,            # 初始学习率
        lrf=0.01,            # 最终学习率 = lr0 * lrf
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=10,    # Warmup 10轮
        cos_lr=True,         # 余弦退火学习率
        project="runs/CMIFE_v2",
        name="train",
        workers=0,           # Windows下设置0，避免多进程问题
        verbose=True,
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: runs/CMIFE_v2/train/weights/best.pt")


if __name__ == '__main__':
    main()
