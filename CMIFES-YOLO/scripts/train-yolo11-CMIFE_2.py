from ultralytics import YOLO
import os


# 核心修复：把训练逻辑放到主函数里，并用 if __name__ == '__main__' 包裹
def main():
    # 获取当前脚本所在目录，拼接yaml绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, r'D:\TestAnalytics\0-ultralytics-main\VisDrone.yaml')

    # 加载预训练模型（已下载完成）
    model = YOLO(r'D:\TestAnalytics\2-ultralytics-main - MDFA\yolov11n_CMIFE_2.yaml')

    # 启动训练（仅保留有效参数，解决Windows多进程问题）
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        patience=50,
        save_period=10,
        project="VisDrone_training",
        name="yolo11n_visdrone",
        workers=0  # 关键：Windows下设置workers=0，禁用多进程加载数据（唯一必需的多进程修复）
    )


# Windows多进程必需的保护语句（核心修复）
if __name__ == '__main__':
    main()