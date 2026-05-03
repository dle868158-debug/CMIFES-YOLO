"""
Grad-CAM 可视化脚本 — 论文质量热力图生成
用法: python grad_cam_vis.py --model best.pt --image path/to/image.jpg

功能:
  1. 标准Grad-CAM实现 (不人为调整权重)
  2. 多层对比可视化: backbone → CMIFE1 → CMIFE2 → 全局融合
  3. 输出论文质量的2x2对比图
  4. 支持批量处理
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# 中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class GradCAM:
    """标准Grad-CAM实现"""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.gradients = {}
        self.hooks = []

    def register_layer(self, name, layer):
        """注册需要可视化的层"""
        h1 = layer.register_forward_hook(self._save_features(name))
        h2 = layer.register_full_backward_hook(self._save_gradients(name))
        self.hooks.append(h1)
        self.hooks.append(h2)

    def _save_features(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def _save_gradients(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook

    def generate(self, input_tensor, target_layer_name):
        """生成Grad-CAM热力图"""
        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)

        # 选择最高置信度作为目标
        if isinstance(output, (list, tuple)):
            score = output[0][..., 4:].max()
        else:
            score = output.max()

        # 反向传播
        score.backward(retain_graph=True)

        # 计算Grad-CAM
        if target_layer_name not in self.features or target_layer_name not in self.gradients:
            print(f"  警告: 层 '{target_layer_name}' 无特征/梯度")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        features = self.features[target_layer_name]
        gradients = self.gradients[target_layer_name]

        # GAP梯度 → 权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 加权求和
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        return cam.squeeze().cpu().numpy()

    def clear(self):
        """清除缓存"""
        self.features.clear()
        self.gradients.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def get_layer_mapping(model):
    """
    自动检测CMIFE-YOLO的关键层.
    返回 {display_name: (layer_index, layer_object)} 的映射
    """
    layers = {}
    model_seq = model.model

    # 遍历模型层，查找CMIFE模块和关键backbone层
    for i, layer in enumerate(model_seq):
        class_name = layer.__class__.__name__

        if class_name == 'SPPF' or class_name == 'C2PSA':
            if 'backbone_end' not in layers:
                layers['backbone_end'] = (i, layer)

        if class_name == 'CMIFE':
            # 根据位置判断是哪个CMIFE
            if 'cmife_1' not in layers:
                layers['cmife_1'] = (i, layer)
            elif 'cmife_2' not in layers:
                layers['cmife_2'] = (i, layer)
            elif 'cmife_3' not in layers:
                layers['cmife_3'] = (i, layer)
            elif 'cmife_4' not in layers:
                layers['cmife_4'] = (i, layer)
            elif 'cmife_5' not in layers:
                layers['cmife_5'] = (i, layer)
            elif 'cmife_6' not in layers:
                layers['cmife_6'] = (i, layer)
            elif 'global_fuse' not in layers:
                layers['global_fuse'] = (i, layer)

    return layers


def overlay_heatmap(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """将热力图叠加到原图上"""
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlay


def visualize_single_image(model_path, image_path, output_dir, imgsz=640):
    """对单张图片生成Grad-CAM可视化"""
    # 加载模型
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()

    # 检测关键层
    layer_map = get_layer_mapping(model)
    print(f"检测到 {len(layer_map)} 个关键层:")
    for name, (idx, _) in layer_map.items():
        print(f"  {name}: layer[{idx}]")

    # 选择可视化的4个层 (论文用)
    vis_layers = {}
    if 'backbone_end' in layer_map:
        vis_layers['(a) Backbone P3特征'] = layer_map['backbone_end']
    if 'cmife_1' in layer_map:
        vis_layers['(b) 第一次CMIFE增强'] = layer_map['cmife_1']
    if 'cmife_2' in layer_map:
        vis_layers['(c) 第二次CMIFE增强'] = layer_map['cmife_2']
    if 'global_fuse' in layer_map:
        vis_layers['(d) 跨尺度全局融合'] = layer_map['global_fuse']

    # 如果检测不到足够层，用前4个
    if len(vis_layers) < 2:
        vis_layers = {}
        for i, (name, (idx, layer)) in enumerate(layer_map.items()):
            vis_layers[f"({chr(97+i)}) {name}"] = (idx, layer)
            if i >= 3:
                break

    # 创建GradCAM
    grad_cam = GradCAM(model)
    layer_names = []
    for display_name, (idx, layer) in vis_layers.items():
        layer_name = f"layer_{idx}"
        layer_names.append((display_name, layer_name))
        grad_cam.register_layer(layer_name, layer)

    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (imgsz, imgsz))

    # 准备输入
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).requires_grad_(True)

    if torch.cuda.is_available():
        tensor = tensor.cuda()
        model = model.cuda()

    # 生成热力图
    cam_results = {}
    for display_name, layer_name in layer_names:
        grad_cam.clear()
        cam = grad_cam.generate(tensor, layer_name)
        cam_results[display_name] = cam

    # 绘制2x2对比图
    n = len(cam_results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), dpi=300)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, cam) in enumerate(cam_results.items()):
        overlay = overlay_heatmap(img_resized, cam, alpha=0.5)
        axes[i].imshow(overlay)
        axes[i].set_title(title, fontsize=14,
                          fontweight='bold' if '全局融合' in title or 'global' in title else 'normal')
        axes[i].axis('off')

    # 隐藏多余子图
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=2.0)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    img_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"gradcam_{img_name}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 同时保存单独的热力图
    for i, (title, cam) in enumerate(cam_results.items()):
        overlay = overlay_heatmap(img_resized, cam, alpha=0.5)
        single_path = os.path.join(output_dir, f"gradcam_{img_name}_{chr(97+i)}.png")
        cv2.imwrite(single_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM可视化已保存到: {output_path}")

    grad_cam.remove_hooks()
    return output_path


def batch_visualize(model_path, image_dir, output_dir, num_images=10, imgsz=640):
    """批量生成Grad-CAM可视化"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [p for p in Path(image_dir).iterdir()
              if p.suffix.lower() in image_extensions]

    images = sorted(images)[:num_images]
    print(f"将处理 {len(images)} 张图片")

    for img_path in images:
        print(f"\n处理: {img_path.name}")
        try:
            visualize_single_image(model_path, str(img_path), output_dir, imgsz)
        except Exception as e:
            print(f"  处理失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='CMIFE-YOLO Grad-CAM可视化')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径 (.pt)')
    parser.add_argument('--image', type=str, default=None, help='单张图片路径')
    parser.add_argument('--image_dir', type=str, default=None, help='图片目录 (批量模式)')
    parser.add_argument('--output', type=str, default='runs/gradcam', help='输出目录')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--num', type=int, default=10, help='批量模式处理数量')
    args = parser.parse_args()

    if args.image:
        visualize_single_image(args.model, args.image, args.output, args.imgsz)
    elif args.image_dir:
        batch_visualize(args.model, args.image_dir, args.output, args.num, args.imgsz)
    else:
        print("请指定 --image (单张) 或 --image_dir (批量)")
        print("示例:")
        print(f"  python grad_cam_vis.py --model best.pt --image test.jpg")
        print(f"  python grad_cam_vis.py --model best.pt --image_dir C:/dataset/images --num 20")


if __name__ == '__main__':
    main()
