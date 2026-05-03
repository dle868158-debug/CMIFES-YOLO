"""
脚本1: 绘制 CMIFE-YOLO v7 整体网络结构图
基于 yolov11n_CMIFE_v7.yaml 加载模型，使用 torchview 生成计算图

运行方式:
    conda activate yolov11-visdrone
    python "1_visualize_full_model.py"

输出: output/CMIFE_YOLO_v7_architecture.png / .gv
"""

import os
import sys

# ── 路径设置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional Graphviz path, for example set GRAPHVIZ_BIN to your local Graphviz bin directory.
graphviz_bin = os.environ.get("GRAPHVIZ_BIN")
if graphviz_bin and os.path.isdir(graphviz_bin) and graphviz_bin not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_bin

sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# ── 导入 ──────────────────────────────────────────────────────────────────────
import torch
from torchview import draw_graph

print("正在加载 CMIFE-YOLO v7 模型...")
from ultralytics import YOLO

yaml_path = os.path.join(PROJECT_DIR, "ultralytics_cfg", "models", "11", "yolov11n_CMIFE_v7.yaml")
model = YOLO(yaml_path)
torch_model = model.model.eval()

total_params = sum(p.numel() for p in torch_model.parameters())
print(f"模型参数量: {total_params/1e6:.2f}M")

# ── torchview 绘图 ────────────────────────────────────────────────────────────
print("正在生成网络结构图 (depth=3)...")

dummy_input = torch.zeros(1, 3, 640, 640)

model_graph = draw_graph(
    torch_model,
    input_data=dummy_input,
    depth=3,                    # 展开深度: 3 层足够展示主要模块
    expand_nested=False,        # 不展开嵌套子模块，保持整体清晰
    show_shapes=True,           # 显示张量形状
    graph_name="CMIFE_YOLO_v7",
    roll=True,                  # 合并重复节点
    device="cpu",
)

# 渲染并保存
out_path = os.path.join(OUTPUT_DIR, "CMIFE_YOLO_v7_architecture")
model_graph.visual_graph.render(
    filename=out_path,
    format="png",
    cleanup=True,               # 删除中间 .gv 文件
)
print(f"✅ 已保存: {out_path}.png")

# 同时保存 SVG (矢量图，适合论文)
model_graph.visual_graph.render(
    filename=out_path + "_svg",
    format="svg",
    cleanup=True,
)
print(f"✅ 已保存: {out_path}_svg.svg")
