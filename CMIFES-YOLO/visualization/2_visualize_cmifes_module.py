"""
脚本2: 绘制 CMIFES 模块内部结构图
直接从 CMIFES.py 导入模块，使用 torchview 可视化单输入与多输入两种模式

运行方式:
    conda activate yolov11-visdrone
    python "2_visualize_cmifes_module.py"

输出:
    output/CMIFES_single_input.png/.svg  — 单路输入（精炼模式）
    output/CMIFES_multi_input.png/.svg   — 双路输入（跨尺度融合模式）
"""

import os
import sys

# ── 路径设置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, r"..\2-ultralytics-main - MDFA"))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 添加 Graphviz 到 PATH (winget 默认安装位置)
graphviz_bin = r"C:\Program Files\Graphviz\bin"
if os.path.isdir(graphviz_bin) and graphviz_bin not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_bin

sys.path.insert(0, PROJECT_DIR)

# ── 导入 ──────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchview import draw_graph
from ultralytics.nn.modules.CMIFES import CMIFES


# ── 辅助：多输入包装器（torchview 需要 forward 接受独立参数）──────────────────
class CMIFESMultiWrapper(nn.Module):
    """将 CMIFES 的 list 输入改成两个独立张量，方便 torchview 追踪。"""
    def __init__(self, cmifes_module):
        super().__init__()
        self.cmifes = cmifes_module

    def forward(self, x1, x2):
        return self.cmifes([x1, x2])


def render(graph, out_path):
    """渲染为 PNG 和 SVG。"""
    graph.visual_graph.render(filename=out_path, format="png", cleanup=True)
    print(f"✅ 已保存: {out_path}.png")
    graph.visual_graph.render(filename=out_path + "_svg", format="svg", cleanup=True)
    print(f"✅ 已保存: {out_path}_svg.svg")


# ── 1. 单路输入模式（CMIFES_P3 #2 级联精炼）────────────────────────────────
print("\n[1/2] 生成 CMIFES 单路输入结构图 (256ch, 80×80)...")

cmifes_single = CMIFES(in_channels=256, out_channels=256).eval()
dummy_single  = torch.zeros(1, 256, 80, 80)

g_single = draw_graph(
    cmifes_single,
    input_data=dummy_single,
    depth=5,              # 展开至子模块内部
    expand_nested=True,
    show_shapes=True,
    graph_name="CMIFES_single_input",
    roll=True,
    device="cpu",
)
render(g_single, os.path.join(OUTPUT_DIR, "CMIFES_single_input"))


# ── 2. 双路跨尺度融合模式（CMIFES_P3 #1，backbone P3 + neck P3 融合）─────────
print("\n[2/2] 生成 CMIFES 双路输入结构图 ([256, 256] → 256ch)...")

cmifes_multi   = CMIFES(in_channels=[256, 256], out_channels=256).eval()
wrapper        = CMIFESMultiWrapper(cmifes_multi).eval()
dummy_x1       = torch.zeros(1, 256, 80, 80)   # backbone P3
dummy_x2       = torch.zeros(1, 256, 80, 80)   # neck P3

g_multi = draw_graph(
    wrapper,
    input_data=(dummy_x1, dummy_x2),
    depth=5,
    expand_nested=True,
    show_shapes=True,
    graph_name="CMIFES_multi_input",
    roll=True,
    device="cpu",
)
render(g_multi, os.path.join(OUTPUT_DIR, "CMIFES_multi_input"))

print("\n全部完成！输出目录:", OUTPUT_DIR)
