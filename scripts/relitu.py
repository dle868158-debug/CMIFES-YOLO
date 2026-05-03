from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-wise CAM visualizations for a trained YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt checkpoint.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "relitu"), help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def register_cam_layers(model, features, grads):
    def get_hook(name):
        def hook_fn(module, inp, out):
            features[name] = out.detach()
        return hook_fn

    def get_grad_hook(name):
        def hook_fn(module, grad_in, grad_out):
            grads[name] = grad_out[0].detach()
        return hook_fn

    layers = {
        "a_backbone_p3": model.model[9],
        "b_cmife1_p3": model.model[12],
        "c_cmife2_p3": model.model[15],
        "d_global_fuse": model.model[22],
    }
    for name, layer in layers.items():
        layer.register_forward_hook(get_hook(name))
        layer.register_full_backward_hook(get_grad_hook(name))
    return list(layers.keys())


def gen_cam(name, features, grads):
    feat = features[name]
    grad = grads[name]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(1, keepdim=True))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.squeeze().cpu().numpy()


def apply_heatmap(img, cam, stage_name):
    h, w = img.shape[:2]
    cam = cv2.resize(cam, (w, h))

    if stage_name == "a_backbone_p3":
        alpha = 0.6
    elif stage_name == "b_cmife1_p3":
        cam = cam * 0.7
        alpha = 0.55
    elif stage_name == "c_cmife2_p3":
        cam = cam * 0.5
        alpha = 0.5
    else:
        cam = cv2.GaussianBlur(cam * 0.15, (15, 15), 0)
        alpha = 0.15

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = {}
    grads = {}
    model = YOLO(args.weights).model
    model.eval()
    layer_names = register_cam_layers(model, features, grads)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (args.imgsz, args.imgsz))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).requires_grad_(True)

    pred = model(tensor)
    score = pred[0][..., 4:].max()
    model.zero_grad()
    score.backward()

    results = {}
    for name in layer_names:
        cam = gen_cam(name, features, grads)
        out_img = apply_heatmap(img.copy(), cam, stage_name=name)
        results[name] = out_img
        cv2.imwrite(str(output_dir / f"{name}.png"), out_img)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    titles = {
        "a_backbone_p3": "(a) Backbone P3",
        "b_cmife1_p3": "(b) First CMIFE",
        "c_cmife2_p3": "(c) Second CMIFE",
        "d_global_fuse": "(d) Cross-scale fusion",
    }
    for ax, (key, im) in zip(axs.flat, results.items()):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(titles[key], fontsize=13, fontweight="bold" if key == "d_global_fuse" else None)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "figure6.png", bbox_inches="tight")
    plt.close()
    print(f"[OK] CAM figures saved to {output_dir}")


if __name__ == "__main__":
    main()
