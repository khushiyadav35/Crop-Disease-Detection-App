"""
ONNX & TorchScript Model Export Pipeline
Crop Disease Detection App for Farmers
Author: Khushi Yadav
"""

import os
import json
import argparse
from pathlib import Path

import torch
from src.model import CropDiseaseMobileNetV2


def export_models(
    weights_path: str = "checkpoints/best_crop_mobilenetv2.pth",
    output_onnx: str = "models/mobilenetv2_plantvillage.onnx",
    output_torchscript: str = "models/mobilenetv2_plantvillage.pt",
    num_classes: int = 38,
):
    print("[*] Initializing MobileNetV2 architecture for export...")
    model = CropDiseaseMobileNetV2(num_classes=num_classes, pretrained=False)

    if Path(weights_path).exists():
        print(f"[*] Loading fine-tuned weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[!] Checkpoint not found; exporting architecture blueprint.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # 1. Export TorchScript
    Path(output_torchscript).parent.mkdir(parents=True, exist_ok=True)
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(output_torchscript)
        size_mb = os.path.getsize(output_torchscript) / (1024 * 1024)
        print(f"[+] TorchScript model exported: {output_torchscript} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] TorchScript export failed: {e}")

    # 2. Export ONNX for Browser / Edge Inference
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
        print(f"[+] ONNX model exported: {output_onnx} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] Note on ONNX export: {e}")


if __name__ == "__main__":
    export_models()
