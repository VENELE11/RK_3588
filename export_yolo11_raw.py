#!/usr/bin/env python3
"""Export YOLO11 Detect feature maps before DFL/decode/sigmoid/concat.

The exported outputs are three NCHW tensors, one per stride.  Their channel
layout is the native Ultralytics Detect head layout: 4*reg_max regression
logits followed by nc classification logits.  All decode operations remain in
the consumer, which makes the graph suitable for RKNN INT8 experiments.

Usage: python export_yolo11_raw.py yolov11s.pt yolov11s_raw.onnx
"""
from pathlib import Path
import sys

import torch


class RawDetect(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Ultralytics' Detect module returns the three feature maps while in
        # training mode, before DFL, decode and sigmoid.  Keep only this
        # module's native model graph so export has no post-processing ops.
        backbone = self.model.model[:-1]
        detect = self.model.model[-1]
        y = x
        outputs = []
        for layer in backbone:
            if layer.f != -1:
                y = outputs[layer.f] if isinstance(layer.f, int) else [y if j == -1 else outputs[j] for j in layer.f]
            y = layer(y)
            outputs.append(y)
        features = y if isinstance(y, (list, tuple)) else outputs[-1]
        detect.train(True)
        raw = detect(features)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise RuntimeError("Detect did not return three raw feature maps")
        return tuple(raw)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    from ultralytics import YOLO
    weights, output = Path(sys.argv[1]), Path(sys.argv[2])
    yolo = YOLO(str(weights))
    yolo.model.eval()
    wrapper = RawDetect(yolo.model).eval()
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        raw = wrapper(dummy)
    print("raw outputs:", [tuple(t.shape) for t in raw])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, dummy, str(output), opset_version=13,
                      input_names=["images"],
                      output_names=["p3_raw", "p4_raw", "p5_raw"],
                      dynamic_axes=None, do_constant_folding=True)
    print("wrote", output)


if __name__ == "__main__":
    main()
