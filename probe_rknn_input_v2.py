#!/usr/bin/env python3
"""Probe UINT8/INT8 input paths with identical letterbox preprocessing.

Usage:
  python probe_rknn_input_v2.py model.rknn image.jpg --scale 0.0039215686 --zp 0
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from rknn.api import RKNN

SIZE = 640

def letterbox(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nh, nw = round(h * r), round(w * r)
    small = cv2.resize(bgr, (nw, nh))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    dy, dx = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = small
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def infer(model, inp, name, out_dir):
    r = RKNN(verbose=False)
    assert r.load_rknn(str(model)) == 0
    assert r.init_runtime(target="rk3588") == 0
    try:
        outs = r.inference(inputs=[inp], data_format="nhwc")
        print(name, "input", inp.dtype, inp.shape, "outputs",
              [tuple(np.asarray(x).shape) for x in outs])
        for i, out in enumerate(outs):
            np.save(out_dir / (name + "_output_%d.npy" % i), np.asarray(out))
    finally:
        r.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--zp", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or a.model.parent / "input_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = letterbox(a.image)
    f32 = rgb.astype(np.float32)[None]
    u8 = rgb[None]
    i8 = np.clip(np.rint(f32 / a.scale + a.zp), -128, 127).astype(np.int8)
    np.save(out_dir / "input_rgb_float32.npy", f32)
    np.save(out_dir / "input_rgb_uint8.npy", u8)
    np.save(out_dir / "input_rgb_int8_manual.npy", i8)
    print("declared input scale=%g zp=%d" % (a.scale, a.zp))
    for name, inp in (("uint8_pt0", u8), ("int8_manual", i8), ("float32_pt1", f32)):
        try:
            infer(a.model, inp, name, out_dir)
        except Exception as exc:
            print(name, "FAILED:", exc)

if __name__ == "__main__":
    main()
