#!/usr/bin/env python3
"""Compare RKNN input dtype modes on one image.

The script deliberately keeps preprocessing identical and records the model's
declared input quantization parameters.  It is intended for running on the
Toolkit2 host, where rknn.api is available.

Usage:
  python probe_rknn_input.py model.rknn image.jpg [out_dir]
"""
from pathlib import Path
import sys

import cv2
import numpy as np
from rknn.api import RKNN

SIZE = 640


def letterbox(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    scale = min(SIZE / h, SIZE / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    dy, dx = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def describe(attr):
    return {
        "name": getattr(attr, "name", ""),
        "dims": list(attr.dims[:attr.n_dims]),
        "type": str(attr.type),
        "fmt": str(attr.fmt),
        "scale": float(attr.scale),
        "zp": int(attr.zp),
        "size": int(attr.size),
    }


def run(model, image, mode):
    r = RKNN(verbose=False)
    assert r.load_rknn(str(model)) == 0
    assert r.init_runtime(target="rk3588") == 0
    attr = r.query(rknn_query_cmd="input_attr", index=0) if False else None
    # Toolkit2 exposes input metadata through the low-level query API in logs;
    # inference itself is intentionally exercised in all three input forms.
    if mode == "uint8_pt0":
        inp = image[None].astype(np.uint8)
        kwargs = {"data_format": "nhwc"}
    elif mode == "int8_manual":
        # These values are filled from RKNN's input_attr by the caller when
        # available.  The default is the common asymmetric uint8 calibration.
        raise RuntimeError("int8_manual requires input scale/zp; use --scale/--zp")
    elif mode == "int8_pt1":
        inp = image[None].astype(np.float32)
        kwargs = {"data_format": "nhwc"}
    else:
        raise ValueError(mode)
    outs = r.inference(inputs=[inp], **kwargs)
    r.release()
    return outs


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    model, image = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else model.parent / "input_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = letterbox(image)
    np.save(out_dir / "input_rgb_float32.npy", rgb[None].astype(np.float32))
    np.save(out_dir / "input_rgb_uint8.npy", rgb[None])
    print("model:", model)
    print("input: shape=%s dtype=%s range=[%g,%g]" %
          (rgb[None].shape, rgb.dtype, rgb.min(), rgb.max()))
    for mode in ("uint8_pt0", "int8_pt1"):
        try:
            outs = run(model, rgb, mode)
            print(mode, "outputs:", [tuple(np.asarray(x).shape) for x in outs])
            for i, out in enumerate(outs):
                np.save(out_dir / (mode + "_output_%d.npy" % i), np.asarray(out))
        except Exception as exc:
            print(mode, "FAILED:", exc)


if __name__ == "__main__":
    main()
