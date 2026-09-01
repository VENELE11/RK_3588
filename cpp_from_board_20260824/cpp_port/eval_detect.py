#!/usr/bin/env python3
"""Run detection inference with v5s/v8s/v11s rknn models on an image list,
output COCO-style predictions JSON.
Usage: python eval_detect.py <model_type> <model_path> <images_dir> <image_list.txt> <out.json>
image_list.txt: one image filename per line (looked up in images_dir)
"""
import json
import os
import sys

import cv2
import numpy as np
from rknn.api import RKNN

IMG_SIZE = 640
OBJ_THRESH = 0.01   # mAP 评估惯例: 低阈值收集候选, 由 P-R 曲线覆盖全部阈值
NMS_THRESH = 0.45
MAX_DETS = 500      # NMS 前按分数截断, 防候选爆炸 (COCO 惯例 max_det)

ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
           [59, 119], [116, 90], [156, 198], [373, 326]]
ANCHOR_MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def letterbox(img, size=IMG_SIZE):
    """等比缩放 + 灰边(114)填充, 返回 (画布, 缩放比, 左上偏移)."""
    h, w = img.shape[:2]
    r = min(float(size) / h, float(size) / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def nms_boxes(boxes, scores, thresh=NMS_THRESH):
    """boxes: [N,4] xyxy, scores: [N]. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def apply_nms(dets):
    """dets: ndarray [N,6] or list of [x1,y1,x2,y2,score,cls]. Returns same format."""
    if isinstance(dets, list):
        if not dets:
            return []
        dets = np.array(dets, dtype=np.float32)
    elif dets is None or len(dets) == 0:
        return []
    d = dets.astype(np.float32)
    if len(d) > MAX_DETS:
        idx = d[:, 4].argsort()[::-1][:MAX_DETS]
        d = d[idx]
    boxes = d[:, :4]
    scores = d[:, 4]
    classes = d[:, 5].astype(np.int32)
    out = []
    for c in set(classes.tolist()):
        idx = np.where(classes == c)[0]
        keep = nms_boxes(boxes[idx], scores[idx])
        for k in keep:
            i = idx[k]
            out.append([float(x) for x in d[i]])
    return out


# ---------------- v5s ----------------
def post_v5s(outs):
    pieces = []
    for o, mask in zip(outs, ANCHOR_MASKS):
        o = np.asarray(o).reshape(3, 85, o.shape[-2], o.shape[-1])
        h, w = o.shape[-2:]
        stride = IMG_SIZE // h
        col, row = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((col, row), 0)  # [2,h,w]
        anchors = np.array([ANCHORS[i] for i in mask], dtype=np.float32)

        box = sigmoid(o[:, :4])                     # [3,4,h,w]
        obj = sigmoid(o[:, 4])[:, None]             # [3,1,h,w]
        cls = sigmoid(o[:, 5:])                     # [3,80,h,w]

        xy = box[:, :2] * 2 - 0.5 + grid            # [3,2,h,w]
        wh = (box[:, 2:] * 2) ** 2 * anchors.reshape(3, 2, 1, 1)
        xy = xy * stride
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2

        score = obj * cls.max(1, keepdims=True)     # [3,1,h,w]
        cid = cls.argmax(1)                         # [3,h,w]

        for a in range(3):
            s = score[a, 0]
            m = s > OBJ_THRESH
            if m.sum() == 0:
                continue
            det = np.stack((x1y1[a, 0][m], x1y1[a, 1][m], x2y2[a, 0][m], x2y2[a, 1][m],
                            s[m], cid[a][m].astype(np.float32)), 1)
            pieces.append(det)
    if not pieces:
        return []
    return apply_nms(np.concatenate(pieces, 0))


# ---------------- v8s ----------------
def dfl_softmax(position):
    """position: [4,16,h,w] DFL logits -> [4,h,w] decoded offsets."""
    e = np.exp(np.clip(position, -30.0, 30.0))
    s = e.sum(1, keepdims=True)
    w = e / s
    acc = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    return (w * acc).sum(1)


def post_v8s(outs):
    pieces = []
    for i in range(3):
        box_branch = np.asarray(outs[2 * i])        # [1,64,h,w]
        cls_branch = np.asarray(outs[2 * i + 1])    # [1,80,h,w]
        h, w = box_branch.shape[-2:]
        stride = IMG_SIZE // h
        box = box_branch[0].reshape(4, 16, h, w)
        dist = dfl_softmax(box)                     # [4,h,w]

        col, row = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((col, row), 0)              # [2,h,w]
        xy1 = (grid + 0.5 - dist[0:2]) * stride     # [2,h,w]
        xy2 = (grid + 0.5 + dist[2:4]) * stride

        cls = sigmoid(cls_branch[0])                # [80,h,w]
        score = cls.max(0)                          # [h,w]
        cid = cls.argmax(0)

        m = score > OBJ_THRESH
        if m.sum() == 0:
            continue
        det = np.stack((xy1[0][m], xy1[1][m], xy2[0][m], xy2[1][m],
                        score[m], cid[m].astype(np.float32)), 1)
        pieces.append(det)
    if not pieces:
        return []
    return apply_nms(np.concatenate(pieces, 0))


# ---------------- v11s ----------------
def _fused_head_dets(p):
    """v8s/v11s 融合头通用处理: p=[N,84], box 前4列(640空间 cxcywh 或归一化), cls 后80列.
    返回 apply_nms 后的 dets. box_scale: box 列的缩放因子(归一化模型=640, 绝对坐标=1)."""
    box = p[:, :4] * BOX_SCALE
    cls = p[:, 4:]
    score = cls.max(1)
    cid = cls.argmax(1)
    m = score > OBJ_THRESH
    if m.sum() == 0:
        return []
    x1 = box[m, 0] - box[m, 2] / 2
    y1 = box[m, 1] - box[m, 3] / 2
    x2 = box[m, 0] + box[m, 2] / 2
    y2 = box[m, 1] + box[m, 3] / 2
    dets = np.stack((x1, y1, x2, y2, score[m], cid[m].astype(np.float32)), 1)
    return apply_nms(dets)


def post_v11s(outs):
    p = np.asarray(outs[0])[0]                      # [8400,84]
    box = p[:, :4]                                  # absolute cx,cy,w,h in 640 space
    cls = sigmoid(p[:, 4:])
    score = cls.max(1)
    cid = cls.argmax(1)
    m = score > OBJ_THRESH
    if m.sum() == 0:
        return []
    x1 = box[m, 0] - box[m, 2] / 2
    y1 = box[m, 1] - box[m, 3] / 2
    x2 = box[m, 0] + box[m, 2] / 2
    y2 = box[m, 1] + box[m, 3] / 2
    dets = [[float(x1[k]), float(y1[k]), float(x2[k]), float(y2[k]),
             float(score[m][k]), int(cid[m][k])] for k in range(x1.shape[0])]
    return apply_nms(dets)


# ---------------- 归一化融合头 (P0/P1 新转换模型) ----------------
BOX_SCALE = 1.0

def post_v11s_norm(outs):
    """v11s_norm: [1,8400,84], box 归一化0-1, cls 已 sigmoid."""
    global BOX_SCALE
    BOX_SCALE = 640.0
    p = np.asarray(outs[0])[0]
    r = _fused_head_dets(p)
    BOX_SCALE = 1.0
    return r


def post_v8s_norm(outs):
    """v8s_norm: [1,84,8400] -> [8400,84], box 归一化0-1, cls 已 sigmoid."""
    global BOX_SCALE
    BOX_SCALE = 640.0
    p = np.asarray(outs[0])[0].T                      # [84,8400] -> [8400,84]
    r = _fused_head_dets(p)
    BOX_SCALE = 1.0
    return r


def post_v5s_norm(outs):
    """v5s_norm(2): 3 输出 [1,N,85] 或 [1,85,N], box 归一化0-1 cxcywh, obj/cls 已 sigmoid."""
    pieces = []
    for o in outs:
        p = np.asarray(o)[0]
        if p.shape[0] == 85 and p.shape[1] != 85:
            p = p.T                                   # [85,N] -> [N,85]
        box = p[:, :4] * 640.0
        obj = p[:, 4]
        cls = p[:, 5:]
        score = obj * cls.max(1)
        cid = cls.argmax(1)
        m = score > OBJ_THRESH
        if m.sum() == 0:
            continue
        x1 = box[m, 0] - box[m, 2] / 2
        y1 = box[m, 1] - box[m, 3] / 2
        x2 = box[m, 0] + box[m, 2] / 2
        y2 = box[m, 1] + box[m, 3] / 2
        det = np.stack((x1, y1, x2, y2, score[m], cid[m].astype(np.float32)), 1)
        pieces.append(det)
    if not pieces:
        return []
    return apply_nms(np.concatenate(pieces, 0))


def post_v5s_dec(outs):
    """v5s_norm3: 3 输出 [1,255,h,w], 255=3x85, box 归一化0-1 cxcywh, obj/cls 已 sigmoid(解码版)."""
    pieces = []
    for o in outs:
        o = np.asarray(o).reshape(3, 85, o.shape[-2], o.shape[-1])
        box = o[:, :4] * 640.0
        obj = o[:, 4]
        cls = o[:, 5:]
        score = obj * cls.max(1)
        cid = cls.argmax(1)
        for a in range(3):
            s = score[a]
            m = s > OBJ_THRESH
            if m.sum() == 0:
                continue
            x1 = box[a, 0][m] - box[a, 2][m] / 2
            y1 = box[a, 1][m] - box[a, 3][m] / 2
            x2 = box[a, 0][m] + box[a, 2][m] / 2
            y2 = box[a, 1][m] + box[a, 3][m] / 2
            det = np.stack((x1, y1, x2, y2, s[m], cid[a][m].astype(np.float32)), 1)
            pieces.append(det)
    if not pieces:
        return []
    return apply_nms(np.concatenate(pieces, 0))


def post_v5s_abs(outs):
    """v5s FP16 版: 3 输出 [1,N,85], box 绝对坐标 cxcywh, obj/cls 已 sigmoid."""
    pieces = []
    for o in outs:
        p = np.asarray(o)[0]                       # [N,85]
        box = p[:, :4]
        obj = p[:, 4]
        cls = p[:, 5:]
        score = obj * cls.max(1)
        cid = cls.argmax(1)
        m = score > OBJ_THRESH
        if m.sum() == 0:
            continue
        x1 = box[m, 0] - box[m, 2] / 2
        y1 = box[m, 1] - box[m, 3] / 2
        x2 = box[m, 0] + box[m, 2] / 2
        y2 = box[m, 1] + box[m, 3] / 2
        det = np.stack((x1, y1, x2, y2, score[m], cid[m].astype(np.float32)), 1)
        pieces.append(det)
    if not pieces:
        return []
    return apply_nms(np.concatenate(pieces, 0))


POST = {"v5s": post_v5s, "v8s": post_v8s, "v11s": post_v11s,
        "v11s_norm": post_v11s_norm, "v8s_norm": post_v8s_norm,
        "v5s_norm": post_v5s_norm,
        "v11s_norm2": post_v8s_norm,   # [1,84,8400] 布局同 v8s_norm
        "v5s_norm2": post_v5s_norm,    # [1,85,N] 自适应转置
        "v5s_norm3": post_v5s_dec,     # [1,255,h,w] 解码版
        "v5s_abs": post_v5s_abs}       # [1,N,85] 绝对坐标解码版


def main():
    args = sys.argv[1:]
    model_type, model_path, images_dir, image_list, out_json = args[:5]
    gt_json = args[5] if len(args) > 5 else None

    # COCO category_id 有跳号(如 12 缺失), 必须用真实 id 映射
    cat_ids = None
    if gt_json:
        import json as _json
        gt = _json.load(open(gt_json))
        cat_ids = [c["id"] for c in sorted(gt["categories"], key=lambda x: x["id"])]

    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(model_path) == 0, "load_rknn failed"
    assert rknn.init_runtime(target="rk3588") == 0, "init_runtime failed"

    names = [l.strip() for l in open(image_list) if l.strip()]
    preds = []
    for n, name in enumerate(names):
        img = cv2.imread(os.path.join(images_dir, name))
        if img is None:
            print("WARN missing image:", name, flush=True)
            continue
        orig_h, orig_w = img.shape[:2]
        canvas, r, dx, dy = letterbox(img)
        inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.uint8)[None]  # uint8 直传, 省 float32 转换 (-12%)
        outs = rknn.inference(inputs=[inp], data_format="nhwc")
        dets = POST[model_type](outs)
        for d in dets:
            x1, y1, x2, y2, sc, cid = d
            # 还原到原图坐标 (letterbox 逆变换)
            ox1 = (x1 - dx) / r
            oy1 = (y1 - dy) / r
            ox2 = (x2 - dx) / r
            oy2 = (y2 - dy) / r
            ox1 = max(0.0, min(float(orig_w), ox1))
            oy1 = max(0.0, min(float(orig_h), oy1))
            ox2 = max(0.0, min(float(orig_w), ox2))
            oy2 = max(0.0, min(float(orig_h), oy2))
            preds.append({"image_id": int(name.split(".")[0]),
                          "category_id": (cat_ids[int(cid)] if cat_ids is not None else int(cid) + 1),
                          "bbox": [ox1, oy1, ox2 - ox1, oy2 - oy1],
                          "score": sc})
        if (n + 1) % 10 == 0:
            print("processed %d/%d, total dets=%d" % (n + 1, len(names), len(preds)), flush=True)

    json.dump({"predictions": preds}, open(out_json, "w"))
    print("DONE: %d images -> %d predictions -> %s" % (len(names), len(preds), out_json), flush=True)
    rknn.release()


if __name__ == "__main__":
    main()
