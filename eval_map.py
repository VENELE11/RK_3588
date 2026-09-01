#!/usr/bin/env python3
"""COCO bbox evaluation.

Usage: python eval_map.py <gt.json> <predictions.json>
Reports COCO AP@[0.50:0.95], AP50 and AP75.  If pycocotools is installed,
its official COCOeval implementation is used; otherwise a compatible
bbox-only fallback is used for this project's subset format.
"""
import json
import sys
from collections import defaultdict

import numpy as np


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, a[2]) * max(0.0, a[3]) + max(0.0, b[2]) * max(0.0, b[3]) - inter
    return inter / union if union > 0.0 else 0.0


def _fallback(gt, pred):
    """COCO bbox AP for area=all and maxDets=100."""
    iou_thresholds = np.arange(0.50, 0.951, 0.05)
    recall_thresholds = np.arange(0.0, 1.001, 0.01)
    gt_by_key = defaultdict(list)
    dt_by_key = defaultdict(list)
    for ann in gt.get("annotations", []):
        if not ann.get("ignore", False):
            gt_by_key[(ann["image_id"], ann["category_id"])].append(ann)
    for det in pred.get("predictions", pred if isinstance(pred, list) else []):
        dt_by_key[(det["image_id"], det["category_id"])].append(det)
    categories = sorted({c for _, c in gt_by_key})
    images = sorted({i for i, _ in gt_by_key} | {i for i, _ in dt_by_key})
    results = np.zeros((len(iou_thresholds), len(categories)), dtype=np.float64)

    for ti, threshold in enumerate(iou_thresholds):
        for ci, category in enumerate(categories):
            scores, matches = [], []
            npos = 0
            for image_id in images:
                gts = gt_by_key.get((image_id, category), [])
                dts = sorted(dt_by_key.get((image_id, category), []),
                             key=lambda x: -float(x.get("score", 0.0)))[:100]
                npos += len(gts)
                used = [False] * len(gts)
                for det in dts:
                    best_j, best_iou = -1, threshold
                    for j, ann in enumerate(gts):
                        if used[j]:
                            continue
                        value = _box_iou(det["bbox"], ann["bbox"])
                        if value >= best_iou:
                            best_iou, best_j = value, j
                    scores.append(float(det.get("score", 0.0)))
                    if best_j >= 0:
                        used[best_j] = True
                        matches.append(1.0)
                    else:
                        matches.append(0.0)
            if npos == 0 or not scores:
                continue
            order = np.argsort(-np.asarray(scores), kind="stable")
            tp = np.asarray(matches, dtype=np.float64)[order]
            fp = 1.0 - tp
            recall = np.cumsum(tp) / float(npos)
            precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            values = [precision[recall >= t].max() if np.any(recall >= t) else 0.0
                      for t in recall_thresholds]
            results[ti, ci] = float(np.mean(values))

    ap5095 = float(results.mean()) if results.size else 0.0
    ap50 = float(results[0].mean()) if results.size else 0.0
    ap75 = float(results[5].mean()) if results.shape[0] > 5 else 0.0
    return ap5095, ap50, ap75, len(categories)


def _official(gt_path, pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(gt_path)
    pred = _load(pred_path)
    detections = pred.get("predictions", pred)
    dt = gt.loadRes(detections)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats, len(ev.params.catIds), "pycocotools.COCOeval"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python eval_map.py <gt.json> <predictions.json>")
    gt_path, pred_path = sys.argv[1:]
    try:
        stats, class_count, backend = _official(gt_path, pred_path)
        print("backend = %s" % backend)
        print("classes evaluated = %d" % class_count)
        print("AP@[.50:.95] = %.4f" % float(stats[0]))
        print("AP@.50 = %.4f" % float(stats[1]))
        print("AP@.75 = %.4f" % float(stats[2]))
    except ImportError:
        gt, pred = _load(gt_path), _load(pred_path)
        ap5095, ap50, ap75, class_count = _fallback(gt, pred)
        print("backend = pure-python COCO bbox fallback (install pycocotools for official COCOeval)")
        print("classes evaluated = %d" % class_count)
        print("AP@[.50:.95] = %.4f" % ap5095)
        print("AP@.50 = %.4f" % ap50)
        print("AP@.75 = %.4f" % ap75)


if __name__ == "__main__":
    main()