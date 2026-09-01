#!/usr/bin/env python3
"""Compare two COCO-style prediction JSON files produced by this project."""

import json
import sys
from collections import defaultdict


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0.0 else 0.0


def grouped(path):
    data = json.load(open(path))["predictions"]
    groups = defaultdict(list)
    for p in data:
        groups[(p["image_id"], p["category_id"])].append(p)
    return data, groups


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_predictions.py python.json cpp.json")
    py_all, py = grouped(sys.argv[1])
    cpp_all, cpp = grouped(sys.argv[2])
    keys = sorted(set(py) | set(cpp))
    count_mismatch = []
    ious, score_diffs, bbox_diffs = [], [], []
    unmatched_python = unmatched_cpp = 0
    worst = (2.0, None, None, None)

    for key in keys:
        left = sorted(py.get(key, []), key=lambda p: -p["score"])
        right = sorted(cpp.get(key, []), key=lambda p: -p["score"])
        if len(left) != len(right):
            count_mismatch.append((key, len(left), len(right)))
        available = list(range(len(right)))
        for p in left:
            if not available:
                unmatched_python += 1
                continue
            same_score = [j for j in available if abs(p["score"] - right[j]["score"]) <= 1e-6]
            pool = same_score if same_score else available
            j = max(pool, key=lambda n: iou(p["bbox"], right[n]["bbox"]))
            q = right[j]
            available.remove(j)
            pair_iou = iou(p["bbox"], q["bbox"])
            ious.append(pair_iou)
            score_diffs.append(abs(p["score"] - q["score"]))
            bbox_diffs.append(max(abs(a - b) for a, b in zip(p["bbox"], q["bbox"])))
            if pair_iou < worst[0]:
                worst = (pair_iou, key, p, q)
        unmatched_cpp += len(available)

    print("python_predictions=%d" % len(py_all))
    print("cpp_predictions=%d" % len(cpp_all))
    print("count_mismatch_groups=%d" % len(count_mismatch))
    print("unmatched_python=%d" % unmatched_python)
    print("unmatched_cpp=%d" % unmatched_cpp)
    if ious:
        print("matched=%d" % len(ious))
        print("iou_min=%.9f" % min(ious))
        print("iou_mean=%.9f" % (sum(ious) / len(ious)))
        print("iou_ge_0.99=%d" % sum(x >= 0.99 for x in ious))
        print("iou_ge_0.50=%d" % sum(x >= 0.50 for x in ious))
        print("max_score_abs_diff=%.9g" % max(score_diffs))
        print("max_bbox_abs_diff=%.9g" % max(bbox_diffs))
        print("worst_key=%s" % (worst[1],))
        print("worst_python=%s" % worst[2])
        print("worst_cpp=%s" % worst[3])
    for item in count_mismatch[:10]:
        print("count_mismatch=%s python=%d cpp=%d" % (item[0], item[1], item[2]))


if __name__ == "__main__":
    main()
