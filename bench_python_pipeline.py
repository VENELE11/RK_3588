#!/usr/bin/env python3
"""End-to-end Python image/video benchmark with single or multiple RKNN workers."""
import argparse
import queue
import threading
import time

import cv2
from rknn.api import RKNN

from eval_detect import POST, letterbox


MODEL_TYPES = {"v5": "v5s", "v8": "v8s", "v11_norm": "v11s_norm2"}


def load_worker(model_path):
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(model_path) == 0, "load_rknn failed"
    assert rknn.init_runtime(target="rk3588") == 0, "init_runtime failed"
    return rknn


def infer_one(rknn, model_type, frame):
    canvas, _, _, _ = letterbox(frame)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    outputs = rknn.inference(inputs=[rgb[None]], data_format="nhwc")
    return len(POST[MODEL_TYPES[model_type]](outputs))


def warmup(workers, model_type, frame):
    for rknn in workers:
        infer_one(rknn, model_type, frame)


def run_single_images(model_path, model_type, image_dir, image_list):
    names = [line.strip() for line in open(image_list) if line.strip()]
    workers = [load_worker(model_path)]
    first = cv2.imread(image_dir + "/" + names[0])
    warmup(workers, model_type, first)
    start = time.perf_counter()
    total = 0
    processed = 0
    for name in names:
        frame = cv2.imread(image_dir + "/" + name)
        if frame is not None:
            total += infer_one(workers[0], model_type, frame)
            processed += 1
    elapsed = time.perf_counter() - start
    workers[0].release()
    return processed, total, elapsed


def run_single_video(model_path, model_type, video_path, max_frames):
    workers = [load_worker(model_path)]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video: " + video_path)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("video has no frames")
    warmup(workers, model_type, first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start = time.perf_counter()
    frames = 0
    total = 0
    while (max_frames < 0 or frames < max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        total += infer_one(workers[0], model_type, frame)
        frames += 1
    elapsed = time.perf_counter() - start
    cap.release()
    workers[0].release()
    return frames, total, elapsed


def run_multi(model_path, model_type, workers_count, source, image_dir, image_list, max_frames):
    workers = [load_worker(model_path) for _ in range(workers_count)]
    if source == "images":
        names = [line.strip() for line in open(image_list) if line.strip()]
        first = cv2.imread(image_dir + "/" + names[0])
    else:
        cap = cv2.VideoCapture(image_list)
        if not cap.isOpened():
            raise RuntimeError("cannot open video: " + image_list)
        ok, first = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("video has no frames")
    warmup(workers, model_type, first)

    tasks = queue.Queue(maxsize=max(2, workers_count * 2))
    counts = [0] * workers_count
    errors = []

    def worker_loop(index):
        while True:
            item = tasks.get()
            try:
                if item is None:
                    return
                counts[index] += infer_one(workers[index], model_type, item)
            except Exception as exc:
                errors.append(exc)
            finally:
                tasks.task_done()

    threads = [threading.Thread(target=worker_loop, args=(i,)) for i in range(workers_count)]
    for thread in threads:
        thread.start()
    start = time.perf_counter()
    submitted = 0
    if source == "images":
        for name in names:
            frame = cv2.imread(image_dir + "/" + name)
            if frame is not None:
                tasks.put(frame)
                submitted += 1
    else:
        cap = cv2.VideoCapture(image_list)
        if not cap.isOpened():
            raise RuntimeError("cannot open video: " + image_list)
        while max_frames < 0 or submitted < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            tasks.put(frame)
            submitted += 1
        cap.release()
    for _ in threads:
        tasks.put(None)
    tasks.join()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    for rknn in workers:
        rknn.release()
    if errors:
        raise errors[0]
    return submitted, sum(counts), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=sorted(MODEL_TYPES))
    parser.add_argument("model_path")
    parser.add_argument("source", choices=("images", "video"))
    parser.add_argument("path")
    parser.add_argument("workers", type=int)
    parser.add_argument("max_frames", type=int)
    parser.add_argument("--image-list", default="/tmp/coco_eval/image_list.txt")
    args = parser.parse_args()

    if args.source == "images":
        if args.workers == 1:
            count, detections, elapsed = run_single_images(
                args.model_path, args.model_type, args.path, args.image_list)
        else:
            count, detections, elapsed = run_multi(
                args.model_path, args.model_type, args.workers, "images",
                args.path, args.image_list, args.max_frames)
    elif args.workers == 1:
        count, detections, elapsed = run_single_video(
            args.model_path, args.model_type, args.path, args.max_frames)
    else:
        count, detections, elapsed = run_multi(
            args.model_path, args.model_type, args.workers, "video",
            "", args.path, args.max_frames)
    print("PYBENCH source=%s model=%s workers=%d count=%d elapsed_s=%.3f fps=%.3f avg_detections=%.3f" %
          (args.source, args.model_type, args.workers, count, elapsed, count / max(elapsed, 1e-9),
           detections / max(count, 1)), flush=True)


if __name__ == "__main__":
    main()
