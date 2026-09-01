import time, threading
import numpy as np
from rknn.api import RKNN

MODEL = "/home/firefly/Tingshuo/rknn-3588-npu-yolo-accelerate/weights/yolov5s.rknn"
imgs = np.random.randint(0, 255, (100, 640, 640, 3), dtype=np.uint8)

def run_threads(n_threads):
    rknns = []
    for _ in range(n_threads):
        r = RKNN(verbose=False)
        assert r.load_rknn(MODEL) == 0
        assert r.init_runtime(target="rk3588") == 0
        rknns.append(r)
    for i in range(min(n_threads, 5)):
        rknns[i % n_threads].inference(inputs=[imgs[i]], data_format="nhwc")
    out = [None] * n_threads
    def worker(t):
        start = (100 * t) // n_threads
        end = (100 * (t + 1)) // n_threads
        for i in range(start, end):
            rknns[t].inference(inputs=[imgs[i]], data_format="nhwc")
        out[t] = end - start
    t0 = time.time()
    ts = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    dt = time.time() - t0
    for r in rknns: r.release()
    return dt

for n in (1, 2, 3, 4, 5):
    dt = run_threads(n)
    print("%d-thread 100 imgs: %.2f s (%.2f FPS)" % (n, dt, 100 / dt), flush=True)
