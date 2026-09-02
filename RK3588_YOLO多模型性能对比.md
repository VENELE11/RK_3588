# RK3588 NPU 上 YOLO 系列模型性能对比

> 记录日期：2026-08-13
> 测试目标：在瑞芯微 RK3588 开发板上，对比 YOLOv5s / YOLOv8s / YOLOv11s 三款模型在 NPU 上的推理性能（纯推理延迟 + 端到端 FPS）。

---

## 一、环境介绍

### 1.1 板端环境（目标设备）

| 项目 | 内容 |
|------|------|
| 主板 | 瑞芯微 RK3588 |
| IP | `192.168.10.72` |
| 用户 | `firefly`（密码 `firefly`） |
| 系统 | Debian GNU/Linux 11 |
| 内核 | `5.10.66` |
| 架构 | `aarch64` |
| CPU | 8 核 ARM，约 15 GiB 内存，无 Swap |
| NPU | RK3588 三核心 NPU，实测运行频率 **1 GHz**（`/sys/class/devfreq/fdab0000.npu`，min 300 MHz / max 1 GHz） |
| 编译器 | gcc/g++ 10.2.1（`/usr/bin/gcc` 与 `/usr/bin/aarch64-linux-gnu-gcc` 均可用），cmake 3.28.0 |
| OpenCV | 4.5.1（`/usr/lib/aarch64-linux-gnu/libopencv*`） |

### 1.2 RKNN 运行时环境

| 项目 | 内容 |
|------|------|
| RKNN Runtime | API `1.5.3b6` / Driver `0.7.2` |
| 系统运行时库 | `/usr/lib/librknnrt.so` |
| rknn-toolkit2 | **2.3.2**（已 pip 安装于 `/dl/python`，含 torch 1.10.2 / onnx 1.10.0） |
| rknn-toolkit-lite2 | 源码在 `/home/firefly/rknn-toolkit2-2.3.2/rknn-toolkit-lite2`（未 pip 安装到 `/dl/python`） |
| rknn_model_zoo | **2.3.2**（完整克隆于 `/home/firefly/Tingshuo/rknn_model_zoo` 和 `/home/firefly/Projects/rknn_model_zoo-2.3.2`） |
| Python | `/dl/python/bin/python`（3.6.8，含 `rknn.api`、onnx、onnxruntime、numpy、opencv-python） |

### 1.3 访问方式

使用非交互式 SSH，通过 `expect` 辅助密码登录（本机无 `sshpass`，`BatchMode` 会禁用密码输入）：

```bash
ssh -o StrictHostKeyChecking=accept-new firefly@192.168.10.72 '命令'
```

- SSH 会话非持久，需要时重建连接。
- `post-quantum key exchange` 警告为密钥交换算法提示，不影响连接。
- 复杂远程脚本通过 `base64` 编码传递，规避 expect/Tcl 对 `$`、`[...]` 等字符的转义问题。

---

## 二、任务目标

在 RK3588 上对比三款 YOLO 模型的性能，口径：

1. **纯 NPU 推理延迟**：`rknn_benchmark` 测 `rknn_run` 单次耗时（三核 / 单核）。
2. **端到端 FPS**：各模型 demo 处理 1280×720@60fps 视频的吞吐。

暂不对比检测效果（mAP / 可视化），只看性能。

---

## 三、执行过程

### 3.1 模型准备

| 模型 | 来源 | 处理方式 |
|------|------|----------|
| YOLOv5s | 当前项目 `weights/yolov5s.rknn` | 现成，直接测试 |
| YOLOv8s | `/home/firefly/Tingshuo/yolov8_rknn/yolov8s.rknn` | 现成，直接测试 |
| YOLOv11s | `/home/firefly/model/yolov11s/yolov11s_T.onnx` | **板端现场转换** |

YOLOv11s 转换（`yolov11s_T.onnx` 输入 `[1,3,640,640]` NCHW → 输出 `[1,8400,84]`，opset 12）：

```python
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
rknn.load_onnx(model='/home/firefly/model/yolov11s/yolov11s_T.onnx')
rknn.build(do_quantization=True, dataset='/home/firefly/Tingshuo/rknn_model_zoo/datasets/COCO/coco_subset_20.txt')
rknn.export_rknn('/home/firefly/model/yolov11s/yolov11s_640.rknn')
```

- 量化校准集：COCO 20 张子集（`datasets/COCO/subset/`）。
- 转换结果：全 INT8（权重 9.96 MB），无算子回退 CPU 的警告，仅有一条离群值量化提示。

### 3.2 编译 rknn_benchmark（官方 benchmark）

官方 `rknn_benchmark` 位于 `rknn-toolkit2-2.3.2/rknpu2/examples/rknn_benchmark`，使用 `stb_image` 读图（**不依赖 OpenCV**）。板端本地 g++ 手动编译：

```bash
cd /home/firefly/rknn-toolkit2-2.3.2/rknpu2/examples/rknn_benchmark
g++ -std=c++14 -O2 \
  -I src -I ../3rdparty \
  -I /home/firefly/Tingshuo/rknn_model_zoo/3rdparty/rknpu2/include \
  src/rknn_benchmark.cpp src/cnpy/cnpy.cpp \
  -L/usr/lib -lrknnrt -lpthread \
  -o /tmp/rknn_benchmark
```

> 注意：stb 头文件在 `examples/3rdparty`（即 `../3rdparty`），而非 `rknn_benchmark/3rdparty`。

### 3.3 编译 yolo11 官方 demo（单图推理）

```bash
cd /home/firefly/Tingshuo/rknn_model_zoo
./build-linux.sh -t rk3588 -a aarch64 -d yolo11
```

产出：`install/rk3588_linux_aarch64/rknn_yolo11_demo/`（含 `rknn_yolo11_demo` 与 `rknn_yolo11_demo_zero_copy`，使用 RGA 硬件预处理）。

---

## 四、性能对比结果

> 统一条件：640×640 输入、INT8 量化、NPU 恒 1 GHz、100 次循环取均值。

### 4.1 纯 NPU 推理延迟（rknn_benchmark，最公平）

| 模型 | 权重 | 三核(core_mask=7) | 三核 FPS | 单核(core_mask=1) | 单核 FPS |
|------|------|:---:|:---:|:---:|:---:|
| **YOLOv5s** | 7.32 MB | **36.35 ms** | 27.5 | **34.84 ms** | 28.7 |
| **YOLOv8s** | 11.24 MB | **36.34 ms** | 27.5 | 45.67 ms | 21.9 |
| **YOLOv11s** | 9.96 MB | **89.90 ms** | 11.1 | 82.08 ms | 12.2 |

### 4.2 端到端 FPS（1280×720@60fps 视频）

| 模型 | 端到端 FPS | 配置 | 说明 |
|------|:---:|------|------|
| YOLOv5s | **~79** | 4 线程 | 当前项目 `yolov5_thread_pool` |
| YOLOv8s | **~82** | 6 线程 | `rknn_deploy`（73~88 FPS 波动） |
| YOLOv11s | **~11 单流 / 理论上限 ~36 多核** | — | 单图 demo 端到端 ~440ms（含模型加载 ~350ms） |

> 口径说明：v5s / v8s 为多线程视频流水线实测 FPS；v11s 官方 demo 为单图推理，故其视频端到端 FPS 以纯推理吞吐（11 FPS 单流）及多核并行上限（~36 FPS）表述。

---

## 五、关键结论

1. **v5s ≈ v8s**：三核延迟几乎相同（36.35 vs 36.34 ms），端到端都约 80 FPS，均超 60 FPS 实时。v8s 精度通常更优，是「实时 + 精度」的更优选。

2. **v11s 显著更慢（约 2.5 倍）**：三核 90 ms，是 v5s/v8s 的 2.5 倍。根因是 YOLOv11 引入的 **C2PSA（attention/softmax/matmul）模块对 RK3588 NPU 不友好**（全 INT8、无 CPU 回退，纯结构计算量大）。

3. **多核并行规律差异**：
   - v5s：单核(34.8ms)已最优，加核无益 → 适合少量线程。
   - v8s：三核(36.3ms)明显快于单核(45.7ms)，并行收益显著 → 6 线程 82 FPS 正好吃满三核。
   - v11s：三核(90ms)反比单核(82ms)慢 → 内部串行瓶颈，多核无收益。

4. **选型建议**：720p 实时检测选 v5s / v8s（~80 FPS）；追求更高精度且能接受 ~30 FPS 上限再考虑 v11s。

---

## 六、产出物（板端路径）

| 产出 | 路径 |
|------|------|
| 转换模型（v11s INT8） | `/home/firefly/model/yolov11s/yolov11s_640.rknn` |
| 官方 benchmark 工具 | `/tmp/rknn_benchmark` |
| yolo11 官方 demo | `/home/firefly/Tingshuo/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolo11_demo/` |
| v11s 转换脚本 | `/tmp/convert_yolo11.py` |

---

## 七、附录：关键命令速查

```bash
# 纯 NPU 延迟测试
/tmp/rknn_benchmark <model.rknn> <bus.jpg> 100 7    # 100次循环，三核
/tmp/rknn_benchmark <model.rknn> <bus.jpg> 100 1    # 100次循环，单核

# v5s 端到端（当前项目）
cd /home/firefly/Tingshuo/rknn-3588-npu-yolo-accelerate
./build/yolov5_thread_pool ./weights/yolov5s.rknn ./720p60hz.mp4 4

# v8s 端到端
cd /home/firefly/Tingshuo/yolov8_rknn/c++
./build/rknn_deploy --model ../yolov8s.rknn --video <video.mp4> --labels coco_80_labels_list.txt

# v11s 单图推理
cd /home/firefly/Tingshuo/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolo11_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./rknn_yolo11_demo /home/firefly/model/yolov11s/yolov11s_640.rknn ./model/bus.jpg
```

## 当前统一 500 张验收基线（2026-09-02）

后续精度判断统一采用 COCO val2017 500 张子集（`/tmp/coco_eval_500/`，标注 `coco_val_subset_500.json`，列表 `image_list_500.txt`）。此前 100 张结果保留为历史阶段记录，不作为当前验收结论。

| 模型 | AP@[0.5:0.95] | FPS | 用途 |
|---|---:|---:|---|
| ONNX FP32 | 48.72% | — | 精度基准 |
| RKNN FP16 | 48.65% | 5.63 | 部署精度基准 |
| 当前 INT8 | 15.6%～16.3% | 约 16.5 | 待修复 |

INT8 第一阶段验收目标：AP ≥45%，且与 RKNN FP16 的差距不超过 3 个百分点。执行优先级为：输入 dtype 验证 → 原始多尺度检测头 → C++ float32 DFL/解码 → FP16 对齐 → 默认 INT8/MMSE/per-channel → 逐层 hybrid quant → 必要时 QAT。
## 2026-09-02 三模型 500 张复测

使用同一板端评测脚本、500 张 COCO val2017 子集、114 灰边、conf=0.01、NMS=0.45：

| 模型 | AP@[0.5:0.95] | AP50 | AP75 | FPS |
|---|---:|---:|---:|---:|
| 原始解耦头 ONNX FP32 | 48.74% | 65.67% | 53.01% | — |
| 原始解耦头 RKNN FP16 | 48.61% | 65.67% | 52.98% | 5.91 |
| 原始解耦头 RKNN INT8 | 48.03% | 65.31% | 52.18% | 15.65 |

INT8 相比 FP16 下降 0.59 个百分点，满足 AP≥45% 且差距≤3个百分点的第一阶段目标。

## 2026-09-02 YOLOv5s / YOLOv8s / YOLO11s 同标准横向复测

统一实测条件：RK3588 板端、640×640、114 灰边 letterbox、RGB、uint8 NHWC、conf=0.01、NMS=0.45、500 张 COCO val2017 子集、官方 `pycocotools.COCOeval`，指标为 bbox AP@[0.50:0.95]。FPS 为板端 Python RKNN 纯推理吞吐（100 次、5 次 warmup），不含图像预处理和后处理。

| 模型 | 官方 COCO AP@[.50:.95] | 板端 AP@[.50:.95] | 板端 AP50 | 板端 AP75 | 板端纯 NPU FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 37.4% | 41.28% | 59.91% | 45.01% | 5.81 |
| YOLOv8s | 44.9% | 45.35% | 61.57% | 48.80% | 16.90 |
| YOLO11s | 47.0% | 48.03% | 65.31% | 52.18% | 15.91 |

官方数据为 Ultralytics 640px、COCO val2017 单模型单尺度结果；官方速度是 CPU ONNX/A100 或 TensorRT 等不同平台数据，不能直接与 RK3588 NPU FPS 数值相减比较。板端三模型均为当前可用 RKNN INT8 产物，其中 YOLO11s 使用原始解耦检测头版本。
