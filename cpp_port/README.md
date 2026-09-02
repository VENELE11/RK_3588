# YOLO RKNN Python→C++ 移植与对齐

这是 Python 基线的前四轮迁移版本：

1. 以 `eval_detect.py` 的 YOLOv5s 为基线，固定 640、letterbox、114 灰边、RGB、uint8、NHWC、阈值 0.01、NMS 0.45。
2. C++ 加载 `.rknn`，查询输入输出，执行单张或批量推理，并直接处理量化输出。
3. C++ 实现 YOLOv5s 三个输出层的 sigmoid、anchor 解码、类别筛选、按类别 NMS 和坐标还原。

板端编译示例：

```bash
cmake -S . -B build \
  -DRKNN_ROOT=/home/firefly/rknn-toolkit2-2.3.2/rknpu2/runtime/Linux/librknn_api \
  -DRGA_ROOT=/home/firefly/Tingshuo/rknn-cpp-Multithreading/include/3rdparty/rga/RK3588
cmake --build build -j4
./build/yolov5_rknn /path/to/yolov5s.rknn /path/to/image.jpg
```

当前程序已加入视频循环、端到端 FPS 统计、RGA 预处理和统一多模型多实例并行。

视频模式：

```bash
./build/yolov5_rknn model.rknn --video 720p60hz.mp4 60
```

最后一个参数是最多处理的帧数，不填则处理到视频结束。

## 已完成的板端验证

- CMake + g++ 编译通过，RKNN 输出为 3 个 YOLOv5s 检测头：`[1,255,80,80]`、`[1,255,40,40]`、`[1,255,20,20]`。
- 同一模型和 4 张 COCO 图片均能完成端到端推理，检测数量为 45、10、46、31。
- Python 与 C++ 已统一 OpenCV 评测预处理、COCO 类别映射、Top-500 稳定排序、分类 NMS 和坐标边界。
- COCO 100 张子集验收：v5 为 2124 个预测、双方 mAP@0.5 均为 0.5857；v8 为 2179 个预测、均为 0.6112；v11s_norm2 为 3354 个预测、均为 0.4439。
- v5 的 2124 个框全部逐框匹配，最小 IoU 0.99998；v8/v11 每图每类检测数量完全一致。
- 60 帧 720p 视频端到端测试：量化输出版从 9.50 FPS 提升到 10.39 FPS；取消中间 float 输出数组后达到 11.19 FPS，检测结果数量保持为 48.65/帧。
- 已接入板端 RGA 的等比缩放路径，并保留 OpenCV 回退；当前统一多线程框架已覆盖多实例并行、零拷贝 IO 和分阶段计时。

## 统一多线程框架

`yolov5_rknn_multi` 现在统一支持 v5、v8、v11 和 v11_norm。每个工作线程拥有独立的 RKNN context、输入输出内存、预处理缓存和计时统计，通过有界队列分发视频帧；任一 Worker 出错时会通知生产者和其他 Worker，主线程安全收尾并返回错误。

```bash
./build/yolov5_rknn_multi v5 model.rknn video.mp4 4 60
./build/yolov5_rknn_multi v8 yolov8s.rknn video.mp4 4 60
./build/yolov5_rknn_multi v11_norm yolov11s_norm2.rknn video.mp4 4 60
```

图片多 worker 测试：

```bash
./build/yolov5_rknn_multi v5 model.rknn --images /tmp/coco_eval/images /tmp/coco_eval/image_list.txt 5 100
```

参数依次为模型类型、模型文件、视频、线程数和最多处理帧数；旧版 v5 命令（省略模型类型）仍兼容。

此前 v5 专用框架按照《RK3588_YOLO测试方法.md》的 300 帧视频口径实测：1 线程 `15.09 FPS`、2 线程 `40.84 FPS`、3 线程 `65.88 FPS`、4 线程 `73.46 FPS`、5 线程 `82.05 FPS`；统一框架复测结果见下方。

统一框架复测：v5 5 线程 300 帧为 `92.93 FPS`，v8 5 线程 60 帧为 `69.78 FPS`，v11_norm 5 线程 60 帧为 `54.74 FPS`。输出同时打印各阶段平均耗时。

## v8/v11 迁移

已新增 `yolo_modern`：

```bash
./build/yolo_modern v8 yolov8s.rknn image.jpg
./build/yolo_modern v11_norm /home/firefly/model/yolov11s/yolov11s_norm2.rknn image.jpg
```

- v8：迁移三个尺度的 DFL 16-bin 边界回归、类别 sigmoid、NMS 和坐标还原；板端 `yolov8s.rknn` 输出为 6 个张量 `[1,64,H,W] + [1,80,H,W]`，单图已运行成功。
- v11：迁移普通融合输出和 `v11s_norm2` 归一化融合输出 `[1,84,N]` 或 `[1,N,84]` 的中心点、宽高、类别概率、NMS 和坐标还原，并兼容量化输出。使用 `/home/firefly/model/yolov11s/yolov11s_norm2.rknn` 已完成实机验证，输出 `[1,84,8400]`，单图检测框 40 个。

## COCO 批量评测

v5：

```bash
./build/yolov5_rknn yolov5s.rknn --eval \
  /tmp/coco_eval/images /tmp/coco_eval/image_list.txt /tmp/cpp_v5.json
```

v8/v11：

```bash
./build/yolo_modern v8 yolov8s.rknn --eval \
  /tmp/coco_eval/images /tmp/coco_eval/image_list.txt /tmp/cpp_v8.json
./build/yolo_modern v11_norm yolov11s_norm2.rknn --eval \
  /tmp/coco_eval/images /tmp/coco_eval/image_list.txt /tmp/cpp_v11.json
```

计算 mAP 和逐框比较：

```bash
/dl/python/bin/python /tmp/eval_map.py /tmp/coco_eval/coco_val_subset_100.json /tmp/cpp_v5.json
/dl/python/bin/python /tmp/compare_predictions.py /tmp/python_v5.json /tmp/cpp_v5.json
```

## 统一测试口径的纯 NPU 结果

使用 `/tmp/rknn_benchmark <model> bus.jpg 100 7`，前 5 次 warmup 不计：

| 模型 | 平均推理时间 | 平均 FPS |
|---|---:|---:|
| v5s | 33.48 ms | 29.872 |
| v8s | 40.68 ms | 24.584 |
| v11s_norm2 | 52.72 ms | 18.970 |

## C++ 分阶段计时与后处理优化（2026-08-31）

批量评测会输出一行 `PROFILE`，分别统计预处理、输入提交、NPU前向、输出读取、CPU后处理、输出释放和端到端耗时。

优化前100张平均耗时：

| 模型 | 预处理 | 输入提交 | NPU前向 | 输出读取 | 后处理 | 端到端 |
|---|---:|---:|---:|---:|---:|---:|
| v5 | 5.55 ms | 0.76 ms | 35.73 ms | 4.53 ms | 9.03 ms | 55.60 ms |
| v8 | 5.32 ms | 0.65 ms | 44.40 ms | 3.27 ms | 14.44 ms | 68.07 ms |
| v11_norm2 | 6.76 ms | 0.54 ms | 61.53 ms | 0.59 ms | 16.93 ms | 86.35 ms |

后处理优化：

1. 先扫描目标/类别分数，稳定选出Top-500，再解码候选框。
2. v5只为Top-500执行anchor坐标解码，v8只为Top-500执行DFL softmax。
3. 直接在INT8/UINT8量化域执行类别argmax，仅对最大类别值反量化。
4. Top-K使用“分数+原始顺序”确定性比较，避免同分候选改变结果。
5. NMS按固定80类数组分组，去掉`std::map`和重复排序，并缓存框面积。
6. 预分配候选和结果容器，减少循环内动态扩容。

最终连续复测的保守结果：

| 模型 | 优化前后处理 | 优化后后处理 | 降幅 | 最终预测数 | mAP@0.5 |
|---|---:|---:|---:|---:|---:|
| v5 | 9.03 ms | 2.51 ms | 72.2% | 2124 | 0.5857 |
| v8 | 14.44 ms | 8.75 ms | 39.4% | 2179 | 0.6112 |
| v11_norm2 | 16.93 ms | 13.45 ms | 20.5% | 3354 | 0.4439 |

独立运行时后处理最低实测为v5 2.30 ms、v8 8.57 ms、v11 12.64 ms。连续测试会受到NPU温度、CPU调度和动态频率影响，因此表格采用更保守的连续复测值。

优化前后的三份COCO JSON逐值比较结果均为：预测数一致、每图每类数量一致、最大分数误差0、最大坐标误差0。

## 零拷贝与 ARM NEON（2026-08-31）

已将三个 C++ 入口统一改为固定 IO 内存：输入使用 `rknn_create_mem` 复用一块缓冲区，输出使用复用的量化张量缓冲区，通过 `rknn_mem_sync` 完成 CPU/NPU 缓存同步，不再每帧申请和释放 `rknn_output`。

v8/v11 的通道优先 INT8/UINT8 类别扫描使用 ARM NEON 16 元素并行比较；v5 实测目标通道筛选较稀疏，保留原有“先目标筛选、后类别扫描”的标量路径，避免无效的全网格扫描。

板端 100 张 COCO 子集的最终连续测量（预处理直接写入 RKNN 输入缓冲区）：

| 模型 | 预处理 | 输入同步 | NPU前向 | 输出同步 | 后处理 | 端到端 | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 | 3.004 ms | 0.112 ms | 39.155 ms | 0.146 ms | 2.674 ms | 45.090 ms | 22.178 |
| v8 | 3.931 ms | 0.096 ms | 46.323 ms | 0.111 ms | 5.445 ms | 55.907 ms | 17.887 |
| v11_norm2 | 4.356 ms | 0.088 ms | 62.482 ms | 0.078 ms | 4.301 ms | 71.306 ms | 14.024 |

与上一版零拷贝实现相比，预处理耗时下降约 26%~46%，输入同步下降约 77%~80%；与原始分阶段基线相比，端到端耗时下降约 17%~19%。最终三套预测仍与已对齐基线逐值一致，mAP@0.5 保持 0.5857/0.6112/0.4439。300 帧视频复测：单线程 15.51 FPS，5 线程 86.81 FPS，平均检测框 51.757/帧。

## Python/C++ 完整测试矩阵（2026-08-31）

测试板为 RK3588，Python 使用 toolkit2.3.2；图片为 COCO 100 张子集，视频为 `720p60hz.mp4` 前 300 帧。FPS 为端到端吞吐，包含图片读取或视频解码、预处理、推理和后处理；多 worker 均为 5 个独立 RKNN 实例。

准确度验收：

| 模型 | Python预测数 | C++预测数 | Python mAP@0.5 | C++ mAP@0.5 | 对齐结果 |
|---|---:|---:|---:|---:|---|
| v5s | 2124 | 2124 | 0.5857 | 0.5857 | 逐框匹配，最小 IoU 0.99998 |
| v8s | 2179 | 2179 | 0.6112 | 0.6112 | 全部 IoU≥0.5，2178/2179 IoU≥0.99 |
| v11s_norm2 | 3354 | 3354 | 0.4439 | 0.4439 | 数量一致，浮点末位误差 |

单进程速度：

| 模型 | Python 图片 FPS | C++ 图片 FPS | Python 视频 FPS | C++ 视频 FPS |
|---|---:|---:|---:|---:|
| v5s | 9.07 | 22.46 | 9.86 | 17.02 |
| v8s | 8.21 | 17.58 | 8.27 | 13.55 |
| v11s_norm2 | 7.66 | 13.62 | 8.30 | 12.27 |

5 worker 速度：

| 模型 | Python 图片 FPS | C++ 图片 FPS | Python 视频 FPS | C++ 视频 FPS |
|---|---:|---:|---:|---:|
| v5s | 44.69 | 99.25 | 41.64 | 92.70 |
| v8s | 49.32 | 78.74 | 47.72 | 76.49 |
| v11s_norm2 | 49.97 | 58.36 | 47.57 | 56.11 |

复现 Python 端到端测试：

```bash
/dl/python/bin/python /tmp/bench_python_pipeline.py v5 model.rknn images /tmp/coco_eval/images 1 100 --image-list /tmp/coco_eval/image_list.txt
/dl/python/bin/python /tmp/bench_python_pipeline.py v5 model.rknn video.mp4 5 300
```

## 2026-09-01 新 IP 板端 C++ 后处理复测

测试板端：`firefly@192.168.10.72`；测试集：`/tmp/coco_eval/` 中 COCO val 100 张子集；后处理：C++，使用 `yolov5_rknn`（v5）和 `yolo_modern`（v8/v11_norm）；输入统一为 640×640、letterbox、阈值 0.01、NMS 0.45。

| 模型 | C++ 预测数 | C++ mAP@0.5 | 端到端耗时 | 端到端 FPS |
|---|---:|---:|---:|---:|
| v5s | 2124 | **0.5857** | 46.079 ms | 21.702 |
| v8s | 2179 | **0.6112** | 58.206 ms | 17.180 |
| v11s_norm2 | 3354 | **0.4439** | 57.779 ms | 17.307 |

与板端 Python 对齐结果：

- v5：2124/2124 框匹配，最小 IoU `0.999980`，最大分数误差 `5.96e-08`。
- v8：2179/2179 框匹配，2178 个 IoU ≥ `0.99`，全部 IoU ≥ `0.50`。
- v11_norm2：3354/3354 框匹配，3350 个 IoU ≥ `0.99`，最大坐标误差 `3.05e-05`；其余差异来自边界裁剪后的等价框表示。

复测结论：三个模型的 C++ 后处理精度均正常，mAP 与既有结果一致；v8s 仍为精度最高模型，v5s 速度优先，v11s_norm2 可用但精度和模型结构仍弱于 v5/v8。
## 2026-09-01 COCO 官方常用 AP 指标复测

已将 `eval_map.py` 更新为 COCO 常用指标：IoU 阈值 `0.50:0.05:0.95`，101 个召回率采样点，单图最多保留 100 个检测；脚本优先调用 `pycocotools.COCOeval`，板端未安装该库时使用同口径纯 Python bbox 回退实现，同时输出 AP@[.50:.95]、AP50 和 AP75。

本轮仍使用现有 COCO 100 张子集（项目中没有其余 COCO 图片，无法安全扩大测试集）：

| 模型 | AP@[.50:.95] | AP@.50 | AP@.75 |
|---|---:|---:|---:|
| v5s | **0.4013** | 0.5856 | 0.4322 |
| v8s | **0.4696** | 0.6092 | 0.4996 |
| v11s_norm2 | **0.2194** | 0.4439 | 0.1921 |

说明：本轮 C++ 预测数量仍为 v5/v8/v11 的 2124/2179/3354，AP@[.50:.95] 低于 AP50 是正常现象，因为它对 IoU=0.55~0.95 的定位精度提出了更严格要求。以上结果是 100 张子集结果，不代表完整 COCO val2017 官方成绩；板端本轮使用纯 Python COCO bbox 回退实现，安装 `pycocotools` 后可再做官方库交叉核验。
## 2026-09-01 COCO 500 张官方 COCOeval 复测

在原 100 张子集基础上，从 COCO val2017 补充下载 400 张图片，板端测试目录为 `/tmp/coco_eval_500/`，共 500 张图片。使用 C++ 后处理生成预测，并使用已安装的 `pycocotools.COCOeval` 计算官方常用指标 `AP@[0.50:0.95]`（area=all，maxDets=100）。

| 模型 | C++ 预测数 | AP@[.50:.95] | AP@.50 | AP@.75 |
|---|---:|---:|---:|---:|
| v5s | 10526 | **0.3998** | 0.5980 | 0.4372 |
| v8s | 10868 | **0.4538** | 0.6167 | 0.4875 |
| v11s_norm2 | 16808 | **0.1556** | 0.3768 | 0.1159 |

端到端 C++ 图片吞吐：v5s `21.844 FPS`，v8s `17.562 FPS`，v11s_norm2 `16.744 FPS`。500 张结果比 100 张子集更稳定：v5/v8 结果基本保持，v11 从子集 AP `0.2194` 降至 `0.1556`，此前 100 张结果对 v11 明显偏乐观。完整 COCO val2017 仍需其余图片全部可用后才能评估。
## v11s 低精度定位完成（500 张 COCO）

原始 yolov11s_T.onnx 与 yolov11s_T_norm2.onnx 使用 ONNX Runtime 和同等后处理分别得到 AP@[0.5:0.95] 0.4906、0.4872；yolov11s_norm2.rknn 使用 C++ 后处理仅 0.1556，yolov11s_norm100.rknn 为 0 个有效检测。由此确认主要损失位于 ONNX→RKNN 的量化/校准或输出反量化阶段，C++ 后处理和 COCO 指标公式不是主因。下一次转换应先验证 FP16，再用与 640 输入预处理一致的 500 张 calibration dataset 重做 INT8，并检查检测头输出量化参数。


### Toolkit2 重建结论

使用 Toolkit2 2.3.2 和 500 张校准图重建：原图校准 INT8 AP@[0.5:0.95]=0.1631，部署一致 letterbox 校准 INT8=0.1567；FP16 RKNN=0.4865，接近 ONNX=0.4872。因此问题不在 C++ 后处理或评估公式，INT8 主要受 outlier/检测头量化影响。当前推荐 FP16；若必须 INT8，使用 hybrid quant，将检测头敏感层保留 FP16。C++ 已补充 RKNN FLOAT16 张量读取支持。


### Hybrid quant 结论

Toolkit2 hybrid 测试：仅 new_output FP16 得到 AP 0.1575；model.23 检测头相关层整体 FP16 得到 AP 0.1582，提升很小。说明 INT8 误差在更早的主干/特征融合阶段形成。当前部署使用 yolov11s_fp16_500.rknn；若必须 INT8，应扩大敏感层范围或采用量化感知训练。


## 当前统一 500 张验收基线（2026-09-02）

后续精度判断统一采用 COCO val2017 500 张子集（`/tmp/coco_eval_500/`，标注 `coco_val_subset_500.json`，列表 `image_list_500.txt`）。此前 100 张结果保留为历史阶段记录，不作为当前验收结论。

| 模型 | AP@[0.5:0.95] | FPS | 用途 |
|---|---:|---:|---|
| ONNX FP32 | 48.72% | — | 精度基准 |
| RKNN FP16 | 48.65% | 5.63 | 部署精度基准 |
| 当前 INT8 | 15.6%～16.3% | 约 16.5 | 待修复 |

INT8 第一阶段验收目标：AP ≥45%，且与 RKNN FP16 的差距不超过 3 个百分点。执行优先级为：输入 dtype 验证 → 原始多尺度检测头 → C++ float32 DFL/解码 → FP16 对齐 → 默认 INT8/MMSE/per-channel → 逐层 hybrid quant → 必要时 QAT。
## 2026-09-02 板端环境核验

已通过交互式 SSH 核对 RK3588：

- 系统：Linux firefly 5.10.66，aarch64。
- Python：3.6.8；`rknn.api` 可导入。
- 当前评测集：`/tmp/coco_eval_500/`，存在 `coco_val_subset_500.json` 和 `image_list_500.txt`。
- YOLO11 目录：`/home/firefly/model/yolov11s/`，存在 `yolo11s.pt`、`yolov11s_T.onnx`、`yolov11s_T_norm2.onnx`、`yolov11s_fp16_500.rknn`、`yolov11s_int8_500.rknn` 以及 hybrid 版本。
- 板端没有安装 `ultralytics`，因此 `export_yolo11_raw.py` 暂不能直接运行；板端已有 RKNN Model Zoo 的 `examples/yolo11`，其模型结构已经是解耦的回归/分类输出，可优先复用其导出与转换链路。

因此当前 500 张验收集和既有 FP16/INT8 产物均已确认，下一步不是重新下载数据，而是基于 Model Zoo 的解耦输出或补齐主机端 Ultralytics 导出环境，生成三尺度 raw-head 模型。
## 2026-09-02 原始解耦头首轮实测

板端从 RKNN Model Zoo 获取 YOLO11 解耦 ONNX，输出为 9 个张量：三个尺度各包含回归 `[1,64,H,W]`、分类 `[1,80,H,W]`、score `[1,1,H,W]`。使用 500 张 `calibration_letterbox_500.txt` 校准并统一 114 灰边/conf=0.01/NMS=0.45 评测：FP16 AP@[0.5:0.95]=0.4861，INT8=0.4803，AP50/AP75=0.6531/0.5218。INT8 相比 FP16 下降 0.59 个百分点，超过第一阶段目标；纯 NPU Python 100 次推理约 15.42 FPS。

该结果说明主要问题确实来自融合输出图的量化方式，而不是 YOLO11 主干本身。后续优先查询 9 个输出的逐张量 `type/scale/zp`，再进行 MMSE、per-channel 和回归/分类 hybrid 消融。
## 2026-09-02 输入 dtype 验证结果

同一 `yolov11s_zoo_int8_letterbox500.rknn` 与同一张 640×640 RGB 输入对比：`uint8` 输入可正常推理；`float32` 输入触发 Toolkit2 错误 `current input size(4915200) > need input size(1228800)`。这确认该 INT8 模型要求 1,228,800 字节的 8-bit 输入，C++ 不能再无条件强制 float32 或依赖错误的 UINT8/INT8 解释。后续必须读取 `RKNN_QUERY_INPUT_ATTR` 的 `type/scale/zp`，分别实现 UINT8、INT8 手动量化和 pass-through 测试。
## 2026-09-02 三模型 500 张复测

使用同一板端评测脚本、500 张 COCO val2017 子集、114 灰边、conf=0.01、NMS=0.45：

| 模型 | AP@[0.5:0.95] | AP50 | AP75 | FPS |
|---|---:|---:|---:|---:|
| 原始解耦头 ONNX FP32 | 48.74% | 65.67% | 53.01% | — |
| 原始解耦头 RKNN FP16 | 48.61% | 65.67% | 52.98% | 5.91 |
| 原始解耦头 RKNN INT8 | 48.03% | 65.31% | 52.18% | 15.65 |

INT8 相比 FP16 下降 0.59 个百分点，满足 AP≥45% 且差距≤3个百分点的第一阶段目标。
