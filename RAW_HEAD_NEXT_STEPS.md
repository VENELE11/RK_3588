# YOLO11 raw-head INT8 实验记录入口

已加入两条首轮验证路径：

- `export_yolo11_raw.py`：从 `yolov11s.pt` 导出三个原始检测头，当前板端实际采用的官方解耦格式为 9 个张量：`[1,64,H,W]`、`[1,80,H,W]`、`[1,1,H,W]` × 3；自定义 `[1,144,H,W]` 合并格式仍可作为后续 C++ 入口的另一种输入协议。
- `cpp_port/src/raw_main.cpp`：读取三组解耦输出，在 C++ float32 中执行 DFL、sigmoid、网格解码、Top-500 和按类 NMS。

板端生成/编译后，先用单图做结构验收：

```bash
python export_yolo11_raw.py yolov11s.pt yolov11s_raw.onnx
python -m onnx.checker yolov11s_raw.onnx
./build/yolo11_raw yolov11s_raw_fp16.rknn bus.jpg
```

输入 dtype 探针：

```bash
python probe_rknn_input_v2.py yolov11s_raw_int8.rknn bus.jpg --scale <input_scale> --zp <input_zp>
```

探针固定相同的 letterbox/RGB 输入，保存三种输入和输出张量，供 ONNX Runtime 首层/输出对比。`int8_manual` 的 scale/zp 必须来自 `RKNN_QUERY_INPUT_ATTR`，不能猜测；`float32_pt1` 仅在模型允许 pass-through float32 时有效。

注意：当前已有 `main.cpp`/`modern_main.cpp` 的零拷贝结构仍默认构造 UINT8 输入，因此原始头模型的首轮验收应使用此独立 runner，并同时查看它打印的 input/output `type/scale/zp`。确认模型输入属性后，再把 dtype 选择合并回长期运行的零拷贝框架。

## 2026-09-02 板端环境核验

已通过交互式 SSH 核对 RK3588：

- 系统：Linux firefly 5.10.66，aarch64。
- Python：3.6.8；`rknn.api` 可导入。
- 当前评测集：`/tmp/coco_eval_500/`，存在 `coco_val_subset_500.json` 和 `image_list_500.txt`。
- YOLO11 目录：`/home/firefly/model/yolov11s/`，存在 `yolo11s.pt`、`yolov11s_T.onnx`、`yolov11s_T_norm2.onnx`、`yolov11s_fp16_500.rknn`、`yolov11s_int8_500.rknn` 以及 hybrid 版本。
- 板端没有安装 `ultralytics`，因此 `export_yolo11_raw.py` 暂不能直接运行；板端已有 RKNN Model Zoo 的 `examples/yolo11`，其模型结构已经是解耦的回归/分类输出，可优先复用其导出与转换链路。

因此当前 500 张验收集和既有 FP16/INT8 产物均已确认，下一步不是重新下载数据，而是基于 Model Zoo 的解耦输出或补齐主机端 Ultralytics 导出环境，生成三尺度 raw-head 模型。
## 2026-09-02 输入 dtype 验证结果

同一 `yolov11s_zoo_int8_letterbox500.rknn` 与同一张 640×640 RGB 输入对比：`uint8` 输入可正常推理；`float32` 输入触发 Toolkit2 错误 `current input size(4915200) > need input size(1228800)`。这确认该 INT8 模型要求 1,228,800 字节的 8-bit 输入，C++ 不能再无条件强制 float32 或依赖错误的 UINT8/INT8 解释。后续必须读取 `RKNN_QUERY_INPUT_ATTR` 的 `type/scale/zp`，分别实现 UINT8、INT8 手动量化和 pass-through 测试。
## 2026-09-02 三模型 500 张复测结果

统一条件：`/tmp/coco_eval_500/` 500 张 COCO val2017 子集、114 灰边、conf=0.01、NMS=0.45。

| 模型 | AP@[0.5:0.95] | AP50 | AP75 | FPS |
|---|---:|---:|---:|---:|
| 原始解耦头 ONNX FP32 | 48.74% | 65.67% | 53.01% | — |
| 原始解耦头 RKNN FP16 | 48.61% | 65.67% | 52.98% | 5.91 |
| 原始解耦头 RKNN INT8 | 48.03% | 65.31% | 52.18% | 15.65 |

INT8 与 FP16 相差 0.59 个百分点，达到第一阶段目标。
