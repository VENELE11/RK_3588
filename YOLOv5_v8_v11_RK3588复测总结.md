# YOLO11s RK3588 修改与验证总结

## 1. 修改目标

原 YOLO11s 融合检测头模型在 RKNN INT8 下 AP 仅约 15.6%～16.3%，而 ONNX FP32 和 RKNN FP16 约 48%。主要修复方向是停止继续扩大末端检测头 FP16 范围，改正模型输出结构、输入类型和量化校准方式。

## 2. YOLO11 模型结构修改

采用 RKNN Model Zoo 的 YOLO11s 原始解耦检测头 ONNX，输出三个尺度的 9 个张量：

- 回归分支：`[1,64,80,80]`、`[1,64,40,40]`、`[1,64,20,20]`
- 分类分支：`[1,80,80,80]`、`[1,80,40,40]`、`[1,80,20,20]`
- score 分支：`[1,1,80,80]`、`[1,1,40,40]`、`[1,1,20,20]`

不再把 DFL、sigmoid、坐标解码和最终拼接放进 ONNX/RKNN 图中，相关逻辑迁移到浮点后处理。这样避免了 box 大数值和分类概率共用输出量化 scale，也避免融合输出在 INT8 下破坏检测头数值分布。

## 3. 输入与量化修改

- 固定部署输入为 640×640、114 灰边 letterbox、RGB、uint8、NHWC。
- INT8 使用与部署预处理一致的 500 张图片生成校准集：`calibration_letterbox_500.txt`。
- 验证发现 INT8 RKNN 输入必须是 8-bit；float32 直传会触发输入尺寸不匹配。
- 增加输入探针脚本，后续可根据 `RKNN_QUERY_INPUT_ATTR` 的 `type/scale/zp` 分别验证 UINT8、INT8 手动量化和 pass-through。
- 当前首轮采用 Toolkit2 2.3.2 默认 asymmetric INT8；MMSE、per-channel 和 hybrid quant 作为后续消融方向。

## 4. C++ 与评测工具修改

- 新增 `cpp_port/src/raw_main.cpp`，提供 raw-head 推理和浮点 DFL/解码入口。
- 新增 `cpp_port/CMakeLists_raw.txt`。
- 修正 RKNN FLOAT16 输出读取，统一转换为 float 后再进行后处理。
- 新增 `export_yolo11_raw.py`、`probe_rknn_input.py`、`probe_rknn_input_v2.py`。
- 评测统一使用 500 张 COCO val2017 子集、114 灰边、conf=0.01、NMS=0.45 和官方 `pycocotools.COCOeval`。

## 5. YOLO11 500 张复测结果

| 模型 | AP@[0.50:0.95] | AP50 | AP75 | 纯 NPU FPS |
|---|---:|---:|---:|---:|
| 原始解耦头 ONNX FP32 | 48.74% | 65.67% | 53.01% | — |
| 原始解耦头 RKNN FP16 | 48.61% | 65.67% | 52.98% | 5.91 |
| 原始解耦头 RKNN INT8 | 48.03% | 65.31% | 52.18% | 15.65 |

INT8 相比 FP16 仅下降 0.59 个百分点，达到第一阶段目标：AP≥45%，且与 FP16 差距不超过 3 个百分点。

## 6. 与 YOLOv5/v8 的背景对照

相同 500 张子集和 COCOeval 条件下，当前板端 INT8 结果为：YOLOv5s 41.28%、YOLOv8s 45.35%、YOLO11s 48.03%。这部分仅用于模型横向背景，不改变本次工作的主体——YOLO11 结构和量化修复。

Ultralytics 官方完整 COCO val2017 参考 AP@[.50:.95]：YOLOv5s 37.4%、YOLOv8s 44.9%、YOLO11s 47.0%。由于官方使用完整验证集、板端使用 500 张子集，结果仅作趋势对比。

## 7. 文件与模型产物

- `model_outputs/yolov11s_zoo_fp16.rknn`
- `model_outputs/yolov11s_zoo_int8_letterbox500.rknn`
- `export_yolo11_raw.py`
- `probe_rknn_input.py`
- `probe_rknn_input_v2.py`
- `cpp_port/src/raw_main.cpp`
- `cpp_port/CMakeLists_raw.txt`
- `RAW_HEAD_NEXT_STEPS.md`

## 8. 后续建议

优先用 Toolkit2 `accuracy_analysis` 定位首层、P3/P4/P5、neck Concat 和 DFL 前输出的误差，再进行 MMSE、per-channel、检测头 hybrid quant。只有 raw-head INT8 AP 再次低于约 42% 时，才进入 QAT。
