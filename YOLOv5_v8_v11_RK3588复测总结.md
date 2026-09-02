# YOLOv5s / YOLOv8s / YOLO11s RK3588 复测总结

## 结论

在同一块 RK3588、同一套 500 张 COCO val2017 子集、同一预处理和官方 `pycocotools.COCOeval` 下，三种模型均使用当前可用的 RKNN INT8 产物进行复测。精度排序为 YOLO11s > YOLOv8s > YOLOv5s；纯 NPU 推理吞吐为 YOLOv8s > YOLO11s > YOLOv5s。

## 统一测试条件

- 输入：640×640，letterbox，114 灰边，RGB，uint8，NHWC。
- 评测集：500 张 COCO val2017 子集，标注 `coco_val_subset_500.json`。
- 后处理：conf=0.01，NMS=0.45，COCO category id 映射，最多 100 个检测参与 COCOeval。
- 指标：bbox AP@[0.50:0.95]、AP50、AP75。
- FPS：RK3588 板端 Python RKNN 纯推理，100 次、5 次 warmup，不含图像预处理和后处理。

## 复测结果与官方数据

| 模型 | 板端模型 | 官方 AP@[.50:.95] | 板端 AP@[.50:.95] | 板端 AP50 | 板端 AP75 | 板端 FPS |
|---|---|---:|---:|---:|---:|---:|
| YOLOv5s | `yolov5s.rknn` | 37.4% | 41.28% | 59.91% | 45.01% | 5.81 |
| YOLOv8s | `yolov8s.rknn` | 44.9% | 45.35% | 61.57% | 48.80% | 16.90 |
| YOLO11s | `yolov11s_zoo_int8_letterbox500.rknn` | 47.0% | 48.03% | 65.31% | 52.18% | 15.91 |

官方值为 Ultralytics 640px、完整 COCO val2017、单模型单尺度结果：

- YOLOv5/v8：<https://docs.ultralytics.com/compare/yolov5-vs-yolov8>
- YOLO11/v8：<https://docs.ultralytics.com/compare/yolo11-vs-yolov8>

由于板端使用 500 张子集，官方值使用完整 val2017，二者用于趋势对比，不应视为同数据集的严格复现。官方速度测试平台为 CPU ONNX、A100/TensorRT 等，与 RK3588 NPU FPS 不直接横向换算。

## YOLO11 INT8 修复结果

此前融合检测头 YOLO11 INT8 只有约 15.6%～16.3% AP。改用官方 RKNN Model Zoo 的原始解耦检测头输出，将 DFL、sigmoid、坐标解码和 NMS 放在 Python/C++ 后处理中，并使用与部署一致的 500 张 letterbox 校准集后：

- ONNX FP32：48.74% AP。
- RKNN FP16：48.61% AP，5.91 FPS。
- RKNN INT8：48.03% AP，15.65 FPS。
- INT8 相比 FP16 下降 0.59 个百分点，达到 AP≥45%、差距≤3个百分点的第一阶段目标。

输入验证显示 INT8 RKNN 模型要求 8-bit 输入；float32 直传会因输入尺寸不匹配而失败，后续 C++ 必须依据模型输入 `type/scale/zp` 正确处理 UINT8、INT8 和 pass-through。

## 本次同步内容

- 新增原始检测头导出、输入探针和评测辅助脚本。
- 新增 raw-head C++ 后处理入口及构建配置。
- 新增 YOLO11 FP16/INT8 模型产物。
- 更新测试方法、性能数据、模型对比、优化报告和任务总结文档。
