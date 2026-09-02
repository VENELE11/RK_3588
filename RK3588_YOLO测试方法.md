# RK3588 YOLO 测试方法文档

> 记录日期：2026-08-17
> 适用范围：RK3588 板端 YOLO 系列模型（v5s/v8s/v11s）的性能与精度评测
> 原则：**口径统一、公平对比、可复现、交叉验证**

---

## 一、测试目标与总原则

| 原则 | 说明 |
|------|------|
| **口径统一** | 同一指标必须用同一工具/参数/循环次数测量 |
| **环境固定** | NPU 频率固定 1GHz（userspace governor）、同一运行时（librknnrt 1.5.3b6 / driver 0.7.2） |
| **多次取均值** | 推理延迟 100 次循环取均值；端到端多帧统计 |
| **交叉验证** | mAP 用自研脚本 + pycocotools 官方 COCOeval 双向核对 |
| **日志留痕** | 所有测试输出重定向到日志文件，可追溯 |

---

## 二、测试环境

| 项 | 值 |
|----|----|
| 板端 | RK3588，`firefly@192.168.10.72`，Debian 11，aarch64 |
| NPU | 三核 @ 1GHz（`/sys/class/devfreq/fdab0000.npu`，userspace 固定） |
| 运行时 | librknnrt 1.5.3b6 / driver 0.7.2 |
| 转换工具 | rknn-toolkit2 2.3.2（板端 `/dl/python`，Python 3.6.8） |
| 测试图 | bus.jpg（640×640 输入统一） |
| 测试视频 | 300 帧短视频（扫描用）、720p60hz.mp4 5400 帧（验证用） |

---

## 三、远程访问与操作规范

### 3.1 SSH 连接（expect 密码登录）

```bash
expect ssh_rk3588.exp '<远程命令>'   # 脚本见项目文件夹
```

### 3.2 复杂脚本传输（base64 规避转义）

```bash
SCRIPT='多行脚本内容'
B64=$(printf '%s' "$SCRIPT" | base64)
expect ssh_rk3588.exp "echo $B64 | base64 -d | bash"
```

### 3.3 文件传输

```bash
expect scp_rk3588.exp <本地文件> <板端路径>        # 上传
expect scp_from_rk3588.exp <板端路径> <本地文件>    # 下载
```

### 3.4 长任务后台运行

```bash
nohup bash /tmp/run.sh > /tmp/run.log 2>&1 &   # 后台执行
tail /tmp/run.log                                # 轮询进度
```

---

## 四、性能测试方法

### 4.1 纯 NPU 推理延迟（rknn_benchmark）

**工具**：`/tmp/rknn_benchmark`（官方工具，stb_image 读图，不依赖 OpenCV）

```bash
/tmp/rknn_benchmark <model.rknn> <bus.jpg> 100 7   # 三核（core_mask=7）
/tmp/rknn_benchmark <model.rknn> <bus.jpg> 100 1   # 单核（core_mask=1）
```

- 100 次循环取均值（前 5 次 warmup 不计）
- 读取日志中 `Avg Time`（ms）与 `Avg FPS`
- **口径**：只含 `rknn_run`（NPU 前向），不含预处理/后处理
- **用途**：模型间最公平的对比；三核/单核差值反映多核并行收益

### 4.2 端到端 FPS（视频流水线）

```bash
# v5s（线程池，4 线程）
cd /home/firefly/Tingshuo/rknn-3588-npu-yolo-accelerate
./build/yolov5_thread_pool ./weights/yolov5s.rknn <video.mp4> 4

# v8s（rknn_deploy）
cd /home/firefly/Tingshuo/yolov8_rknn/c++
./build/rknn_deploy --model ../yolov8s.rknn --video <video.mp4> --labels coco_80_labels_list.txt
```

- 日志每秒输出一行 `Method2 Time/FPS/Frame Count`，取全部行平均
- **口径**：解码+预处理+推理+后处理全链路
- **注意**：v5s 必须从项目根目录运行（labels 相对路径）；各模型程序不同，端到端仅做量级对比，严格对比看 4.1

### 4.3 线程数扫描（确定最优线程数）

```bash
for T in 1 2 3 4 5; do
  ./build/yolov5_thread_pool ./weights/yolov5s.rknn /tmp/short_300.mp4 $T
done
```

- 用 300 帧短视频（5s@60fps）控制总时长
- 绘制"线程数-FPS"曲线找饱和点（本板实测 v5s：4 线程已近饱和，5 线程达上限 94%）

### 4.4 单图端到端（Python 统一口径）

```python
# 计时：load / init / 单次 inference / 首次全流程
# 见项目文件夹 eval_detect.py 的 benchmark 逻辑
```

- **注意**：Python 推理（~78ms）比 C++ benchmark（~33ms）慢 2.4 倍（numpy 转换/GIL 开销）——单图口径只做功能验证与相对对比，不做跨语言绝对值比较

---

## 五、精度测试方法（mAP@0.5）

### 5.1 COCO 子集准备

```bash
# 100 张 COCO val 有标注图片 + 子集标注 JSON（本地准备，已存档在项目文件夹）
# 板端部署：/tmp/coco_eval_500/（images/ + coco_val_subset_500.json + image_list_500.txt）
```

- 子集标注包含 `categories`（80 类）——**mAP 计算的 id 映射来源**

### 5.2 批量推理（eval_detect.py）

```bash
/dl/python/bin/python /tmp/eval_detect.py <model_type> <model.rknn> \
    images image_list.txt pred.json <coco_val_subset_100.json>
```

**脚本关键实现**：
- 统一 letterbox 预处理（等比缩放 + 灰边 114）+ uint8 直传（`astype(np.uint8)`）
- 后处理按模型类型分发：`v5s`（anchor 解码 logit）/ `v8s`（DFL 6张量）/ `v11s`（融合头）/ `v5s_abs`（FP16 绝对坐标）/ `v8s_norm` `v11s_norm2`（归一化融合头）
- conf=0.01 收集候选（mAP 阈值无关，P-R 曲线覆盖全部阈值）、NMS 0.45、NMS 前 top-500 截断
- 输出 COCO 格式 predictions JSON

### 5.3 mAP 计算（eval_map.py）

```bash
/dl/python/bin/python /tmp/eval_map.py <coco_val_subset_100.json> pred.json
```

- 每类：按分数排序 → IoU>0.5 贪心匹配（GT 去重）→ 累计 TP/FP → P-R 曲线 → **101 点插值 AP**
- mAP = 所有出现类别 AP 的平均

### 5.4 pycocotools 交叉验证（本地）

```bash
pip3 install pycocotools
python3 - <<EOF
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt = COCO("coco_val_subset_100.json")
dt = gt.loadRes(pred["predictions"])
ev = COCOeval(gt, dt, "bbox"); ev.evaluate(); ev.accumulate()
print(f"mAP@0.5 = {ev.stats[1]:.4f}")
EOF
```

**验证结果**：自研 eval_map 与 pycocotools 误差 <0.003（v5s 0.5856 vs 0.5842 等），链路可靠。

### 5.5 ⚠️ 关键注意事项：COCO category_id 跳号

COCO 80 类的 `category_id` 分布在 **1~90 且不连续**（如 id=12 缺失）：
- 预测输出必须用 `cat_ids[cid]`（GT json 中按 id 排序的 80 个真实 id）映射，**不能**用 `cid+1`
- 违反此规则 → 类别错位 → mAP 全面虚低（实测从 0.586 掉到 0.079，真实案例）

---

## 六、优化验证方法

### 6.1 uint8 vs float32 对比

```python
# 同一模型同一图，分别传 uint8 / float32，各 10 次计时 + 输出对比
# 判定标准：耗时下降 + 输出 max diff = 0（逐位一致）
# 实测：41.59 → 36.43 ms（-12.4%），diff 全 0
```

### 6.2 多实例并行验证

```bash
/dl/python/bin/python /tmp/bench_multithread.py   # 单线程 vs 4 线程，100 张图计时
```

- 每线程独立 RKNN 实例（独立 context）
- 实测：14.8 → 66.6 FPS（4.5×）

### 6.3 后处理正确性验证（sanity check）

```python
# 用 bus.jpg 跑各模型，打印 top 检测（类名/分数/框），与已知真值对照
# 例：v5s/v8s 应检出 bus ~0.86 + 多个 person；坐标应与参考 demo 一致
```

---

## 七、模型转换方法（含 ONNX 修复经验）

```python
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]],
            target_platform='rk3588')
rknn.load_onnx(model='model.onnx')
rknn.build(do_quantization=True, dataset='/tmp/coco_eval_500/calibration_letterbox_500.txt')
rknn.export_rknn('model.rknn')
```

**已知坑（务必先检查 ONNX）**：
1. **opset 必须 ≤15**（2.3.2 限制；opset 17 需本地降级，注意删 Reshape 的 allowzero 属性并验证输出一致）
2. **输出层避免 axis=2（最后一维）Slice**（rknn 会置零通道）——先 Transpose 到 axis=1
3. **融合输出（box 坐标+cls 概率混一个张量）量化损失大**——box 先归一化 ÷640 使值域统一 0~1
4. **v5s 解码版 ONNX 的 INT8 量化会置零 obj/cls**（FP16 正常）——用 raw logit 版 ONNX
5. **mmse 量化 + 大校准集在板端不可行**（实测 22h+）——用默认 normal 算法，100 校准仅 3~8 分钟

**转换验证三件套**：
```bash
# ① 输出布局/值域 dump（形状、min/max、逐通道统计）
# ② sanity check（bus.jpg 检出是否合理）
# ③ benchmark + mAP 全套（回归对比）
```

---

## 八、测试口径对照表

| 口径 | 工具 | 含什么 | 用途 |
|------|------|--------|------|
| 纯推理延迟 | rknn_benchmark | 仅 rknn_run | 模型公平对比（主） |
| 端到端 FPS | 视频流水线 | 全链路 | 真实应用吞吐 |
| 单图端到端 | Python 计时 | 加载+推理 | 功能验证 |
| mAP@0.5 | eval_map / pycocotools | 检测精度 | 精度对比 |

---

## 九、踩坑清单（影响测试准确性）

| # | 坑 | 影响 | 规避 |
|---|----|------|------|
| 1 | COCO category_id 跳号 | mAP 类别错位（0.586→0.079） | 用 cat_ids 映射 + pycocotools 交叉验证 |
| 2 | Python vs C++ 推理差异 | 绝对值差 2.4× | 跨语言只比趋势 |
| 3 | 多线程流水线 vs 纯推理 | FPS 差异大 | 明确口径再对比 |
| 4 | letterbox 与直接 resize 混淆 | 坐标还原错 | 统一 letterbox(114) + 逆变换 |
| 5 | 校准集预处理与推理不一致 | 量化统计偏差 | 校准图不 letterbox（rknn 内部拉伸） |
| 6 | 运行时版本混用 | 结果不可比 | 固定 librknnrt 1.5.3b6 |

## 2026-09-02 原始解耦检测头首轮结果

采用 RKNN Model Zoo YOLO11 解耦 ONNX（9 输出：`[64,80,1] × 3`），校准文件为 `/tmp/coco_eval_500/calibration_letterbox_500.txt`，使用 500 张 COCO、114 灰边、conf=0.01、NMS=0.45 统一评测：FP16 AP=0.4861，INT8 AP=0.4803，AP50/AP75=0.6531/0.5218。INT8 与 FP16 差距仅 0.59 个百分点，已达到 AP≥45% 且差距≤3个百分点的第一阶段目标。
## 2026-09-02 三模型 500 张复测结果

统一条件：`/tmp/coco_eval_500/` 500 张 COCO val2017 子集、114 灰边、conf=0.01、NMS=0.45。

| 模型 | AP@[0.5:0.95] | AP50 | AP75 | FPS |
|---|---:|---:|---:|---:|
| 原始解耦头 ONNX FP32 | 48.74% | 65.67% | 53.01% | — |
| 原始解耦头 RKNN FP16 | 48.61% | 65.67% | 52.98% | 5.91 |
| 原始解耦头 RKNN INT8 | 48.03% | 65.31% | 52.18% | 15.65 |

INT8 与 FP16 相差 0.59 个百分点，达到第一阶段目标。

## 2026-09-02 YOLOv5s / YOLOv8s / YOLO11s 同标准横向复测

统一实测条件：RK3588 板端、640×640、114 灰边 letterbox、RGB、uint8 NHWC、conf=0.01、NMS=0.45、500 张 COCO val2017 子集、官方 `pycocotools.COCOeval`，指标为 bbox AP@[0.50:0.95]。FPS 为板端 Python RKNN 纯推理吞吐（100 次、5 次 warmup），不含图像预处理和后处理。

| 模型 | 官方 COCO AP@[.50:.95] | 板端 AP@[.50:.95] | 板端 AP50 | 板端 AP75 | 板端纯 NPU FPS |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 37.4% | 41.28% | 59.91% | 45.01% | 5.81 |
| YOLOv8s | 44.9% | 45.35% | 61.57% | 48.80% | 16.90 |
| YOLO11s | 47.0% | 48.03% | 65.31% | 52.18% | 15.91 |

官方数据为 Ultralytics 640px、COCO val2017 单模型单尺度结果；官方速度是 CPU ONNX/A100 或 TensorRT 等不同平台数据，不能直接与 RK3588 NPU FPS 数值相减比较。板端三模型均为当前可用 RKNN INT8 产物，其中 YOLO11s 使用原始解耦检测头版本。
