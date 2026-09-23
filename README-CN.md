# reCamera Pro RKNN 模型基准测试

[![模型](https://img.shields.io/badge/models-25-blue)](./models/)
[![设备](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

对 Seeed reCamera Pro（Rockchip RV1126B，2 GB RAM，aarch64）进行了全面的 RKNN 模型基准测试。所有模型均转换为 RKNN 格式，并在设备上使用真实的 NPU 推理进行基准测试。

---

## 快速入门

```bash
# 克隆仓库（使用 Git LFS 下载模型权重）
git lfs install
git clone https://github.com/Seeed-Projects/recamera-pro-benchmark.git
cd recamera-pro-benchmark

# 部署任意模型到设备
scp models/yolo/yolov8_det/yolov8n_640x640_W8A8.rknn root@192.168.3.207:/userdata/benchmark/
scp models/yolo/yolov8_det/infer.py root@192.168.3.207:/userdata/benchmark/
```

---

## 仓库结构

```
recamera-pro-benchmark/
├── README.md                           # 本文件
├── AGENTS.md                           # 详细的基准测试方法论与规则
│
├── models/                             # 所有模型按系列组织
│   ├── yolo/                           # YOLO 系列（v8, 11, 26）
│   │   ├── tools/                      # ONNX 优化脚本
│   │   │   ├── optimize_yolo_onnx.py   # YOLOv8/YOLO11 ONNX 优化器
│   │   │   └── optimize_yolo26_onnx.py # YOLO26 ONNX 优化器
│   │   ├── yolov8_det/                 # YOLOv8 目标检测 (n, s)
│   │   ├── yolov8_pose/                # YOLOv8 姿态估计
│   │   ├── yolov8_obb/                 # YOLOv8 旋转边界框
│   │   ├── yolov8_seg/                 # YOLOv8 实例分割
│   │   ├── yolo11_det/                 # YOLO11 目标检测 (n, s)
│   │   ├── yolo11_pose/               # YOLO11 姿态估计
│   │   ├── yolo11_obb/                # YOLO11 旋转边界框
│   │   ├── yolo11_seg/                # YOLO11 实例分割
│   │   ├── yolo11_cls/                # YOLO11 图像分类
│   │   ├── yolo26_det/                # YOLO26 目标检测
│   │   ├── yolo26_pose/               # YOLO26 姿态估计
│   │   ├── yolo26_obb/                # YOLO26 旋转边界框
│   │   ├── yolo26_seg/                # YOLO26 实例分割
│   │   ├── yolo26_cls/                # YOLO26 图像分类
│   │   └── yolo26_depth/              # YOLO26 深度估计
│   │
│   ├── ppocr/                          # OCR 流水线
│   │   └── ppocrv4/                    # PPOCRv4 检测 + 识别 (INT8)
│   │
│   └── zipformer/                      # 语音转文字
│       └── (encoder + decoder + joiner, FP16)
│
├── benchmark_results/                  # 基准测试数据与日志
├── scripts/                            # 转换与基准测试脚本
├── assets/                             # 测试图像
├── calibration/                        # 校准数据集
└── doc/                                # 参考文档
```

每个模型子目录均为自包含的，包含：
- `*.onnx` — 原始 ONNX 模型
- `*_optimized.onnx` — 优化后的 ONNX（如适用）
- `*_W8A8.rknn` / `*_FP16.rknn` — 转换后的 RKNN 模型
- `convert_to_rknn.py` — ONNX → RKNN 转换脚本
- `infer.py` — 设备推理脚本（图像 + 摄像头模式）
- `calibration_200.txt` — 校准图像列表
- `result_*.jpg` — 示例推理结果
- `README.md` — 模型专项文档

---

## 模型概览

**25 个 RKNN 模型**，覆盖 3 个系列、9 个任务类别，全部在 RV1126B NPU 上进行基准测试。

### 目标检测

| 模型 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| YOLO26n | 640×640 | INT8 | 37.48 | 26.7 | 5.2 |
| YOLO11n | 640×640 | INT8 | 38.70 | 25.8 | 5.2 |
| YOLO11s | 640×640 | INT8 | 58.50 | 17.1 | 5.2 |
| YOLOv8n | 640×640 | INT8 | 36.35 | 27.5 | 5.1 |
| YOLOv8s | 640×640 | INT8 | 37.25 | 26.8 | 5.1 |

### 姿态估计

| 模型 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| YOLO26n-pose | 640×640 | INT8 | 37.31 | 26.8 | 5.2 |
| YOLO11n-pose | 640×640 | INT8 | 38.19 | 26.2 | 5.6 |
| YOLOv8n-pose | 640×640 | INT8 | 36.92 | 27.1 | 5.3 |

### 实例分割

| 模型 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| YOLO26n-seg | 640×640 | INT8 | 55.11 | 18.1 | 5.4 |
| YOLO11n-seg | 640×640 | INT8 | 54.59 | 18.3 | 5.4 |
| YOLOv8n-seg | 640×640 | INT8 | 61.89 | 16.2 | 5.4 |

### 旋转边界框 (OBB)

| 模型 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| YOLO26n-obb | 640×640 | INT8 | 33.57 | 29.8 | 5.2 |
| YOLO11n-obb | 640×640 | INT8 | 34.40 | 29.1 | 5.3 |
| YOLOv8n-obb | 640×640 | INT8 | 35.56 | 28.1 | 5.3 |

### 图像分类

| 模型 | FPS | 延迟 | 大小 |
|------|-----|------|------|
| yolo26n-cls | 152.2 | 6.6 ms | 3.65 MB |
| yolo11n-cls | 65.7 | 15.2 ms | — |

### 深度估计

| 模型 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| YOLO26n-depth | 640×640 | INT8 | 99.16 | 10.1 | 5.2 |

### OCR (PPOCRv4)

| 组件 | 输入 | 精度 | 延迟 (ms)* | FPS* | 功耗 (W) |
|------|------|------|-----------|------|----------|
| ppocrv4_det | 480×480 | INT8 | 58.62 | 17.1 | 5.0 |
| ppocrv4_rec | 48×320 | INT8 | (合计) | (合计) | |

\* OCR 延迟/FPS 为检测 + 识别完整流水线的合计值，速度随文本内容变化。

### 语音转文字 (Zipformer)

| 组件 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| zipformer_encoder | 103×80 | FP16 | 1834.9（整组合计） | / | 5.2 |
| zipformer_decoder | 1×2 | FP16 | | / | |
| zipformer_joiner | 1×512 | FP16 | | / | |

| 音频 | 总延迟 | RTF |
|------|--------|-----|
| 5.61 s test.wav | 1834.9 ms | 0.33 |
| 5.00 s 麦克风 | 1580.1 ms | 0.32 |

RTF < 1.0 表示实时处理（推理速度快于音频时长）。

---

## 模型目录使用方法

每个模型目录包含转换和部署所需的全部文件：

### ONNX → RKNN 转换（x86_64 主机）

```bash
cd models/yolo/yolov8_det
python3 convert_to_rknn.py
```

使用 RKNN-Toolkit2，`target_platform='rv1126b'`，200 张校准图像。

### 部署与运行（设备端）

```bash
# 推送模型和脚本到设备
scp models/yolo/yolov8_det/yolov8n_640x640_W8A8.rknn root@192.168.3.207:/userdata/benchmark/
scp models/yolo/yolov8_det/infer.py root@192.168.3.207:/userdata/benchmark/

# 图像推理（保存标注结果）
ssh root@192.168.3.207
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --image /userdata/benchmark/bus.jpg \
    --output /userdata/benchmark/result_bus.jpg

# 摄像头推理（20 次运行，5 次预热，输出平均延迟）
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn
```

### infer.py 模式

- **图像模式**（指定 `--input`）：执行推理，在最后一帧绘制结果，保存到设备并拉取回主机。
- **摄像头模式**（不指定 `--input`）：从 `/dev/video13`（1920×1080 NV12）读取，运行 20 次迭代（5 次预热），输出平均推理速度。不绘制帧。

---

## ONNX 优化

从 PyTorch（`*.pt`）导出的 YOLO 模型包含不适合 NPU 执行的后处理解码操作。`models/yolo/tools/` 中的优化脚本会剥离这些 CPU 端操作：

```bash
# 优化 YOLOv8/YOLO11 ONNX
python3 models/yolo/tools/optimize_yolo_onnx.py --model yolov8n.onnx --output yolov8n_optimized.onnx

# 优化 YOLO26 ONNX
python3 models/yolo/tools/optimize_yolo26_onnx.py --model yolo26n.onnx --output yolo26n_optimized.onnx
```

主要优化内容：
- 剥离 DFL 解码、锚点生成、NMS 和 sigmoid 操作
- 输出原始空间特征图供 NPU 加速推理
- 后处理在推理脚本中由 CPU 执行

---

## 设备环境

| 组件 | 规格 |
|------|------|
| 设备 | Seeed reCamera Pro |
| SoC | Rockchip RV1126B |
| CPU | ARM Cortex-A7, 4 核 |
| RAM | 1.9 GB（总计） |
| NPU | 2.0 TOPS |
| RKNN Runtime | 2.3.2 |
| 内核 | Linux 6.1.x |
| 主机工具包 | RKNN-Toolkit2 2.3.2 (x86_64) |

---

## 主机环境与工具链

在转换或基准测试模型之前，主机必须安装以下工具链。

| 组件 | 版本 | 用途 |
|------|------|------|
| 主机操作系统 | Ubuntu 22.04 (x86_64) / WSL2 | 构建与转换主机 |
| Python | 3.10 | 转换与推理脚本的运行时 |
| Conda 环境 | `recamera-rknn-2.3.2` (miniforge3) | ONNX → RKNN 转换 |
| RKNN-Toolkit2 | **2.3.2** | 模型转换，`target_platform='rv1126b'` |
| ONNX | ≥ 1.14 | ONNX 图检查与优化 |
| ONNX Runtime | ≥ 1.16 | CPU 端精度对比 |
| NumPy | 1.24 | 张量运算 |
| OpenCV | 4.x | 图像 I/O、绘制、NMS |
| PyTorch（可选） | 2.x | 通过 ultralytics 导出 YOLO `.pt` → `.onnx` |
| Ultralytics | ≥ 8.4 | YOLO 导出与参考后处理 |
| PaddlePaddle + Paddle2ONNX | 可选 | PPOCRv4 导出流水线 |

转换环境的激活方式：

```bash
# 基于 Miniforge（推荐，recamera-rknn-dev skill 使用）
conda activate recamera-rknn-2.3.2

# 或基于 Miniconda（旧版基准测试环境）
eval "$(/home/seeed/miniconda3/bin/conda shell.bash hook)"
conda activate rknn_bench
```

> **约定**：本仓库中的所有模型均使用 **RKNN-Toolkit2 2.3.2** 转换，目标平台为 `rv1126b`。设备端运行时（`librknnrt.so`）必须与工具包版本匹配——混用版本会在 `load_rknn()` 时失败。

---

## 测试与验证

测试分为三个级别，从快速验证到完整 NPU 基准测试。

### 1. 转换验证（主机，无需设备）

运行 `convert_to_rknn.py` 后，验证 `.rknn` 是否正确生成：

```bash
# 文件存在且大小合理（例如 YOLO nano 约 4 MB）
ls -lh yolov8n_640x640_W8A8.rknn

# 可选：在 Toolkit 模拟器上对比 ONNX 与 RKNN 输出
python3 scripts/compare_onnx_rknn.py \
    --onnx yolov8n_optimized.onnx \
    --rknn yolov8n_640x640_W8A8.rknn \
    --image assets/bus.jpg
```

### 2. 设备端推理测试（功能正确性）

部署模型并运行单次图像推理，验证检测质量：

```bash
# 推送到设备
scp yolov8n_640x640_W8A8.rknn infer.py bus.jpg root@192.168.3.207:/userdata/benchmark/

# SSH 到设备并运行图像模式
ssh root@192.168.3.207
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --input /userdata/benchmark/bus.jpg \
    --output /userdata/benchmark/result_bus.jpg
```

脚本会打印检测数量、类别、置信度，并写入带绘制框的 `result_bus.jpg`。将图像拉回主机进行目视验证。

### 3. NPU 基准测试（性能测量）

运行摄像头模式以获得稳定的计时结果——无绘制开销，20 次迭代，5 次预热：

```bash
# 在设备上，不指定 --input → 通过 /dev/video13 的摄像头模式
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --runs 20 --warmup 5
```

预期输出：

```
Warmup: 5 iterations
Benchmark: 20 iterations
Avg latency: 36.35 ms
Min latency: 34.21 ms
Max latency: 62.17 ms
FPS: 27.5
```

如需扩展基准测试（30 秒运行，完整矩阵），使用项目级运行器：

```bash
python3 scripts/bench_runner.py /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --duration 30 --warmup 10
```

### 通过 / 失败标准

- **转换通过**：`rknn.export_rknn()` 返回 0，文件大小符合预期。
- **精度通过**：在 `bus.jpg` 上 ONNX 与 RKNN 的余弦相似度 ≥ 0.99；在 conf ≥ 0.3 时检测结果差异不超过 ±1 个目标。
- **性能通过**：延迟在上表数值的 ±5% 以内。

---

## 测试方法论

- **迭代次数**：每模型 20 次运行，5 次预热不计入
- **延迟**：端到端处理时间——预处理、NPU 推理（`rknn.inference`）与后处理（适用时使用 RGA 加速）
- **FPS**：`1000 / avg_latency_ms`
- **INT8 校准**：200 张真实图像（COCO val2017 / ImageNet val），非随机噪声
- **输入类型**：INT8 模型使用 uint8，FP16 模型使用 float16
- **转换**：RKNN-Toolkit2 流水线，`target_platform='rv1126b'`

### YOLO 代际对比

各任务最佳模型（端到端延迟，含预处理/后处理）：

| 任务 | 最佳模型 | FPS | 优势 |
|------|---------|-----|------|
| 目标检测 | yolov8n | 27.5 | 导出成熟、算子融合充分 |
| 姿态估计 | yolov8n-pose | 27.1 | 关键点解码优化 |
| 旋转边界框 | yolo26n-obb | 29.8 | 直接角度回归 |
| 实例分割 | yolo11n-seg | 18.3 | 简化掩码原型 |
| 图像分类 | yolo26n-cls | 152.2 | 轻量骨干网络 |

---

## 许可证

RKNN 模型来源于各自的开源项目，许可证请参阅各模型的原始来源。基准测试脚本和结果采用 MIT 许可证。

---

## 相关资源

- [Seeed reCamera Pro](https://www.seeedstudio.com/reCamera-Pro-2GB.html)
- [Rockchip RKNN 模型库](https://github.com/airockchip/rknn_model_zoo)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [K2 Zipformer](https://github.com/k2-fsa/icefall)
