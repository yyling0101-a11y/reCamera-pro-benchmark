# reCamera Pro RKNN 模型基准测试

[![模型](https://img.shields.io/badge/models-40-blue)](./models/)
[![经过](https://img.shields.io/badge/PASS-34-green)](./benchmark_results/summary.md)
[![设备](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

对 Seeed reCamera Pro（Rockchip RV1126B，2GB RAM，aarch64）进行了全面的 RKNN 模型基准测试。所有模型均转换为 RKNN 格式，并在设备上使用真实的 NPU 推理进行基准测试。

---

## 快速入门

```bash
# Clone the repo (with Git LFS for model weights)
git lfs install
git clone https://github.com/Seeed-Projects/recamera-pro-benchmark.git
cd recamera-pro-benchmark

# All RKNN models are in models/ — ready to deploy
scp models/yolov8n_640x640_W8A8.rknn root@<device-ip>:/userdata/benchmark/models/
```

---

## 存储库结构

```
recamera-pro-benchmark/
├── README.md                         # This file
├── AGENTS.md                         # Detailed benchmark methodology & rules
├── .gitattributes                    # Git LFS config for .rknn files
│
├── models/                           # 🎯 RKNN model weights (Git LFS)
│   ├── *.rknn                        # 40 converted models — ready to deploy
│   └── README.md                     # Model download guide
│
├── benchmark_results/                # 📊 Benchmark data
│   ├── model_matrix.csv              # Full result matrix (CSV)
│   ├── summary.md                    # Executive summary with tables
│   ├── reproduce.md                  # Step-by-step reproduction guide
│   ├── environment.log               # Device & host environment info
│   └── logs/                         # Per-model benchmark logs
│
├── scripts/                          # 🔧 Conversion, deploy & benchmark scripts
│   ├── bench_runner.py               # Single-model NPU benchmark
│   ├── audio_multi_runner.py         # Multi-input audio model benchmark
│   ├── pipeline_runner.py            # End-to-end pipeline benchmark
│   ├── yolo26_bench.py               # YOLOv26-specific benchmark runner
│   ├── convert_yolo26_rknn.py        # RKNN conversion (YOLOv26)
│   ├── convert_all_yolo.py           # RKNN conversion (all YOLO)
│   ├── export_yolo26_onnx.py         # PyTorch → ONNX export
│   ├── download_zoo_models.sh        # Download Rockchip RKNN Zoo ONNX files
│   └── ...
│
├── assets/                           # 🖼️ Test images
│   ├── bus.jpg
│   └── bus_result.jpg
│
└── doc/                              # 📄 Reference documents
    └── Rockchip_Benchmark_NPU_Models_CN.pdf
```

> **笔记：** `artifacts/` 包含中间 PT/ONNX 文件和校准数据。这些文件**未**在 Git 中进行跟踪。请参阅 `scripts/` 有关如何从官方来源复现的问题。

---

## 模型概述

**40 个 RKNN 模型**，涵盖 14 个类别，全部在 RV1126B NPU 上进行基准测试。

### 目标检测
| 模型 | 帧率 | 延迟 | 量化 |
|-------|-----|---------|-------|
| yolov8n | 23.1 | 43.4毫秒 | INT8 |
| yolo11n | 21.1 | 47.5毫秒 | INT8 |
| yolov8s | 15.2 | 65.6毫秒 | INT8 |
| yolo11s | 13.8 | 72.3ms | INT8 |
| yolo26n | 23.6 | 42.3毫秒 | INT8 |
| yolo26s | 14.2 | 70.5ms | INT8 |

### 安装估价
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| yolov8n-pose | 22.1 | 45.2毫秒 |
| yolo11n-pose | 16.2 | 61.5毫秒 |
| yolo26n-pose | 20.3 | 49.2毫秒 |

### 实例分割
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| yolov8n 段 | 15.1 | 15.1 66.1 毫秒 |
| yolo11n-seg | yolo11n-seg | 15.5 | 15.5 64.5 毫秒 |
| yolo26n-seg | yolo26n-seg | 15.4 | 15.4 64.8 毫秒 |

### 定向边界框
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| yolov8n-obb | 27.0 | 27.0 37.0 毫秒 |
| yolo11n-obb | 19.5 | 51.1毫秒 |
| yolo26n-obb | 21.9 | 45.7毫秒 |

### 分类
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| yolo11n-cls | 156.4 | 6.4毫秒 |
| yolo26n-cls | 161.2 | 6.2毫秒 |
| mobilenetv2 | 149.9 | 6.7毫秒 |
| resnet50-v2-7 | 64.2 | 15.6毫秒 |

### 深度估计（YOLOv26）
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| yolo26n-深度 (640²) | 6.2 | 160.2毫秒 |
| yolo26s-深度 (640²) | 5.6 | 177.2毫秒 |
| yolo26n-深度 (320²) | 22.4 | 44.8毫秒 |

### 人脸检测
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| retinaface-mobile320 | 96.9 | 10.3毫秒 |

### OCR
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| ppocrv4_det | 23.3 | 23.3 42.9 毫秒 |
| ppocrv4_rec | 24.8 | 40.2毫秒 |

### 车牌识别
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| lprnet | 83.5 | 12.0毫秒 |

### 语义分割
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| pp_liteseg_cityscapes | 4.5 | 223.4毫秒 |

### 提示式分段
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| mobilesam_encoder_tiny | 4.7 | 214.9毫秒 |

### 视觉嵌入
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| clip-vit-base-patch32-vision | 4.7 | 214.7毫秒 |
| clip-vit-base-patch32-text | 317.1 | 3.1毫秒 |

### 音频：语音转文本
| 模型 | 帧率 | 延迟 | 量化 |
|-------|-----|---------|-------|
| whisper_encoder | 2.4 | 422.0毫秒 | INT8 |
| whisper_decoder | 10.5 | 95.3毫秒 | FP16 |
| wav2vec2 | 0.6 | 1763.2毫秒 | INT8 |
| zipformer_decoder | 172.5 | 5.8毫秒 | FP16 |
| zipformer_joiner | 127.1 | 7.8毫秒 | FP16 |

### 音频分类
| 模型 | 帧率 | 延迟 | 加速比 |
|-------|-----|---------|---------|
| yamnet_3s (INT8) | yamnet_3s (INT8) | 48.4 | 48.4 20.6 毫秒 | 15.6× |
| yamnet_3s (FP16) | 3.1 | 319.6毫秒 | — |

### 文本转语音
| 模型 | 帧率 | 延迟 |
|-------|-----|---------|
| mms_tts_encoder | 8.3 | 120.5毫秒 |
| mms_tts_decoder | 1.7 | 603.6毫秒 |

> 完整结果 [`benchmark_results/summary.md`](./benchmark_results/summary.md) 和 [`benchmark_results/model_matrix.csv`](./benchmark_results/model_matrix.csv)。

---

## 下载 RKNN 模型

所有 RKNN 模型均使用 **Git LFS** 进行跟踪。如果您在克隆时未使用 LFS：

```bash
# Install Git LFS and pull model files
git lfs install
git lfs pull

# Or download individual models directly from GitHub Releases
# (see models/README.md for direct download links)
```

单个模型文件位于 `models/`：

```
models/
├── yolov8n_640x640_W8A8.rknn          (4.2 MB)
├── yolo11n_640x640_W8A8.rknn          (4.2 MB)
├── yolo26n_640x640_W8A8.rknn          (4.7 MB)
├── mobilenetv2_224x224_W8A8.rknn      (3.9 MB)
├── retinaface_mobile320_official_W8A8.rknn  (927 KB)
├── yamnet_3s_official_W8A8.rknn       (5.7 MB)
├── whisper_encoder_base_20s_official_W8A8.rknn (22 MB)
├── ...
└── zipformer_encoder_official_FP16.rknn (108 MB)
```

---

## 重现结果

看 [`benchmark_results/reproduce.md`](./benchmark_results/reproduce.md) 完整的分步指南请点击此处。

快速入门：

```bash
# 1. Set up host environment
cd scripts && source activate_env.sh

# 2. Download Rockchip RKNN Zoo ONNX models
bash download_zoo_models.sh

# 3. Convert ONNX → RKNN (example)
python3 convert_all_yolo.py

# 4. Deploy to device
scp models/*.rknn root@192.168.3.206:/userdata/benchmark/models/
scp scripts/bench_runner.py root@192.168.3.206:/userdata/benchmark/

# 5. Run benchmark on device
ssh root@192.168.3.206
cd /userdata/benchmark
python3 bench_runner.py models/yolov8n_640x640_W8A8.rknn --duration 30 --warmup 10
```

---

## 设备环境

| 组件 | 规格 |
|-----------|------|
| 设备 | Seeed reCamera Pro |
| SoC | 瑞芯微 RV1126B |
| CPU | ARM Cortex-A7，4 核 |
| 内存 | 1.9 GB（总计）/ 约 960 MB（可用） |
| NPU驱动程序 | v0.9.8 |
| RKNN 运行时 | 2.3.2 |
| 内核 | Linux 6.1.157 |
| 主机工具包 | RKNN-Toolkit2 2.3.2 (x86_64) |

---

## 测试方法

- **持续时间：**每个模型 30 秒，10 次迭代预热。
- **帧率：** `1000 / avg_latency_ms`
- **INT8 校准：**来自 COCO val2017、ImageNet val、WIDER Face 和 ICDAR 的真实图像——**而非**随机噪声
- **音频 INT8 校准：** 来自目标域音频的真实梅尔频谱图特征
- **输入数据类型：** INT8 模型使用 uint8，FP16 模型使用 float16（由运行程序自动检测）
- **转换：**标准 RKNN-Toolkit2 流程， `target_platform='rv1126b'`

### INT8 加速发现

| 架构 | 典型加速比 | 示例 |
|-------------|----------------|---------|
| CNN主导 | 10–16倍 | yamnet_3s：15.6倍 |
| CNN + 注意力机制混合 | 3–4倍 | whisper_encoder：3.4倍 |
| 全变压器 | 1.4–2倍 | wav2vec2：1.4倍 |

---

## 主要发现

1. **INT8 量化对于 RV1126B 的实时性能至关重要**——音频模型速度最高可提升 15.6 倍
2. **无NMS架构**（YOLOv26）在各项任务中比YOLOv11性能提升2%至25%。
3. **Rockchip Zoo ONNX**（SiLU→ReLU 替换）与自导出的 ONNX 相比，某些模型的性能提升了 5%–14%。
4. **MobileNetV2 以 150 FPS 的速度保持着分类速度之王的地位。
5. NPU上的**Transformer瓶颈**：注意力操作将INT8增益限制在2倍以下
6. **zipformer 子模型** 在 FP16 下已达到 100+ FPS — 无需 INT8

---

## 已屏蔽模型

模型 | 原因 |
|-------|--------|
| yolo26n-sem / yolo26s-sem | RKNN：RV1126B 不支持 int32 输出数据类型 |
| ppocrv5_det / ppocrv5_rec | PaddleOCR 3.7.0 + Paddle 3.0.0-beta2 内核不兼容 |
| mobilesam_decoder | 动态 `Slice` 具有运行时相关维度的运算符 |

---

## 执照

RKNN 模型均源自各自的开源项目。有关许可条款，请参阅各模型的原始源代码。基准测试脚本和结果采用 MIT 许可。

---

## 相关资源

- [Seed reCamera Pro](https://www.seeedstudio.com/reCamera-Pro-2GB.html)
- [Rockchip RKNN 模型库](https://github.com/airockchip/rknn_model_zoo)
- [RKNN-Toolkit2 文档](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
