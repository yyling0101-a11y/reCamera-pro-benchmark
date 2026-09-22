# reCamera Pro RKNN Model Benchmark

[![Models](https://img.shields.io/badge/models-40-blue)](./models/)
[![PASS](https://img.shields.io/badge/PASS-34-green)](./benchmark_results/summary.md)
[![Device](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

Comprehensive RKNN model benchmarking for the **Seeed reCamera Pro** (Rockchip RV1126B, 2GB RAM, aarch64). All models converted to RKNN format and benchmarked on-device with real NPU inference.

---

## Quick Start

```bash
# Clone the repo (with Git LFS for model weights)
git lfs install
git clone https://github.com/Seeed-Projects/recamera-pro-benchmark.git
cd recamera-pro-benchmark

# All RKNN models are in models/ — ready to deploy
scp models/yolov8n_640x640_W8A8.rknn root@<device-ip>:/userdata/benchmark/models/
```

---

## Repository Structure

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

> **Note:** `artifacts/` contains intermediate PT/ONNX files and calibration data. These are **not** tracked in Git. See `scripts/` for how to reproduce from official sources.

---

## Models Overview

**40 RKNN models** across 14 categories, all benchmarked on RV1126B NPU.

### Object Detection
| Model | FPS | Latency | Quant |
|-------|-----|---------|-------|
| yolov8n | 23.1 | 43.4ms | INT8 |
| yolo11n | 21.1 | 47.5ms | INT8 |
| yolov8s | 15.2 | 65.6ms | INT8 |
| yolo11s | 13.8 | 72.3ms | INT8 |
| yolo26n | 23.6 | 42.3ms | INT8 |
| yolo26s | 14.2 | 70.5ms | INT8 |

### Pose Estimation
| Model | FPS | Latency |
|-------|-----|---------|
| yolov8n-pose | 22.1 | 45.2ms |
| yolo11n-pose | 16.2 | 61.5ms |
| yolo26n-pose | 20.3 | 49.2ms |

### Instance Segmentation
| Model | FPS | Latency |
|-------|-----|---------|
| yolov8n-seg | 15.1 | 66.1ms |
| yolo11n-seg | 15.5 | 64.5ms |
| yolo26n-seg | 15.4 | 64.8ms |

### Oriented Bounding Box
| Model | FPS | Latency |
|-------|-----|---------|
| yolov8n-obb | 27.0 | 37.0ms |
| yolo11n-obb | 19.5 | 51.1ms |
| yolo26n-obb | 21.9 | 45.7ms |

### Classification
| Model | FPS | Latency |
|-------|-----|---------|
| yolo11n-cls | 156.4 | 6.4ms |
| yolo26n-cls | 161.2 | 6.2ms |
| mobilenetv2 | 149.9 | 6.7ms |
| resnet50-v2-7 | 64.2 | 15.6ms |

### Depth Estimation (YOLOv26)
| Model | FPS | Latency |
|-------|-----|---------|
| yolo26n-depth (640²) | 6.2 | 160.2ms |
| yolo26s-depth (640²) | 5.6 | 177.2ms |
| yolo26n-depth (320²) | 22.4 | 44.8ms |

### Face Detection
| Model | FPS | Latency |
|-------|-----|---------|
| retinaface-mobile320 | 96.9 | 10.3ms |

### OCR
| Model | FPS | Latency |
|-------|-----|---------|
| ppocrv4_det | 23.3 | 42.9ms |
| ppocrv4_rec | 24.8 | 40.2ms |

### License Plate Recognition
| Model | FPS | Latency |
|-------|-----|---------|
| lprnet | 83.5 | 12.0ms |

### Semantic Segmentation
| Model | FPS | Latency |
|-------|-----|---------|
| pp_liteseg_cityscapes | 4.5 | 223.4ms |

### Prompted Segmentation
| Model | FPS | Latency |
|-------|-----|---------|
| mobilesam_encoder_tiny | 4.7 | 214.9ms |

### Vision Embeddings
| Model | FPS | Latency |
|-------|-----|---------|
| clip-vit-base-patch32-vision | 4.7 | 214.7ms |
| clip-vit-base-patch32-text | 317.1 | 3.1ms |

### Audio: Speech-to-Text
| Model | FPS | Latency | Quant |
|-------|-----|---------|-------|
| whisper_encoder | 2.4 | 422.0ms | INT8 |
| whisper_decoder | 10.5 | 95.3ms | FP16 |
| wav2vec2 | 0.6 | 1763.2ms | INT8 |
| zipformer_decoder | 172.5 | 5.8ms | FP16 |
| zipformer_joiner | 127.1 | 7.8ms | FP16 |

### Audio Classification
| Model | FPS | Latency | Speedup |
|-------|-----|---------|---------|
| yamnet_3s (INT8) | 48.4 | 20.6ms | 15.6× |
| yamnet_3s (FP16) | 3.1 | 319.6ms | — |

### Text-to-Speech
| Model | FPS | Latency |
|-------|-----|---------|
| mms_tts_encoder | 8.3 | 120.5ms |
| mms_tts_decoder | 1.7 | 603.6ms |

> Full results in [`benchmark_results/summary.md`](./benchmark_results/summary.md) and [`benchmark_results/model_matrix.csv`](./benchmark_results/model_matrix.csv).

---

## Downloading RKNN Models

All RKNN models are tracked with **Git LFS**. If you cloned without LFS:

```bash
# Install Git LFS and pull model files
git lfs install
git lfs pull

# Or download individual models directly from GitHub Releases
# (see models/README.md for direct download links)
```

Individual model files are in `models/`:

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

## Reproducing Results

See [`benchmark_results/reproduce.md`](./benchmark_results/reproduce.md) for the full step-by-step guide.

Quick start:

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

## Device Environment

| Component | Spec |
|-----------|------|
| Device | Seeed reCamera Pro |
| SoC | Rockchip RV1126B |
| CPU | ARM Cortex-A7, 4 cores |
| RAM | 1.9 GB (total) / ~960 MB (free) |
| NPU Driver | v0.9.8 |
| RKNN Runtime | 2.3.2 |
| Kernel | Linux 6.1.157 |
| Host Toolkit | RKNN-Toolkit2 2.3.2 (x86_64) |

---

## Methodology

- **Duration:** 30 seconds per model, 10-iteration warmup
- **FPS:** `1000 / avg_latency_ms`
- **INT8 Calibration:** Real images from COCO val2017, ImageNet val, WIDER Face, ICDAR — **not** random noise
- **Audio INT8 Calibration:** Real mel-spectrogram features from target-domain audio
- **Input dtype:** uint8 for INT8 models, float16 for FP16 models (auto-detected by runner)
- **Conversion:** Standard RKNN-Toolkit2 pipeline, `target_platform='rv1126b'`

### INT8 Speedup Findings

| Architecture | Typical Speedup | Example |
|-------------|----------------|---------|
| CNN-dominated | 10–16× | yamnet_3s: 15.6× |
| CNN + Attention hybrid | 3–4× | whisper_encoder: 3.4× |
| Full Transformer | 1.4–2× | wav2vec2: 1.4× |

---

## Key Findings

1. **INT8 quantization is critical** for real-time performance on RV1126B — up to 15.6× speedup for audio models
2. **NMS-free architectures** (YOLOv26) deliver +2–25% vs YOLOv11 across tasks
3. **Rockchip Zoo ONNX** (SiLU→ReLU substitution) gives +5–14% for some models vs self-exported ONNX
4. **MobileNetV2 at 150 FPS** remains the classification speed king
5. **Transformer bottlenecks** on NPU: attention ops limit INT8 gains to <2×
6. **zipformer sub-models** already hit 100+ FPS in FP16 — INT8 not needed

---

## BLOCKED Models

| Model | Reason |
|-------|--------|
| yolo26n-sem / yolo26s-sem | RKNN: int32 output dtype not supported on RV1126B |
| ppocrv5_det / ppocrv5_rec | PaddleOCR 3.7.0 + Paddle 3.0.0-beta2 kernel incompatibility |
| mobilesam_decoder | Dynamic `Slice` operator with runtime-dependent dimensions |

---

## License

RKNN models are derived from their respective open-source projects. See each model's original source for license terms. Benchmark scripts and results are MIT-licensed.

---

## Related Resources

- [Seeed reCamera Pro](https://www.seeedstudio.com/reCamera-Pro-2GB.html)
- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [RKNN-Toolkit2 Documentation](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
