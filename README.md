# reCamera Pro RKNN Model Benchmark

[![Models](https://img.shields.io/badge/models-25-blue)](./models/)
[![Device](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

Comprehensive RKNN model benchmarking for the **Seeed reCamera Pro** (Rockchip RV1126B, 2 GB RAM, aarch64). All models converted to RKNN format and benchmarked on-device with real NPU inference.

---

## Quick Start

```bash
# Clone the repo (with Git LFS for model weights)
git lfs install
git clone https://github.com/Seeed-Projects/recamera-pro-benchmark.git
cd recamera-pro-benchmark

# Deploy any model to the device
scp models/yolo/yolov8_det/yolov8n_640x640_W8A8.rknn root@192.168.3.207:/userdata/benchmark/
scp models/yolo/yolov8_det/infer.py root@192.168.3.207:/userdata/benchmark/
```

---

## Repository Structure

```
recamera-pro-benchmark/
├── README.md                           # This file
├── AGENTS.md                           # Detailed benchmark methodology & rules
│
├── models/                             # All models organized by family
│   ├── yolo/                           # YOLO family (v8, 11, 26)
│   │   ├── tools/                      # ONNX optimization scripts
│   │   │   ├── optimize_yolo_onnx.py   # YOLOv8/YOLO11 ONNX optimizer
│   │   │   └── optimize_yolo26_onnx.py # YOLO26 ONNX optimizer
│   │   ├── yolov8_det/                 # YOLOv8 Detection (n, s)
│   │   ├── yolov8_pose/                # YOLOv8 Pose Estimation
│   │   ├── yolov8_obb/                 # YOLOv8 Oriented Bounding Box
│   │   ├── yolov8_seg/                 # YOLOv8 Instance Segmentation
│   │   ├── yolo11_det/                 # YOLO11 Detection (n, s)
│   │   ├── yolo11_pose/               # YOLO11 Pose Estimation
│   │   ├── yolo11_obb/                # YOLO11 Oriented Bounding Box
│   │   ├── yolo11_seg/                # YOLO11 Instance Segmentation
│   │   ├── yolo11_cls/                # YOLO11 Classification
│   │   ├── yolo26_det/                # YOLO26 Detection
│   │   ├── yolo26_pose/               # YOLO26 Pose Estimation
│   │   ├── yolo26_obb/                # YOLO26 Oriented Bounding Box
│   │   ├── yolo26_seg/                # YOLO26 Instance Segmentation
│   │   ├── yolo26_cls/                # YOLO26 Classification
│   │   └── yolo26_depth/              # YOLO26 Depth Estimation
│   │
│   ├── ppocr/                          # OCR pipeline
│   │   └── ppocrv4/                    # PPOCRv4 Det + Rec (INT8)
│   │
│   └── zipformer/                      # Speech-to-Text
│       └── (encoder + decoder + joiner, FP16)
│
├── benchmark_results/                  # Benchmark data & logs
├── scripts/                            # Conversion & benchmark scripts
├── assets/                             # Test images
├── calibration/                        # Calibration datasets
└── doc/                                # Reference documents
```

Each model sub-directory is self-contained with:
- `*.onnx` — Original ONNX model
- `*_optimized.onnx` — Optimized ONNX for RKNN (where applicable)
- `*_W8A8.rknn` / `*_FP16.rknn` — Converted RKNN model
- `convert_to_rknn.py` — ONNX → RKNN conversion script
- `infer.py` — Device inference script (image + camera modes)
- `calibration_200.txt` — Calibration image list
- `result_*.jpg` — Sample inference results
- `README.md` — Model-specific documentation

---

## Models Overview

**25 RKNN models** across 3 families and 9 task categories, benchmarked on RV1126B NPU.

### Object Detection

| Model | Input | Precision | Latency (ms) | FPS | Power (W) |
|-------|-------|-----------|--------------|-----|-----------|
| YOLO26n | 640×640 | INT8 | 37.48 | 26.7 | 5.2 |
| YOLO11n | 640×640 | INT8 | 38.70 | 25.8 | 5.2 |
| YOLO11s | 640×640 | INT8 | 58.50 | 17.1 | 5.2 |
| YOLOv8n | 640×640 | INT8 | 36.35 | 27.5 | 5.1 |
| YOLOv8s | 640×640 | INT8 | 37.25 | 26.8 | 5.1 |

### Pose Estimation

| Model | Input | Precision | Latency (ms) | FPS | Power (W) |
|-------|-------|-----------|--------------|-----|-----------|
| YOLO26n-pose | 640×640 | INT8 | 37.31 | 26.8 | 5.2 |
| YOLO11n-pose | 640×640 | INT8 | 38.19 | 26.2 | 5.6 |
| YOLOv8n-pose | 640×640 | INT8 | 36.92 | 27.1 | 5.3 |

### Instance Segmentation

| Model | Input | Precision | Latency (ms) | FPS | Power (W) |
|-------|-------|-----------|--------------|-----|-----------|
| YOLO26n-seg | 640×640 | INT8 | 55.11 | 18.1 | 5.4 |
| YOLO11n-seg | 640×640 | INT8 | 54.59 | 18.3 | 5.4 |
| YOLOv8n-seg | 640×640 | INT8 | 61.89 | 16.2 | 5.4 |

### Oriented Bounding Box (OBB)

| Model | Input | Precision | Latency (ms) | FPS | Power (W) |
|-------|-------|-----------|--------------|-----|-----------|
| YOLO26n-obb | 640×640 | INT8 | 33.57 | 29.8 | 5.2 |
| YOLO11n-obb | 640×640 | INT8 | 34.40 | 29.1 | 5.3 |
| YOLOv8n-obb | 640×640 | INT8 | 35.56 | 28.1 | 5.3 |

### Classification

| Model | FPS | Latency | Size |
|-------|-----|---------|------|
| yolo26n-cls | 152.2 | 6.6 ms | 3.65 MB |
| yolo11n-cls | 65.7 | 15.2 ms | — |

### Depth Estimation

| Model | Input | Precision | Latency (ms) | FPS | Power (W) |
|-------|-------|-----------|--------------|-----|-----------|
| YOLO26n-depth | 640×640 | INT8 | 99.16 | 10.1 | 5.2 |

### OCR (PPOCRv4)

| Component | Input | Precision | Latency (ms)* | FPS* | Power (W) |
|-----------|-------|-----------|---------------|------|-----------|
| ppocrv4_det | 480×480 | INT8 | 58.62 | 17.1 | 5.0 |
| ppocrv4_rec | 48×320 | INT8 | (combined) | (combined) | |

\* OCR latency/FPS are the combined detection + recognition pipeline; speed varies with text content.

### Speech-to-Text (Zipformer)

| Component | Input | Precision | Latency (ms) | FPS | Power (W) |
|-----------|-------|-----------|--------------|-----|-----------|
| zipformer_encoder | 103×80 | FP16 | 1834.9 (group total) | / | 5.2 |
| zipformer_decoder | 1×2 | FP16 | | / | |
| zipformer_joiner | 1×512 | FP16 | | / | |

| Audio | Total Latency | RTF |
|-------|--------------|-----|
| 5.61 s test.wav | 1834.9 ms | 0.33 |
| 5.00 s microphone | 1580.1 ms | 0.32 |

RTF < 1.0 means real-time capable (inference faster than audio duration).

---

## Per-Model Directory Usage

Each model directory contains everything needed for conversion and deployment:

### Convert ONNX → RKNN (on x86_64 host)

```bash
cd models/yolo/yolov8_det
python3 convert_to_rknn.py
```

Uses RKNN-Toolkit2 with `target_platform='rv1126b'` and 200-image calibration.

### Deploy & Run (on device)

```bash
# Push model + script to device
scp models/yolo/yolov8_det/yolov8n_640x640_W8A8.rknn root@192.168.3.207:/userdata/benchmark/
scp models/yolo/yolov8_det/infer.py root@192.168.3.207:/userdata/benchmark/

# Image inference (saves annotated result)
ssh root@192.168.3.207
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --image /userdata/benchmark/bus.jpg \
    --output /userdata/benchmark/result_bus.jpg

# Camera inference (20 runs, 5 warmup, reports avg latency)
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn
```

### infer.py Modes

- **Image mode** (`--input` specified): Runs inference, draws results on the last frame, saves to device, and pulls back to host.
- **Camera mode** (no `--input`): Reads from `/dev/video13` (1920×1080 NV12), runs 20 iterations with 5 warmup, reports average inference speed. No frame drawing.

---

## ONNX Optimization

YOLO models exported from PyTorch (`*.pt`) produce ONNX graphs that include post-processing decode operations not well-suited for NPU execution. The optimization scripts in `models/yolo/tools/` strip these CPU-bound operations:

```bash
# Optimize YOLOv8/YOLO11 ONNX
python3 models/yolo/tools/optimize_yolo_onnx.py --model yolov8n.onnx --output yolov8n_optimized.onnx

# Optimize YOLO26 ONNX
python3 models/yolo/tools/optimize_yolo26_onnx.py --model yolo26n.onnx --output yolo26n_optimized.onnx
```

Key optimizations:
- Strip DFL decode, anchor generation, NMS, and sigmoid operations
- Output raw spatial feature maps for NPU-accelerated inference
- Post-processing runs on CPU in the inference script

---

## Device Environment

| Component | Spec |
|-----------|------|
| Device | Seeed reCamera Pro |
| SoC | Rockchip RV1126B |
| CPU | ARM Cortex-A7, 4 cores |
| RAM | 1.9 GB (total) |
| NPU | 2.0 TOPS |
| RKNN Runtime | 2.3.2 |
| Kernel | Linux 6.1.x |
| Host Toolkit | RKNN-Toolkit2 2.3.2 (x86_64) |

---

## Host Environment & Toolchain

Before converting or benchmarking models, the host must have the following toolchain installed.

| Component | Version | Purpose |
|-----------|---------|---------|
| Host OS | Ubuntu 22.04 (x86_64) / WSL2 | Build & conversion host |
| Python | 3.10 | Runtime for conversion & inference scripts |
| Conda env | `recamera-rknn-2.3.2` (miniforge3) | ONNX → RKNN conversion |
| RKNN-Toolkit2 | **2.3.2** | Model conversion, `target_platform='rv1126b'` |
| ONNX | ≥ 1.14 | ONNX graph inspection & optimization |
| ONNX Runtime | ≥ 1.16 | CPU-side accuracy comparison |
| NumPy | 1.24 | Tensor math |
| OpenCV | 4.x | Image I/O, drawing, NMS |
| PyTorch (optional) | 2.x | Export YOLO `.pt` → `.onnx` via ultralytics |
| Ultralytics | ≥ 8.4 | YOLO export & reference post-processing |
| PaddlePaddle + Paddle2ONNX | optional | PPOCRv4 export pipeline |

The conversion environment is activated with:

```bash
# Miniforge-based (recommended, used by recamera-rknn-dev skill)
conda activate recamera-rknn-2.3.2

# Or Miniconda-based (legacy benchmark environment)
eval "$(/home/seeed/miniconda3/bin/conda shell.bash hook)"
conda activate rknn_bench
```

> **Contract**: All models in this repo are converted with **RKNN-Toolkit2 2.3.2** targeting `rv1126b`. The device runtime (`librknnrt.so`) must match the toolkit version — mixing versions will fail at `load_rknn()`.

---

## Testing & Validation

There are three levels of testing, from quick sanity to full NPU benchmark.

### 1. Conversion Sanity (host, no device)

After `convert_to_rknn.py`, verify the `.rknn` was produced correctly:

```bash
# File exists and has reasonable size (e.g., ~4 MB for YOLO nano)
ls -lh yolov8n_640x640_W8A8.rknn

# Optional: compare ONNX vs RKNN outputs on the Toolkit simulator
python3 scripts/compare_onnx_rknn.py \
    --onnx yolov8n_optimized.onnx \
    --rknn yolov8n_640x640_W8A8.rknn \
    --image assets/bus.jpg
```

### 2. On-device Inference Test (functional correctness)

Deploy the model and run a single image inference to validate detection quality:

```bash
# Push to device
scp yolov8n_640x640_W8A8.rknn infer.py bus.jpg root@192.168.3.207:/userdata/benchmark/

# SSH to device and run image mode
ssh root@192.168.3.207
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --input /userdata/benchmark/bus.jpg \
    --output /userdata/benchmark/result_bus.jpg
```

The script prints detection count, classes, confidence, and writes `result_bus.jpg` with drawn boxes. Pull the image back to the host to visually verify.

### 3. NPU Benchmark (performance measurement)

Run camera mode for stable timing — no drawing overhead, 20 iterations with 5 warmup:

```bash
# On device, no --input → camera mode via /dev/video13
python3 /userdata/benchmark/infer.py \
    --model /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --runs 20 --warmup 5
```

Expected output:

```
Warmup: 5 iterations
Benchmark: 20 iterations
Avg latency: 36.35 ms
Min latency: 34.21 ms
Max latency: 62.17 ms
FPS: 27.5
```

For extended benchmarks (30 s runs, full matrix), use the project-level runner:

```bash
python3 scripts/bench_runner.py /userdata/benchmark/yolov8n_640x640_W8A8.rknn \
    --duration 30 --warmup 10
```

### Pass / Fail Criteria

- **Conversion pass**: `rknn.export_rknn()` returns 0 and file size matches expectation.
- **Accuracy pass**: ONNX vs RKNN cosine similarity ≥ 0.99 on `bus.jpg`; detections match within ±1 object at conf ≥ 0.3.
- **Performance pass**: Latency within ±5 % of the numbers in the tables above.

---

## Methodology

- **Iterations**: 20 runs per model, 5 warmup excluded
- **Latency**: end-to-end processing time — preprocessing, NPU inference (`rknn.inference`), and post-processing (RGA-assisted where applicable)
- **FPS**: `1000 / avg_latency_ms`
- **INT8 Calibration**: 200 real images from COCO val2017 / ImageNet val — not random noise
- **Input dtype**: uint8 for INT8 models, float16 for FP16 models
- **Conversion**: RKNN-Toolkit2 pipeline, `target_platform='rv1126b'`

### YOLO Generation Comparison

Best-performing model per task (end-to-end latency, including pre/post-processing):

| Task | Best Model | FPS | Advantage |
|------|-----------|-----|-----------|
| Detection | yolov8n | 27.5 | Mature, well-fused export |
| Pose | yolov8n-pose | 27.1 | Optimized keypoint decode |
| OBB | yolo26n-obb | 29.8 | Direct angle regression |
| Segmentation | yolo11n-seg | 18.3 | Simplified mask proto |
| Classification | yolo26n-cls | 152.2 | Lightweight backbone |

---

## License

RKNN models are derived from their respective open-source projects. See each model's original source for license terms. Benchmark scripts and results are MIT-licensed.

---

## Related Resources

- [Seeed reCamera Pro](https://www.seeedstudio.com/reCamera-Pro-2GB.html)
- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [K2 Zipformer](https://github.com/k2-fsa/icefall)
