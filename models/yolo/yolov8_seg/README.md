# YOLOv8n-seg — RV1126B RKNN Benchmark

## Model Info

| Field | Value |
|-------|-------|
| Model | YOLOv8n-seg |
| Source | Rockchip Model Zoo (`yolov8n-seg_zoo.onnx`) |
| Input | 640×640 RGB uint8 |
| Quantization | W8A8 (INT8) |
| Target | RV1126B |
| Calibration | 200 COCO val2017 images |

## Output Format

13 output tensors (grouped by stride):

Per stride (8, 16, 32):
- bbox DFL: `[1, 64, H, W]` — needs DFL softmax decode
- cls sigmoid: `[1, 80, H, W]` — per-class probabilities (sigmoid in ONNX)
- cls sum: `[1, 1, H, W]` — confidence = sum of sigmoid scores (ReduceSum + Clip)
- mask coeffs: `[1, 32, H, W]` — mask weights

Plus proto: `[1, 32, 160, 160]` — proto mask feature map

## Post-Processing

1. DFL decode (softmax on 64 channels → 4 distance values)
2. Anchor-based bbox decode (distances → xyxy)
3. Confidence from cls_sum output (already computed in ONNX)
4. Class = argmax of 80 sigmoid class scores
5. Mask = sigmoid(mask_coeffs @ proto)
6. NMS on bboxes

## Results (bus.jpg)

| Metric | Value |
|--------|-------|
| Avg latency | 77.88 ms |
| Min latency | 61.37 ms |
| Max latency | 96.58 ms |
| FPS | 12.8 |
| Model size | 4.61 MB |

### Detection Comparison

| # | Ultralytics FP32 | RKNN INT8 |
|---|-----------------|-----------|
| 1 | person 0.888 | person 0.871 |
| 2 | person 0.852 | bus 0.847 |
| 3 | person 0.842 | person 0.839 |
| 4 | bus 0.807 | person 0.804 |
| 5 | person 0.343 | skateboard 0.523 |
| 6 | skateboard 0.335 | person 0.296 |

6/6 objects detected. Same classes, similar confidence. Mask pixel counts differ due to resolution scaling.

## Usage

### Convert

```bash
python convert_to_rknn.py
```

### Inference (on device)

```bash
python3 infer.py \
    --model /userdata/benchmark/models/yolo/yolov8_seg/yolov8n-seg_W8A8.rknn \
    --image /userdata/benchmark/test_bus.jpg \
    --output result_bus.jpg
```
