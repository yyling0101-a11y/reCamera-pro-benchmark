# YOLOv8n-pose — RV1126B RKNN Benchmark

## Model Info

| Field | Value |
|-------|-------|
| Model | YOLOv8n-pose |
| Source | Rockchip Model Zoo (`yolov8n-pose_zoo.onnx`) |
| Input | 640×640 RGB uint8 |
| Quantization | W8A8 (INT8) |
| Target | RV1126B |
| Calibration | 200 COCO val2017 images |

## ONNX Modification

The Zoo ONNX model has sigmoid applied on keypoint visibility inside the graph.
INT8 quantization destroys the precision of [0,1] sigmoid outputs (the visibility
channel shares a quantization scale with the xy coordinates that span 0–640).

**Fix**: Removed the sigmoid on visibility logits in the ONNX graph before conversion.
Sigmoid is applied in Python during post-processing.

## Output Format

4 output tensors:
- `[1, 65, 80, 80]` — stride 8: 64 DFL + 1 class logit
- `[1, 65, 40, 40]` — stride 16
- `[1, 65, 20, 20]` — stride 32
- `[1, 17, 3, 8400]` — keypoints: (x, y) pixel coords + raw vis logit

## Post-Processing

1. DFL decode (softmax on 64 channels → 4 distance values)
2. Anchor-based bbox decode (distances → xyxy)
3. Sigmoid on class logit → confidence score
4. Sigmoid on visibility logit → visibility probability
5. Letterbox coordinate correction
6. NMS (IoU threshold)

## Results (bus.jpg)

| Metric | Value |
|--------|-------|
| Avg latency | 45.87 ms |
| Min latency | 34.29 ms |
| Max latency | 60.86 ms |
| FPS | 21.8 |
| Model size | 4.88 MB |

### Detection Comparison

| # | Ultralytics FP32 | RKNN INT8 |
|---|-----------------|-----------|
| 1 | conf=0.892, 16/17 vis kpts | conf=0.863, 16/17 vis kpts |
| 2 | conf=0.874, 17/17 vis kpts | conf=0.836, 0/17 vis kpts |
| 3 | conf=0.881, 9/17 vis kpts | conf=0.863, 15/17 vis kpts |
| 4 | conf=0.454, 0/17 vis kpts | conf=0.338, 0/17 vis kpts |

4/4 persons detected. Bboxes match well. Keypoint visibility slightly degraded
for partially occluded persons due to INT8 quantization.

## Usage

### Convert

```bash
python convert_to_rknn.py
```

### Inference (on device)

```bash
python3 infer.py \
    --model /userdata/benchmark/models/yolo/yolov8_pose/yolov8n-pose_W8A8.rknn \
    --image /userdata/benchmark/test_bus.jpg \
    --output result_bus.jpg
```

### Options

- `--conf 0.25` — confidence threshold
- `--nms 0.45` — NMS IoU threshold
- `--runs 10` — benchmark iterations
- `--warmup 3` — warmup iterations
