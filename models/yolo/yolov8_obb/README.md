# YOLOv8n-obb — RV1126B RKNN Benchmark

## Model Info

| Field | Value |
|-------|-------|
| Model | YOLOv8n-obb |
| Source | Rockchip Model Zoo (`yolov8n-obb_zoo.onnx`) |
| Input | 640×640 RGB uint8 |
| Quantization | W8A8 (INT8) |
| Target | RV1126B |
| Calibration | 200 COCO val2017 images |
| Training | DOTA (aerial imagery, 15 classes) |

## Output Format

4 output tensors:
- `[1, 79, 80, 80]` — stride 8: 64 DFL + 15 DOTA class logits
- `[1, 79, 40, 40]` — stride 16
- `[1, 79, 20, 20]` — stride 32
- `[1, 1, 8400]` — angle sigmoid (maps to [0, π/4])

## Post-Processing

1. DFL decode (softmax on 64 channels → 4 distance values)
2. Sigmoid on 15 class logits → confidence
3. Anchor-based bbox decode → xyxy
4. Angle = sigmoid_value × π/4
5. Convert to rotated rectangle (cx, cy, w, h, angle)
6. Rotated NMS (cv2.rotatedRectangleIntersection)

## Results (bus.jpg)

Note: OBB is trained on DOTA aerial imagery, so street-level images produce low confidence detections.

| Metric | Value |
|--------|-------|
| Avg latency | 40.17 ms |
| Min latency | 28.76 ms |
| Max latency | 70.01 ms |
| FPS | 24.9 |
| Model size | 4.26 MB |

At conf=0.25: 0 detections (expected for non-aerial images)
At conf=0.05: 13 "small-vehicle" detections (low confidence)

## Usage

### Convert

```bash
python convert_to_rknn.py
```

### Inference (on device)

```bash
python3 infer.py \
    --model /userdata/benchmark/models/yolo/yolov8_obb/yolov8n-obb_W8A8.rknn \
    --image /userdata/benchmark/test_bus.jpg \
    --output result_bus.jpg \
    --conf 0.05
```
