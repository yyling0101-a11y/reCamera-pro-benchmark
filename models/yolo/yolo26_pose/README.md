# YOLO26n Pose - RV1126B

## Model Info
- **Architecture**: YOLO26n-pose
- **Input**: 640x640 RGB
- **Output**: 9 tensors (keypoint×3 + bbox×3 + objectness×3)
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-pose.pt`)

## ONNX Optimization
- `keypoint_stride_N`: [1, 51, H, W] (17 keypoints × 3 values)
- `bbox_stride_N`: [1, 4, H, W]
- `objectness_stride_N`: [1, 1, H, W] (needs sigmoid)

## Files
- `yolo26n-pose.onnx` - Original ONNX export
- `yolo26n-pose_optimized.onnx` - Optimized ONNX
- `yolo26n-pose_640x640_W8A8.rknn` - RKNN INT8 model (4.81 MB)
- `infer.py` - RKNN inference script

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 45.08 ms
- FPS: 22.2
- Detections: 4 persons with 17 keypoints each

## Usage
```bash
python3 infer.py --model yolo26n-pose_640x640_W8A8.rknn --image bus.jpg --output result.jpg
```
