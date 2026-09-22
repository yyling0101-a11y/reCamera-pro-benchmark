# YOLO26n Detection - RV1126B

## Model Info
- **Architecture**: YOLO26n (End-to-end, NMS-free, no DFL)
- **Input**: 640x640 RGB
- **Output**: 6 tensors (bbox×3 + cls×3 strides), grouped by branch
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n.pt`)

## Key Architecture Differences from YOLOv8/YOLO11
- **No DFL**: 4-channel direct bbox regression (not 64-channel DFL)
- **No anchor grid offsets**: Direct ltrb distance regression from cell centers
- **End-to-end**: Model includes decode in-graph (but we strip it for NPU optimization)

## ONNX Optimization
The optimized ONNX strips the post-decode nodes and outputs spatial features:
- `bbox_stride_N`: [1, 4, H, W] - direct bbox regression
- `cls_stride_N`: [1, 80, H, W] - sigmoid class scores

## Files
- `yolo26n.pt` - PyTorch weights
- `yolo26n.onnx` - Original ONNX export (opset 12)
- `yolo26n_optimized.onnx` - Optimized ONNX for RKNN
- `yolo26n_640x640_W8A8.rknn` - RKNN INT8 model (4.22 MB)
- `infer.py` - RKNN inference script

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 44.44 ms
- FPS: 22.5
- Detections: 5 (bus:0.88, 4 persons)

## Usage
```bash
python3 infer.py --model yolo26n_640x640_W8A8.rknn --image bus.jpg --output result.jpg
```
