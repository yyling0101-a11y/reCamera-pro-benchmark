# YOLO26n OBB - RV1126B

## Model Info
- **Architecture**: YOLO26n-obb (DOTA dataset, 15 classes)
- **Input**: 640x640 RGB
- **Output**: 9 tensors (cls×3 + bbox×3 + angle×3)
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-obb.pt`)

## ONNX Optimization
- `cls_stride_N`: [1, 15, H, W] (sigmoid, DOTA classes)
- `bbox_stride_N`: [1, 4, H, W]
- `angle_stride_N`: [1, 1, H, W] (needs sigmoid decode)

Angle decode: `(sigmoid(raw) - 0.25) * π` → range [-π/4, 3π/4]

## Files
- `yolo26n-obb.onnx` - Original ONNX export
- `yolo26n-obb_optimized.onnx` - Optimized ONNX
- `yolo26n-obb_640x640_W8A8.rknn` - RKNN INT8 model (4.36 MB)
- `infer.py` - RKNN inference script

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 42.53 ms
- FPS: 23.5

## DOTA Classes
plane, ship, storage-tank, baseball-diamond, tennis-court,
basketball-court, ground-track-field, harbor, bridge,
large-vehicle, small-vehicle, helicopter, roundabout,
soccer-ball-field, swimming-pool

## Usage
```bash
python3 infer.py --model yolo26n-obb_640x640_W8A8.rknn --image bus.jpg --output result.jpg
```
