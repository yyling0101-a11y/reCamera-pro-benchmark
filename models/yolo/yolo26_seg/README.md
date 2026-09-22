# YOLO26n Segmentation - RV1126B

## Model Info
- **Architecture**: YOLO26n-seg
- **Input**: 640x640 RGB
- **Output**: 10 tensors (bbox×3 + cls×3 + mask_coeff×3 + proto)
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-seg.pt`)

## ONNX Optimization
Strips decode, outputs spatial features:
- `bbox_stride_N`: [1, 4, H, W]
- `cls_stride_N`: [1, 80, H, W] (sigmoid)
- `mask_coeff_stride_N`: [1, 32, H, W]
- `proto`: [1, 32, 160, 160] (mask prototype)

## Files
- `yolo26n-seg.onnx` - Original ONNX export
- `yolo26n-seg_optimized.onnx` - Optimized ONNX
- `yolo26n-seg_640x640_W8A8.rknn` - RKNN INT8 model (4.76 MB)
- `infer.py` - RKNN inference script

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 67.25 ms
- FPS: 14.9
- Detections: 5 (4 persons, bus with segmentation masks)

## Usage
```bash
python3 infer.py --model yolo26n-seg_640x640_W8A8.rknn --image bus.jpg --output result.jpg
```
