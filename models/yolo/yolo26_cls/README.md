# YOLO26n Classification - RV1126B

## Model Info
- **Architecture**: YOLO26n-cls (ImageNet 1000 classes)
- **Input**: 224x224 RGB
- **Output**: [1, 1000] classification scores
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-cls.pt`)

## Files
- `yolo26n-cls.onnx` - ONNX export (opset 12)
- `yolo26n-cls_224x224_W8A8.rknn` - RKNN INT8 model (3.65 MB)
- `infer.py` - RKNN inference script

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 6.57 ms
- FPS: 152.2
- Top-5 predictions printed on image

## Usage
```bash
python3 infer.py --model yolo26n-cls_224x224_W8A8.rknn --image bus.jpg --output result.jpg
```
