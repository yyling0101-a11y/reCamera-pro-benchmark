# YOLOv8 Detection Models for reCamera Pro (RV1126B)

## Models

### YOLOv8n (Nano)
- **Model**: `yolov8n_W8A8.rknn` (4.2 MB)
- **Source**: Rockchip Model Zoo (`yolov8n.onnx`)
- **Quantization**: INT8 W8A8
- **Input**: 640x640, RGB, uint8
- **Performance**: 48.5ms / 20.6 FPS
- **Test Result**: 4 detections (3 persons + 1 bus)

### YOLOv8s (Small)
- **Model**: `yolov8s_W8A8.rknn` (12 MB)
- **Source**: Rockchip Model Zoo (`yolov8s.onnx`)
- **Quantization**: INT8 W8A8
- **Input**: 640x640, RGB, uint8
- **Performance**: 69.1ms / 14.5 FPS
- **Test Result**: 5 detections (4 persons + 1 bus)

## Files

- `*.onnx` - Original ONNX models from Rockchip Zoo
- `*_W8A8.rknn` - Converted INT8 RKNN models for RV1126B
- `convert_to_rknn.py` - Conversion script
- `infer.py` - Inference script with DFL post-processing
- `calibration_200.txt` - Calibration image list (200 COCO images)
- `result_bus.jpg` - YOLOv8n inference result
- `result_bus_s.jpg` - YOLOv8s inference result

## Usage

### Convert ONNX to RKNN
```bash
python convert_to_rknn.py \
  --model yolov8n.onnx \
  --calibration calibration_200.txt \
  --output yolov8n_W8A8.rknn \
  --target rv1126b
```

### Run Inference on Device
```bash
python infer.py \
  --model yolov8n_W8A8.rknn \
  --image /path/to/test.jpg \
  --output result.jpg \
  --conf 0.25 \
  --nms 0.45 \
  --runs 10 \
  --warmup 3
```

## Post-processing

The YOLOv8 Zoo models output raw feature maps (9 outputs for 3 scales):
- Bounding box features (64 channels, DFL with reg_max=16)
- Class scores (80 classes, sigmoid applied)
- Objectness scores (1 channel, sigmoid applied)

The inference script performs:
1. DFL decode (softmax + weighted sum)
2. Anchor generation for 3 scales (8, 16, 32)
3. Bounding box decoding
4. Confidence filtering (objectness × max_class_score)
5. Non-maximum suppression (NMS)

## Performance Comparison

| Model | Size | FPS | Latency | Detections |
|-------|------|-----|---------|------------|
| YOLOv8n | 4.2 MB | 20.6 | 48.5ms | 4 |
| YOLOv8s | 12 MB | 14.5 | 69.1ms | 5 |

## Notes

- Models use Rockchip Zoo preprocessing: mean=[0,0,0], std=[255,255,255]
- Input should be uint8 [0-255], normalized internally
- Calibration used 200 COCO val2017 images
- Tested on reCamera Pro (RV1126B) with NPU
