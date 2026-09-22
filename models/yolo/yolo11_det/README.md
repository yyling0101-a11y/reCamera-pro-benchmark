# YOLO11 Detection Models

## 模型信息
- **模型**: yolo11n, yolo11s
- **任务**: 目标检测 (80 classes, COCO)
- **输入**: RGB 图像, 640×640
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n.onnx` / `yolo11s.onnx`: ONNX 格式模型
- `yolo11n_640x640_W8A8.rknn` / `yolo11s_640x640_W8A8.rknn`: RKNN 量化模型
- `infer.py`: 推理脚本
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表
- `result_bus_n.jpg` / `result_bus_s.jpg`: 推理结果示例

## 使用方法

### 推理
```bash
python3 infer.py --model yolo11n_640x640_W8A8.rknn --image test_bus.jpg --output result.jpg
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- **yolo11n**: 20.7 FPS, 48.3 ms
- **yolo11s**: 13.4 FPS, 74.4 ms
