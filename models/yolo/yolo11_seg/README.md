# YOLO11 Instance Segmentation

## 模型信息
- **模型**: yolo11n-seg
- **任务**: 实例分割 (80 classes, COCO)
- **输入**: RGB 图像, 640×640
- **输出**: 检测框 + 分割掩码 (160×160)
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n-seg.onnx`: ONNX 格式模型
- `yolo11n-seg_640x640_W8A8.rknn`: RKNN 量化模型
- `infer.py`: 推理脚本
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表
- `result_bus.jpg`: 推理结果示例

## 使用方法

### 推理
```bash
python3 infer.py --model yolo11n-seg_640x640_W8A8.rknn --image test_bus.jpg --output result.jpg
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- **FPS**: 6.6
- **延迟**: 151 ms
