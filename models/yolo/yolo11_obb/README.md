# YOLO11 Oriented Bounding Box (OBB)

## 模型信息
- **模型**: yolo11n-obb
- **任务**: 旋转边界框检测 (15 classes, DOTA aerial)
- **输入**: RGB 图像, 640×640
- **输出**: 旋转边界框 + 类别
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n-obb.onnx`: ONNX 格式模型
- `yolo11n-obb_640x640_W8A8.rknn`: RKNN 量化模型
- `infer.py`: 推理脚本
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表

## 使用方法

### 推理
```bash
python3 infer.py --model yolo11n-obb_640x640_W8A8.rknn --image test_bus.jpg --output result.jpg
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- 待测试
