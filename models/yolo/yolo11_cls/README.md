# YOLO11 Image Classification

## 模型信息
- **模型**: yolo11n-cls
- **任务**: 图像分类 (1000 classes, ImageNet)
- **输入**: RGB 图像, 224×224
- **输出**: 1000类别概率
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n-cls.onnx`: ONNX 格式模型
- `yolo11n-cls_224x224_W8A8.rknn`: RKNN 量化模型
- `infer.py`: 推理脚本
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表
- `result_bus.jpg`: 推理结果示例

## 使用方法

### 推理
```bash
python3 infer.py --model yolo11n-cls_224x224_W8A8.rknn --image test_bus.jpg --output result.jpg --imgsz 224
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- **FPS**: 65.7
- **延迟**: 15.2 ms
