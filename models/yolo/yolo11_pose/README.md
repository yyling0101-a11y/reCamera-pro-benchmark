# YOLO11 Pose Estimation

## 模型信息
- **模型**: yolo11n-pose
- **任务**: 人体姿态估计 (COCO 17 keypoints)
- **输入**: RGB 图像, 640×640
- **输出**: 检测框 + 17个关键点
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n-pose.onnx`: ONNX 格式模型
- `yolo11n-pose_640x640_W8A8.rknn`: RKNN 量化模型
- `infer.py`: 推理脚本
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表
- `result_bus.jpg`: 推理结果示例

## 使用方法

### 推理
```bash
python3 infer.py --model yolo11n-pose_640x640_W8A8.rknn --image test_bus.jpg --output result.jpg
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- **FPS**: 7.3
- **延迟**: 137 ms
