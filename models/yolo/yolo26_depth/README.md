# YOLO26n-Depth - RV1126B 深度估计

## 模型信息
- **架构**: YOLO26n-depth (DepthAnythingV2-based)
- **任务**: 单目深度估计
- **输入**: 640×640 RGB 图像
- **输出**: 160×160 深度图 (NPU) → 上采样至原图尺寸
- **量化**: INT8 W8A8
- **参数量**: 6.36M
- **来源**: ultralytics v8.4.154 (`yolo26n-depth.pt`)

## 性能 (RV1126B, 640×640, 100次推理)
| 指标 | 值 |
|------|-----|
| 平均延迟 | 123.04 ms |
| 最小延迟 | 98.02 ms |
| 最大延迟 | 140.84 ms |
| FPS | 8.1 |
| 模型大小 | 6.7 MB |

## 深度解码
NPU 输出原始深度预测 [1,1,160,160]，需在 CPU 端解码：
```python
depth = exp(clip(raw_depth, -10, 10))
depth = depth ** cal_a  # cal_a = 1.0
depth = depth * exp(cal_b)  # cal_b = -0.1938
```
解码后深度范围约 1.0-13.6 米。

## 文件说明
- `yolo26n-depth.pt` - PyTorch 权重
- `yolo26n-depth.onnx` - 原始 ONNX (opset 12)
- `yolo26n-depth_optimized.onnx` - 优化 ONNX (剥离后处理)
- `yolo26n-depth_640x640_W8A8.rknn` - RKNN INT8 模型
- `infer.py` - RKNN 推理脚本
- `convert_to_rknn.py` - RKNN 转换脚本
- `optimize_onnx.py` - ONNX 优化脚本

## 使用方法
```bash
python3 infer.py \
    --model yolo26n-depth_640x640_W8A8.rknn \
    --image bus.jpg \
    --output result_bus.jpg \
    --max-depth 15.0
```

## 输出格式
- `result_bus.jpg` - 原图与深度图并排对比
- `result_bus_depth.jpg` - 纯深度热力图 (INFERNO colormap)

## 优化说明
ONNX 优化剥离了 10 个后处理节点：
- Clip, Exp, Pow, Mul (深度解码)
- Resize (160×160 → 640×640 上采样)

这些操作在 CPU 端执行，减少 NPU 负担并提高量化精度。
