# PPOCRv4 OCR Pipeline (INT8)

PPOCRv4 文字检测 + 文字识别 完整 OCR 工作流，INT8 (W8A8) 量化，适用于 Seeed reCamera Pro (RV1126B)。

## 模型信息

本目录包含 PPOCRv4 OCR 完整流程的两个模型：

### 检测模型 (Det)
- **来源**: PaddleOCR PP-OCRv4 Detection (DB)
- **ONNX 来源**: RKNN Model Zoo
- **输入**: `[1, 3, 480, 480]` NCHW, uint8
- **输出**: `[1, 1, 480, 480]` 概率图 (DB 分割)
- **归一化**: ImageNet (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
- **模型大小**: 2.55 MB

### 识别模型 (Rec)
- **来源**: PaddleOCR PP-OCRv4 Recognition (SVTR_LCNet)
- **ONNX 来源**: RKNN Model Zoo
- **输入**: `[1, 3, 48, 320]` NCHW, uint8
- **输出**: `[1, T, 6625]` 字符概率 (CTC, T 动态)
- **归一化**: 1/255 (mean=[0,0,0], std=[255,255,255])
- **模型大小**: 4.59 MB
- **字典**: `ppocr_keys_v1.txt` (6624 中英文字符 + 空格 + blank)

## reCamera Pro (RV1126B) 上的性能

| 模式 | 平均延迟 | 最小 | 最大 | FPS |
|------|----------|------|------|-----|
| 图像 (单帧完整流程) | 596.1 ms (16 区域) | - | - | - |
| 摄像头 (全流程) | 58.62 ms | 46.07 ms | 75.49 ms | 17.1 |

- 摄像头基准: 20 次推理, 5 次预热
- 全流程包含: 检测 → 裁剪文字区域 → 逐区域识别 → 后处理
- 检测单模型: ~40ms, 识别单模型: ~24ms
- 总延迟取决于检测到的文字区域数量

## 文件清单

| 文件 | 说明 |
|------|------|
| `ppocrv4_det.onnx` | 检测 ONNX 模型 |
| `ppocrv4_rec.onnx` | 识别 ONNX 模型 |
| `ppocrv4_det_480x480_det.rknn` | 检测 INT8 RKNN 模型 |
| `ppocrv4_rec_48x320_rec.rknn` | 识别 INT8 RKNN 模型 |
| `convert_to_rknn.py` | ONNX → RKNN 转换脚本 (同时转换两个模型) |
| `infer.py` | 设备推理脚本 (Det+Rec 完整流程) |
| `ppocr_keys_v1.txt` | 中文字符字典 |
| `calibration.txt` | 量化校准图片列表 |
| `test_ocr.jpg` | 测试图片 |
| `result_ocr.jpg` | 推理结果图片 (带文字框和识别文字) |

## 使用方法

### 转换 ONNX 到 RKNN

在 x86_64 主机上使用 `recamera-rknn-2.3.2` conda 环境:

```bash
conda run -n recamera-rknn-2.3.2 python convert_to_rknn.py
```

可选参数:
- `--det-onnx` — 检测 ONNX 路径 (默认: `ppocrv4_det.onnx`)
- `--rec-onnx` — 识别 ONNX 路径 (默认: `ppocrv4_rec.onnx`)
- `--calibration` — 校准图片列表 (默认: `calibration.txt`)
- `--det-output` — 检测 RKNN 输出路径 (默认: `ppocrv4_det_480x480_det.rknn`)
- `--rec-output` — 识别 RKNN 输出路径 (默认: `ppocrv4_rec_48x320_rec.rknn`)
- `--target` — 目标平台 (默认: `rv1126b`)

### 设备上运行推理

将 `.rknn`、`ppocr_keys_v1.txt`、`infer.py` 和测试图片推送到设备后:

**图片模式** (检测 → 裁剪 → 识别 → 绘制结果保存):
```bash
python3 infer.py \
    --det-model ppocrv4_det_480x480_det.rknn \
    --rec-model ppocrv4_rec_48x320_rec.rknn \
    --dict ppocr_keys_v1.txt \
    --input test_ocr.jpg \
    --output result_ocr.jpg
```

**摄像头模式** (20 次全流程推理, 5 次预热, 输出平均延迟):
```bash
python3 infer.py \
    --det-model ppocrv4_det_480x480_det.rknn \
    --rec-model ppocrv4_rec_48x320_rec.rknn \
    --dict ppocr_keys_v1.txt
```

可选参数:
- `--runs` — 基准测试次数 (默认: 20)
- `--warmup` — 预热次数 (默认: 5)

摄像头通过 GStreamer 管线读取 `/dev/video13` (1920×1080 NV12)。

## OCR 流程说明

1. **检测 (Det)**: 将整图缩放到 480×480，送入检测模型，输出概率图
2. **DB 后处理**: 二值化 → 轮廓提取 → 最小外接矩形 → 多边形扩展 → 过滤小框
3. **裁剪**: 对每个检测到的文字区域做透视变换，裁剪为矩形
4. **识别 (Rec)**: 将每个裁剪区域缩放到 48×320，送入识别模型
5. **CTC 解码**: 对识别输出做 CTC 去重 → 去除 blank → 映射到字符

## 测试结果示例

在 `test_ocr.jpg` (产品标签图) 上检测到 16 个文字区域，识别结果包括:
- "适用人群: 适合所有肤质"
- "净含量: 220ml"
- "产品编号: YM-X-301"

INT8 量化对中文字符准确度有一定影响，英文和数字识别效果较好。如需更高准确度，可考虑使用 FP16 量化。

## 注意事项

- INT8 识别在 rv1126b 上是实验性的，官方 Model Zoo 推荐 FP16。如果准确度不足可改用 FP16
- 检测模型使用 ImageNet 归一化，识别模型使用 1/255 归一化，两者不同
- 摄像头模式下不做绘制，仅执行推理并报告延迟
- 图片模式下最后一次推理结果会被绘制并保存到设备后拉取到本地

## 更新日志

**2026-09-22**: 优化检测框扩展和裁剪参数
- 检测阈值降低 (0.3→0.2)，框分数阈值降低 (0.6→0.5)
- 多边形扩展比例增大 (1.5→3.0)
- 新增 40% 边距扩展，确保文字区域完整
- 识别准确度显著提升，英文文字识别置信度从 ~0.45-0.55 提升到 0.60-0.70
- 摄像头全流程性能：平均 58.90 ms，17.0 FPS
