# PPOCRv4 Detection (INT8)

PPOCRv4 text detection model converted to RKNN INT8 (W8A8) for Seeed reCamera Pro (RV1126B).

## Model Info

- **Source**: PaddleOCR PP-OCRv4 Detection
- **ONNX Source**: RKNN Model Zoo (`ppocrv4_det.onnx`)
- **Input**: `[1, 3, 480, 480]` NCHW, uint8 [0-255]
- **Output**: `[1, 1, 480, 480]` probability map (DB segmentation)
- **Quantization**: INT8 W8A8
- **Normalization**: ImageNet (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
- **Model Size**: 2.55 MB

## Performance on reCamera Pro (RV1126B)

| Mode | Avg Latency | Min | Max | FPS |
|------|-------------|-----|-----|-----|
| Image (single) | 36.36 ms | - | - | ~27.5 |
| Camera (video13) | 40.58 ms | 33.48 ms | 59.94 ms | 24.6 |

Camera benchmark: 20 runs, 5 warmup iterations. Latency is NPU inference only (excludes pre/post-processing).

## Files

| File | Description |
|------|-------------|
| `ppocrv4_det.onnx` | Original ONNX model |
| `ppocrv4_det_480x480_det.rknn` | Converted INT8 RKNN model |
| `convert_to_rknn.py` | ONNX → RKNN conversion script |
| `infer.py` | On-device inference script |
| `calibration.txt` | Calibration image list (for INT8) |
| `test_ocr.jpg` | Test image with text |
| `result_ocr.jpg` | Inference result with bounding boxes |

## Usage

### Convert ONNX to RKNN

Run on x86_64 host with `recamera-rknn-2.3.2` conda environment:

```bash
conda run -n recamera-rknn-2.3.2 python convert_to_rknn.py
```

Optional arguments:
- `--model` — ONNX model path (default: `ppocrv4_det.onnx`)
- `--calibration` — Calibration image list (default: `calibration.txt`)
- `--output` — Output RKNN path (default: `ppocrv4_det_480x480_det.rknn`)
- `--target` — Target platform (default: `rv1126b`)

### Run Inference on Device

Push `.rknn`, `infer.py`, and test image to the device, then:

**Image mode** (runs inference and saves result with bounding boxes):
```bash
python3 infer.py --model ppocrv4_det_480x480_det.rknn --input test_ocr.jpg --output result_ocr.jpg
```

**Camera mode** (20 runs, 5 warmup, reports average latency):
```bash
python3 infer.py --model ppocrv4_det_480x480_det.rknn
```

Optional arguments:
- `--runs` — Number of benchmark runs (default: 20)
- `--warmup` — Warmup iterations (default: 5)

Camera input uses `/dev/video13` via GStreamer pipeline (1920×1080 NV12).

## Post-processing

Detection output uses DB (Differentiable Binarization) post-processing:
1. Threshold binarization on the probability map
2. Contour finding and filtering
3. Minimum bounding rectangle extraction
4. Polygon expansion (unclip)
5. Size filtering (min width/height > 3px)

Results are drawn as green polygons on the output image.

## Notes

- This is the **detection-only** model. For full OCR, combine with `ppocrv4_rec` (recognition).
- INT8 quantization uses 20 calibration images from the PPOCR dataset.
- The detection model outputs text region bounding boxes without recognized text content.
