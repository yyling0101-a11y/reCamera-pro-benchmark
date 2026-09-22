# PPOCRv4 Recognition (INT8)

PPOCRv4 text recognition model converted to RKNN INT8 (W8A8) for Seeed reCamera Pro (RV1126B).

## Model Info

- **Source**: PaddleOCR PP-OCRv4 Recognition (SVTR_LCNet)
- **ONNX Source**: RKNN Model Zoo (`ppocrv4_rec.onnx`)
- **Input**: `[1, 3, 48, 320]` NCHW, uint8 [0-255]
- **Output**: `[1, 80, 6625]` character probabilities (CTC)
- **Quantization**: INT8 W8A8 (experimental for rec on rv1126b)
- **Normalization**: 1/255 scale (mean=[0,0,0], std=[255,255,255])
- **Model Size**: 4.59 MB
- **Dictionary**: `ppocr_keys_v1.txt` (6624 Chinese/English characters + space + blank)

## Performance on reCamera Pro (RV1126B)

| Mode | Avg Latency | Min | Max | FPS |
|------|-------------|-----|-----|-----|
| Image (single) | 21.73 ms | - | - | ~46.0 |
| Camera (video13) | 24.11 ms | 16.22 ms | 46.79 ms | 41.5 |

Camera benchmark: 20 runs, 5 warmup iterations. Latency is NPU inference only (excludes pre/post-processing).

## Files

| File | Description |
|------|-------------|
| `ppocrv4_rec.onnx` | Original ONNX model |
| `ppocrv4_rec_48x320_rec.rknn` | Converted INT8 RKNN model |
| `convert_to_rknn.py` | ONNX → RKNN conversion script |
| `infer.py` | On-device inference script |
| `ppocr_keys_v1.txt` | Character dictionary (6625 classes) |
| `calibration.txt` | Calibration image list (for INT8) |
| `test_ocr.jpg` | Test image with text |
| `result_ocr.jpg` | Inference result with recognized text |

## Usage

### Convert ONNX to RKNN

Run on x86_64 host with `recamera-rknn-2.3.2` conda environment:

```bash
conda run -n recamera-rknn-2.3.2 python convert_to_rknn.py
```

Optional arguments:
- `--model` — ONNX model path (default: `ppocrv4_rec.onnx`)
- `--calibration` — Calibration image list (default: `calibration.txt`)
- `--output` — Output RKNN path (default: `ppocrv4_rec_48x320_rec.rknn`)
- `--target` — Target platform (default: `rv1126b`)

### Run Inference on Device

Push `.rknn`, `ppocr_keys_v1.txt`, `infer.py`, and test image to the device, then:

**Image mode** (runs inference and saves result with recognized text):
```bash
python3 infer.py --model ppocrv4_rec_48x320_rec.rknn --input test_ocr.jpg --dict ppocr_keys_v1.txt --output result_ocr.jpg
```

**Camera mode** (20 runs, 5 warmup, reports average latency):
```bash
python3 infer.py --model ppocrv4_rec_48x320_rec.rknn --dict ppocr_keys_v1.txt
```

Optional arguments:
- `--runs` — Number of benchmark runs (default: 20)
- `--warmup` — Warmup iterations (default: 5)

Camera input uses `/dev/video13` via GStreamer pipeline (1920×1080 NV12).

## Post-processing

Recognition output uses CTC (Connectionist Temporal Classification) decoding:
1. Argmax over 6625 character classes at each time step
2. Remove consecutive duplicate characters
3. Remove blank tokens (index 0)
4. Map indices to characters via dictionary

## Notes

- This is the **recognition-only** model. For full OCR, first detect text regions with `ppocrv4_det`, crop each region, then run recognition.
- INT8 quantization for rec on rv1126b is **experimental**. The official RKNN Model Zoo recommends FP16 for recognition on this platform. INT8 may reduce character-level accuracy.
- The model expects a single text line image resized to 48×320. For best results, crop detected text regions before feeding to the recognition model.
- The dictionary `ppocr_keys_v1.txt` contains 6623 Chinese characters, 10 digits, 26 lowercase + 26 uppercase English letters, plus space and CTC blank.
