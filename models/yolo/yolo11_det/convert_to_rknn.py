#!/usr/bin/env python3
"""Convert optimized YOLO11n detection ONNX to RKNN for RV1126B."""
from rknn.api import RKNN

ONNX_MODEL = 'yolo11n_optimized.onnx'
RKNN_MODEL = 'yolo11n_optimized.rknn'
DATASET_FILE = 'calibration_200.txt'
TARGET = 'rv1126b'

print(f"Converting {ONNX_MODEL} to {RKNN_MODEL}...")

rknn = RKNN()

print("\n==> Configuring RKNN...")
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET,
    quantized_dtype='w8a8',
    optimization_level=3
)

print("\n==> Loading ONNX model...")
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print(f"ERROR: Failed to load ONNX model, ret={ret}")
    exit(ret)

print("\n==> Building RKNN model...")
ret = rknn.build(do_quantization=True, dataset=DATASET_FILE)
if ret != 0:
    print(f"ERROR: Failed to build RKNN model, ret={ret}")
    exit(ret)

print("\n==> Exporting RKNN model...")
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print(f"ERROR: Failed to export RKNN model, ret={ret}")
    exit(ret)

rknn.release()
print(f"\n✓ Conversion complete! Output: {RKNN_MODEL}")
