#!/usr/bin/env python3
"""Convert RK official yolov8n to RKNN INT8 for RV1126B."""
import os
from rknn.api import RKNN

MODEL = "yolov8n_zoo.onnx"
CALIBRATION = "calibration_200.txt"
OUTPUT = "yolov8n_zoo_640x640_W8A8.rknn"

print(f"Converting {MODEL} -> {OUTPUT}")
rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rv1126b",
    quantized_dtype="w8a8",
    optimization_level=3,
)
ret = rknn.load_onnx(model=MODEL)
if ret != 0: print(f"load_onnx failed: {ret}"); exit(1)
ret = rknn.build(do_quantization=True, dataset=CALIBRATION)
if ret != 0: print(f"build failed: {ret}"); exit(1)
ret = rknn.export_rknn(OUTPUT)
if ret != 0: print(f"export failed: {ret}"); exit(1)
rknn.release()
print(f"SUCCESS: {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.2f} MB)")
