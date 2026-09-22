#!/usr/bin/env python3
"""Convert RK official yolo11n using RK's EXACT conversion settings."""
import os
from rknn.api import RKNN

MODEL = "yolo11n_zoo.onnx"
CALIBRATION = "calibration_200.txt"
OUTPUT = "yolo11n_zoo_640x640_W8A8.rknn"

print(f"Converting {MODEL} -> {OUTPUT}")
print("Using RK official conversion settings (no extra opts)")

rknn = RKNN(verbose=True)
# EXACT same config as RK zoo convert.py - minimal settings
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rv1126b",
)
ret = rknn.load_onnx(model=MODEL)
if ret != 0: print(f"load_onnx failed: {ret}"); exit(1)
ret = rknn.build(do_quantization=True, dataset=CALIBRATION)
if ret != 0: print(f"build failed: {ret}"); exit(1)
ret = rknn.export_rknn(OUTPUT)
if ret != 0: print(f"export failed: {ret}"); exit(1)
rknn.release()
print(f"SUCCESS: {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.2f} MB)")
