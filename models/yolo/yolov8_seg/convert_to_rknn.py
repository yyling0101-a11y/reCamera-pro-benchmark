#!/usr/bin/env python3
"""Convert YOLOv8n-seg Zoo ONNX to RKNN INT8 for RV1126B."""
import os
from rknn.api import RKNN

MODEL = "yolov8n-seg.onnx"
CALIBRATION = "calibration_200.txt"
OUTPUT = "yolov8n-seg_W8A8.rknn"
TARGET = "rv1126b"

def main():
    print(f"Converting {MODEL} -> {OUTPUT}")
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET,
        quantized_dtype="w8a8",
        optimization_level=3,
    )
    print("Loading ONNX...")
    ret = rknn.load_onnx(model=MODEL)
    if ret != 0: return 1
    print("Building RKNN (INT8)...")
    ret = rknn.build(do_quantization=True, dataset=CALIBRATION)
    if ret != 0: return 1
    print(f"Exporting {OUTPUT}...")
    ret = rknn.export_rknn(OUTPUT)
    if ret != 0: return 1
    rknn.release()
    print(f"SUCCESS: {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.2f} MB)")
    return 0

if __name__ == "__main__": exit(main())
