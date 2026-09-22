#!/usr/bin/env python3
"""
Convert YOLOv8-pose Zoo ONNX to RKNN INT8 for RV1126B.

Usage:
    python convert_to_rknn.py
"""

import os
from rknn.api import RKNN

MODEL = "yolov8n-pose.onnx"
CALIBRATION = "calibration_200.txt"
OUTPUT = "yolov8n-pose_W8A8.rknn"
TARGET = "rv1126b"


def main():
    print(f"Converting {MODEL} -> {OUTPUT}")
    print(f"Target: {TARGET}")

    with open(CALIBRATION, "r") as f:
        cal_images = [line.strip() for line in f if line.strip()]
    print(f"Calibration images: {len(cal_images)}")

    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET,
        quantized_dtype="w8a8",
        optimization_level=3,
    )

    print("\nLoading ONNX model...")
    ret = rknn.load_onnx(model=MODEL)
    if ret != 0:
        print(f"ERROR: load_onnx failed: {ret}")
        return 1

    print("\nBuilding RKNN (INT8)...")
    ret = rknn.build(do_quantization=True, dataset=CALIBRATION)
    if ret != 0:
        print(f"ERROR: build failed: {ret}")
        return 1

    print(f"\nExporting to {OUTPUT}...")
    ret = rknn.export_rknn(OUTPUT)
    if ret != 0:
        print(f"ERROR: export_rknn failed: {ret}")
        return 1

    rknn.release()
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\nSUCCESS: {OUTPUT} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    exit(main())
