#!/usr/bin/env python3
"""
Convert YOLOv8 Zoo ONNX models to RKNN INT8 for RV1126B.

Usage:
    python convert_to_rknn.py --model yolov8n_zoo.onnx --calibration /path/to/calibration_images.txt

Calibration file should contain paths to images, one per line.
"""

import argparse
import os
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='ONNX model path')
    parser.add_argument('--calibration', required=True, help='Calibration image list file')
    parser.add_argument('--output', help='Output RKNN path (default: same name with .rknn)')
    parser.add_argument('--target', default='rv1126b', help='Target platform')
    args = parser.parse_args()

    model_name = os.path.basename(args.model).replace('.onnx', '')
    output_path = args.output or f'{model_name}.rknn'

    print(f"Converting {args.model} -> {output_path}")
    print(f"Target: {args.target}")
    print(f"Calibration: {args.calibration}")

    # Load calibration images
    with open(args.calibration, 'r') as f:
        cal_images = [line.strip() for line in f if line.strip()]
    print(f"Calibration images: {len(cal_images)}")

    # Create RKNN model
    rknn = RKNN(verbose=True)

    # Configure for YOLO Zoo models
    # Rockchip Zoo models expect: mean=[0,0,0], std=[255,255,255]
    # Input: uint8 [0-255], will be normalized internally
    print("\nConfiguring RKNN...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
        quantized_dtype='w8a8',
        optimization_level=3,
    )

    # Load ONNX
    print("\nLoading ONNX model...")
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        print(f"ERROR: load_onnx failed with code {ret}")
        return 1

    # Build RKNN with quantization
    print("\nBuilding RKNN model (INT8 quantization)...")
    ret = rknn.build(do_quantization=True, dataset=args.calibration)
    if ret != 0:
        print(f"ERROR: build failed with code {ret}")
        return 1

    # Export RKNN
    print(f"\nExporting RKNN model to {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"ERROR: export_rknn failed with code {ret}")
        return 1

    rknn.release()
    print(f"\nSUCCESS: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    return 0


if __name__ == '__main__':
    exit(main())
