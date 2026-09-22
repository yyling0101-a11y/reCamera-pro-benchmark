#!/usr/bin/env python3
"""
Convert PPOCRv4 Detection ONNX model to RKNN INT8 for RV1126B.

Usage:
    python convert_to_rknn.py

Optional arguments:
    --model       ONNX model path (default: ppocrv4_det.onnx)
    --calibration Calibration image list file (default: calibration.txt)
    --output      Output RKNN path (default: ppocrv4_det_480x480_det.rknn)
    --target      Target platform (default: rv1126b)
"""

import argparse
import os
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description='Convert PPOCRv4 Det ONNX to RKNN INT8')
    parser.add_argument('--model', default='ppocrv4_det.onnx', help='ONNX model path')
    parser.add_argument('--calibration', default='calibration.txt', help='Calibration image list file')
    parser.add_argument('--output', default='ppocrv4_det_480x480_det.rknn', help='Output RKNN path')
    parser.add_argument('--target', default='rv1126b', help='Target platform')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return 1

    if not os.path.exists(args.calibration):
        print(f"ERROR: Calibration file not found: {args.calibration}")
        return 1

    print(f"Converting {args.model} -> {args.output}")
    print(f"Target: {args.target}")
    print(f"Calibration: {args.calibration}")

    with open(args.calibration, 'r') as f:
        cal_images = [line.strip() for line in f if line.strip()]
    print(f"Calibration images: {len(cal_images)}")

    rknn = RKNN(verbose=True)

    # PPOCRv4 Det preprocessing:
    # Original normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=1/255
    # RKNN equivalent (applied to uint8 [0,255]):
    #   mean_values = mean * 255 = [123.675, 116.28, 103.53]
    #   std_values = std * 255 = [58.395, 57.12, 57.375]
    print("\nConfiguring RKNN...")
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        target_platform=args.target,
        quantized_dtype='w8a8',
        optimization_level=3,
    )

    print("\nLoading ONNX model...")
    ret = rknn.load_onnx(model=args.model)
    if ret != 0:
        print(f"ERROR: load_onnx failed with code {ret}")
        return 1

    print("\nBuilding RKNN model (INT8 quantization)...")
    ret = rknn.build(do_quantization=True, dataset=args.calibration)
    if ret != 0:
        print(f"ERROR: build failed with code {ret}")
        return 1

    print(f"\nExporting RKNN model to {args.output}...")
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print(f"ERROR: export_rknn failed with code {ret}")
        return 1

    rknn.release()
    print(f"\nSUCCESS: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    return 0


if __name__ == '__main__':
    exit(main())
