#!/usr/bin/env python3
"""
Convert PPOCRv4 Detection + Recognition ONNX models to RKNN INT8 for RV1126B.

Usage:
    python convert_to_rknn.py

This script converts both the detection and recognition models as a pair.

Optional arguments:
    --det-onnx     Detection ONNX model (default: ppocrv4_det.onnx)
    --rec-onnx     Recognition ONNX model (default: ppocrv4_rec.onnx)
    --calibration  Calibration image list file (default: calibration.txt)
    --det-output   Detection RKNN output (default: ppocrv4_det_480x480_det.rknn)
    --rec-output   Recognition RKNN output (default: ppocrv4_rec_48x320_rec.rknn)
    --target       Target platform (default: rv1126b)
"""

import argparse
import os
from rknn.api import RKNN


def convert_det(onnx_path, output_path, calibration, target):
    """Convert PPOCRv4 Detection ONNX to RKNN INT8."""
    print(f"\n{'='*60}")
    print(f"Converting Detection Model")
    print(f"{'='*60}")
    print(f"  ONNX:  {onnx_path}")
    print(f"  RKNN:  {output_path}")
    print(f"  Target: {target}")

    with open(calibration, 'r') as f:
        cal_images = [line.strip() for line in f if line.strip()]
    print(f"  Calibration images: {len(cal_images)}")

    rknn = RKNN(verbose=False)

    # ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    # Scaled to uint8 [0,255]: mean*255, std*255
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        target_platform=target,
        quantized_dtype='w8a8',
        optimization_level=3,
    )

    print("  Loading ONNX...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  ERROR: load_onnx failed with code {ret}")
        return False

    print("  Building (INT8 quantization)...")
    ret = rknn.build(do_quantization=True, dataset=calibration)
    if ret != 0:
        print(f"  ERROR: build failed with code {ret}")
        return False

    print(f"  Exporting to {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"  ERROR: export_rknn failed with code {ret}")
        return False

    rknn.release()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  SUCCESS: {output_path} ({size_mb:.2f} MB)")
    return True


def convert_rec(onnx_path, output_path, calibration, target):
    """Convert PPOCRv4 Recognition ONNX to RKNN INT8."""
    print(f"\n{'='*60}")
    print(f"Converting Recognition Model")
    print(f"{'='*60}")
    print(f"  ONNX:  {onnx_path}")
    print(f"  RKNN:  {output_path}")
    print(f"  Target: {target}")
    print(f"  NOTE: INT8 for rec on rv1126b is experimental.")

    with open(calibration, 'r') as f:
        cal_images = [line.strip() for line in f if line.strip()]
    print(f"  Calibration images: {len(cal_images)}")

    rknn = RKNN(verbose=False)

    # scale=1/255, mean=[0,0,0], std=[1,1,1]
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target,
        quantized_dtype='w8a8',
        optimization_level=3,
    )

    print("  Loading ONNX...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  ERROR: load_onnx failed with code {ret}")
        return False

    print("  Building (INT8 quantization)...")
    ret = rknn.build(do_quantization=True, dataset=calibration)
    if ret != 0:
        print(f"  ERROR: build failed with code {ret}")
        return False

    print(f"  Exporting to {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"  ERROR: export_rknn failed with code {ret}")
        return False

    rknn.release()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  SUCCESS: {output_path} ({size_mb:.2f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert PPOCRv4 Det+Rec ONNX models to RKNN INT8')
    parser.add_argument('--det-onnx', default='ppocrv4_det.onnx',
                        help='Detection ONNX model path')
    parser.add_argument('--rec-onnx', default='ppocrv4_rec.onnx',
                        help='Recognition ONNX model path')
    parser.add_argument('--calibration', default='calibration.txt',
                        help='Calibration image list file')
    parser.add_argument('--det-output', default='ppocrv4_det_480x480_det.rknn',
                        help='Detection RKNN output path')
    parser.add_argument('--rec-output', default='ppocrv4_rec_48x320_rec.rknn',
                        help='Recognition RKNN output path')
    parser.add_argument('--target', default='rv1126b',
                        help='Target platform')
    args = parser.parse_args()

    for path in [args.det_onnx, args.rec_onnx, args.calibration]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return 1

    print("PPOCRv4 Det + Rec -> RKNN INT8 Conversion")
    print(f"Target platform: {args.target}")

    det_ok = convert_det(args.det_onnx, args.det_output,
                         args.calibration, args.target)
    rec_ok = convert_rec(args.rec_onnx, args.rec_output,
                         args.calibration, args.target)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Detection:    {'OK' if det_ok else 'FAILED'} -> {args.det_output}")
    print(f"  Recognition:  {'OK' if rec_ok else 'FAILED'} -> {args.rec_output}")

    if not (det_ok and rec_ok):
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
