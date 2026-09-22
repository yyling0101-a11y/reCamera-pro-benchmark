#!/usr/bin/env python3
"""
Convert Zipformer (encoder + decoder + joiner) ONNX models to RKNN FP16 for RV1126B.

Usage:
    python convert_to_rknn.py

This script converts all three sub-models as FP16 (the recommended quantization
for zipformer on rv1126b).

Optional arguments:
    --encoder  Encoder ONNX path  (default: zipformer_encoder.onnx)
    --decoder  Decoder ONNX path  (default: zipformer_decoder.onnx)
    --joiner   Joiner ONNX path   (default: zipformer_joiner.onnx)
    --target   Target platform    (default: rv1126b)
"""

import argparse
import os
from rknn.api import RKNN


def convert_model(onnx_path, output_path, target, label=""):
    """Convert one ONNX model to RKNN FP16."""
    print(f"\n{'='*60}")
    print(f"Converting {label}")
    print(f"{'='*60}")
    print(f"  ONNX:  {onnx_path}")
    print(f"  RKNN:  {output_path}")
    print(f"  Target: {target}")

    rknn = RKNN(verbose=False)

    print("  Configuring (FP16, no quantization)...")
    rknn.config(target_platform=target)

    print("  Loading ONNX...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  ERROR: load_onnx failed with code {ret}")
        return False

    print("  Building (FP16)...")
    ret = rknn.build(do_quantization=False)
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
        description='Convert Zipformer ONNX models to RKNN FP16')
    parser.add_argument('--encoder', default='zipformer_encoder.onnx',
                        help='Encoder ONNX model path')
    parser.add_argument('--decoder', default='zipformer_decoder.onnx',
                        help='Decoder ONNX model path')
    parser.add_argument('--joiner', default='zipformer_joiner.onnx',
                        help='Joiner ONNX model path')
    parser.add_argument('--target', default='rv1126b',
                        help='Target platform')
    args = parser.parse_args()

    for path in [args.encoder, args.decoder, args.joiner]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return 1

    print("Zipformer Encoder + Decoder + Joiner -> RKNN FP16 Conversion")
    print(f"Target platform: {args.target}")

    enc_ok = convert_model(
        args.encoder,
        args.encoder.replace('.onnx', '.rknn'),
        args.target, "Encoder")
    dec_ok = convert_model(
        args.decoder,
        args.decoder.replace('.onnx', '.rknn'),
        args.target, "Decoder")
    joi_ok = convert_model(
        args.joiner,
        args.joiner.replace('.onnx', '.rknn'),
        args.target, "Joiner")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Encoder:  {'OK' if enc_ok else 'FAILED'}")
    print(f"  Decoder:  {'OK' if dec_ok else 'FAILED'}")
    print(f"  Joiner:   {'OK' if joi_ok else 'FAILED'}")

    return 0 if (enc_ok and dec_ok and joi_ok) else 1


if __name__ == '__main__':
    exit(main())
