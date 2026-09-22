#!/usr/bin/env python3
"""
YOLO26n-depth RKNN inference script with depth decoding and visualization.
"""
import argparse
import cv2
import numpy as np
import time
import os
from rknnlite.api import RKNNLite

# Calibration parameters from ONNX
CAL_A = 1.0
CAL_B = -0.19384766

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding."""
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def decode_depth(raw_depth, cal_a=CAL_A, cal_b=CAL_B):
    """
    Decode raw depth prediction to metric depth.
    Formula from ONNX: clip -> exp -> pow(cal_a) -> mul(exp(cal_b))
    """
    # Clip to reasonable range
    depth = np.clip(raw_depth, -10, 10)
    # Exponential
    depth = np.exp(depth)
    # Power calibration
    depth = np.power(depth, cal_a)
    # Multiply by exp(cal_b)
    depth = depth * np.exp(cal_b)
    return depth

def visualize_depth(depth_map, max_depth=10.0):
    """Convert depth map to colored visualization."""
    # Normalize to 0-255
    depth_norm = np.clip(depth_map / max_depth, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap (INFERNO is good for depth)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    return depth_colored

TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO26-depth RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO26-depth RKNN Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Mode:  {mode}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print(f"ERROR: load_rknn failed: {ret}"); return 1
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f"ERROR: init_runtime failed: {ret}"); return 1

    img_input = None
    ratio, dw, dh = 1.0, 0.0, 0.0
    orig_img = None

    if mode == "camera":
        gst_pipeline = ("v4l2src device=/dev/video13 ! "
                       "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! "
                       "videoscale ! video/x-raw,width=640,height=480 ! "
                       "videoconvert ! video/x-raw,format=BGR ! "
                       "appsink drop=true max-buffers=1")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("ERROR: Cannot open camera via GStreamer (/dev/video13)"); rknn.release(); return 1
        print("Camera /dev/video13 opened via GStreamer")

        print(f"\nWarmup ({WARMUP_RUNS} iters)...")
        for _ in range(WARMUP_RUNS):
            ret_cam, frame = cap.read()
            if ret_cam:
                img_lb, _, _ = letterbox(frame, (args.imgsz, args.imgsz))
                inp = img_lb[np.newaxis, :, :, :].astype(np.uint8)
                rknn.inference(inputs=[inp])

        print(f"Benchmark ({TOTAL_RUNS} runs, skipping first {WARMUP_RUNS})...")
        latencies = []
        for i in range(TOTAL_RUNS):
            ret_cam, frame = cap.read()
            if not ret_cam:
                continue
            img_lb, _, _ = letterbox(frame, (args.imgsz, args.imgsz))
            inp = img_lb[np.newaxis, :, :, :].astype(np.uint8)
            t0 = time.perf_counter()
            rknn.inference(inputs=[inp])
            t1 = time.perf_counter()
            if i >= WARMUP_RUNS:
                latencies.append((t1 - t0) * 1000)

        cap.release()
        latencies = np.array(latencies)
        print(f"\nBenchmark Results ({TOTAL_RUNS} runs, {WARMUP_RUNS} warmup):")
        print(f"  Avg: {np.mean(latencies):.2f} ms")
        print(f"  Min: {np.min(latencies):.2f} ms")
        print(f"  Max: {np.max(latencies):.2f} ms")
        print(f"  FPS: {1000.0 / np.mean(latencies):.1f}")

        rknn.release()
        print(f"{'='*60}\n")
        return 0

    # Image mode
    print(f"Image: {args.input}")
    orig_img = cv2.imread(args.input)
    if orig_img is None:
        print(f"ERROR: Cannot read {args.input}"); rknn.release(); return 1
    print(f"Image size: {orig_img.shape}")

    img_lb, ratio, (dw, dh) = letterbox(orig_img, (args.imgsz, args.imgsz))
    img_input = img_lb[np.newaxis, :, :, :].astype(np.uint8)

    print(f"\nWarmup ({WARMUP_RUNS} iters)...")
    for _ in range(WARMUP_RUNS):
        rknn.inference(inputs=[img_input])

    print(f"Benchmark ({TOTAL_RUNS} runs, skipping first {WARMUP_RUNS})...")
    latencies = []
    last_outputs = None
    for i in range(TOTAL_RUNS):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[img_input])
        t1 = time.perf_counter()
        if i >= WARMUP_RUNS:
            latencies.append((t1 - t0) * 1000)
        last_outputs = outputs

    latencies = np.array(latencies)
    print(f"\nBenchmark Results ({TOTAL_RUNS} runs, {WARMUP_RUNS} warmup):")
    print(f"  Avg: {np.mean(latencies):.2f} ms")
    print(f"  Min: {np.min(latencies):.2f} ms")
    print(f"  Max: {np.max(latencies):.2f} ms")
    print(f"  FPS: {1000.0 / np.mean(latencies):.1f}")

    print(f"\nPost-processing...")
    raw_depth = last_outputs[0]
    print(f"Raw depth shape: {raw_depth.shape}, range: [{raw_depth.min():.3f}, {raw_depth.max():.3f}]")
    depth_160 = raw_depth[0, 0]
    depth_metric = decode_depth(depth_160)
    print(f"Metric depth range: [{depth_metric.min():.3f}, {depth_metric.max():.3f}] m")
    depth_640 = cv2.resize(depth_metric, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
    depth_crop = depth_640[int(dh):int(args.imgsz - dh), int(dw):int(args.imgsz - dw)]
    orig_h, orig_w = orig_img.shape[:2]
    depth_final = cv2.resize(depth_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    depth_colored = visualize_depth(depth_final, max_depth=10.0)
    result = np.hstack([orig_img, depth_colored])
    fps_val = 1000.0 / np.mean(latencies)
    cv2.putText(result, f"Depth ({fps_val:.1f} FPS)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img_draw = result

    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"result_{input_basename}.jpg")
    cv2.imwrite(output_path, img_draw)
    print(f"Saved: {output_path}")

    rknn.release()
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())
