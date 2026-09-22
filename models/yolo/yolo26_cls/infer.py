#!/usr/bin/env python3
"""
YOLO26 classification RKNN inference on RV1126B.

Usage:
    python3 infer.py --model yolo26_cls.rknn --image test.jpg --output result.jpg

Model output: [1, 1000] classification scores (ImageNet 1000 classes)
Input size: 224x224
"""

import argparse, os, time, sys
import cv2
import numpy as np


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def letterbox(img, new_shape=(224, 224), color=(114, 114, 114)):
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def postprocess_cls(outputs, topk=5):
    """Apply softmax to the first output and return top-k predictions."""
    raw = outputs[0].astype(np.float32).reshape(1, -1)
    probs = softmax(raw, axis=1).flatten()
    top_indices = np.argsort(probs)[::-1][:topk]
    results = []
    for idx in top_indices:
        results.append({"class": int(idx), "conf": float(probs[idx])})
    return results


def draw_cls(img, predictions, topk=5):
    """Draw top-k classification predictions on the image."""
    overlay = img.copy()
    h_img = img.shape[0]

    # Determine font scale based on image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.2, h_img / 400.0))
    thickness = max(1, int(h_img / 300))
    line_height = int(25 * font_scale)
    y_start = int(line_height * 1.5)

    # Draw semi-transparent background bar
    bar_height = line_height * (topk + 1) + 10
    cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_height), (40, 40, 40), -1)
    result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Draw header
    cv2.putText(result, "YOLO26-cls Top-5 Predictions",
                (10, y_start), font, font_scale, (0, 255, 255), thickness)
    y_start += line_height + 5

    # Draw each prediction
    colors = [(0, 255, 0), (0, 220, 80), (80, 200, 80), (120, 180, 60), (160, 160, 40)]
    for i, pred in enumerate(predictions[:topk]):
        color = colors[i] if i < len(colors) else (0, 255, 0)
        label = f"#{i+1} class_{pred['class']:>4d}: {pred['conf']:.4f} ({pred['conf']*100:.1f}%)"
        cv2.putText(result, label, (10, y_start + i * line_height),
                    font, font_scale * 0.85, color, thickness)

    return result


TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Classification RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=224, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO26 Classification RKNN Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Mode:  {mode}")

    from rknnlite.api import RKNNLite
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
    predictions = postprocess_cls(last_outputs, topk=5)
    print(f"\nTop-5 Predictions:")
    for i, pred in enumerate(predictions):
        bar = "█" * int(pred['conf'] * 40)
        print(f"  #{i+1}  class_{pred['class']:>4d}  {pred['conf']:.4f}  ({pred['conf']*100:6.2f}%)  {bar}")
    img_draw = draw_cls(orig_img, predictions, topk=5)

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
