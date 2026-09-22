#!/usr/bin/env python3
"""
YOLOv8n-obb RKNN inference on RV1126B.
Corrected postprocessing matching Ultralytics dist2rbox.

Output structure (4 tensors):
  [0]: [1, 79, 80, 80] stride 8  (64 DFL + 15 DOTA classes)
  [1]: [1, 79, 40, 40] stride 16
  [2]: [1, 79, 20, 20] stride 32
  [3]: [1, 1, 8400]    angle (sigmoid applied)

Angle decode: (sigmoid_value - 0.25) * pi → range [-pi/4, 3pi/4]
Center decode: dist2rbox (rotate anchor offset by angle)
"""

import argparse
import math
import os
import time
import cv2
import numpy as np

DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
    'soccer-ball-field', 'swimming-pool',
]
COLORS = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
          for _ in range(15)]


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def dfl_decode(x, reg_max=16):
    b, c, h, w = x.shape
    x = x.reshape(b, 4, reg_max, h, w)
    x = softmax(x, axis=2)
    weights = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    return (x * weights).sum(axis=2)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def obb_nms(rboxes, scores, iou_thres=0.45):
    """Rotated NMS using cv2.rotatedRectangleIntersection."""
    if len(rboxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        overlaps = []
        for j_idx in range(1, len(order)):
            j = order[j_idx]
            ret, pts = cv2.rotatedRectangleIntersection(rboxes[i], rboxes[j])
            if ret == cv2.INTERSECT_NONE or pts is None:
                inter = 0.0
            else:
                inter = cv2.contourArea(pts)
            area_i = rboxes[i][1][0] * rboxes[i][1][1]
            area_j = rboxes[j][1][0] * rboxes[j][1][1]
            union = area_i + area_j - inter
            iou = inter / (union + 1e-6)
            overlaps.append(iou)
        overlaps = np.array(overlaps)
        inds = np.where(overlaps <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """Post-process YOLOv8n-obb RKNN outputs (corrected dist2rbox)."""
    strides = [8, 16, 32]
    reg_max = 16
    num_classes = 15

    # Decode angle: (sigmoid - 0.25) * pi → [-pi/4, 3pi/4]
    angle_raw = outputs[3].astype(np.float32).flatten()
    angles_rad = (angle_raw - 0.25) * math.pi

    all_cxcywh = []
    all_scores = []
    all_classes = []
    all_angles = []

    for i, stride in enumerate(strides):
        feat = outputs[i].astype(np.float32)
        _, c, h, w = feat.shape
        n_anchors = h * w

        bbox_feat = feat[:, :64, :, :]
        cls_scores = feat[:, 64:, :, :]

        # DFL decode → [l, t, r, b] in grid units
        bbox = dfl_decode(bbox_feat, reg_max)
        bbox_np = bbox[0]
        l = bbox_np[0].flatten()
        t = bbox_np[1].flatten()
        r = bbox_np[2].flatten()
        b = bbox_np[3].flatten()

        # Class scores with sigmoid
        cls_scores = cls_scores[0].transpose(1, 2, 0).reshape(-1, num_classes)
        cls_scores = 1.0 / (1.0 + np.exp(-np.clip(cls_scores, -50, 50)))

        # Anchor points in grid units
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchor_x = (xv.ravel() + 0.5).astype(np.float32)
        anchor_y = (yv.ravel() + 0.5).astype(np.float32)

        # dist2rbox: rotate center offset by angle
        xf = (r - l) / 2.0
        yf = (b - t) / 2.0

        offset = sum([imgsz * imgsz // (s * s) for s in strides[:i]])
        stride_angles = angles_rad[offset:offset + n_anchors]

        cos_a = np.cos(stride_angles)
        sin_a = np.sin(stride_angles)

        x_rot = xf * cos_a - yf * sin_a
        y_rot = xf * sin_a + yf * cos_a

        # Center in grid units, then convert to pixels
        cx_px = (x_rot + anchor_x) * stride
        cy_px = (y_rot + anchor_y) * stride
        w_px = (l + r) * stride
        h_px = (t + b) * stride

        max_score = cls_scores.max(axis=1)
        max_cls = cls_scores.argmax(axis=1)

        all_cxcywh.append(np.stack([cx_px, cy_px, w_px, h_px], axis=-1))
        all_scores.append(max_score)
        all_classes.append(max_cls)
        all_angles.append(stride_angles)

    cxcywh = np.concatenate(all_cxcywh, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    angles_all = np.concatenate(all_angles, axis=0)

    # Filter
    mask = scores > conf_thres
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    det_cxcywh = cxcywh[indices]
    det_scores = scores[indices]
    det_classes = classes[indices]
    det_angles = angles_all[indices]

    # Convert to original image coordinates
    dw, dh = pad
    rboxes = []
    for i in range(len(det_cxcywh)):
        cx, cy, w, h = det_cxcywh[i]
        cx_orig = (cx - dw) / ratio
        cy_orig = (cy - dh) / ratio
        w_orig = w / ratio
        h_orig = h / ratio
        angle_deg = math.degrees(det_angles[i])
        rboxes.append(((cx_orig, cy_orig), (w_orig, h_orig), angle_deg))

    # Rotated NMS
    keep = obb_nms(rboxes, det_scores, iou_thres)
    rboxes = [rboxes[k] for k in keep]
    det_scores = det_scores[keep]
    det_classes = det_classes[keep]

    detections = []
    for i in range(len(rboxes)):
        detections.append({
            "rbox": rboxes[i],
            "conf": float(det_scores[i]),
            "class": int(det_classes[i]),
        })
    return detections


def draw_obb(img, detections):
    for det in detections:
        cls = det["class"]
        conf = det["conf"]
        rbox = det["rbox"]
        color = COLORS[cls % len(COLORS)]
        corners = cv2.boxPoints(rbox).astype(np.int32)
        cv2.polylines(img, [corners], True, color, 2)
        cx, cy = int(rbox[0][0]), int(rbox[0][1])
        label = f"{DOTA_CLASSES[cls] if cls < len(DOTA_CLASSES) else cls}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (cx, cy - th - 6), (cx + tw, cy), color, -1)
        cv2.putText(img, label, (cx, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO11 OBB RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO11 OBB RKNN Inference")
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
    detections = postprocess(last_outputs, ratio, (dw, dh),
                              conf_thres=args.conf, iou_thres=args.nms, imgsz=args.imgsz)
    print(f"Detections: {len(detections)}")
    for i, det in enumerate(detections[:15]):
        cls_name = DOTA_CLASSES[det['class']] if det['class'] < len(DOTA_CLASSES) else str(det['class'])
        rbox = det['rbox']
        print(f"  [{i}] {cls_name}: {det['conf']:.3f} center=({rbox[0][0]:.0f},{rbox[0][1]:.0f}) size=({rbox[1][0]:.0f}x{rbox[1][1]:.0f}) angle={rbox[2]:.1f}deg")
    img_draw = draw_obb(orig_img.copy(), detections)

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
