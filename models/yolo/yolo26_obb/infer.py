#!/usr/bin/env python3
"""
YOLO26-OBB RKNN inference on RV1126B.
9 outputs, no DFL, direct bbox regression + per-stride angle.

Output structure (9 tensors):
  [0] cls_stride_0:   [1, 15, 80, 80]  stride 8, sigmoid applied, DOTA 15 classes
  [1] bbox_stride_0:  [1,  4, 80, 80]  stride 8, raw bbox regression
  [2] angle_stride_0: [1,  1, 80, 80]  stride 8, needs sigmoid then decode
  [3] cls_stride_1:   [1, 15, 40, 40]  stride 16
  [4] bbox_stride_1:  [1,  4, 40, 40]  stride 16
  [5] angle_stride_1: [1,  1, 40, 40]  stride 16
  [6] cls_stride_2:   [1, 15, 20, 20]  stride 32
  [7] bbox_stride_2:  [1,  4, 20, 20]  stride 32
  [8] angle_stride_2: [1,  1, 20, 20]  stride 32

Bbox decode (no DFL):
  lt = anchor - bbox[0:2], rb = anchor + bbox[2:4]
  xyxy = (lt, rb) * stride
  center = (xy1+xy2)/2, width = x2-x1, height = y2-y1

Angle decode:
  angle = (sigmoid(raw) - 0.25) * pi → range [-pi/4, 3pi/4]
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
COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
          for _ in range(15)]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Aspect-preserving resize with padding."""
    shape = img.shape[:2]  # h, w
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


def axis_aligned_nms(boxes_xyxy, scores, iou_thres=0.45):
    """Standard axis-aligned NMS."""
    if len(boxes_xyxy) == 0:
        return []
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """Post-process YOLO26-OBB RKNN outputs (no DFL, direct bbox regression)."""
    strides = [8, 16, 32]
    num_classes = 15

    all_cxcywh = []
    all_scores = []
    all_classes = []
    all_angles = []

    for si, stride in enumerate(strides):
        # Outputs grouped by branch: cls*3, bbox*3, angle*3
        cls_raw = outputs[si].astype(np.float32)           # [1, 15, H, W]
        bbox_raw = outputs[si + 3].astype(np.float32)      # [1, 4, H, W]
        angle_raw = outputs[si + 6].astype(np.float32)     # [1, 1, H, W]

        _, _, h, w = cls_raw.shape
        n_anchors = h * w

        # Class scores — already sigmoid from model output
        cls_scores = cls_raw[0].transpose(1, 2, 0).reshape(-1, num_classes)  # [N, 15]
        # Ensure values are in [0,1] range (in case model output isn't strictly sigmoided)
        cls_scores = np.clip(cls_scores, 0.0, 1.0)

        # Bbox: direct 4-channel regression [l, t, r, b] in grid units
        bbox_np = bbox_raw[0]  # [4, H, W]
        l = bbox_np[0].flatten()  # [N]
        t = bbox_np[1].flatten()
        r = bbox_np[2].flatten()
        b = bbox_np[3].flatten()

        # Angle: sigmoid then decode
        angle_vals = angle_raw[0, 0].flatten()  # [N]
        sigmoid_angle = 1.0 / (1.0 + np.exp(-np.clip(angle_vals, -50, 50)))
        angles_rad = (sigmoid_angle - 0.25) * math.pi  # [-pi/4, 3pi/4]

        # Anchor points in grid units (center of each cell)
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchor_x = (xv.ravel() + 0.5).astype(np.float32)
        anchor_y = (yv.ravel() + 0.5).astype(np.float32)

        # Bbox decode: lt = anchor - bbox[0:2], rb = anchor + bbox[2:4]
        # In pixel space (multiply by stride)
        x1 = (anchor_x - l) * stride
        y1 = (anchor_y - t) * stride
        x2 = (anchor_x + r) * stride
        y2 = (anchor_y + b) * stride

        # Convert xyxy to cxcywh
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        # Filter by confidence
        max_score = cls_scores.max(axis=1)
        mask = max_score > conf_thres

        if mask.sum() > 0:
            cxcywh = np.stack([cx[mask], cy[mask], bw[mask], bh[mask]], axis=1)
            scores = max_score[mask]
            classes = cls_scores[mask].argmax(axis=1)
            angles = angles_rad[mask]

            all_cxcywh.append(cxcywh)
            all_scores.append(scores)
            all_classes.append(classes)
            all_angles.append(angles)

    if not all_cxcywh:
        return []

    cxcywh = np.concatenate(all_cxcywh, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    angles_all = np.concatenate(all_angles, axis=0)

    # Convert to original image coordinates
    dw, dh = pad
    rboxes = []
    enclosing_boxes = []

    for i in range(len(cxcywh)):
        cx, cy, bw, bh = cxcywh[i]
        angle = angles_all[i]

        # Map to original image coordinates
        cx_orig = (cx - dw) / ratio
        cy_orig = (cy - dh) / ratio
        w_orig = bw / ratio
        h_orig = bh / ratio

        # Clamp to reasonable size
        w_orig = max(1.0, w_orig)
        h_orig = max(1.0, h_orig)

        angle_deg = math.degrees(angle)

        # Build rotated rectangle
        rbox = ((cx_orig, cy_orig), (w_orig, h_orig), angle_deg)
        rboxes.append(rbox)

        # Compute enclosing axis-aligned bbox for NMS
        corners = cv2.boxPoints(rbox)
        x_min = corners[:, 0].min()
        y_min = corners[:, 1].min()
        x_max = corners[:, 0].max()
        y_max = corners[:, 1].max()
        enclosing_boxes.append([x_min, y_min, x_max, y_max])

    # Axis-aligned NMS on enclosing bounding boxes as approximation
    enclosing_boxes = np.array(enclosing_boxes)
    keep = axis_aligned_nms(enclosing_boxes, scores, iou_thres)

    rboxes = [rboxes[k] for k in keep]
    det_scores = scores[keep]
    det_classes = classes[keep]

    detections = []
    for i in range(len(rboxes)):
        detections.append({
            "rbox": rboxes[i],
            "conf": float(det_scores[i]),
            "class": int(det_classes[i]),
        })
    return detections


def draw_obb(img, detections):
    """Draw rotated bounding boxes with class labels."""
    for det in detections:
        cls_id = det["class"]
        conf = det["conf"]
        rbox = det["rbox"]
        color = COLORS[cls_id % len(COLORS)]

        # Get 4 corner points of the rotated rectangle
        corners = cv2.boxPoints(rbox).astype(np.int32)
        cv2.polylines(img, [corners], True, color, 2)

        # Label at top-left corner of the rotated box
        cx, cy = int(rbox[0][0]), int(rbox[0][1])
        label = f"{DOTA_CLASSES[cls_id] if cls_id < len(DOTA_CLASSES) else cls_id}: {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Place label above the center
        ly = min(corners[:, 1]) - 2
        lx = int(corners[:, 0].min())
        cv2.rectangle(img, (lx, ly - th - 6), (lx + tw, ly), color, -1)
        cv2.putText(img, label, (lx, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO26-OBB RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO26-OBB RKNN Inference")
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
