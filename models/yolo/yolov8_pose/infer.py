#!/usr/bin/env python3
"""
YOLOv8-pose Zoo RKNN inference on RV1126B.

Handles 4-output Zoo pose model (modified ONNX — vis sigmoid removed):
  output[0]: [1, 65, 80, 80]  stride 8  (64 DFL + 1 class logit)
  output[1]: [1, 65, 40, 40]  stride 16
  output[2]: [1, 65, 20, 20]  stride 32
  output[3]: [1, 17, 3, 8400] keypoints (xy pixel + vis RAW logit)

Usage (on device):
    python3 infer.py \
        --model /userdata/benchmark/models/yolo/yolov8_pose/yolov8n-pose_W8A8.rknn \
        --image /userdata/benchmark/test_bus.jpg \
        --output /userdata/benchmark/models/yolo/yolov8_pose/result_bus.jpg
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]
KPT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]
KPT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def dfl_decode(x, reg_max=16):
    """DFL decode: [B, 4*reg_max, H, W] -> [B, 4, H, W]."""
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


def nms(detections, iou_thres=0.45):
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["conf"] for d in detections])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
    return [detections[i] for i in keep]


def postprocess(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """
    Post-process YOLOv8-pose Zoo RKNN outputs.

    outputs[0..2]: stride-level features [1, 65, H, W]
        channels [0:64] = bbox DFL features (need softmax)
        channel  [64]   = class logit (need sigmoid)
    outputs[3]: [1, 17, 3, 8400]
        [:, :, 0, :] = x pixel coord (in letterbox space)
        [:, :, 1, :] = y pixel coord (in letterbox space)
        [:, :, 2, :] = raw visibility logit (need sigmoid)
    """
    strides = [8, 16, 32]
    reg_max = 16

    all_bboxes = []
    all_scores = []

    for i, stride in enumerate(strides):
        feat = outputs[i].astype(np.float32)
        _, c, h, w = feat.shape

        bbox_feat = feat[:, :64, :, :]
        cls_logit = feat[:, 64, :, :]

        # DFL decode bbox
        bbox = dfl_decode(bbox_feat, reg_max)  # [1, 4, H, W]
        bbox = bbox[0].transpose(1, 2, 0).reshape(-1, 4)  # [N, 4]

        # Sigmoid on class logit
        cls_score = sigmoid(cls_logit[0].flatten())  # [N]

        # Generate anchors for this stride
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchors = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)
        anchors = (anchors + 0.5) * stride  # [N, 2]

        # Decode bbox: distances -> xyxy
        bx1 = anchors[:, 0] - bbox[:, 0] * stride
        by1 = anchors[:, 1] - bbox[:, 1] * stride
        bx2 = anchors[:, 0] + bbox[:, 2] * stride
        by2 = anchors[:, 1] + bbox[:, 3] * stride

        all_bboxes.append(np.stack([bx1, by1, bx2, by2], axis=-1))
        all_scores.append(cls_score)

    # Concatenate all strides
    bboxes = np.concatenate(all_bboxes, axis=0)  # [8400, 4]
    scores = np.concatenate(all_scores, axis=0)  # [8400]

    # Decode keypoints from output[3]
    kpt_out = outputs[3].astype(np.float32)  # [1, 17, 3, 8400]
    kpts_xy = kpt_out[0, :, :2, :]  # [17, 2, 8400] - pixel coords in letterbox
    kpts_vis_logit = kpt_out[0, :, 2, :]  # [17, 8400] - raw logit
    kpts_vis = sigmoid(kpts_vis_logit)  # [17, 8400]

    # Filter by confidence
    mask = scores > conf_thres
    indices = np.where(mask)[0]

    detections = []
    dw, dh = pad
    for idx in indices:
        # Letterbox -> original image coords for bbox
        ox1 = (bboxes[idx, 0] - dw) / ratio
        oy1 = (bboxes[idx, 1] - dh) / ratio
        ox2 = (bboxes[idx, 2] - dw) / ratio
        oy2 = (bboxes[idx, 3] - dh) / ratio

        # Letterbox -> original coords for keypoints
        kpts = []
        for j in range(17):
            kx = (kpts_xy[j, 0, idx] - dw) / ratio
            ky = (kpts_xy[j, 1, idx] - dh) / ratio
            kv = float(kpts_vis[j, idx])
            kpts.append([float(kx), float(ky), kv])

        detections.append({
            "bbox": [float(ox1), float(oy1), float(ox2), float(oy2)],
            "conf": float(scores[idx]),
            "keypoints": kpts,
        })

    detections = nms(detections, iou_thres)
    return detections


def draw_pose(img, detections, vis_thres=0.5):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["conf"]
        kpts = det["keypoints"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for i, (kx, ky, kv) in enumerate(kpts):
            if kv > vis_thres:
                cv2.circle(img, (int(kx), int(ky)), 4, KPT_COLORS[i], -1)

        for si, (p1, p2) in enumerate(SKELETON):
            k1, k2 = kpts[p1 - 1], kpts[p2 - 1]
            if k1[2] > vis_thres and k2[2] > vis_thres:
                cv2.line(img,
                         (int(k1[0]), int(k1[1])),
                         (int(k2[0]), int(k2[1])),
                         KPT_COLORS[si % len(KPT_COLORS)], 2)
    return img


TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLOv8-pose Zoo RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLOv8-pose Zoo RKNN Inference")
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
    for i, det in enumerate(detections[:10]):
        b = det["bbox"]
        vis_count = sum(1 for kx, ky, kv in det["keypoints"] if kv > 0.5)
        print(f"  [{i}] person: {det['conf']:.3f}  bbox=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]  kpts_vis={vis_count}/17")
    img_draw = draw_pose(orig_img.copy(), detections)

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
