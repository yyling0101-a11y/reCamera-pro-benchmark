#!/usr/bin/env python3
"""
YOLO26n-pose RKNN inference with visualization.

Usage (on device):
    python3 yolo26n_pose_infer.py \
        --model /userdata/benchmark/models/yolo26n-pose_640x640_W8A8.rknn \
        --image /userdata/benchmark/test_bus.jpg \
        --output /userdata/benchmark/output/pose_result.jpg

Usage (on host, for simulation):
    python3 yolo26n_pose_infer.py \
        --model models/yolo26n-pose_640x640_W8A8.rknn \
        --image assets/bus.jpg \
        --output output/pose_result.jpg \
        --rknn-full

Note:
    The RKNN model outputs raw logits (class) and decoded pixel coords (bbox/kpts),
    but sigmoid was removed from the ONNX graph before quantization to avoid
    INT8 precision loss on [0,1] range. This script applies sigmoid in Python.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with aspect ratio preserved, pad with gray."""
    shape = img.shape[:2]
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


def _build_anchors(imgsz=640):
    """Build anchor points and stride array for 3-level detection."""
    strides = [8.0, 16.0, 32.0]
    anchor_points = []
    stride_arr = []
    for s in strides:
        si = int(s)
        h, w = imgsz // si, imgsz // si
        yv, xv = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing='ij',
        )
        pts = np.stack([xv.ravel(), yv.ravel()], axis=-1) + 0.5
        anchor_points.append(pts)
        stride_arr.append(np.full(h * w, s, dtype=np.float32))
    return np.concatenate(anchor_points), np.concatenate(stride_arr)


ANCHORS, STRIDES = _build_anchors()
ANCHOR_T = ANCHORS.T  # [2, 8400]


def postprocess(raw_output, ratio, pad, conf_thres=0.25, iou_thres=0.45):
    """
    Post-process raw RKNN output.

    raw_output shape: [1, 56, 8400]
        [0:4]   bbox distances (letterbox grid units) - need dist2bbox
        [4:5]   class logits - need sigmoid
        [5:56]  kpts [17 * (x_offset, y_offset, vis_logit)] - need anchor+stride decode + sigmoid on vis
    """
    pred = raw_output[0]
    bbox_raw = pred[:4, :]
    class_raw = pred[4:5, :]
    kpts_raw = pred[5:, :]

    # dist2bbox: convert distances to pixel-space xyxy
    lt = ANCHOR_T - bbox_raw[:2, :]
    rb = ANCHOR_T + bbox_raw[2:, :]
    center = (lt + rb) / 2.0 * STRIDES[np.newaxis, :]
    wh = (rb - lt) * STRIDES[np.newaxis, :]
    x1_lb = center[0:1, :] - wh[0:1, :] / 2.0
    y1_lb = center[1:2, :] - wh[1:2, :] / 2.0
    x2_lb = center[0:1, :] + wh[0:1, :] / 2.0
    y2_lb = center[1:2, :] + wh[1:2, :] / 2.0

    # sigmoid on class logits
    class_conf = 1.0 / (1.0 + np.exp(-np.clip(class_raw, -50, 50)))

    # decode keypoints
    kpts_r = kpts_raw.reshape(17, 3, -1)
    kpt_xy = kpts_r[:, :2, :]
    kpt_vis = kpts_r[:, 2:3, :]
    kpt_xy_dec = (kpt_xy + ANCHOR_T[np.newaxis, :, :]) * STRIDES[np.newaxis, np.newaxis, :]
    kpt_vis_dec = 1.0 / (1.0 + np.exp(-np.clip(kpt_vis, -50, 50)))

    # letterbox -> original image coords
    dw, dh = pad
    x1 = (x1_lb - dw) / ratio
    y1 = (y1_lb - dh) / ratio
    x2 = (x2_lb - dw) / ratio
    y2 = (y2_lb - dh) / ratio

    # collect detections above threshold
    mask = class_conf[0] > conf_thres
    indices = np.where(mask)[0]

    detections = []
    for idx in indices:
        bbox = [float(x1[0, idx]), float(y1[0, idx]), float(x2[0, idx]), float(y2[0, idx])]
        kpts = [
            [
                float((kpt_xy_dec[j, 0, idx] - dw) / ratio),
                float((kpt_xy_dec[j, 1, idx] - dh) / ratio),
                float(kpt_vis_dec[j, 0, idx]),
            ]
            for j in range(17)
        ]
        detections.append({'bbox': bbox, 'conf': float(class_conf[0, idx]), 'keypoints': kpts})

    # NMS
    if detections:
        detections = _nms(detections, iou_thres)

    return detections


def _nms(detections, iou_thres):
    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['conf'] for d in detections])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter = (
            np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
            * np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        )
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return [detections[i] for i in keep]


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


def draw_pose(img, detections, vis_thres=0.5):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        kpts = det['keypoints']
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img, f"person {det['conf']:.2f}",
            (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        for i, (kx, ky, kv) in enumerate(kpts):
            if kv > vis_thres:
                cv2.circle(img, (int(kx), int(ky)), 4, KPT_COLORS[i], -1)
        for i, (p1, p2) in enumerate(SKELETON):
            k1, k2 = kpts[p1 - 1], kpts[p2 - 1]
            if k1[2] > vis_thres and k2[2] > vis_thres:
                cv2.line(
                    img,
                    (int(k1[0]), int(k1[1])), (int(k2[0]), int(k2[1])),
                    KPT_COLORS[i % len(KPT_COLORS)], 2,
                )
    return img


def main():
    parser = argparse.ArgumentParser(description='YOLO26n-pose RKNN inference')
    parser.add_argument('--model', required=True, help='Path to .rknn model')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', default=None, help='Path to save result image')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--rknn-full', action='store_true',
                        help='Use full rknn.api.RKNN (host) instead of rknnlite (device)')
    parser.add_argument('--show', action='store_true', help='Display result in window')
    args = parser.parse_args()

    # load runtime
    if args.rknn_full:
        from rknn.api import RKNN
        rknn = RKNN()
        rknn_cls = RKNN
    else:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        rknn_cls = RKNNLite

    print(f"Loading model: {args.model}")
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        sys.exit(f"load_rknn failed: {ret}")

    print("Initializing runtime...")
    if args.rknn_full:
        ret = rknn.init_runtime(target='rv1126b')
    else:
        ret = rknn.init_runtime(core_mask=rknn_cls.NPU_CORE_AUTO)
    if ret != 0:
        sys.exit(f"init_runtime failed: {ret}")

    # load and preprocess image
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read image: {args.image}")
    print(f"Image: {img.shape}")

    img_lb, ratio, (dw, dh) = letterbox(img, (args.imgsz, args.imgsz))

    if args.rknn_full:
        # host RKNN expects NCHW float32 (or uint8 with batch dim)
        img_input = img_lb[np.newaxis, :, :, :].astype(np.uint8)
    else:
        # RKNNLite on device expects NHWC uint8
        img_input = img_lb[np.newaxis, :, :, :].astype(np.uint8)

    # inference
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[img_input])
    t1 = time.perf_counter()
    print(f"Inference: {(t1 - t0) * 1000:.1f} ms")

    raw = outputs[0]
    print(f"Output: {raw.shape}, dtype={raw.dtype}, range=[{raw.min():.3f}, {raw.max():.3f}]")

    # postprocess
    detections = postprocess(raw, ratio, (dw, dh), conf_thres=args.conf, iou_thres=args.iou)
    print(f"Detections: {len(detections)}")
    for i, det in enumerate(detections):
        b = det['bbox']
        print(f"  [{i}] conf={det['conf']:.3f} bbox=[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]")
        for j in range(17):
            kx, ky, kv = det['keypoints'][j]
            vis_str = '*' if kv > 0.5 else ' '
            print(f"      kpt[{j:2d}]{vis_str} ({kx:6.1f}, {ky:6.1f}) vis={kv:.3f}")

    # draw
    img_draw = draw_pose(img.copy(), detections, vis_thres=0.5)

    # save / show
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        cv2.imwrite(args.output, img_draw)
        print(f"Saved: {args.output}")

    if args.show:
        cv2.imshow('YOLO26n-pose', img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    rknn.release()


if __name__ == '__main__':
    main()
