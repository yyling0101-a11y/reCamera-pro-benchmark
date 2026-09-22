#!/usr/bin/env python3
"""
Batch YOLO RKNN inference with performance benchmarking + visual verification.
Runs inference 10 times, measures NPU latency, saves visual result from first run.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np


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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def nms(boxes, scores, iou_thres=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter = np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])) * \
                np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


def postprocess_detect(raw, ratio, pad, conf_thres=0.25):
    pred = raw[0]
    if pred.shape[0] == 84:
        bbox = pred[:4, :]
        cls = pred[4:, :]
    elif pred.shape[0] == 6:
        bbox = pred[:4, :]
        conf = pred[4:5, :]
        cls = conf
    else:
        print(f"Unknown output shape: {pred.shape}")
        return []

    if pred.shape[0] == 84:
        strides = [8.0, 16.0, 32.0]
        anchors = []
        for s in strides:
            si = int(s)
            h, w = 640 // si, 640 // si
            yv, xv = np.meshgrid(np.arange(h, dtype=np.float32),
                                 np.arange(w, dtype=np.float32), indexing='ij')
            pts = np.stack([xv.ravel(), yv.ravel()], axis=-1) + 0.5
            anchors.append(pts)
        anchors = np.concatenate(anchors).T
        lt = anchors - bbox[:2, :]
        rb = anchors + bbox[2:, :]
        center = (lt + rb) / 2.0
        wh = rb - lt
        bbox = np.vstack([center, wh])

    if cls.max() > 1.0 or cls.min() < 0.0:
        cls = 1.0 / (1.0 + np.exp(-np.clip(cls, -50, 50)))

    if pred.shape[0] == 84:
        best_cls = cls.max(axis=0)
        cls_id = cls.argmax(axis=0)
    else:
        best_cls = cls[0]
        cls_id = np.zeros_like(best_cls, dtype=int)

    dw, dh = pad
    cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x1 = (cx - w/2 - dw) / ratio
    y1 = (cy - h/2 - dh) / ratio
    x2 = (cx + w/2 - dw) / ratio
    y2 = (cy + h/2 - dh) / ratio

    mask = best_cls > conf_thres
    dets = []
    for i in np.where(mask)[0]:
        dets.append({
            'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            'conf': float(best_cls[i]),
            'class': int(cls_id[i])
        })

    if dets:
        boxes = np.array([d['bbox'] for d in dets])
        scores = np.array([d['conf'] for d in dets])
        keep = nms(boxes, scores, 0.45)
        dets = [dets[i] for i in keep]

    return dets


def postprocess_pose(raw, ratio, pad, conf_thres=0.25):
    pred = raw[0]
    bbox_raw = pred[:4, :]
    cls_raw = pred[4:5, :]
    kpts_raw = pred[5:, :]

    strides = [8.0, 16.0, 32.0]
    anchors = []
    stride_arr = []
    for s in strides:
        si = int(s)
        h, w = 640 // si, 640 // si
        yv, xv = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing='ij')
        pts = np.stack([xv.ravel(), yv.ravel()], axis=-1) + 0.5
        anchors.append(pts)
        stride_arr.append(np.full(h * w, s, dtype=np.float32))
    anchors = np.concatenate(anchors)
    stride_arr = np.concatenate(stride_arr)
    anchor_t = anchors.T

    lt = anchor_t - bbox_raw[:2, :]
    rb = anchor_t + bbox_raw[2:, :]
    center = (lt + rb) / 2.0 * stride_arr[np.newaxis, :]
    wh = (rb - lt) * stride_arr[np.newaxis, :]

    cls_conf = 1.0 / (1.0 + np.exp(-np.clip(cls_raw, -50, 50)))

    kpts_r = kpts_raw.reshape(17, 3, -1)
    kpt_xy = kpts_r[:, :2, :]
    kpt_vis = kpts_r[:, 2:3, :]
    kpt_xy_dec = (kpt_xy + anchor_t[np.newaxis, :, :]) * stride_arr[np.newaxis, np.newaxis, :]
    kpt_vis_dec = 1.0 / (1.0 + np.exp(-np.clip(kpt_vis, -50, 50)))

    dw, dh = pad
    x1 = (center[0:1, :] - wh[0:1, :]/2 - dw) / ratio
    y1 = (center[1:2, :] - wh[1:2, :]/2 - dh) / ratio
    x2 = (center[0:1, :] + wh[0:1, :]/2 - dw) / ratio
    y2 = (center[1:2, :] + wh[1:2, :]/2 - dh) / ratio

    mask = cls_conf[0] > conf_thres
    dets = []
    for i in np.where(mask)[0]:
        kpts = [[float((kpt_xy_dec[j, 0, i] - dw) / ratio),
                 float((kpt_xy_dec[j, 1, i] - dh) / ratio),
                 float(kpt_vis_dec[j, 0, i])] for j in range(17)]
        dets.append({
            'bbox': [float(x1[0, i]), float(y1[0, i]), float(x2[0, i]), float(y2[0, i])],
            'conf': float(cls_conf[0, i]),
            'class': 0,
            'keypoints': kpts
        })

    if dets:
        boxes = np.array([d['bbox'] for d in dets])
        scores = np.array([d['conf'] for d in dets])
        keep = nms(boxes, scores, 0.45)
        dets = [dets[i] for i in keep]

    return dets


def postprocess_seg(raw, proto, ratio, pad, conf_thres=0.25):
    pred = raw[0]
    bbox_raw = pred[:4, :]
    cls_raw = pred[4:84, :]

    strides = [8.0, 16.0, 32.0]
    anchors = []
    stride_arr = []
    for s in strides:
        si = int(s)
        h, w = 640 // si, 640 // si
        yv, xv = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing='ij')
        pts = np.stack([xv.ravel(), yv.ravel()], axis=-1) + 0.5
        anchors.append(pts)
        stride_arr.append(np.full(h * w, s, dtype=np.float32))
    anchors = np.concatenate(anchors)
    stride_arr = np.concatenate(stride_arr)
    anchor_t = anchors.T

    lt = anchor_t - bbox_raw[:2, :]
    rb = anchor_t + bbox_raw[2:, :]
    center = (lt + rb) / 2.0 * stride_arr[np.newaxis, :]
    wh = (rb - lt) * stride_arr[np.newaxis, :]

    cls_conf = 1.0 / (1.0 + np.exp(-np.clip(cls_raw, -50, 50)))
    best_cls = cls_conf.max(axis=0)
    cls_id = cls_conf.argmax(axis=0)

    dw, dh = pad
    x1 = (center[0:1, :] - wh[0:1, :]/2 - dw) / ratio
    y1 = (center[1:2, :] - wh[1:2, :]/2 - dh) / ratio
    x2 = (center[0:1, :] + wh[0:1, :]/2 - dw) / ratio
    y2 = (center[1:2, :] + wh[1:2, :]/2 - dh) / ratio

    mask = best_cls > conf_thres
    dets = []
    for i in np.where(mask)[0]:
        dets.append({
            'bbox': [float(x1[0, i]), float(y1[0, i]), float(x2[0, i]), float(y2[0, i])],
            'conf': float(best_cls[i]),
            'class': int(cls_id[i])
        })

    if dets:
        boxes = np.array([d['bbox'] for d in dets])
        scores = np.array([d['conf'] for d in dets])
        keep = nms(boxes, scores, 0.45)
        dets = [dets[i] for i in keep]

    return dets


def postprocess_obb(raw, ratio, pad, conf_thres=0.25):
    pred = raw[0]
    bbox_raw = pred[:4, :]
    angle = pred[4:5, :]
    cls_raw = pred[5:, :]

    strides = [8.0, 16.0, 32.0]
    anchors = []
    stride_arr = []
    for s in strides:
        si = int(s)
        h, w = 640 // si, 640 // si
        yv, xv = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing='ij')
        pts = np.stack([xv.ravel(), yv.ravel()], axis=-1) + 0.5
        anchors.append(pts)
        stride_arr.append(np.full(h * w, s, dtype=np.float32))
    anchors = np.concatenate(anchors)
    stride_arr = np.concatenate(stride_arr)
    anchor_t = anchors.T

    lt = anchor_t - bbox_raw[:2, :]
    rb = anchor_t + bbox_raw[2:, :]
    center = (lt + rb) / 2.0 * stride_arr[np.newaxis, :]
    wh = (rb - lt) * stride_arr[np.newaxis, :]

    cls_conf = 1.0 / (1.0 + np.exp(-np.clip(cls_raw, -50, 50)))
    best_cls = cls_conf.max(axis=0)
    cls_id = cls_conf.argmax(axis=0)

    dw, dh = pad
    x1 = (center[0:1, :] - wh[0:1, :]/2 - dw) / ratio
    y1 = (center[1:2, :] - wh[1:2, :]/2 - dh) / ratio
    x2 = (center[0:1, :] + wh[0:1, :]/2 - dw) / ratio
    y2 = (center[1:2, :] + wh[1:2, :]/2 - dh) / ratio

    mask = best_cls > conf_thres
    dets = []
    for i in np.where(mask)[0]:
        dets.append({
            'bbox': [float(x1[0, i]), float(y1[0, i]), float(x2[0, i]), float(y2[0, i])],
            'conf': float(best_cls[i]),
            'class': int(cls_id[i])
        })

    if dets:
        boxes = np.array([d['bbox'] for d in dets])
        scores = np.array([d['conf'] for d in dets])
        keep = nms(boxes, scores, 0.45)
        dets = [dets[i] for i in keep]

    return dets


def postprocess_cls(raw):
    pred = raw[0]
    if pred.ndim > 1:
        pred = pred.squeeze()
    probs = np.exp(pred) / np.exp(pred).sum()
    top5 = np.argsort(probs)[::-1][:5]
    return [{'class': int(i), 'conf': float(probs[i])} for i in top5]


def draw_detect(img, dets):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        conf = det['conf']
        cls = det.get('class', 0)
        color = colors[cls % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{cls} {conf:.2f}', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_pose(img, dets):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]]
    colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
              (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
              (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
              (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)]
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"person {det['conf']:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        kpts = det.get('keypoints', [])
        for i, (kx, ky, kv) in enumerate(kpts):
            if kv > 0.5:
                cv2.circle(img, (int(kx), int(ky)), 4, colors[i], -1)
        for i, (p1, p2) in enumerate(skeleton):
            k1, k2 = kpts[p1-1], kpts[p2-1]
            if k1[2] > 0.5 and k2[2] > 0.5:
                cv2.line(img, (int(k1[0]), int(k1[1])), (int(k2[0]), int(k2[1])),
                        colors[i % len(colors)], 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--runs', type=int, default=10, help='Number of inference runs for benchmarking')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    parser.add_argument('--rknn-lite', action='store_true')
    args = parser.parse_args()

    # Load runtime
    if args.rknn_lite:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
    else:
        from rknn.api import RKNN
        rknn = RKNN()

    model_name = os.path.basename(args.model).replace('.rknn', '')
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    print(f"Loading {args.model}...")
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        sys.exit(f"load_rknn failed: {ret}")

    if args.rknn_lite:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    else:
        ret = rknn.init_runtime(target='rv1126b')
    if ret != 0:
        sys.exit(f"init_runtime failed: {ret}")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read {args.image}")

    # Determine model type and input size
    if 'cls' in model_name or '224x224' in model_name:
        task = 'cls'
        imgsz = 224
    elif 'pose' in model_name:
        task = 'pose'
        imgsz = 640
    elif 'seg' in model_name:
        task = 'seg'
        imgsz = 640
    elif 'obb' in model_name:
        task = 'obb'
        imgsz = 640
    elif 'depth' in model_name:
        task = 'depth'
        imgsz = 640
    else:
        task = 'detect'
        imgsz = 640

    print(f"Task: {task}, Input: {imgsz}x{imgsz}")

    # Preprocess
    img_lb, ratio, (dw, dh) = letterbox(img, (imgsz, imgsz))
    img_input = img_lb[np.newaxis, :, :, :].astype(np.uint8)

    # Warmup
    print(f"Warming up ({args.warmup} iters)...")
    for _ in range(args.warmup):
        rknn.inference(inputs=[img_input])

    # Benchmark: run N times, measure NPU time
    print(f"Benchmarking ({args.runs} runs)...")
    latencies = []
    first_output = None
    for i in range(args.runs):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[img_input])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if i == 0:
            first_output = outputs

    latencies = np.array(latencies)
    avg_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    fps = 1000.0 / avg_ms

    print(f"\nNPU Performance:")
    print(f"  Runs: {args.runs}")
    print(f"  Avg latency: {avg_ms:.2f} ms")
    print(f"  Min latency: {min_ms:.2f} ms")
    print(f"  Max latency: {max_ms:.2f} ms")
    print(f"  FPS: {fps:.1f}")

    # Postprocess first result
    print(f"\nVisual verification:")
    if task == 'detect':
        dets = postprocess_detect(first_output[0], ratio, (dw, dh), args.conf)
        print(f"  Detections: {len(dets)}")
        for d in dets[:5]:
            print(f"    class={d['class']}, conf={d['conf']:.3f}, bbox={[f'{v:.1f}' for v in d['bbox']]}")
        img_out = draw_detect(img.copy(), dets)
    elif task == 'pose':
        dets = postprocess_pose(first_output[0], ratio, (dw, dh), args.conf)
        print(f"  Poses: {len(dets)}")
        for d in dets[:5]:
            print(f"    conf={d['conf']:.3f}, bbox={[f'{v:.1f}' for v in d['bbox']]}")
        img_out = draw_pose(img.copy(), dets)
    elif task == 'seg':
        dets = postprocess_seg(first_output[0], first_output[1] if len(first_output) > 1 else None,
                               ratio, (dw, dh), args.conf)
        print(f"  Segmentations: {len(dets)}")
        for d in dets[:5]:
            print(f"    class={d['class']}, conf={d['conf']:.3f}")
        img_out = draw_detect(img.copy(), dets)
    elif task == 'obb':
        dets = postprocess_obb(first_output[0], ratio, (dw, dh), args.conf)
        print(f"  OBB: {len(dets)}")
        for d in dets[:5]:
            print(f"    class={d['class']}, conf={d['conf']:.3f}")
        img_out = draw_detect(img.copy(), dets)
    elif task == 'cls':
        top5 = postprocess_cls(first_output[0])
        print(f"  Top-5 classes:")
        for t in top5:
            print(f"    class={t['class']}, prob={t['conf']:.3f}")
        img_out = img.copy()
        y = 30
        for t in top5:
            cv2.putText(img_out, f"cls={t['class']} {t['conf']:.3f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30
    elif task == 'depth':
        depth = first_output[0].squeeze()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
        img_out = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
        img_out = cv2.resize(img_out, (img.shape[1], img.shape[0]))
        print(f"  Depth map generated")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, img_out)
    print(f"\nSaved: {args.output}")

    rknn.release()
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
