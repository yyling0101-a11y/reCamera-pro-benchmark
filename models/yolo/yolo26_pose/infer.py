#!/usr/bin/env python3
"""
YOLO26-pose RKNN inference & benchmark on RV1126B.

9 outputs (no DFL, direct bbox regression + objectness + keypoints):
  Per stride s in {8, 16, 32}:
    keypoint : [1, 51, H_s, W_s]   (17 kpts * 3: x_off, y_off, vis)
    bbox     : [1,  4, H_s, W_s]   (ltrb offsets, no DFL)
    objectness:[1,  1, H_s, W_s]   (needs sigmoid)
"""

import argparse, math, os, time, cv2, numpy as np

# ── COCO pose skeleton (1-indexed in spec, converted to 0-indexed below) ──────
SKELETON = [
    [16,14],[14,12],[17,15],[15,13],[12,13],
    [6,12],[7,13],[6,7],[6,8],[7,9],
    [8,10],[9,11],[2,3],[1,2],[1,3],
    [2,4],[3,5],[4,6],[5,7]
]
# Convert to 0-indexed
SKELETON_IDX = [[a - 1, b - 1] for a, b in SKELETON]

LIMB_COLORS = [
    (0,255,0),   (0,255,0),   (0,255,255), (0,255,255), (255,128,0),
    (255,128,0), (255,0,255), (255,0,255), (255,0,0),   (255,0,0),
    (255,255,0), (255,255,0), (0,128,255), (0,128,255), (0,128,255),
    (128,0,255), (128,0,255), (128,255,0), (128,255,0)
]

KPT_COLORS = [
    (255,0,0),   (255,0,0),   (255,0,0),   (255,0,0),   (255,0,0),
    (0,0,255),   (0,0,255),   (0,0,255),   (0,0,255),   (0,0,255),
    (0,255,0),   (0,255,0),   (0,255,0),   (0,255,0),   (0,255,0),
    (255,255,255),(255,255,255)
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


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


def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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


# ── YOLO26-pose post-processing ───────────────────────────────────────────────

def postprocess_pose(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """
    Decode 9 RKNN outputs into pose detections.

    Output mapping per stride group (stride 8 / 16 / 32):
      keypoint  : [1, 51, H, W]
      bbox      : [1,  4, H, W]   (direct ltrb offsets)
      objectness: [1,  1, H, W]   (sigmoid → confidence)
    """
    strides = [8, 16, 32]
    num_kpts = 17

    all_bboxes = []
    all_scores = []
    all_keypoints = []

    for si, stride in enumerate(strides):
        # Outputs grouped by branch: keypoint*3, bbox*3, objectness*3
        kpt_raw  = outputs[si].astype(np.float32)         # [1, 51, H, W]
        bbox_raw = outputs[si + 3].astype(np.float32)     # [1,  4, H, W]
        obj_raw  = outputs[si + 6].astype(np.float32)     # [1,  1, H, W]

        _, _, h, w = bbox_raw.shape

        # Objectness → confidence
        obj_conf = sigmoid(obj_raw[0, 0])                  # [H, W]

        # Bbox: direct regression [4, H, W] → ltrb
        bbox = bbox_raw[0]                                 # [4, H, W]
        bbox = bbox.transpose(1, 2, 0).reshape(-1, 4)      # [H*W, 4]

        # Anchor centres
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchors = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)  # [H*W, 2]
        anchors_abs = (anchors + 0.5) * stride             # pixel-space centres

        # Decode bbox (ltrb offsets from anchor)
        bx1 = anchors_abs[:, 0] - bbox[:, 0] * stride
        by1 = anchors_abs[:, 1] - bbox[:, 1] * stride
        bx2 = anchors_abs[:, 0] + bbox[:, 2] * stride
        by2 = anchors_abs[:, 1] + bbox[:, 3] * stride
        boxes_stride = np.stack([bx1, by1, bx2, by2], axis=-1)  # [H*W, 4]

        scores_stride = obj_conf.flatten()                   # [H*W]

        # Keypoints: [51, H, W] → [17, 3, H, W]
        kpt = kpt_raw[0].reshape(num_kpts, 3, h, w)        # [17, 3, H, W]
        # xy offsets: first 2 channels per keypoint
        kpt_xy = kpt[:, :2, :, :]                          # [17, 2, H, W]
        kpt_xy = kpt_xy.transpose(2, 3, 0, 1)              # [H, W, 17, 2]
        kpt_xy = kpt_xy.reshape(-1, num_kpts, 2)           # [H*W, 17, 2]
        # Absolute keypoint positions
        kpt_xy_abs = (anchors[:, np.newaxis, :] + kpt_xy) * stride  # [H*W, 17, 2]

        # Visibility: sigmoid of last channel per keypoint
        kpt_vis = kpt[:, 2, :, :]                          # [17, H, W]
        kpt_vis = sigmoid(kpt_vis)                          # [17, H, W]
        kpt_vis = kpt_vis.transpose(1, 2, 0).reshape(-1, num_kpts)  # [H*W, 17]

        all_bboxes.append(boxes_stride)
        all_scores.append(scores_stride)
        all_keypoints.append(
            np.concatenate([kpt_xy_abs, kpt_vis[..., np.newaxis]], axis=-1)
        )  # [H*W, 17, 3]

    bboxes = np.concatenate(all_bboxes, axis=0)        # [N, 4]
    scores = np.concatenate(all_scores, axis=0)        # [N]
    keypoints = np.concatenate(all_keypoints, axis=0)  # [N, 17, 3]

    # Confidence filter
    mask = scores > conf_thres
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    bboxes_filt = bboxes[indices]
    scores_filt = scores[indices]
    kpts_filt = keypoints[indices]

    # Map back to original image coordinates
    dw, dh = pad
    boxes_orig = bboxes_filt.copy()
    boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - dw) / ratio
    boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - dh) / ratio

    kpts_orig = kpts_filt.copy()
    kpts_orig[..., 0] = (kpts_orig[..., 0] - dw) / ratio
    kpts_orig[..., 1] = (kpts_orig[..., 1] - dh) / ratio

    keep = nms(boxes_orig, scores_filt, iou_thres)

    results = []
    for k in keep:
        results.append({
            "bbox": boxes_orig[k].tolist(),
            "conf": float(scores_filt[k]),
            "keypoints": kpts_orig[k]   # [17, 3] (x, y, vis)
        })
    return results


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_pose(img, dets):
    overlay = img.copy()

    for d in dets:
        b = d["bbox"]
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        c = (0, 255, 0)
        kpts = d["keypoints"]  # [17, 3]

        # Draw bones
        for li, (i, j) in enumerate(SKELETON_IDX):
            if kpts[i, 2] > 0.5 and kpts[j, 2] > 0.5:
                pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                lc = LIMB_COLORS[li % len(LIMB_COLORS)]
                cv2.line(overlay, pt1, pt2, lc, 2)

        # Draw keypoints
        for ki in range(len(kpts)):
            if kpts[ki, 2] > 0.5:
                kp = (int(kpts[ki, 0]), int(kpts[ki, 1]))
                kc = KPT_COLORS[ki % len(KPT_COLORS)]
                cv2.circle(overlay, kp, 3, kc, -1)
                cv2.circle(overlay, kp, 4, (0, 0, 0), 1)

        # Draw bbox + label
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        lbl = f"person: {d['conf']:.2f}"
        cv2.putText(overlay, lbl, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

    return overlay


# ── Main ───────────────────────────────────────────────────────────────────────

TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO26-pose RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO26-pose RKNN Inference")
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
    dets = postprocess_pose(last_outputs, ratio, (dw, dh),
                             args.conf, args.nms, args.imgsz)
    print(f"Detections: {len(dets)}")
    for i, d in enumerate(dets[:10]):
        b = d["bbox"]
        vis_count = int((d["keypoints"][:, 2] > 0.5).sum())
        print(f"  [{i}] person: {d['conf']:.3f} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] kpts_vis={vis_count}/17")
    img_draw = draw_pose(orig_img.copy(), dets)

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
