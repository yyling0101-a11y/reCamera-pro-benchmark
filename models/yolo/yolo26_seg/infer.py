#!/usr/bin/env python3
"""
YOLO26-seg RKNN inference on RV1126B.
10 outputs, no DFL, direct 4-channel bbox regression + mask proto.
"""

import argparse, os, time, cv2, numpy as np

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
    'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','wine glass','cup',
    'fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

COLORS = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in range(80)]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


import sys as _sys
_tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools')
if os.path.isdir(_tools_dir) and _tools_dir not in _sys.path:
    _sys.path.insert(0, _tools_dir)
from rga_preprocess import RGAPreprocessor

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w, h
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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

def postprocess_seg(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """
    YOLO26-seg 10 outputs (no DFL):
      stride 8:  bbox[1,4,80,80]  cls[1,80,80,80]  mask_coeff[1,32,80,80]
      stride 16: bbox[1,4,40,40]  cls[1,80,40,40]  mask_coeff[1,32,40,40]
      stride 32: bbox[1,4,20,20]  cls[1,80,20,20]  mask_coeff[1,32,20,20]
      proto:     [1,32,160,160]
    """
    strides = [8, 16, 32]
    num_classes = 80
    num_mask_coeffs = 32

    # proto mask prototype [1, 32, 160, 160] → [32, 160*160]
    proto = outputs[9].astype(np.float32)
    proto_flat = proto[0].reshape(num_mask_coeffs, -1)  # [32, 25600]
    proto_h, proto_w = proto.shape[2], proto.shape[3]   # 160, 160

    all_bboxes = []
    all_scores = []
    all_classes = []
    all_mask_coeffs = []

    for i, stride in enumerate(strides):
        # Outputs grouped by branch: bbox*3, cls*3, mask_coeff*3, proto
        bbox_raw = outputs[i].astype(np.float32)          # [1, 4, H, W]
        cls_raw = outputs[i + 3].astype(np.float32)       # [1, 80, H, W]
        mask_coeff_raw = outputs[i + 6].astype(np.float32) # [1, 32, H, W]

        _, _, gh, gw = bbox_raw.shape

        # Build grid anchors: center = grid_pos + 0.5
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        anchors = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32) + 0.5  # [H*W, 2]

        # Decode bbox: [1, 4, H, W] → [H*W, 4]
        bbox_data = bbox_raw[0].reshape(4, -1).transpose(1, 0)  # [H*W, 4]

        # lt = anchor - bbox[0:2], rb = anchor + bbox[2:4]
        lt = anchors - bbox_data[:, 0:2]
        rb = anchors + bbox_data[:, 2:4]

        # Convert to pixel coords in the letterboxed image
        xyxy = np.concatenate([lt, rb], axis=-1) * stride  # [H*W, 4]

        # Class scores: [1, 80, H, W] → [H*W, 80]
        cls_scores = cls_raw[0].reshape(num_classes, -1).transpose(1, 0)  # [H*W, 80]

        # Mask coefficients: [1, 32, H, W] → [32, H*W]
        mask_coeffs = mask_coeff_raw[0].reshape(num_mask_coeffs, -1)  # [32, H*W]

        max_score = cls_scores.max(axis=1)
        max_cls = cls_scores.argmax(axis=1)

        all_bboxes.append(xyxy)
        all_scores.append(max_score)
        all_classes.append(max_cls)
        all_mask_coeffs.append(mask_coeffs)

    bboxes = np.concatenate(all_bboxes, axis=0)          # [total, 4]
    scores = np.concatenate(all_scores, axis=0)           # [total]
    classes = np.concatenate(all_classes, axis=0)          # [total]
    mask_coeffs_all = np.concatenate(all_mask_coeffs, axis=1)  # [32, total]

    # Confidence filter
    mask = scores > conf_thres
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    # Map boxes back to original image coordinates
    dw, dh = pad
    boxes_orig = []
    for idx in indices:
        x1 = (bboxes[idx, 0] - dw) / ratio
        y1 = (bboxes[idx, 1] - dh) / ratio
        x2 = (bboxes[idx, 2] - dw) / ratio
        y2 = (bboxes[idx, 3] - dh) / ratio
        boxes_orig.append([x1, y1, x2, y2])
    boxes_orig = np.array(boxes_orig)

    scores_filt = scores[indices]
    classes_filt = classes[indices]
    coeffs_filt = mask_coeffs_all[:, indices]  # [32, num_candidates]

    # NMS
    keep = nms(boxes_orig, scores_filt, iou_thres)
    if len(keep) == 0:
        return []

    keep_set = set(keep)
    keep = sorted(keep, key=lambda k: -scores_filt[k])

    results = []
    for k in keep:
        if k not in keep_set:
            continue
        # Generate mask for this detection
        coeff = coeffs_filt[:, k]  # [32]
        # mask = sigmoid(coeff @ proto_flat)  → [proto_h * proto_w]
        mask_raw = coeff @ proto_flat  # [25600]
        mask_sigmoid = sigmoid(mask_raw)
        mask_2d = mask_sigmoid.reshape(proto_h, proto_w)  # [160, 160]

        # Convert mask from letterbox proto coords to original image coords
        # First crop the letterbox padding from the proto mask
        # Proto is at 160x160, letterbox image is imgsz x imgsz
        # Scale factor from proto to letterbox: imgsz / proto_w
        # Padding in proto coords:
        proto_pad_x = dw * (proto_w / imgsz)
        proto_pad_y = dh * (proto_h / imgsz)
        # Crop to content region in proto
        cx1 = int(round(proto_pad_x))
        cy1 = int(round(proto_pad_y))
        cx2 = proto_w - int(round(proto_pad_x))
        cy2 = proto_h - int(round(proto_pad_y))
        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(proto_w, cx2)
        cy2 = min(proto_h, cy2)

        mask_cropped = mask_2d[cy1:cy2, cx1:cx2]
        if mask_cropped.size == 0:
            continue

        # Resize to original image size
        # The content region maps to the full original image
        # Calculate the original image dimensions
        orig_h = int(round((imgsz - 2 * dh) / ratio))
        orig_w = int(round((imgsz - 2 * dw) / ratio))
        if orig_h <= 0 or orig_w <= 0:
            continue
        mask_resized = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Clip boxes to image bounds
        bx1 = max(0, boxes_orig[k, 0])
        by1 = max(0, boxes_orig[k, 1])
        bx2 = min(orig_w, boxes_orig[k, 2])
        by2 = min(orig_h, boxes_orig[k, 3])

        results.append({
            "class": int(classes_filt[k]),
            "conf": float(scores_filt[k]),
            "bbox": [bx1, by1, bx2, by2],
            "mask": mask_binary.astype(np.float32)
        })

    return results

def draw_seg(img, dets):
    overlay = img.copy()
    for d in dets:
        c = COLORS[d["class"] % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        mask = d.get("mask")
        if mask is not None:
            ih, iw = overlay.shape[:2]
            mf = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR) if mask.shape != (ih, iw) else mask
            colored = np.full_like(overlay, c, dtype=np.uint8)
            mb = (mf > 0.5).astype(np.uint8)
            overlay = np.where(mb[:, :, np.newaxis] == 1,
                               (overlay * 0.5 + colored * 0.5).astype(np.uint8), overlay)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), c, 2)
        lbl = f"{COCO_CLASSES[d['class']] if d['class'] < len(COCO_CLASSES) else d['class']}: {d['conf']:.2f}"
        cv2.putText(overlay, lbl, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return overlay

TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO26-seg RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO26-seg RKNN Inference")
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
    preproc = RGAPreprocessor(target_size=(args.imgsz, args.imgsz))
    print(f"Preprocessor: {'RGA' if preproc.available else 'CPU'} letterbox")


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
            print("ERROR: Cannot open camera via GStreamer (/dev/video13)"); preproc.release(); rknn.release(); return 1
        print("Camera /dev/video13 opened via GStreamer")

        print(f"\nWarmup ({WARMUP_RUNS} iters)...")
        for _ in range(WARMUP_RUNS):
            ret_cam, frame = cap.read()
            if ret_cam:
                img_lb, _, _ = preproc.process(frame)
                inp = img_lb[np.newaxis, :, :, :].astype(np.uint8)
                rknn.inference(inputs=[inp])

        print(f"Benchmark ({TOTAL_RUNS} runs, skipping first {WARMUP_RUNS})...")
        latencies = []
        for i in range(TOTAL_RUNS):
            ret_cam, frame = cap.read()
            if not ret_cam:
                continue
            t0 = time.perf_counter()
            img_lb, _, _ = preproc.process(frame)
            inp = img_lb[np.newaxis, :, :, :].astype(np.uint8)
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

        preproc.release()

        rknn.release()
        print(f"{'='*60}\n")
        return 0

    # Image mode
    print(f"Image: {args.input}")
    orig_img = cv2.imread(args.input)
    if orig_img is None:
        print(f"ERROR: Cannot read {args.input}"); preproc.release(); rknn.release(); return 1
    print(f"Image size: {orig_img.shape}")

    img_lb, ratio, (dw, dh) = letterbox(orig_img, (args.imgsz, args.imgsz))
    img_input = img_lb[np.newaxis, :, :, :].astype(np.uint8)

    print(f"\nWarmup ({WARMUP_RUNS} iters)...")
    for _ in range(WARMUP_RUNS):
        img_lb_w, _, _ = preproc.process(orig_img)
        img_input_w = img_lb_w[np.newaxis, :, :, :].astype(np.uint8)
        rknn.inference(inputs=[img_input_w])

    print(f"Benchmark ({TOTAL_RUNS} runs, skipping first {WARMUP_RUNS})...")
    latencies = []
    last_outputs = None
    for i in range(TOTAL_RUNS):
        t0 = time.perf_counter()
        img_lb, ratio, (dw, dh) = preproc.process(orig_img)
        img_input = img_lb[np.newaxis, :, :, :].astype(np.uint8)
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
    dets = postprocess_seg(last_outputs, ratio, (dw, dh), args.conf, args.nms, args.imgsz)
    print(f"Detections: {len(dets)}")
    for i, d in enumerate(dets[:10]):
        b = d["bbox"]
        cls_name = COCO_CLASSES[d["class"]] if d["class"] < len(COCO_CLASSES) else str(d["class"])
        mask_px = int((d["mask"] > 0.5).sum()) if d.get("mask") is not None else 0
        print(f"  [{i}] {cls_name}: {d['conf']:.3f} bbox=[{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}] mask={mask_px}px")
    img_draw = draw_seg(orig_img.copy(), dets)

    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"result_{input_basename}.jpg")
    cv2.imwrite(output_path, img_draw)
    print(f"Saved: {output_path}")

    preproc.release()

    rknn.release()
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    exit(main())
