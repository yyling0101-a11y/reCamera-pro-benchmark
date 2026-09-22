#!/usr/bin/env python3
"""
YOLOv8n-seg Zoo RKNN inference on RV1126B.

Output structure (13 tensors):
  [0-3]  stride 8:  bbox_dfl[1,64,80,80], cls_sig[1,80,80,80], cls_sum[1,1,80,80], mask_coeff[1,32,80,80]
  [4-7]  stride 16: same pattern at 40x40
  [8-11] stride 32: same pattern at 20x20
  [12]   proto: [1,32,160,160]

Usage (on device):
    python3 infer.py \
        --model /userdata/benchmark/models/yolo/yolov8_seg/yolov8n-seg_W8A8.rknn \
        --image /userdata/benchmark/test_bus.jpg \
        --output /userdata/benchmark/models/yolo/yolov8_seg/result_bus.jpg
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

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
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]
COLORS = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
          for _ in range(80)]


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


def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
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
    return keep


def postprocess(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """Post-process YOLOv8n-seg Zoo RKNN outputs."""
    strides = [8, 16, 32]
    reg_max = 16

    all_bboxes = []
    all_scores = []
    all_classes = []
    all_mask_coeffs = []
    all_anchors_lb = []  # anchor positions in letterbox coords

    proto = outputs[12].astype(np.float32)[0]  # [32, 160, 160]

    for i, stride in enumerate(strides):
        base = i * 4
        bbox_dfl = outputs[base + 0].astype(np.float32)    # [1, 64, H, W]
        cls_sigmoid = outputs[base + 1].astype(np.float32)  # [1, 80, H, W]
        cls_sum = outputs[base + 2].astype(np.float32)      # [1, 1, H, W]
        mask_coeff = outputs[base + 3].astype(np.float32)   # [1, 32, H, W]

        _, _, h, w = bbox_dfl.shape

        # DFL decode bbox
        bbox = dfl_decode(bbox_dfl, reg_max)  # [1, 4, H, W]
        bbox = bbox[0].transpose(1, 2, 0).reshape(-1, 4)  # [N, 4]

        # Class scores (already sigmoid)
        cls_scores = cls_sigmoid[0].transpose(1, 2, 0).reshape(-1, 80)  # [N, 80]
        # Confidence = cls_sum (already ReduceSum + Clip)
        conf = cls_sum[0, 0].flatten()  # [N]

        # Mask coefficients
        mc = mask_coeff[0].transpose(1, 2, 0).reshape(-1, 32)  # [N, 32]

        # Generate anchors
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchors = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)
        anchors_lb = (anchors + 0.5) * stride  # [N, 2] in letterbox pixel

        # Decode bbox: distances from anchor center -> xyxy in letterbox
        bx1 = anchors_lb[:, 0] - bbox[:, 0] * stride
        by1 = anchors_lb[:, 1] - bbox[:, 1] * stride
        bx2 = anchors_lb[:, 0] + bbox[:, 2] * stride
        by2 = anchors_lb[:, 1] + bbox[:, 3] * stride

        # Class = argmax of per-class sigmoid scores
        cls_idx = cls_scores.argmax(axis=1)  # [N]
        cls_max_score = cls_scores[np.arange(len(cls_idx)), cls_idx]  # [N]

        all_bboxes.append(np.stack([bx1, by1, bx2, by2], axis=-1))
        all_scores.append(conf)
        all_classes.append(cls_idx)
        all_mask_coeffs.append(mc)
        all_anchors_lb.append(anchors_lb)

    # Concatenate all strides
    bboxes = np.concatenate(all_bboxes, axis=0)  # [8400, 4]
    scores = np.concatenate(all_scores, axis=0)  # [8400]
    classes = np.concatenate(all_classes, axis=0)  # [8400]
    mask_coeffs = np.concatenate(all_mask_coeffs, axis=0)  # [8400, 32]
    anchors_lb = np.concatenate(all_anchors_lb, axis=0)  # [8400, 2]

    # Filter by confidence
    mask = scores > conf_thres
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return [], np.zeros((imgsz, imgsz), dtype=np.uint8)

    # Letterbox -> original coords
    dw, dh = pad
    det_bboxes = bboxes[indices]
    det_bboxes[:, [0, 2]] = (det_bboxes[:, [0, 2]] - dw) / ratio
    det_bboxes[:, [1, 3]] = (det_bboxes[:, [1, 3]] - dh) / ratio

    det_scores = scores[indices]
    det_classes = classes[indices]
    det_mask_coeffs = mask_coeffs[indices]
    det_anchors_lb = anchors_lb[indices]

    # NMS
    keep = nms(det_bboxes, det_scores, iou_thres)
    det_bboxes = det_bboxes[keep]
    det_scores = det_scores[keep]
    det_classes = det_classes[keep]
    det_mask_coeffs = det_mask_coeffs[keep]
    det_anchors_lb = det_anchors_lb[keep]

    # Generate masks: mask_coeffs @ proto -> [N, 160, 160]
    # proto: [32, 160*160], mask_coeffs: [N, 32]
    proto_flat = proto.reshape(32, -1)  # [32, 25600]
    masks = det_mask_coeffs @ proto_flat  # [N, 25600]
    masks = 1.0 / (1.0 + np.exp(-masks))  # sigmoid
    masks = masks.reshape(-1, 160, 160)

    # Upscale masks to original image size
    orig_h = int(round(imgsz / ratio - 2 * dh / ratio + pad[1] * 2 / ratio))
    # Actually, let's compute from letterbox: masks are at 160x160 (1/4 of 640)
    # Scale mask to letterbox size (640x640), then crop to original region
    masks_lb = np.zeros((len(masks), imgsz, imgsz), dtype=np.float32)
    for m in range(len(masks)):
        masks_lb[m] = cv2.resize(masks[m], (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # Crop letterbox padding and scale to original
    orig_w = int(round((imgsz - 2 * dw) / ratio))
    orig_h_real = int(round((imgsz - 2 * dh) / ratio))

    masks_orig = masks_lb[:, int(dh):imgsz - int(dh), int(dw):imgsz - int(dw)]
    if masks_orig.shape[1] > 0 and masks_orig.shape[2] > 0:
        masks_resized = np.zeros((len(masks), orig_h_real, orig_w), dtype=np.float32)
        for m in range(len(masks)):
            masks_resized[m] = cv2.resize(masks_orig[m], (orig_w, orig_h_real),
                                           interpolation=cv2.INTER_LINEAR)
    else:
        masks_resized = masks_orig

    detections = []
    for i in range(len(det_bboxes)):
        detections.append({
            "bbox": det_bboxes[i].tolist(),
            "conf": float(det_scores[i]),
            "class": int(det_classes[i]),
            "mask": masks_resized[i],
        })

    return detections


def draw_seg(img, detections):
    overlay = img.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls = det["class"]
        conf = det["conf"]
        color = COLORS[cls % len(COLORS)]

        # Draw mask overlay
        mask = det["mask"]
        if mask.shape[0] > 0 and mask.shape[1] > 0:
            h, w = mask.shape
            ih, iw = overlay.shape[:2]
            # Resize mask to match image
            if h != ih or w != iw:
                mask_full = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
            else:
                mask_full = mask
            colored = np.full_like(overlay, color, dtype=np.uint8)
            mask_binary = (mask_full > 0.5).astype(np.uint8)
            overlay = np.where(mask_binary[:, :, np.newaxis] == 1,
                             (overlay * 0.5 + colored * 0.5).astype(np.uint8),
                             overlay)

        # Draw bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else cls}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(overlay, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay



TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n-seg Zoo RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLOv8n-seg Zoo RKNN Inference")
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
        cls_name = COCO_CLASSES[det['class']] if det['class'] < len(COCO_CLASSES) else str(det['class'])
        b = det['bbox']
        mask_area = int((det['mask'] > 0.5).sum())
        print(f"  [{i}] {cls_name}: {det['conf']:.3f} bbox=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] mask_pixels={mask_area}")
    img_draw = draw_seg(orig_img.copy(), detections)

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
