#!/usr/bin/env python3
"""
YOLO11 series RKNN inference on RV1126B.
Supports: yolo11n, yolo11s, yolo11n-pose, yolo11n-seg, yolo11n-cls
"""

import argparse, math, os, time, cv2, numpy as np

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

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
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

def postprocess_det(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    strides = [8, 16, 32]
    reg_max = 16
    num_classes = 80
    all_bboxes, all_scores, all_classes = [], [], []
    for i, stride in enumerate(strides):
        base = i * 3
        bbox_dfl = outputs[base + 0].astype(np.float32)
        cls_sig = outputs[base + 1].astype(np.float32)
        cls_sum = outputs[base + 2].astype(np.float32)
        _, _, h, w = bbox_dfl.shape
        bbox = dfl_decode(bbox_dfl, reg_max)
        bbox = bbox[0].transpose(1, 2, 0).reshape(-1, 4)
        cls_scores = cls_sig[0].transpose(1, 2, 0).reshape(-1, num_classes)
        conf = cls_sum[0, 0].flatten()
        yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        anchors = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)
        anchors = (anchors + 0.5) * stride
        bx1 = anchors[:, 0] - bbox[:, 0] * stride
        by1 = anchors[:, 1] - bbox[:, 1] * stride
        bx2 = anchors[:, 0] + bbox[:, 2] * stride
        by2 = anchors[:, 1] + bbox[:, 3] * stride
        max_score = cls_scores.max(axis=1) * conf
        max_cls = cls_scores.argmax(axis=1)
        all_bboxes.append(np.stack([bx1, by1, bx2, by2], axis=-1))
        all_scores.append(max_score)
        all_classes.append(max_cls)
    bboxes = np.concatenate(all_bboxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    mask = scores > conf_thres
    indices = np.where(mask)[0]
    if len(indices) == 0: return []
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
    keep = nms(boxes_orig, scores_filt, iou_thres)
    detections = []
    for k in keep:
        detections.append({"bbox": boxes_orig[k].tolist(), "conf": float(scores_filt[k]), "class": int(classes[indices[k]])})
    return detections

def postprocess_pose(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    pred = outputs[0].astype(np.float32)[0]  # [56, 8400]
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    conf = pred[4]  # already sigmoid (range [0,1])
    kpts_raw = pred[5:].reshape(17, 3, -1)
    kpt_x, kpt_y, kpt_vis = kpts_raw[:, 0], kpts_raw[:, 1], sigmoid(kpts_raw[:, 2])
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    mask = conf > conf_thres
    indices = np.where(mask)[0]
    if len(indices) == 0: return []
    dw, dh = pad
    boxes_orig, kpts_orig, scores_filt = [], [], []
    for idx in indices:
        bx1 = (x1[idx] - dw) / ratio
        by1 = (y1[idx] - dh) / ratio
        bx2 = (x2[idx] - dw) / ratio
        by2 = (y2[idx] - dh) / ratio
        boxes_orig.append([bx1, by1, bx2, by2])
        scores_filt.append(conf[idx])
        kp = []
        for j in range(17):
            kx = (kpt_x[j, idx] - dw) / ratio
            ky = (kpt_y[j, idx] - dh) / ratio
            kp.append([kx, ky, kpt_vis[j, idx]])
        kpts_orig.append(kp)
    boxes_orig = np.array(boxes_orig)
    scores_filt = np.array(scores_filt)
    keep = nms(boxes_orig, scores_filt, iou_thres)
    detections = []
    for k in keep:
        detections.append({"bbox": boxes_orig[k].tolist(), "conf": float(scores_filt[k]), "class": 0, "keypoints": kpts_orig[k]})
    return detections

def postprocess_seg(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    pred = outputs[0].astype(np.float32)[0]  # [116, 8400]
    proto = outputs[1].astype(np.float32)[0]  # [32, 160, 160]
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    cls_scores = pred[4:84]  # already sigmoid
    mask_coeff = pred[84:116]
    max_score = cls_scores.max(axis=0)
    max_cls = cls_scores.argmax(axis=0)
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    mask = max_score > conf_thres
    indices = np.where(mask)[0]
    if len(indices) == 0: return []
    dw, dh = pad
    det_bboxes, det_scores, det_classes, det_coeffs = [], [], [], []
    for idx in indices:
        bx1 = (x1[idx] - dw) / ratio
        by1 = (y1[idx] - dh) / ratio
        bx2 = (x2[idx] - dw) / ratio
        by2 = (y2[idx] - dh) / ratio
        det_bboxes.append([bx1, by1, bx2, by2])
        det_scores.append(max_score[idx])
        det_classes.append(max_cls[idx])
        det_coeffs.append(mask_coeff[:, idx])
    det_bboxes = np.array(det_bboxes)
    det_scores = np.array(det_scores)
    keep = nms(det_bboxes, det_scores, iou_thres)
    detections = []
    for k in keep:
        coeff = det_coeffs[k]
        m = sigmoid(coeff @ proto.reshape(32, -1)).reshape(160, 160)
        orig_h = int((imgsz - 2*dh) / ratio)
        orig_w = int((imgsz - 2*dw) / ratio)
        lb_top = int(dh * 160 / imgsz)
        lb_left = int(dw * 160 / imgsz)
        lb_bot = int((imgsz - dh) * 160 / imgsz)
        lb_right = int((imgsz - dw) * 160 / imgsz)
        m_crop = m[lb_top:lb_bot, lb_left:lb_right]
        m_resized = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        detections.append({"bbox": det_bboxes[k].tolist(), "conf": float(det_scores[k]), "class": int(det_classes[k]), "mask": m_resized})
    return detections

def postprocess_cls(outputs, ratio, pad, conf_thres=0.25, iou_thres=0.45, imgsz=224):
    pred = outputs[0].astype(np.float32)[0]
    scores = pred  # already softmax probabilities
    top5_idx = np.argsort(scores)[::-1][:5]
    return [{"class": int(i), "conf": float(scores[i])} for i in top5_idx]

SKELETON = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

def draw_det(img, dets):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        c = COLORS[d["class"] % len(COLORS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), c, 2)
        lbl = f"{COCO_CLASSES[d['class']] if d['class'] < len(COCO_CLASSES) else d['class']}: {d['conf']:.2f}"
        cv2.putText(img, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return img

def draw_pose(img, dets):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"person: {d['conf']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        kpts = d.get("keypoints", [])
        if len(kpts) == 17:
            for kp in kpts:
                if kp[2] > 0.5: cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0,0,255), -1)
            for s in SKELETON:
                p1, p2 = s[0]-1, s[1]-1
                if kpts[p1][2] > 0.5 and kpts[p2][2] > 0.5:
                    cv2.line(img, (int(kpts[p1][0]), int(kpts[p1][1])), (int(kpts[p2][0]), int(kpts[p2][1])), (255,0,0), 2)
    return img

def draw_seg(img, dets):
    overlay = img.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        c = COLORS[d["class"] % len(COLORS)]
        mask = d.get("mask")
        if mask is not None:
            ih, iw = overlay.shape[:2]
            mf = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR) if mask.shape != (ih, iw) else mask
            colored = np.full_like(overlay, c, dtype=np.uint8)
            mb = (mf > 0.5).astype(np.uint8)
            overlay = np.where(mb[:,:,np.newaxis]==1, (overlay*0.5+colored*0.5).astype(np.uint8), overlay)
        cv2.rectangle(overlay, (x1,y1), (x2,y2), c, 2)
        lbl = f"{COCO_CLASSES[d['class']] if d['class'] < len(COCO_CLASSES) else d['class']}: {d['conf']:.2f}"
        cv2.putText(overlay, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return overlay

def draw_cls(img, dets):
    y = 30
    for d in dets:
        cv2.putText(img, f"class_{d['class']}: {d['conf']:.4f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 30
    return img

def detect_model_type(outputs):
    if len(outputs) == 1:
        s = outputs[0].shape
        if len(s) == 2 and s[1] == 1000: return "cls"
        elif len(s) == 3 and s[1] == 56: return "pose"
        elif len(s) == 3 and s[1] == 20: return "obb"
        else: return "unknown"
    elif len(outputs) == 2: return "seg"
    elif len(outputs) == 9: return "det"
    return "unknown"

TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Pose RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLO11 Pose RKNN Inference")
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
    detections = postprocess_pose(last_outputs, ratio, (dw, dh),
                                   conf_thres=args.conf, iou_thres=args.nms, imgsz=args.imgsz)
    print(f"Detections: {len(detections)}")
    for i, d in enumerate(detections[:10]):
        b = d["bbox"]
        print(f"  [{i}] person: {d['conf']:.3f} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
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
