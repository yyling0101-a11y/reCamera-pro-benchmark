#!/usr/bin/env python3
"""
YOLOv8 Zoo RKNN inference script with DFL post-processing.

Supports YOLOv8n and YOLOv8s Zoo models exported from Rockchip Model Zoo.

Usage:
    python infer.py \
        --model yolov8n_640x640_W8A8.rknn \
        --image /path/to/test.jpg \
        --output result.jpg \
        --conf 0.25 \
        --nms 0.45
"""

import argparse
import os
import time
import numpy as np
import cv2


def softmax(x, axis=1):
    """Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def dfl_decode(x, reg_max=16):
    """
    DFL (Distribution Focal Loss) decode.
    Input: [B, 4*reg_max, H, W]
    Output: [B, 4, H, W]
    """
    b, c, h, w = x.shape
    x = x.reshape(b, 4, reg_max, h, w)
    x = softmax(x, axis=2)
    weights = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    x = (x * weights).sum(axis=2)
    return x


def generate_anchors(img_size, strides=(8, 16, 32)):
    """Generate anchor points for YOLOv8."""
    anchors = []
    for stride in strides:
        feat_size = img_size // stride
        y, x = np.meshgrid(
            np.arange(feat_size, dtype=np.float32),
            np.arange(feat_size, dtype=np.float32),
            indexing='ij'
        )
        anchor = np.stack([x, y], axis=-1).reshape(-1, 2)
        anchor = (anchor + 0.5) * stride
        anchors.append(anchor)
    return np.concatenate(anchors, axis=0)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with aspect ratio preserved, pad with gray."""
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


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Filter out zero-area boxes
    valid = areas > 0
    if not valid.any():
        return []
    
    boxes = boxes[valid]
    scores = scores[valid]
    areas = areas[valid]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
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
        
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def postprocess_zoo(outputs, anchors, img_shape, ratio, pad, conf_thres=0.25, nms_thres=0.45):
    """
    Post-process YOLOv8 Zoo model outputs.
    
    outputs: list of 9 tensors
        [0]: bbox features stride 8  [1, 64, 80, 80]
        [1]: class scores stride 8   [1, 80, 80, 80]
        [2]: objectness stride 8     [1, 1, 80, 80]
        [3]: bbox features stride 16 [1, 64, 40, 40]
        [4]: class scores stride 16  [1, 80, 40, 40]
        [5]: objectness stride 16    [1, 1, 40, 40]
        [6]: bbox features stride 32 [1, 64, 20, 20]
        [7]: class scores stride 32  [1, 80, 20, 20]
        [8]: objectness stride 32    [1, 1, 20, 20]
    """
    strides = [8, 16, 32]
    reg_max = 16
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for i, stride in enumerate(strides):
        bbox_feat = outputs[i * 3]
        cls_scores = outputs[i * 3 + 1]
        obj_scores = outputs[i * 3 + 2]
        
        # DFL decode bbox
        bbox = dfl_decode(bbox_feat, reg_max)
        
        _, _, h, w = bbox.shape
        bbox = bbox[0].transpose(1, 2, 0).reshape(-1, 4)
        cls_scores = cls_scores[0].transpose(1, 2, 0).reshape(-1, 80)
        obj_scores = obj_scores[0, 0].flatten()
        
        # Get anchors for this scale
        n_anchors = h * w
        anchor_start = sum([(640 // s) ** 2 for s in strides[:i]])
        anchor_end = anchor_start + n_anchors
        anchor = anchors[anchor_start:anchor_end]
        
        # Decode bbox
        x1 = anchor[:, 0] - bbox[:, 0] * stride
        y1 = anchor[:, 1] - bbox[:, 1] * stride
        x2 = anchor[:, 0] + bbox[:, 2] * stride
        y2 = anchor[:, 1] + bbox[:, 3] * stride
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Filter: use combined score (obj * max_cls) > threshold
        cls_max = cls_scores.max(axis=1)
        combined_scores = obj_scores * cls_max
        
        mask = combined_scores > conf_thres
        if not mask.any():
            continue
        
        boxes = boxes[mask]
        cls_scores = cls_scores[mask]
        combined_scores = combined_scores[mask]
        cls_ids = cls_scores.argmax(axis=1)
        
        all_boxes.append(boxes)
        all_scores.append(combined_scores)
        all_classes.append(cls_ids)
    
    if not all_boxes:
        return []
    
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    
    # Convert from letterbox to original image coordinates
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    
    # Clip to image boundaries
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_shape[0])
    
    # NMS
    keep = nms(boxes, scores, nms_thres)
    
    detections = []
    for i in keep:
        detections.append({
            'bbox': boxes[i].tolist(),
            'score': float(scores[i]),
            'class': int(classes[i])
        })
    
    return detections


def draw_detections(img, detections, class_names=None):
    """Draw bounding boxes and labels."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        score = det['score']
        cls = det['class']
        
        color = colors[cls % len(colors)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        text = f'{label}: {score:.2f}'
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img



TOTAL_RUNS = 20
WARMUP_RUNS = 5

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Zoo RKNN inference")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--input", default=None, help="Input image path (omit for camera mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input size")
    args = parser.parse_args()

    mode = "image" if args.input else "camera"
    print(f"\n{'='*60}")
    print(f"YOLOv8 Zoo RKNN Inference")
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
    anchors = generate_anchors(args.imgsz)
    detections = postprocess_zoo(last_outputs, anchors, orig_img.shape, ratio, (dw, dh),
                                  conf_thres=args.conf, nms_thres=args.nms)
    class_names = ['person','bicycle','car','motorcycle','airplane','bus','train','truck',
        'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
        'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
        'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
        'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
        'chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
        'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    print(f"Detections: {len(detections)}")
    for i, det in enumerate(detections[:10]):
        cls_name = class_names[det['class']] if det['class'] < len(class_names) else str(det['class'])
        print(f"  [{i}] {cls_name}: {det['score']:.3f} bbox={[f'{v:.1f}' for v in det['bbox']]}")
    img_draw = draw_detections(orig_img.copy(), detections, class_names)

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
