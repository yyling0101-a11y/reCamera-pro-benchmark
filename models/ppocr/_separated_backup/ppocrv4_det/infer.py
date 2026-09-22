#!/usr/bin/env python3
"""
PPOCRv4 Detection RKNN inference script for RV1126B.

Usage:
    # Image mode (saves result with bounding boxes):
    python infer.py --model ppocrv4_det_480x480_det.rknn --input test_ocr.jpg

    # Camera mode (benchmark with /dev/video13 via GStreamer):
    python infer.py --model ppocrv4_det_480x480_det.rknn
"""

import argparse
import os
import time
import numpy as np
import cv2

DET_INPUT_H = 480
DET_INPUT_W = 480


def preprocess_image(img, input_h=DET_INPUT_H, input_w=DET_INPUT_W):
    """Resize image to model input size and return preprocessed array + shape info."""
    src_h, src_w = img.shape[:2]
    resized = cv2.resize(img, (input_w, input_h))
    ratio_h = input_h / float(src_h)
    ratio_w = input_w / float(src_w)
    return resized, [src_h, src_w, ratio_h, ratio_w]


def order_points_clockwise(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype="float32")


def get_mini_boxes(contour):
    """Get minimum bounding rectangle vertices."""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, box):
    """Calculate box score using bbox mean."""
    h, w = bitmap.shape[:2]
    box_copy = box.copy()
    xmin = np.clip(np.floor(box_copy[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box_copy[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box_copy[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box_copy[:, 1].max()).astype(np.int32), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_copy[:, 0] = box_copy[:, 0] - xmin
    box_copy[:, 1] = box_copy[:, 1] - ymin
    cv2.fillPoly(mask, box_copy.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    """Expand polygon using pyclipper."""
    try:
        from shapely.geometry import Polygon
        import pyclipper
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
    except ImportError:
        return box.reshape(-1, 1, 2)


def postprocess_det(pred, shape_list, thresh=0.3, box_thresh=0.6,
                    max_candidates=1000, unclip_ratio=1.5):
    """DB post-processing for detection output."""
    pred = pred[0, 0, :, :]
    segmentation = pred > thresh
    src_h, src_w, ratio_h, ratio_w = shape_list[0]
    bitmap = segmentation.astype(np.uint8)

    outs = cv2.findContours((bitmap * 255).astype(np.uint8),
                            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[0] if len(outs) == 2 else outs[1]
    num_contours = min(len(contours), max_candidates)

    boxes = []
    for i in range(num_contours):
        contour = contours[i]
        points, sside = get_mini_boxes(contour)
        if sside < 3:
            continue
        points = np.array(points)
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue

        box_expanded = unclip(points, unclip_ratio)
        if box_expanded.ndim < 2 or len(box_expanded) < 4:
            continue
        box_expanded = box_expanded.reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box_expanded)
        if sside < 5:
            continue
        box = np.array(box)

        height, width = bitmap.shape
        box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)
        boxes.append(box.astype(np.int32))

    filtered = []
    for box in boxes:
        box = order_points_clockwise(box)
        rect_w = int(np.linalg.norm(box[0] - box[1]))
        rect_h = int(np.linalg.norm(box[0] - box[3]))
        if rect_w <= 3 or rect_h <= 3:
            continue
        filtered.append(box)
    return filtered


def draw_det_results(img, boxes):
    """Draw detection results on image."""
    for box in boxes:
        box = np.array(box).astype(np.int32)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
        for pt in box:
            cv2.circle(img, tuple(pt), 3, (0, 0, 255), -1)
    return img


def open_camera():
    """Open camera via GStreamer pipeline for reCamera Pro."""
    pipeline = ('v4l2src device=/dev/video13 '
                '! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 '
                '! videoconvert ! appsink')
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture("/dev/video13")
    return cap


def run_inference(rknn, img_input):
    """Run RKNN inference and return output."""
    outputs = rknn.inference(inputs=[img_input])
    return outputs


def main():
    parser = argparse.ArgumentParser(description='PPOCRv4 Det RKNN Inference')
    parser.add_argument('--model', required=True, help='RKNN model path')
    parser.add_argument('--input', help='Input image path (omit for camera mode)')
    parser.add_argument('--output', default='result_ocr.jpg', help='Output image path (image mode)')
    parser.add_argument('--runs', type=int, default=20, help='Number of inference runs (camera mode)')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations (camera mode)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PPOCRv4 Detection RKNN Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")

    from rknnlite.api import RKNNLite
    rknn = RKNNLite()

    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print(f"ERROR: load_rknn failed with code {ret}")
        return 1

    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f"ERROR: init_runtime failed with code {ret}")
        return 1

    if args.input:
        # Image mode
        print(f"Image: {args.input}")
        img = cv2.imread(args.input)
        if img is None:
            print(f"ERROR: Cannot read {args.input}")
            return 1

        print(f"Original size: {img.shape}")
        resized, shape_list = preprocess_image(img)
        img_input = resized[np.newaxis, :, :, :].astype(np.uint8)

        print("Warmup...")
        for _ in range(3):
            run_inference(rknn, img_input)

        print("Running inference...")
        t0 = time.perf_counter()
        outputs = run_inference(rknn, img_input)
        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000
        print(f"Latency: {latency:.2f} ms")

        preds = {'maps': outputs[0].astype(np.float32)}
        boxes = postprocess_det(preds['maps'], [shape_list])
        print(f"Detected {len(boxes)} text regions")

        for i, box in enumerate(boxes[:10]):
            print(f"  [{i}] box={box.tolist()}")

        img_draw = draw_det_results(img.copy(), boxes)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        cv2.imwrite(args.output, img_draw)
        print(f"Saved: {args.output}")

    else:
        # Camera mode
        print(f"Camera: /dev/video13 (GStreamer)")
        print(f"Runs: {args.runs}, Warmup: {args.warmup}")

        cap = open_camera()
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return 1

        print(f"Warmup ({args.warmup} iterations)...")
        for _ in range(args.warmup):
            ret_frame, frame = cap.read()
            if not ret_frame:
                print("ERROR: Cannot read frame")
                cap.release()
                return 1
            resized, shape_list = preprocess_image(frame)
            img_input = resized[np.newaxis, :, :, :].astype(np.uint8)
            run_inference(rknn, img_input)

        print(f"Benchmark ({args.runs} runs)...")
        latencies = []
        for i in range(args.runs):
            ret_frame, frame = cap.read()
            if not ret_frame:
                print(f"ERROR: Cannot read frame at iteration {i}")
                break
            resized, shape_list = preprocess_image(frame)
            img_input = resized[np.newaxis, :, :, :].astype(np.uint8)
            t0 = time.perf_counter()
            run_inference(rknn, img_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        cap.release()

        if latencies:
            latencies = np.array(latencies)
            print(f"\nPerformance (NPU inference only):")
            print(f"  Avg: {np.mean(latencies):.2f} ms")
            print(f"  Min: {np.min(latencies):.2f} ms")
            print(f"  Max: {np.max(latencies):.2f} ms")
            print(f"  FPS: {1000.0 / np.mean(latencies):.1f}")

    rknn.release()
    print(f"\n{'='*60}\n")
    return 0


if __name__ == '__main__':
    exit(main())
