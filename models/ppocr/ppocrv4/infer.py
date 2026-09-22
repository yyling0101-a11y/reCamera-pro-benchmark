#!/usr/bin/env python3
"""
PPOCRv4 OCR Pipeline RKNN inference for RV1126B.
Detection + Recognition combined workflow.

Usage:
    # Image mode (det -> crop -> rec -> draw results):
    python infer.py --det-model ppocrv4_det_480x480_det.rknn \
                    --rec-model ppocrv4_rec_48x320_rec.rknn \
                    --dict ppocr_keys_v1.txt \
                    --input test_ocr.jpg

    # Camera mode (benchmark, 20 runs, 5 warmup):
    python infer.py --det-model ppocrv4_det_480x480_det.rknn \
                    --rec-model ppocrv4_rec_48x320_rec.rknn \
                    --dict ppocr_keys_v1.txt
"""

import argparse
import os
import time
import numpy as np
import cv2

DET_INPUT_H = 480
DET_INPUT_W = 480
REC_INPUT_H = 48
REC_INPUT_W = 320

# Det post-processing parameters (tuned for better crop coverage)
DET_THRESH = 0.2
DET_BOX_THRESH = 0.5
DET_UNCLIP_RATIO = 3.0
DET_MAX_CANDIDATES = 1000

# Crop expansion ratio (add padding around detected text region)
CROP_PAD_RATIO = 0.4


# ─── Detection post-processing ────────────────────────────────────────

def order_points_clockwise(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype="float32")


def get_mini_boxes(contour):
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


def expand_box(box, img_h, img_w, pad_ratio=CROP_PAD_RATIO):
    """Expand a 4-point box by pad_ratio in all directions, clipped to image bounds."""
    pts = box.astype(np.float32)
    # Compute center and half-widths
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    half_w = (pts[:, 0].max() - pts[:, 0].min()) / 2.0
    half_h = (pts[:, 1].max() - pts[:, 1].min()) / 2.0
    # Expand
    new_half_w = half_w * (1.0 + pad_ratio)
    new_half_h = half_h * (1.0 + pad_ratio)
    # Rebuild box
    new_pts = np.array([
        [cx - new_half_w, cy - new_half_h],
        [cx + new_half_w, cy - new_half_h],
        [cx + new_half_w, cy + new_half_h],
        [cx - new_half_w, cy + new_half_h],
    ], dtype=np.float32)
    # Clip to image bounds
    new_pts[:, 0] = np.clip(new_pts[:, 0], 0, img_w - 1)
    new_pts[:, 1] = np.clip(new_pts[:, 1], 0, img_h - 1)
    return new_pts.astype(np.int32)


def postprocess_det(pred, src_h, src_w,
                    thresh=DET_THRESH, box_thresh=DET_BOX_THRESH,
                    max_candidates=DET_MAX_CANDIDATES,
                    unclip_ratio=DET_UNCLIP_RATIO):
    """DB post-processing: returns list of 4-point quadrilaterals in original coords."""
    pred = pred[0, 0, :, :]
    segmentation = pred > thresh
    bitmap = segmentation.astype(np.uint8)

    outs = cv2.findContours((bitmap * 255).astype(np.uint8),
                            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[0] if len(outs) == 2 else outs[1]
    num_contours = min(len(contours), max_candidates)
    h, w = bitmap.shape

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
        if sside < 3:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / w * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / h * src_h), 0, src_h)

        # Expand box further for better rec coverage
        box = expand_box(box, src_h, src_w, CROP_PAD_RATIO)

        # Re-check minimum size after expansion
        rect_w = int(np.linalg.norm(box[0] - box[1]))
        rect_h = int(np.linalg.norm(box[0] - box[3]))
        if rect_w <= 3 or rect_h <= 3:
            continue
        boxes.append(box)

    return boxes


# ─── Recognition post-processing ──────────────────────────────────────

class CTCLabelDecode:
    def __init__(self, character_dict_path, use_space_char=True):
        self.character_str = []
        with open(character_dict_path, "rb") as fin:
            for line in fin:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if use_space_char:
            self.character_str.append(" ")
        dict_character = ['blank'] + self.character_str
        self.character = dict_character

    def decode(self, text_index, text_prob=None):
        result_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for token in [0]:
                selection &= text_index[batch_idx] != token
            char_list = [
                self.character[tid]
                for tid in text_index[batch_idx][selection]
                if tid < len(self.character)
            ]
            conf_list = (text_prob[batch_idx][selection]
                         if text_prob is not None else [1.0] * len(char_list))
            if len(conf_list) == 0:
                conf_list = [0.0]
            text = ''.join(char_list)
            result_list.append((text, float(np.mean(conf_list))))
        return result_list

    def __call__(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        return self.decode(preds_idx, preds_prob)


# ─── Crop helper ──────────────────────────────────────────────────────

def crop_text_region(img, box):
    """Perspective-warp a 4-point quadrilateral to a rectangle."""
    pts = box.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]),
                np.linalg.norm(pts[1] - pts[2])))
    if w <= 0 or h <= 0:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(img, M, (w, h))
    return crop


# ─── Drawing helper (PIL-based, supports CJK) ─────────────────────────

def draw_ocr_results(img, boxes, texts):
    """Draw OCR results using PIL for proper CJK text rendering."""
    from PIL import Image, ImageDraw, ImageFont

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Try to load a CJK font
    font = None
    font_paths = [
        "/oem/usr/share/SourceHanSansCN.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 18)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    for box, (text, conf) in zip(boxes, texts):
        pts = np.array(box, dtype=np.int32)
        # Draw polygon
        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            draw.line([p1, p2], fill=(0, 255, 0), width=2)

        if text:
            label = f"{text} ({conf:.2f})"
            x_min = int(pts[:, 0].min())
            y_min = int(pts[:, 1].min())
            # Get text bounding box
            bbox = draw.textbbox((x_min + 2, max(y_min - 24, 0)), label, font=font)
            tw = bbox[2] - bbox[0] + 4
            th = bbox[3] - bbox[1] + 4
            # Draw label background
            label_y = max(y_min - th - 2, 0)
            draw.rectangle(
                [x_min, label_y, x_min + tw, label_y + th],
                fill=(0, 255, 0)
            )
            draw.text((x_min + 2, label_y + 1), label,
                      fill=(0, 0, 0), font=font)

    img_out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_out


# ─── Camera helper ────────────────────────────────────────────────────

def open_camera():
    pipeline = ('v4l2src device=/dev/video13 '
                '! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 '
                '! videoconvert ! appsink')
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture("/dev/video13")
    return cap


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PPOCRv4 OCR Pipeline')
    parser.add_argument('--det-model', required=True, help='Det RKNN model path')
    parser.add_argument('--rec-model', required=True, help='Rec RKNN model path')
    parser.add_argument('--dict', default='ppocr_keys_v1.txt', help='Character dictionary')
    parser.add_argument('--input', help='Input image (omit for camera mode)')
    parser.add_argument('--output', default='result_ocr.jpg', help='Output image path')
    parser.add_argument('--runs', type=int, default=20, help='Benchmark runs')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PPOCRv4 OCR Pipeline")
    print(f"{'='*60}")
    print(f"Det model: {args.det_model}")
    print(f"Rec model: {args.rec_model}")
    print(f"Dict:      {args.dict}")

    from rknnlite.api import RKNNLite

    # Load detection model
    det_engine = RKNNLite()
    ret = det_engine.load_rknn(args.det_model)
    if ret != 0:
        print(f"ERROR: det load_rknn failed ({ret})")
        return 1
    ret = det_engine.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f"ERROR: det init_runtime failed ({ret})")
        return 1

    # Load recognition model
    rec_engine = RKNNLite()
    ret = rec_engine.load_rknn(args.rec_model)
    if ret != 0:
        print(f"ERROR: rec load_rknn failed ({ret})")
        return 1
    ret = rec_engine.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f"ERROR: rec init_runtime failed ({ret})")
        return 1

    # Load dictionary
    ctc_decoder = CTCLabelDecode(args.dict)

    def run_ocr_pipeline(img):
        """Full det->rec pipeline on one image. Returns (boxes, texts)."""
        src_h, src_w = img.shape[:2]

        # ── Detection ──
        det_input = cv2.resize(img, (DET_INPUT_W, DET_INPUT_H))
        det_input = det_input[np.newaxis, :, :, :].astype(np.uint8)
        det_out = det_engine.inference(inputs=[det_input])
        boxes = postprocess_det(det_out[0].astype(np.float32), src_h, src_w)

        # ── Recognition ──
        texts = []
        for box in boxes:
            crop = crop_text_region(img, box)
            if crop is None:
                texts.append(("", 0.0))
                continue
            rec_input = cv2.resize(crop, (REC_INPUT_W, REC_INPUT_H))
            rec_input = rec_input[np.newaxis, :, :, :].astype(np.uint8)
            rec_out = rec_engine.inference(inputs=[rec_input])
            preds = rec_out[0].astype(np.float32)
            if preds.ndim == 2:
                preds = preds[np.newaxis, :, :]
            results = ctc_decoder(preds)
            texts.append(results[0] if results else ("", 0.0))

        return boxes, texts

    if args.input:
        # ── Image mode ──
        print(f"Image: {args.input}")
        img = cv2.imread(args.input)
        if img is None:
            print(f"ERROR: Cannot read {args.input}")
            return 1
        print(f"Size: {img.shape}")

        # Warmup
        for _ in range(3):
            run_ocr_pipeline(img)

        # Final run with timing
        t0 = time.perf_counter()
        boxes, texts = run_ocr_pipeline(img)
        total_ms = (time.perf_counter() - t0) * 1000

        print(f"\nDetected {len(boxes)} text region(s), total {total_ms:.1f} ms")
        for i, (box, (text, conf)) in enumerate(zip(boxes, texts)):
            print(f"  [{i}] \"{text}\" (conf={conf:.3f})")

        # Draw results using PIL (supports CJK)
        img_draw = draw_ocr_results(img, boxes, texts)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        cv2.imwrite(args.output, img_draw)
        print(f"Saved: {args.output}")

    else:
        # ── Camera mode ──
        print(f"Camera: /dev/video13 (GStreamer)")
        print(f"Runs: {args.runs}, Warmup: {args.warmup}")

        cap = open_camera()
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return 1

        # Warmup
        print(f"Warmup ({args.warmup} iterations)...")
        for _ in range(args.warmup):
            ret_frame, frame = cap.read()
            if not ret_frame:
                break
            run_ocr_pipeline(frame)

        # Benchmark
        print(f"Benchmark ({args.runs} runs)...")
        latencies = []
        for i in range(args.runs):
            ret_frame, frame = cap.read()
            if not ret_frame:
                break
            t0 = time.perf_counter()
            run_ocr_pipeline(frame)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        cap.release()

        if latencies:
            latencies = np.array(latencies)
            print(f"\nPerformance (full det+rec pipeline):")
            print(f"  Avg: {np.mean(latencies):.2f} ms")
            print(f"  Min: {np.min(latencies):.2f} ms")
            print(f"  Max: {np.max(latencies):.2f} ms")
            print(f"  FPS: {1000.0 / np.mean(latencies):.1f}")

    det_engine.release()
    rec_engine.release()
    print(f"\n{'='*60}\n")
    return 0


if __name__ == '__main__':
    exit(main())
