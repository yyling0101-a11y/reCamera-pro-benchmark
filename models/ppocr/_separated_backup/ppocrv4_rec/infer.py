#!/usr/bin/env python3
"""
PPOCRv4 Recognition RKNN inference script for RV1126B.

Usage:
    # Image mode (recognizes text in image):
    python infer.py --model ppocrv4_rec_48x320_rec.rknn --input test_ocr.jpg --dict ppocr_keys_v1.txt

    # Camera mode (benchmark with /dev/video13 via GStreamer):
    python infer.py --model ppocrv4_rec_48x320_rec.rknn --dict ppocr_keys_v1.txt
"""

import argparse
import os
import time
import numpy as np
import cv2

REC_INPUT_H = 48
REC_INPUT_W = 320


class CTCLabelDecode:
    """CTC label decoder for PPOCR recognition."""

    def __init__(self, character_dict_path, use_space_char=True):
        self.character_str = []
        if character_dict_path:
            with open(character_dict_path, "rb") as fin:
                for line in fin:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
        else:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz ")

        dict_character = ['blank'] + self.character_str
        self.character = dict_character
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i

    def decode(self, text_index, text_prob=None):
        """Convert text-index to text-label with CTC dedup."""
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for token in ignored_tokens:
                selection &= text_index[batch_idx] != token
            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
                if text_id < len(self.character)
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1.0] * len(char_list)
            if len(conf_list) == 0:
                conf_list = [0.0]
            text = ''.join(char_list)
            result_list.append((text, float(np.mean(conf_list))))
        return result_list

    def __call__(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        return self.decode(preds_idx, preds_prob)


def preprocess_rec_image(img, input_h=REC_INPUT_H, input_w=REC_INPUT_W):
    """Resize image to rec model input size."""
    resized = cv2.resize(img, (input_w, input_h))
    return resized


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
    """Run RKNN inference."""
    outputs = rknn.inference(inputs=[img_input])
    return outputs


def main():
    parser = argparse.ArgumentParser(description='PPOCRv4 Rec RKNN Inference')
    parser.add_argument('--model', required=True, help='RKNN model path')
    parser.add_argument('--input', help='Input image path (omit for camera mode)')
    parser.add_argument('--output', default='result_ocr.jpg', help='Output image path (image mode)')
    parser.add_argument('--dict', default='ppocr_keys_v1.txt', help='Character dictionary path')
    parser.add_argument('--runs', type=int, default=20, help='Number of inference runs (camera mode)')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations (camera mode)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PPOCRv4 Recognition RKNN Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dict: {args.dict}")

    if not os.path.exists(args.dict):
        print(f"ERROR: Dictionary file not found: {args.dict}")
        return 1
    ctc_decoder = CTCLabelDecode(args.dict)

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
        resized = preprocess_rec_image(img)
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

        preds = outputs[0].astype(np.float32)
        if preds.ndim == 2:
            preds = preds[np.newaxis, :, :]
        results = ctc_decoder(preds)

        print(f"\nRecognition results:")
        for i, (text, conf) in enumerate(results):
            print(f"  [{i}] text='{text}' conf={conf:.4f}")

        img_draw = img.copy()
        if results and results[0][0]:
            text_str = results[0][0]
            conf_val = results[0][1]
            label = f"{text_str} ({conf_val:.2f})"
            font_scale = min(1.0, img.shape[1] / 500.0)
            cv2.rectangle(img_draw, (5, 5), (img.shape[1] - 5, 50), (0, 0, 0), -1)
            cv2.putText(img_draw, label, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

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
            resized = preprocess_rec_image(frame)
            img_input = resized[np.newaxis, :, :, :].astype(np.uint8)
            run_inference(rknn, img_input)

        print(f"Benchmark ({args.runs} runs)...")
        latencies = []
        for i in range(args.runs):
            ret_frame, frame = cap.read()
            if not ret_frame:
                print(f"ERROR: Cannot read frame at iteration {i}")
                break
            resized = preprocess_rec_image(frame)
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
