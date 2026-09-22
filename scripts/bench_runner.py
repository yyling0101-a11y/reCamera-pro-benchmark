#!/usr/bin/env python3
"""NPU benchmark runner for reCamera Pro RV1126B. Supports vision + audio models."""
import sys, time, os, json, argparse, numpy as np

def run_benchmark(model_path, duration_sec=30, warmup_iters=10):
    from rknnlite.api import RKNNLite
    
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"FATAL: load_rknn failed ret={ret}")
        return None
    
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"FATAL: init_runtime failed ret={ret}")
        return None
    
    basename = os.path.basename(model_path).lower()
    
    # ── Audio models ──
    if any(k in basename for k in ['wav2vec2', 'whisper', 'yamnet', 'zipformer', 'mms_tts']):
        if 'yamnet' in basename:
            input_shape = [1, 48000]
        elif 'wav2vec2' in basename:
            input_shape = [1, 320000]
        elif 'whisper_encoder' in basename:
            input_shape = [1, 80, 2000]
        elif 'whisper_decoder' in basename:
            input_shape = [1, 1]
        elif 'zipformer' in basename:
            input_shape = [1, 80, 2000]
        elif 'mms_tts' in basename:
            input_shape = [1, 200]
        else:
            input_shape = [1, 48000]
        input_dtype = 'float32'
        print(f"Audio model detected, shape={input_shape}")
    
    # ── Text models ──
    elif 'clip_text' in basename or 'clip-text' in basename:
        input_shape = [1, 77]
        input_dtype = 'int64'
        print(f"CLIP text model detected, shape={input_shape}")
    elif 'clip_images' in basename or 'clip-vision' in basename:
        input_shape = [1, 3, 224, 224]
        input_dtype = 'float16'
        print(f"CLIP vision model detected, shape={input_shape}")
    
    # ── Vision models ──
    else:
        # Determine by filename pattern
        import re
        match = re.search(r'(\d+)x(\d+)', basename)
        if match:
            h, w = int(match.group(1)), int(match.group(2))
            input_shape = [1, 3, h, w]
        elif 'mobilenetv2' in basename or 'mobilenet' in basename:
            input_shape = [1, 3, 224, 224]
        elif 'resnet' in basename:
            input_shape = [1, 3, 224, 224]
        elif 'retinaface' in basename:
            input_shape = [1, 3, 320, 320]
        elif 'lprnet' in basename:
            input_shape = [1, 3, 24, 94]
        elif 'ppocrv4_det' in basename:
            input_shape = [1, 3, 480, 480]
        elif 'ppocrv4_rec' in basename:
            input_shape = [1, 3, 48, 320]
        elif 'pp_liteseg' in basename:
            input_shape = [1, 3, 512, 512]
        elif 'mobilesam' in basename:
            input_shape = [1, 3, 1024, 1024]
        elif 'cls' in basename or '_cls' in basename:
            input_shape = [1, 3, 224, 224]
        else:
            input_shape = [1, 3, 640, 640]
        input_dtype = 'unknown'
        print(f"Vision model detected, shape={input_shape}")
    
    # ── dtype ──
    if 'clip_text' in basename or 'clip-text' in basename:
        input_data = np.random.randint(0, 49408, [int(s) for s in input_shape], dtype=np.int64)
        print(f"Text token input: shape={input_data.shape}, int64")
    elif '_w8a8' in basename or 'int8' in basename.lower():
        input_data = np.random.randint(0, 256, [int(s) for s in input_shape], dtype=np.uint8)
        print(f"INT8 input: shape={input_data.shape}, uint8")
    elif '_fp16' in basename or 'official_fp16' in basename:
        input_data = np.random.randn(*[int(s) for s in input_shape]).astype(np.float16)
        print(f"FP16 input: shape={input_data.shape}, float16")
    else:
        input_data = np.random.randn(*[int(s) for s in input_shape]).astype(np.float32)
        print(f"FP32 input: shape={input_data.shape}, float32")
    
    # ── Warmup ──
    print(f"Warming up ({warmup_iters} iterations)...")
    for i in range(warmup_iters):
        outputs = rknn.inference(inputs=[input_data])
    print("Warmup done.")
    
    # ── Benchmark ──
    print(f"Benchmarking for {duration_sec} seconds...")
    latencies = []
    start_time = time.time()
    iters = 0
    while time.time() - start_time < duration_sec:
        iter_start = time.time()
        outputs = rknn.inference(inputs=[input_data])
        lat = (time.time() - iter_start) * 1000.0
        latencies.append(lat)
        iters += 1
    
    elapsed = time.time() - start_time
    latencies = np.array(latencies)
    
    results = {
        'model': os.path.basename(model_path),
        'duration_sec': elapsed,
        'total_iters': iters,
        'fps': iters / elapsed,
        'latency_avg_ms': float(np.mean(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'input_shape': [int(s) for s in input_shape],
        'input_dtype': str(input_data.dtype),
    }
    
    rknn.release()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output')
    args = parser.parse_args()
    
    results = run_benchmark(args.model_path, args.duration, args.warmup)
    if results:
        print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        print("BENCHMARK FAILED")
        sys.exit(1)
