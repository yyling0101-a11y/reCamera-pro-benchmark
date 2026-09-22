#!/usr/bin/env python3
"""End-to-end pipeline benchmark for multi-stage RKNN models on RV1126B."""
import sys, time, os, json, argparse, numpy as np
from rknnlite.api import RKNNLite

# ─── Pipeline: Whisper (encoder → decoder autoregressive) ───
def bench_whisper(enc_path, dec_path, duration=30, warmup=3):
    print(f"=== Whisper Pipeline ===")
    print(f"  Encoder: {os.path.basename(enc_path)}")
    print(f"  Decoder: {os.path.basename(dec_path)}")
    
    # Load both models
    enc = RKNNLite()
    if enc.load_rknn(enc_path) != 0:
        print("FATAL: encoder load failed"); return None
    enc.init_runtime()
    
    dec = RKNNLite()
    if dec.load_rknn(dec_path) != 0:
        print("FATAL: decoder load failed"); return None
    dec.init_runtime()
    
    # Encoder input: mel spectrogram [1, 80, 2000]
    mel_input = np.random.randn(1, 80, 2000).astype(np.float32) * 0.5 - 2.0
    if '_w8a8' in os.path.basename(enc_path).lower():
        mel_input = np.random.randint(0, 256, (1, 80, 2000), dtype=np.uint8)
    
    # Whisper tokens: SOS=50257, EN=50258, ZH=50259, TRANSCRIBE=50360, NOTIMESTAMPS=50364
    # The decoder expects [1, 12] tokens (fixed shape from ONNX export)
    SOS = 50258
    init_tokens = np.array([[SOS, SOS, SOS, SOS, SOS, SOS, SOS, SOS, SOS, SOS, SOS, SOS]], dtype=np.int64)
    
    # Warmup
    print(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        enc_out = enc.inference(inputs=[mel_input])
        # Decoder: tokens [1,12] + encoder audio [1,1000,512]
        dec.inference(inputs=[init_tokens, enc_out[0]])
    print(f"  Warmup done.")
    
    # Benchmark - measure encoder + single decoder pass
    print(f"  Benchmarking ({duration}s)...")
    latencies = []
    start_t = time.time()
    iters = 0
    while time.time() - start_t < duration:
        t0 = time.time()
        enc_out = enc.inference(inputs=[mel_input])
        logits = dec.inference(inputs=[init_tokens, enc_out[0]])
        lat = (time.time() - t0) * 1000.0
        latencies.append(lat)
        iters += 1
    
    elapsed = time.time() - start_t
    enc.release()
    dec.release()
    
    lats = np.array(latencies)
    return {
        'pipeline': 'whisper',
        'description': 'encoder + 1 decoder pass (12 token context)',
        'duration_sec': elapsed,
        'total_iters': iters,
        'fps': iters / elapsed,
        'latency_avg_ms': float(np.mean(lats)),
        'latency_min_ms': float(np.min(lats)),
        'latency_max_ms': float(np.max(lats)),
        'latency_p50_ms': float(np.percentile(lats, 50)),
    }

# ─── Pipeline: CLIP (vision + text → embedding) ───
def bench_clip(vis_path, txt_path, duration=30, warmup=5):
    print(f"=== CLIP Pipeline ===")
    
    vis = RKNNLite()
    vis.load_rknn(vis_path); vis.init_runtime()
    
    txt = RKNNLite()
    txt.load_rknn(txt_path); txt.init_runtime()
    
    img = np.random.randn(1, 3, 224, 224).astype(np.float16)
    text = np.random.randint(0, 49408, (1, 77), dtype=np.int64)
    
    print(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        vis.inference(inputs=[img])
        txt.inference(inputs=[text])
    print(f"  Warmup done.")
    
    print(f"  Benchmarking ({duration}s)...")
    latencies, start_t, iters = [], time.time(), 0
    while time.time() - start_t < duration:
        t0 = time.time()
        v = vis.inference(inputs=[img])
        t = txt.inference(inputs=[text])
        latencies.append((time.time() - t0) * 1000)
        iters += 1
    
    vis.release(); txt.release()
    lats = np.array(latencies)
    return {
        'pipeline': 'clip',
        'description': 'vision + text encoding (parallel)',
        'duration_sec': time.time() - start_t,
        'total_iters': iters,
        'fps': iters / (time.time() - start_t),
        'latency_avg_ms': float(np.mean(lats)),
        'latency_min_ms': float(np.min(lats)),
        'latency_max_ms': float(np.max(lats)),
        'latency_p50_ms': float(np.percentile(lats, 50)),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', choices=['whisper','clip'], required=True)
    parser.add_argument('--enc')
    parser.add_argument('--dec')
    parser.add_argument('--vis')
    parser.add_argument('--txt')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output')
    args = parser.parse_args()
    
    if args.pipeline == 'whisper':
        results = bench_whisper(args.enc, args.dec, args.duration, args.warmup)
    elif args.pipeline == 'clip':
        results = bench_clip(args.vis, args.txt, args.duration, args.warmup)
    
    if results:
        print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        print("PIPELINE FAILED")
        sys.exit(1)
