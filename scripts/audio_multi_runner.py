#!/usr/bin/env python3
"""Multi-input audio benchmark runner for RKNN models."""
import sys, time, os, json, numpy as np
from rknnlite.api import RKNNLite

def run_benchmark_multi(model_path, inputs_config, warmup_iters=10, duration_sec=30):
    """
    inputs_config: list of dict:
      [{'shape': [1,12], 'dtype': 'int64'}, {'shape': [1,1000,512], 'dtype': 'float16'}]
    Or simple: list of (shape, dtype) tuples.
    """
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"FATAL: load_rknn failed ret={ret}")
        return None
    
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"FATAL: init_runtime failed ret={ret}")
        return None
    
    # Build input data
    input_datas = []
    for cfg in inputs_config:
        shape = cfg['shape']
        dtype = cfg.get('dtype', 'float32')
        if dtype == 'int64':
            data = np.random.randint(0, 51865, shape, dtype=np.int64)
        elif dtype == 'float16':
            data = np.random.randn(*shape).astype(np.float16)
        elif dtype == 'float32':
            data = np.random.randn(*shape).astype(np.float32)
        elif dtype == 'uint8':
            data = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            data = np.random.randn(*shape).astype(np.float32)
        input_datas.append(data)
        print(f"Input: shape={data.shape}, dtype={data.dtype}")
    
    # Warmup
    print(f"Warming up ({warmup_iters} iterations)...")
    for i in range(warmup_iters):
        outputs = rknn.inference(inputs=input_datas)
    print("Warmup done.")
    
    # Benchmark
    print(f"Benchmarking for {duration_sec} seconds...")
    latencies = []
    start_time = time.time()
    iters = 0
    while time.time() - start_time < duration_sec:
        iter_start = time.time()
        outputs = rknn.inference(inputs=input_datas)
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
        'num_inputs': len(input_datas),
        'input_dtypes': [str(d.dtype) for d in input_datas],
    }
    
    rknn.release()
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--config')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config) as f:
            inputs_config = json.load(f)
    else:
        # Auto-detect from model name
        basename = os.path.basename(args.model_path).lower()
        if 'whisper_decoder' in basename:
            inputs_config = [
                {'shape': [1, 12], 'dtype': 'int64'},
                {'shape': [1, 1000, 512], 'dtype': 'float16'},
            ]
        elif 'zipformer_joiner' in basename:
            inputs_config = [
                {'shape': [1, 512], 'dtype': 'float32'},
                {'shape': [1, 512], 'dtype': 'float32'},
            ]
        elif 'zipformer_decoder' in basename:
            inputs_config = [
                {'shape': [1, 2], 'dtype': 'int64'},
            ]
        elif 'zipformer_encoder' in basename:
            inputs_config = [{'shape': [1, 103, 80], 'dtype': 'float32'}]
            print("WARNING: zipformer_encoder needs 35 inputs for full state management. Using only audio input for raw encode test.")
        else:
            print("Auto-detect failed, using single float32 input")
            inputs_config = [{'shape': [1, 48000], 'dtype': 'float32'}]
    
    results = run_benchmark_multi(args.model_path, inputs_config, args.warmup, args.duration)
    if results:
        print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
    else:
        print("BENCHMARK FAILED")
        sys.exit(1)
