#!/usr/bin/env python3
"""YOLOv26 benchmark runner for reCamera Pro device.

Usage: python yolo26_bench.py <model.rknn> <task> <imgsz> [--iterations N] [--warmup N]
Tasks: detect, segment, pose, obb, classify
"""

import sys, os, time, json
import numpy as np

def bench_model(rknn_path, task, imgsz, iterations=200, warmup=30):
    from rknnlite.api import RKNNLite
    
    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print(f"ERROR: load_rknn failed: {ret}")
        return None
    
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f"ERROR: init_runtime failed: {ret}")
        return None
    
    # Create dummy input
    if task == 'classify':
        input_shape = (1, 3, imgsz, imgsz)
    else:
        input_shape = (1, 3, imgsz, imgsz)
    
    dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...", flush=True)
    for _ in range(warmup):
        rknn.inference(inputs=[dummy_input])
    
    # Benchmark
    print(f"  Benchmarking ({iterations} iterations)...", flush=True)
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        outputs = rknn.inference(inputs=[dummy_input])
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    avg = np.mean(times)
    min_t = np.min(times)
    max_t = np.max(times)
    fps = 1000.0 / avg
    
    results = {
        'model': os.path.basename(rknn_path),
        'task': task,
        'imgsz': imgsz,
        'iterations': iterations,
        'warmup': warmup,
        'avg_latency_ms': round(avg, 2),
        'min_latency_ms': round(min_t, 2),
        'max_latency_ms': round(max_t, 2),
        'fps': round(fps, 2),
        'std_latency_ms': round(np.std(times), 2),
        'output_shapes': [o.shape for o in outputs],
    }
    
    print(f"  Results: FPS={fps:.2f}, Latency={avg:.2f}ms (min={min_t:.2f}, max={max_t:.2f})", flush=True)
    
    rknn.release()
    return results

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python yolo26_bench.py <model.rknn> <task> [imgsz] [--iterations N]")
        sys.exit(1)
    
    rknn_path = sys.argv[1]
    task = sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 640
    
    iterations = 200
    warmup = 30
    for i, arg in enumerate(sys.argv):
        if arg == '--iterations' and i + 1 < len(sys.argv):
            iterations = int(sys.argv[i + 1])
        if arg == '--warmup' and i + 1 < len(sys.argv):
            warmup = int(sys.argv[i + 1])
    
    if task == 'classify':
        imgsz = 224  # Override for classification
    
    print(f"Model: {os.path.basename(rknn_path)}")
    print(f"Task: {task}, Input: {imgsz}x{imgsz}")
    print(f"Iterations: {iterations}, Warmup: {warmup}")
    print(f"Device: reCamera Pro (RV1126B)")
    
    results = bench_model(rknn_path, task, imgsz, iterations, warmup)
    if results:
        print(f"\n=== FINAL RESULTS ===")
        print(json.dumps(results, indent=2))
    else:
        print("ERROR: Benchmark failed")
        sys.exit(1)
