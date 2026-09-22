#!/bin/sh
# Batch benchmark runner for device
MODEL_DIR="/userdata/benchmark/models"
LOG_DIR="/userdata/benchmark/logs"

for rknn in $MODEL_DIR/*.rknn; do
    name=$(basename "$rknn" .rknn)
    log="$LOG_DIR/${name}.json"
    
    if [ -f "$log" ]; then
        echo "SKIP $name: already done"
        continue
    fi
    
    echo "=== BENCHMARKING $name ==="
    python3 /userdata/benchmark/bench_runner.py "$rknn" --duration 30 --warmup 10 --output "$log"
    echo "=== DONE $name ==="
    echo ""
done
echo "ALL DONE"
