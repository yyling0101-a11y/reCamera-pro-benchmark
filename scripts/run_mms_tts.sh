#!/bin/sh
# Benchmark mms_tts encoder
cd /userdata/benchmark
python3 -c '
import json, sys
sys.path.insert(0, "/userdata/benchmark")
from audio_multi_runner import run_benchmark_multi
config = [
    {"shape": [1, 200], "dtype": "int64"},
    {"shape": [1, 200], "dtype": "int64"},
]
results = run_benchmark_multi("models/mms_tts_eng_encoder_200_official_FP16.rknn", config, 10, 30)
if results:
    with open("logs/mms_tts_encoder.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
else:
    print("ENCODER FAILED")
'
echo "=== ENCODER DONE ==="

# Benchmark mms_tts decoder
python3 -c '
import json, sys
sys.path.insert(0, "/userdata/benchmark")
from audio_multi_runner import run_benchmark_multi
config = [
    {"shape": [1, 1, 400, 200], "dtype": "float16"},
    {"shape": [1, 1, 400], "dtype": "float16"},
    {"shape": [1, 200, 192], "dtype": "float16"},
    {"shape": [1, 200, 192], "dtype": "float16"},
]
results = run_benchmark_multi("models/mms_tts_eng_decoder_200_official_FP16.rknn", config, 10, 30)
if results:
    with open("logs/mms_tts_decoder.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
else:
    print("DECODER FAILED")
'
echo "=== DECODER DONE ==="
