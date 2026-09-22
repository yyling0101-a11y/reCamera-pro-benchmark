#!/usr/bin/env python3
"""MMS-TTS end-to-end runner for power/noise measurement."""
import sys, time, numpy as np
from rknnlite.api import RKNNLite

ENC_PATH = "models/mms_tts_eng_encoder_200_official_FP16.rknn"
DEC_PATH = "models/mms_tts_eng_decoder_200_official_FP16.rknn"

print("Loading encoder...")
enc = RKNNLite()
if enc.load_rknn(ENC_PATH) != 0:
    print("FAIL enc load"); sys.exit(1)
enc.init_runtime()

print("Loading decoder...")
dec = RKNNLite()
if dec.load_rknn(DEC_PATH) != 0:
    print("FAIL dec load"); sys.exit(1)
dec.init_runtime()

# Encoder inputs: text tokens [1, 200], attention mask [1, 200]
text = np.random.randint(0, 1000, (1, 200), dtype=np.int64)
mask = np.ones((1, 200), dtype=np.int64)

# Decoder inputs (fixed from RKNN conversion)
attn     = np.random.randn(1, 1, 400, 200).astype(np.float16)
out_mask = np.ones((1, 1, 400), dtype=np.float16)

print("Warmup...")
for i in range(5):
    enc_out = enc.inference(inputs=[text, mask])
    # enc_out: [log_duration, input_padding_mask, prior_means, prior_log_variances]
    pm = enc_out[2]   # [1, 200, 192]
    plv = enc_out[3]  # [1, 200, 192]
    dec.inference(inputs=[attn, out_mask, pm.astype(np.float16), plv.astype(np.float16)])
print("Warmup done. Running 60s for power measurement...")

start = time.time()
iters = 0
lats = []
while time.time() - start < 60:
    t0 = time.time()
    enc_out = enc.inference(inputs=[text, mask])
    pm  = enc_out[2]
    plv = enc_out[3]
    dec.inference(inputs=[attn, out_mask, pm.astype(np.float16), plv.astype(np.float16)])
    lats.append((time.time() - t0) * 1000)
    iters += 1
    if iters % 10 == 0:
        print(f"  iter {iters}: {lats[-1]:.0f}ms")

elapsed = time.time() - start
lats = np.array(lats)
print(f"\n=== MMS-TTS Pipeline ===")
print(f"Iters: {iters} in {elapsed:.0f}s")
print(f"FPS: {iters/elapsed:.1f}")
print(f"Latency: avg={lats.mean():.0f}ms min={lats.min():.0f}ms max={lats.max():.0f}ms")

enc.release()
dec.release()
print("DONE")
