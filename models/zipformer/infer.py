#!/usr/bin/env python3
"""
Zipformer STT RKNN inference for RV1126B.
Streaming Encoder + Decoder + Joiner pipeline.

Usage:
    # Audio file mode:
    python infer.py --encoder zipformer_encoder_official_FP16.rknn \
                    --decoder zipformer_decoder_official_FP16.rknn \
                    --joiner  zipformer_joiner_official_FP16.rknn \
                    --vocab vocab.txt \
                    --input test.wav

    # Real-time microphone mode (streaming, print results as you speak):
    python infer.py --encoder zipformer_encoder_official_FP16.rknn \
                    --decoder zipformer_decoder_official_FP16.rknn \
                    --joiner  zipformer_joiner_official_FP16.rknn \
                    --vocab vocab.txt \
                    --duration 10
"""

import argparse
import os
import sys
import time
import struct
import wave
import subprocess
import numpy as np

# ─── Audio I/O ────────────────────────────────────────────────────────

SAMPLE_RATE = 16000


def read_wav(path):
    """Read WAV file and return float32 mono audio at 16kHz."""
    with wave.open(path, 'rb') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / rate
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        print(f"  Resampled: {rate} Hz -> {SAMPLE_RATE} Hz")

    return data


# ─── Streaming Fbank Feature Extraction ───────────────────────────────

def mel_filterbank(num_bins, sample_rate, fft_size, low_freq=20, high_freq_offset=-400):
    high_freq = sample_rate / 2.0 + high_freq_offset
    def hz_to_mel(hz):
        return 1127.0 * np.log(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (np.exp(mel / 1127.0) - 1.0)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = mel_to_hz(mel_points)

    num_fft_bins = fft_size // 2 + 1
    filters = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    freq_res = sample_rate / fft_size
    for i in range(num_bins):
        left, center, right = hz_points[i], hz_points[i+1], hz_points[i+2]
        for j in range(num_fft_bins):
            freq = j * freq_res
            if left <= freq <= center:
                filters[i, j] = (freq - left) / (center - left) if center != left else 0
            elif center < freq <= right:
                filters[i, j] = (right - freq) / (right - center) if right != center else 0
    return filters


class StreamingFbank:
    """Incremental fbank extractor for streaming audio."""

    def __init__(self, sample_rate=16000, num_bins=80,
                 frame_length_ms=25, frame_shift_ms=10, high_freq_offset=-400):
        self.sample_rate = sample_rate
        self.num_bins = num_bins
        self.frame_length = int(sample_rate * frame_length_ms / 1000.0)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000.0)
        self.fft_size = 512
        while self.fft_size < self.frame_length:
            self.fft_size *= 2

        self.preemph_coeff = 0.97
        self.window = np.hamming(self.frame_length).astype(np.float32)
        self.mel_fb = mel_filterbank(num_bins, sample_rate, self.fft_size,
                                     high_freq_offset=high_freq_offset)

        # State: overlap audio from previous call
        self.prev_sample = 0.0
        # Audio buffer for overlap (for frame computation)
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.frames_produced = 0

    def reset(self):
        self.prev_sample = 0.0
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.frames_produced = 0

    def process(self, audio_chunk, flush=False):
        """
        Process new audio samples and return fbank frames.
        audio_chunk: float32 array of new samples.
        flush: if True, also extract remaining frames with zero padding.
        Returns: numpy array [N, 80] of log-mel features.
        """
        # Pre-emphasis: connect with previous chunk
        if len(audio_chunk) == 0:
            return np.zeros((0, self.num_bins), dtype=np.float32)

        preemph = np.zeros(len(audio_chunk), dtype=np.float32)
        preemph[0] = audio_chunk[0] - self.preemph_coeff * self.prev_sample
        if len(audio_chunk) > 1:
            preemph[1:] = audio_chunk[1:] - self.preemph_coeff * audio_chunk[:-1]
        self.prev_sample = audio_chunk[-1]

        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, preemph])

        # Extract frames
        half_frame = self.frame_length // 2
        # With snip_edges=False: total samples = original_len + 2*half_frame
        # But for streaming, we compute frames from what we have
        # Frame i starts at sample i * frame_shift (centered)
        features = []
        frame_idx = self.frames_produced

        while True:
            center = frame_idx * self.frame_shift
            start = center - half_frame
            end = start + self.frame_length

            if end > len(self.audio_buffer):
                if flush and start < len(self.audio_buffer):
                    # Pad remaining with zeros for one final frame
                    frame = np.zeros(self.frame_length, dtype=np.float32)
                    if start >= 0:
                        avail = len(self.audio_buffer) - start
                        if avail > 0:
                            frame[:avail] = self.audio_buffer[start:start + avail]
                    else:
                        copy_start = -start
                        frame[copy_start:copy_start + len(self.audio_buffer)] = self.audio_buffer
                    features.append(self._extract_frame(frame))
                    frame_idx += 1
                break

            if start < 0:
                # Pad beginning
                frame = np.zeros(self.frame_length, dtype=np.float32)
                copy_start = -start
                frame[copy_start:] = self.audio_buffer[0:end]
            else:
                frame = self.audio_buffer[start:end].copy()

            features.append(self._extract_frame(frame))
            frame_idx += 1

        # Keep buffer for overlap (we need half_frame samples before next frame start)
        next_start = frame_idx * self.frame_shift - half_frame
        if next_start > 0 and next_start < len(self.audio_buffer):
            self.audio_buffer = self.audio_buffer[next_start:]
        elif next_start >= len(self.audio_buffer):
            self.audio_buffer = np.zeros(0, dtype=np.float32)

        self.frames_produced = frame_idx

        if features:
            return np.array(features, dtype=np.float32)
        return np.zeros((0, self.num_bins), dtype=np.float32)

    def _extract_frame(self, frame):
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed, n=self.fft_size)
        power = np.abs(spectrum) ** 2
        mel_energy = self.mel_fb @ power
        mel_energy = np.maximum(mel_energy, 1e-10)
        return np.log(mel_energy)


def compute_fbank(audio, sample_rate=16000, num_bins=80, frame_length_ms=25,
                  frame_shift_ms=10, high_freq_offset=-400):
    """Batch fbank for file mode."""
    sf = StreamingFbank(sample_rate, num_bins, frame_length_ms,
                        frame_shift_ms, high_freq_offset)
    return sf.process(audio, flush=True)


# ─── Vocab ─────────────────────────────────────────────────────────────

def read_vocab(path):
    vocab = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                vocab[int(parts[1])] = parts[0]
            elif len(parts) == 1:
                vocab[len(vocab)] = parts[0]
    return vocab


# ─── Zipformer Streaming Pipeline ─────────────────────────────────────

BLANK_ID = 0
UNK_ID = 2
SEGMENT = 103
OFFSET = 96
CONTEXT_SIZE = 2
NUM_MEL_BINS = 80

ENCODER_STATE_CONFIG = {
    'cached_len_0': [2, 1], 'cached_len_1': [2, 1],
    'cached_len_2': [2, 1], 'cached_len_3': [2, 1], 'cached_len_4': [2, 1],
    'cached_avg_0': [2, 1, 256], 'cached_avg_1': [2, 1, 256],
    'cached_avg_2': [2, 1, 256], 'cached_avg_3': [2, 1, 256], 'cached_avg_4': [2, 1, 256],
    'cached_key_0': [2, 192, 1, 192], 'cached_key_1': [2, 96, 1, 192],
    'cached_key_2': [2, 48, 1, 192], 'cached_key_3': [2, 24, 1, 192], 'cached_key_4': [2, 96, 1, 192],
    'cached_val_0': [2, 192, 1, 96], 'cached_val_1': [2, 96, 1, 96],
    'cached_val_2': [2, 48, 1, 96], 'cached_val_3': [2, 24, 1, 96], 'cached_val_4': [2, 96, 1, 96],
    'cached_val2_0': [2, 192, 1, 96], 'cached_val2_1': [2, 96, 1, 96],
    'cached_val2_2': [2, 48, 1, 96], 'cached_val2_3': [2, 24, 1, 96], 'cached_val2_4': [2, 96, 1, 96],
    'cached_conv1_0': [2, 1, 256, 30], 'cached_conv1_1': [2, 1, 256, 30],
    'cached_conv1_2': [2, 1, 256, 30], 'cached_conv1_3': [2, 1, 256, 30], 'cached_conv1_4': [2, 1, 256, 30],
    'cached_conv2_0': [2, 1, 256, 30], 'cached_conv2_1': [2, 1, 256, 30],
    'cached_conv2_2': [2, 1, 256, 30], 'cached_conv2_3': [2, 1, 256, 30], 'cached_conv2_4': [2, 1, 256, 30],
}


class ZipformerPipeline:
    def __init__(self, encoder_path, decoder_path, joiner_path):
        from rknnlite.api import RKNNLite

        self.encoder = RKNNLite()
        self.encoder.load_rknn(encoder_path)
        self.encoder.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        self.decoder = RKNNLite()
        self.decoder.load_rknn(decoder_path)
        self.decoder.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        self.joiner = RKNNLite()
        self.joiner.load_rknn(joiner_path)
        self.joiner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

        self.init_state()

    def init_state(self):
        self.enc_state_names = list(ENCODER_STATE_CONFIG.keys())
        self.enc_states = []
        for name in self.enc_state_names:
            shape = ENCODER_STATE_CONFIG[name]
            if 'cached_len' in name:
                self.enc_states.append(np.zeros(shape, dtype=np.int64))
            else:
                self.enc_states.append(np.zeros(shape, dtype=np.float32))

    def run_encoder(self, x):
        inputs = [x.astype(np.float32)]
        for state in self.enc_states:
            inputs.append(state)
        outputs = self.encoder.inference(inputs=inputs)
        encoder_out = outputs[0]
        for idx in range(len(self.enc_states)):
            out_idx = idx + 1
            if out_idx < len(outputs):
                data = outputs[out_idx]
                name = self.enc_state_names[idx]
                if data.ndim == 4 and name in ENCODER_STATE_CONFIG:
                    expected_shape = ENCODER_STATE_CONFIG[name]
                    if list(data.shape) != expected_shape:
                        data = np.transpose(data, (0, 2, 3, 1))
                self.enc_states[idx] = data
        return encoder_out

    def run_decoder(self, decoder_input):
        outputs = self.decoder.inference(inputs=[decoder_input.astype(np.int64)])
        return outputs[0]

    def run_joiner(self, encoder_out, decoder_out):
        outputs = self.joiner.inference(inputs=[
            encoder_out.astype(np.float32),
            decoder_out.astype(np.float32)
        ])
        return outputs[0]

    def transcribe(self, features):
        """Full batch transcription."""
        num_frames = features.shape[0]
        hyp = [BLANK_ID] * CONTEXT_SIZE
        decoder_input = np.array([hyp], dtype=np.int64)
        decoder_out = self.run_decoder(decoder_input)

        timestamp = []
        num_processed = 0
        frame_offset = 0
        total_enc_time = 0.0

        while num_frames - num_processed > 0:
            remaining = num_frames - num_processed
            if remaining < SEGMENT:
                pad = np.zeros((SEGMENT - remaining, NUM_MEL_BINS), dtype=np.float32)
                segment_features = np.concatenate([
                    features[num_processed:num_processed + remaining], pad], axis=0)
            else:
                segment_features = features[num_processed:num_processed + SEGMENT]

            x = segment_features[np.newaxis, :, :]
            t0 = time.perf_counter()
            encoder_out = self.run_encoder(x)
            total_enc_time += (time.perf_counter() - t0)

            enc_out = encoder_out.squeeze(0)
            T_enc = enc_out.shape[0]
            for t in range(T_enc):
                cur_enc = enc_out[t:t + 1]
                joiner_out = self.run_joiner(cur_enc, decoder_out).squeeze(0)
                y = np.argmax(joiner_out)
                if y != BLANK_ID and y != UNK_ID:
                    timestamp.append(frame_offset + t)
                    hyp.append(int(y))
                    dec_input = np.array([hyp[-CONTEXT_SIZE:]], dtype=np.int64)
                    decoder_out = self.run_decoder(dec_input)

            frame_offset += T_enc
            num_processed += OFFSET

        self._last_enc_time = total_enc_time
        return hyp[CONTEXT_SIZE:], timestamp

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.joiner.release()


# ─── Streaming Recognizer ─────────────────────────────────────────────

class StreamingRecognizer:
    """
    Real-time streaming recognizer: continuously reads audio from arecord,
    computes fbank features incrementally, and runs encoder+decoder+joiner
    as soon as enough frames are available.
    """

    def __init__(self, pipeline, vocab, device="hw:0,0"):
        self.pipeline = pipeline
        self.vocab = vocab
        self.device = device
        self.fbank = StreamingFbank()

    def tokens_to_text(self, hyp):
        text = ""
        for idx in hyp:
            if idx in self.vocab:
                text += self.vocab[idx]
        return text.replace("▁", " ").strip()

    def run(self, duration_sec):
        """
        Run real-time streaming recognition for duration_sec seconds.
        Prints recognized text to terminal as it accumulates.
        """
        # Start arecord as subprocess with stdout pipe for streaming
        cmd = [
            "arecord",
            "--device", self.device,
            "-d", str(duration_sec),
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "--buffer-time=100000",  # 100ms buffer
            "-"  # output to stdout
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Each chunk: OFFSET frames of audio
        # frame_shift=10ms => OFFSET=96 frames => 960ms => 15360 samples
        chunk_samples = OFFSET * 160  # 96 * 160 = 15360 samples per chunk
        chunk_bytes = chunk_samples * 2  # S16_LE = 2 bytes per sample

        # Greedy search state
        hyp = [BLANK_ID] * CONTEXT_SIZE
        decoder_input = np.array([hyp], dtype=np.int64)
        decoder_out = self.pipeline.run_decoder(decoder_input)
        frame_offset = 0
        total_segments = 0
        last_text = ""

        print(f"\n{'='*60}")
        print(f"  Real-time streaming recognition")
        print(f"  Duration: {duration_sec}s | Device: {self.device}")
        print(f"  Listening... (speak now)")
        print(f"{'='*60}")
        print()

        t_start = time.perf_counter()
        segment_count = 0

        try:
            while True:
                # Read one chunk of audio from arecord
                raw = proc.stdout.read(chunk_bytes)
                if not raw or len(raw) < chunk_bytes:
                    # End of audio or short read
                    if raw and len(raw) > 0:
                        # Process remaining samples
                        audio_chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        new_frames = self.fbank.process(audio_chunk)
                        if len(new_frames) > 0:
                            self._process_new_frames(
                                new_frames, hyp, decoder_out, frame_offset)
                    break

                # Convert to float32
                audio_chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                # Compute fbank features for this chunk
                new_frames = self.fbank.process(audio_chunk)

                if len(new_frames) == 0:
                    continue

                # Check if we have enough accumulated frames for a segment
                result = self._process_new_frames(
                    new_frames, hyp, decoder_out, frame_offset)
                if result is not None:
                    hyp, decoder_out, frame_offset = result
                    total_segments += 1

                # Print current partial result
                current_text = self.tokens_to_text(hyp[CONTEXT_SIZE:])
                if current_text and current_text != last_text:
                    elapsed = time.perf_counter() - t_start
                    # Use \r to overwrite the current line for a clean display
                    sys.stdout.write(f"\r  [{elapsed:5.1f}s] {current_text}")
                    sys.stdout.flush()
                    last_text = current_text

        except KeyboardInterrupt:
            proc.terminate()
            print(f"\n\n  Interrupted by user")

        proc.wait()
        total_time = time.perf_counter() - t_start

        # Final result
        final_text = self.tokens_to_text(hyp[CONTEXT_SIZE:])
        print(f"\n")
        print(f"{'='*60}")
        print(f"  Final result ({total_time:.1f}s, {total_segments} segments):")
        print(f"  {final_text}")
        print(f"{'='*60}")
        return final_text

    def _process_new_frames(self, new_frames, hyp, decoder_out, frame_offset):
        """
        Append new_frames to internal buffer and process a segment if ready.
        Returns updated (hyp, decoder_out, frame_offset) or None.
        """
        if not hasattr(self, '_feat_buffer'):
            self._feat_buffer = np.zeros((0, NUM_MEL_BINS), dtype=np.float32)

        self._feat_buffer = np.concatenate([self._feat_buffer, new_frames], axis=0)

        if len(self._feat_buffer) < SEGMENT:
            return None

        # Take first SEGMENT frames
        segment_features = self._feat_buffer[:SEGMENT]
        # Keep the rest (OFFSET frames of overlap are implicitly in the next segment)
        self._feat_buffer = self._feat_buffer[OFFSET:]

        x = segment_features[np.newaxis, :, :].astype(np.float32)
        encoder_out = self.pipeline.run_encoder(x)
        enc_out = encoder_out.squeeze(0)
        T_enc = enc_out.shape[0]

        for t in range(T_enc):
            cur_enc = enc_out[t:t + 1]
            joiner_out = self.pipeline.run_joiner(cur_enc, decoder_out).squeeze(0)
            y = np.argmax(joiner_out)
            if y != BLANK_ID and y != UNK_ID:
                hyp.append(int(y))
                dec_input = np.array([hyp[-CONTEXT_SIZE:]], dtype=np.int64)
                decoder_out = self.pipeline.run_decoder(dec_input)

        frame_offset += T_enc
        return (hyp, decoder_out, frame_offset)


# ─── Post-processing ──────────────────────────────────────────────────

def post_process(hyp, vocab, timestamp):
    text = ""
    for idx in hyp:
        if idx in vocab:
            text += vocab[idx]
    text = text.replace("▁", " ").strip()
    frame_shift_s = 10 / 1000.0 * 4
    real_ts = [round(frame_shift_s * t, 2) for t in timestamp]
    return text, real_ts


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Zipformer STT RKNN Inference')
    parser.add_argument('--encoder', required=True, help='Encoder RKNN path')
    parser.add_argument('--decoder', required=True, help='Decoder RKNN path')
    parser.add_argument('--joiner', required=True, help='Joiner RKNN path')
    parser.add_argument('--vocab', default='vocab.txt', help='Vocab file path')
    parser.add_argument('--input', help='Input WAV file (omit for real-time mic)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Streaming duration in seconds (mic mode, default: 10)')
    parser.add_argument('--device', default='hw:0,0',
                        help='ALSA capture device (mic mode)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of inference runs (file mode)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Zipformer STT RKNN Inference")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder}")
    print(f"Decoder: {args.decoder}")
    print(f"Joiner:  {args.joiner}")
    print(f"Vocab:   {args.vocab}")

    vocab = read_vocab(args.vocab)
    print(f"Vocab size: {len(vocab)}")

    print(f"\nLoading models...")
    pipeline = ZipformerPipeline(args.encoder, args.decoder, args.joiner)

    if args.input:
        # ── File mode ──
        print(f"\nInput: {args.input}")
        audio = read_wav(args.input)
        duration = len(audio) / SAMPLE_RATE
        print(f"  Audio: {duration:.2f}s, {len(audio)} samples @ {SAMPLE_RATE} Hz")

        print("  Computing fbank features...")
        features = compute_fbank(audio)
        print(f"  Features: {features.shape}")

        print("  Warmup...")
        pipeline.init_state()
        hyp, ts = pipeline.transcribe(features)

        if args.runs > 1:
            latencies = []
            for i in range(args.runs):
                pipeline.init_state()
                t0 = time.perf_counter()
                hyp, ts = pipeline.transcribe(features)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                print(f"  Run {i+1}: {latencies[-1]:.0f} ms")
            latencies = np.array(latencies)
            print(f"\n  Performance:")
            print(f"    Avg: {np.mean(latencies):.1f} ms")
            print(f"    Min: {np.min(latencies):.1f} ms")
            print(f"    Max: {np.max(latencies):.1f} ms")
        else:
            t0 = time.perf_counter()
            pipeline.init_state()
            hyp, ts = pipeline.transcribe(features)
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"  Total: {total_ms:.1f} ms")

        text, timestamps = post_process(hyp, vocab, ts)
        print(f"\n{'='*60}")
        print(f"Transcription:")
        print(f"  {text}")
        print(f"Timestamps: {timestamps}")
        print(f"{'='*60}")

    else:
        # ── Real-time streaming microphone mode ──
        pipeline.init_state()
        recognizer = StreamingRecognizer(pipeline, vocab, args.device)
        recognizer.run(args.duration)

    pipeline.release()
    print(f"\nDone.\n")
    return 0


if __name__ == '__main__':
    exit(main())
