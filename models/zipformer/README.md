# Zipformer STT (FP16)

K2-Zipformer 流式语音识别模型，中英双语 ASR，适用于 Seeed reCamera Pro (RV1126B)。

## 模型信息

Zipformer 由三个子模型组成 (Encoder + Decoder + Joiner):

| 子模型 | 说明 | ONNX 大小 | RKNN 大小 |
|--------|------|-----------|-----------|
| Encoder | 特征编码 (Zipformer streaming) | 75 MB | 108 MB |
| Decoder | 自回归解码 (Embedding) | 14 MB | 7.7 MB |
| Joiner | 联合解码 (Linear) | 13 MB | 6.3 MB |

- **来源**: [csukuangfj/k2fsa-zipformer-bilingual-zh-en-t](https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t)
- **ONNX 来源**: RKNN Model Zoo
- **量化**: FP16 (rv1126b 上推荐 FP16，INT8 量化效果差)
- **特征**: 80 维 log-mel fbank, 16kHz 采样率
- **词汇表**: 6257 tokens (中英混合 + sentencepiece BPE)
- **流式参数**: segment=103 帧, offset=96 帧

## 关于 INT8 量化

INT8 (W8A8) 量化虽然能将模型体积缩小约 46%、延迟降低约 22%，但对 Zipformer 流式架构的识别准确度影响严重。原因是 Encoder 维护 45 个缓存状态张量，跨段传递信息，INT8 量化误差在流式推理中逐段累积，导致最终识别结果严重退化。因此本项目仅保留 FP16 版本。

## reCamera Pro (RV1126B) 上的性能

| 音频长度 | 总延迟 | RTF (实时因子) |
|----------|--------|---------------|
| 5.61s (test.wav) | 1834.9 ms | 0.33 |
| 5.00s (麦克风) | 1580.1 ms | 0.32 |

RTF < 1.0 表示实时处理（推理速度快于音频时长）。

## 文件清单

| 文件 | 说明 |
|------|------|
| `zipformer_encoder.onnx` | Encoder ONNX 模型 |
| `zipformer_decoder.onnx` | Decoder ONNX 模型 |
| `zipformer_joiner.onnx` | Joiner ONNX 模型 |
| `zipformer_encoder_official_FP16.rknn` | Encoder FP16 RKNN |
| `zipformer_decoder_official_FP16.rknn` | Decoder FP16 RKNN |
| `zipformer_joiner_official_FP16.rknn` | Joiner FP16 RKNN |
| `convert_to_rknn.py` | ONNX → RKNN 转换脚本 |
| `infer.py` | 推理脚本 (WAV 文件 + 麦克风) |
| `vocab.txt` | 词汇表 (6257 tokens) |
| `test.wav` | 测试音频 (5.61s 中文) |

## 使用方法

### 转换 ONNX 到 RKNN

在 x86_64 主机上使用 `recamera-rknn-2.3.2` conda 环境:

```bash
conda run -n recamera-rknn-2.3.2 python convert_to_rknn.py
```

一次性转换 encoder、decoder、joiner 三个模型为 FP16。

### 设备上运行推理

将三个 `.rknn` 文件、`vocab.txt`、`infer.py` 推送到设备后:

**WAV 文件模式:**
```bash
python3 infer.py \
    --encoder zipformer_encoder_official_FP16.rknn \
    --decoder zipformer_decoder_official_FP16.rknn \
    --joiner  zipformer_joiner_official_FP16.rknn \
    --vocab vocab.txt \
    --input test.wav
```

**麦克风模式** (录音后识别):
```bash
python3 infer.py \
    --encoder zipformer_encoder_official_FP16.rknn \
    --decoder zipformer_decoder_official_FP16.rknn \
    --joiner  zipformer_joiner_official_FP16.rknn \
    --vocab vocab.txt \
    --duration 5
```

可选参数:
- `--device` — ALSA 采集设备 (默认: `hw:0,0`)
- `--duration` — 录音时长秒数 (默认: 5)
- `--runs` — 多次推理取平均 (文件模式)

## 流式推理流程

1. **特征提取**: 纯 numpy 实现 80 维 log-mel fbank (匹配 kaldifeat 参数)
2. **分段编码**: 将 fbank 按 103 帧分段，每段 overlap 96 帧送入 encoder
3. **状态缓存**: Encoder 维护 45 个缓存张量 (conv/key/val/avg/len 状态)
4. **贪心解码**: 逐帧 encoder 输出 → joiner + decoder 联合解码
5. **后处理**: 索引映射 → sentencepiece 还原 → 文本输出

## 测试结果

test.wav (5.61s 中文语音):
```
对我做了介绍那么我想说的是大家如果我的研究感兴趣
```

预期结果: "对我做了介绍那么我想说的是大家如果对我的研究感兴趣呢"
