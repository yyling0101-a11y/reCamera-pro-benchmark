#!/bin/bash
# Download all target ONNX models from Rockchip RKNN Model Zoo
BASE="https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples"
DST="/home/seeed/recamera_pro/benchmark/zoo_onnx"
mkdir -p "$DST"

declare -A MODELS=(
  # Object Detection - Rockchip optimized (SiLU->ReLU!)
  ["yolov8/yolov8n.onnx"]="yolov8n_zoo.onnx"
  ["yolov8/yolov8s.onnx"]="yolov8s_zoo.onnx"
  ["yolo11/yolo11n.onnx"]="yolo11n_zoo.onnx"
  ["yolo11/yolo11s.onnx"]="yolo11s_zoo.onnx"
  # Pose / Seg / OBB - Rockchip optimized
  ["yolov8_pose/yolov8n-pose.onnx"]="yolov8n-pose_zoo.onnx"
  ["yolov8_seg/yolov8n-seg.onnx"]="yolov8n-seg_zoo.onnx"
  ["yolov8_obb/yolov8n-obb.onnx"]="yolov8n-obb_zoo.onnx"
  # Classification
  ["mobilenet/mobilenetv2-12.onnx"]="mobilenetv2_zoo.onnx"
  ["resnet/resnet50-v2-7.onnx"]="resnet50-v2-7_zoo.onnx"
  # Face
  ["RetinaFace/RetinaFace_mobile320.onnx"]="retinaface_mobile320.onnx"
  # LPR
  ["LPRNet/lprnet.onnx"]="lprnet.onnx"
  # OCR
  ["PPOCR/ppocrv4_det.onnx"]="ppocrv4_det.onnx"
  ["PPOCR/ppocrv4_rec.onnx"]="ppocrv4_rec.onnx"
  # Semantic Segmentation
  ["ppseg/pp_liteseg_cityscapes.onnx"]="pp_liteseg_cityscapes.onnx"
  # MobileSAM
  ["mobilesam/mobilesam_encoder_tiny.onnx"]="mobilesam_encoder_tiny.onnx"
  ["mobilesam/mobilesam_decoder.onnx"]="mobilesam_decoder.onnx"
  # CLIP
  ["clip/clip_images.onnx"]="clip_images.onnx"
  ["clip/clip_text.onnx"]="clip_text.onnx"
  # Audio ASR
  ["wav2vec2/wav2vec2_base_960h_20s.onnx"]="wav2vec2_base_960h_20s.onnx"
  ["whisper/whisper_encoder_base_20s.onnx"]="whisper_encoder_base_20s.onnx"
  ["whisper/whisper_decoder_base_20s.onnx"]="whisper_decoder_base_20s.onnx"
  ["zipformer/encoder-epoch-99-avg-1.onnx"]="zipformer_encoder.onnx"
  ["zipformer/decoder-epoch-99-avg-1.onnx"]="zipformer_decoder.onnx"
  ["zipformer/joiner-epoch-99-avg-1.onnx"]="zipformer_joiner.onnx"
  # Audio Classification + TTS
  ["yamnet/yamnet_3s.onnx"]="yamnet_3s.onnx"
  ["mms_tts/mms_tts_eng_encoder_200.onnx"]="mms_tts_eng_encoder_200.onnx"
  ["mms_tts/mms_tts_eng_decoder_200.onnx"]="mms_tts_eng_decoder_200.onnx"
)

for path in "${!MODELS[@]}"; do
  fname="${MODELS[$path]}"
  url="$BASE/$path"
  dst="$DST/$fname"
  if [ -f "$dst" ] && [ $(stat -c%s "$dst" 2>/dev/null) -gt 1000 ]; then
    echo "SKIP $fname (exists)"
    continue
  fi
  echo "Downloading $fname..."
  HTTPS_PROXY=http://192.168.3.151:7890 curl -L --connect-timeout 30 --max-time 120 -o "$dst" "$url" 2>&1
  if [ -f "$dst" ]; then
    sz=$(stat -c%s "$dst" 2>/dev/null)
    echo "  -> $sz bytes"
  else
    echo "  -> FAILED"
  fi
done
echo "ALL DONE"
