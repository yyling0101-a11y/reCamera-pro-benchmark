#!/bin/bash
# Activate the benchmark environment (Miniconda3)
eval "$(/home/seeed/miniconda3/bin/conda shell.bash hook)"
conda activate rknn_bench
echo "Environment: rknn_bench (Miniconda3, Python 3.10)"
echo "Key packages:"
python -c "
import torch; print(f'  torch: {torch.__version__}')
import numpy; print(f'  numpy: {numpy.__version__}')
import onnx; print(f'  onnx: {onnx.__version__}')
import paddle; print(f'  paddle: {paddle.__version__}')
import paddle2onnx; print(f'  paddle2onnx: {paddle2onnx.__version__}')
print('  rknn-toolkit2: 2.3.2')
print('  ultralytics: OK')
print('  paramiko: OK')
print('  huggingface_hub: OK')
" 2>/dev/null
