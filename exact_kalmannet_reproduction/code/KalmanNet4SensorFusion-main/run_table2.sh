#!/bin/bash
set -e
mkdir -p results logs

echo "=== Running KalmanNet (paper baseline) ==="
for tbptt in "2_4_50" "2_8_50" "50_50_50" "2_4_100" "2_8_100" "2_4_200" "2_8_200"; do
  echo "KalmanNet TBPTT: $tbptt"
  python train.py \
    --cfg ./configs/nclt/fusion/wheel_gpsfusion_origin.py \
    --tbptt "$tbptt" \
    --output-dir "experiments/knet_${tbptt}" \
    2>&1 | tee "logs/knet_${tbptt}.log"
done

echo "=== Running MambaKalmanNet ==="
for tbptt in "2_4_50" "2_8_50" "50_50_50" "2_4_100" "2_8_100" "2_4_200" "2_8_200"; do
  echo "Mamba TBPTT: $tbptt"
  python train.py \
    --cfg ./configs/nclt/fusion/wheel_gpsfusion_mamba.py \
    --tbptt "$tbptt" \
    --output-dir "experiments/mamba_${tbptt}" \
    2>&1 | tee "logs/mamba_${tbptt}.log"
done

echo "=== Generating Table II comparison ==="
python make_table2.py
