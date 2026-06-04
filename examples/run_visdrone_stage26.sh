#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-data/visdrone_det_coco}"

python scripts/run_stage26_best_candidates_seed42.py \
  --python python \
  --dataset-root "$DATASET_ROOT" \
  --epochs 100 \
  --batch-size 8 \
  --data-num-workers 0 \
  --amp
