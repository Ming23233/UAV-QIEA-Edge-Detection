#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-data/visdrone_det_coco}"

python scripts/stage3_generate_search_space.py
python scripts/stage4_validate_proxy_split.py --dataset-root "$DATASET_ROOT"

python scripts/run_stage5_proxy_search.py \
  --python python \
  --dataset-root "$DATASET_ROOT"

python scripts/run_stage6_qiea_final.py \
  --python python \
  --dataset-root "$DATASET_ROOT" \
  --epochs 100 \
  --batch-size 8 \
  --amp

python scripts/run_stage19_search_algorithm_comparison.py \
  --python python \
  --dataset-root "$DATASET_ROOT"

python scripts/run_stage26_best_candidates_seed42.py \
  --python python \
  --dataset-root "$DATASET_ROOT" \
  --epochs 100 \
  --batch-size 8 \
  --data-num-workers 0 \
  --amp
