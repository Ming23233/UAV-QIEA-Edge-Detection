# Reproducibility Notes

## Training Protocol

The main full-training protocol uses:

- 100 epochs
- batch size 8
- image size 640 x 640
- SGD optimizer with momentum
- learning rate 0.0015
- AMP when CUDA is available
- COCO-style detection metrics

Use fixed seeds when comparing Baseline, +P2, QIEA-Final, and search-best candidates.

## Proxy Search Protocol

The proxy search uses:

- 10 epochs per candidate
- seed 42
- batch size 10
- population size 4
- 4 generations
- 16 total evaluated candidates for QIEA

The compared search methods are QIEA, random search, GA, and an SA/QUBO heuristic reference under the same proxy budget where applicable.

## Diagnostic Analysis

Use `scripts/dataset_statistics.py` to recompute object-scale distributions from COCO annotations. Use `scripts/diagnose_small_object_errors.py` with COCO detection JSON files to reproduce the IoU-based missed-detection, localization-error, and false-positive diagnosis.

## Important Caution

Proxy-search results are not identical to final full-training rankings. The manuscript therefore treats QIEA as a candidate-screening and trade-off analysis tool, not as a guaranteed final-accuracy maximizer.

## Outputs to Archive

For paper evidence, archive:

- final `summary.json`
- final CSV/Markdown result tables
- generated figures
- exact experiment configuration files

For this public repository, the small CSV/JSON/Markdown tables under `results/` have already been included because they are manuscript-level summaries rather than raw training logs or dataset-derived annotation files.

Avoid uploading:

- raw datasets
- checkpoints
- logs with local paths
- large detection-output JSON files
- temporary generated files
