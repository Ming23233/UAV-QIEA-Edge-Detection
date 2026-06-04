# Results Directory

This directory contains lightweight, manuscript-aligned result artifacts for the current paper version. These files are intended for reviewer inspection and reproducibility checks.

Included files are small CSV/JSON/Markdown summaries only. They do not include raw datasets, model checkpoints, pretrained weights, full stdout/stderr logs, or large detection-output JSON files.

## Included Tables

- `table_1_main_multiseed.csv`: three-seed VisDrone results for Baseline, +P2, and QIEA-Final.
- `table_2_scale_distribution.csv`: VisDrone and AU-AIR object-scale statistics.
- `table_3_small_diagnostic_ap.csv`: diagnostic AP for very-small, dense-small, and occluded-small cases.
- `table_4_dense_error_diagnostic.csv`: dense-small error-type diagnosis.
- `table_5_contextual_yolo_records.csv`: contextual YOLOv5n/YOLOv8n records.
- `table_6_ablation.csv`: bounded ablation results.
- `table_7_search_algorithm_comparison.csv`: proxy-stage search comparison.
- `table_8_search_candidate_fulltrain_seed42.csv`: full 100-epoch verification of proxy-best candidates.
- `table_9_efficiency_deployment.csv`: parameter, FLOPs, latency, memory, and AP-small cost results.
- `table_10_auair_case.csv`: AU-AIR zero-shot and 30% fine-tuning engineering case.
- `table_11_accuracy_cost_pareto.csv`: marginal AP-small-to-cost analysis.

`paper_results_summary.json` mirrors the same values in machine-readable form. `diagnostic_definitions.md` documents the scale bins and diagnostic rules used in the manuscript.
