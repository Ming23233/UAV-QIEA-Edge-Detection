# Open-Source Audit for the Current Manuscript

This checklist maps the current manuscript's reproducibility claims to files in this repository.

## Included

- Training scripts for Baseline, +P2, QIEA-Final, search comparison, efficiency evaluation, and Stage26 full-training verification: `scripts/`.
- Model and experiment configuration files: `third_party/ByteTrack/exps/example/uav/` and `configs/`.
- Modified YOLOX/ByteTrack detector implementation with P2, attention, context, fusion, small-object loss weighting, and center-radius hooks: `third_party/ByteTrack/yolox/`.
- Dataset conversion, scale-statistics, and small-object error-diagnosis helpers: `third_party/ByteTrack/tools/`, `scripts/dataset_statistics.py`, and `scripts/diagnose_small_object_errors.py`.
- Manuscript-aligned small result summaries: `results/`.
- Diagnostic definitions for scale bins and error types: `results/diagnostic_definitions.md`.

## Intentionally Excluded

- Raw VisDrone, AU-AIR, UAVDT, DroneVehicle, or other dataset files.
- Converted COCO annotation JSON files derived from third-party datasets.
- Model checkpoints, pretrained weights, and exported deployment models.
- Full stdout/stderr logs and local run folders.
- Large detection-output JSON files.
- Manuscript drafts and local PDF/DOCX files.

## Current Consistency Notes

- The paper uses 640 x 640 VisDrone full training with 100 epochs and seeds 42/43/44.
- The QIEA proxy search uses 16 candidates, 4 generations, population size 4, 10 proxy epochs, and seed 42.
- The selected QIEA-Final configuration is P2 + CA, with small-object loss weight 1.25 and center radius 2.5.
- Random-best, GA-best, and SA/QUBO-best are provided as explicit full-training experiment files under `third_party/ByteTrack/exps/example/uav/generated_upgrade/`.
