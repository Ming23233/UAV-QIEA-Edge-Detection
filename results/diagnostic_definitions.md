# Diagnostic Definitions

The manuscript uses COCO-style object area definitions and an additional very-small bin for UAV diagnostics.

## Scale Bins

- Very-small: bounding-box area `< 16^2` pixels.
- COCO-small: bounding-box area `< 32^2` pixels.
- Medium: bounding-box area in `[32^2, 96^2)` pixels.
- Large: bounding-box area `>= 96^2` pixels.

## Diagnostic Subsets

- Very-small objects: ground-truth boxes with area `< 16^2` pixels.
- Dense-small images: VisDrone validation images whose COCO-small object count is no lower than the 75th percentile of the validation split. In the reported split, the threshold is 66 small objects per image and selects 139 of 548 validation images.
- Occluded-small objects: COCO-small VisDrone ground-truth boxes with a nonzero occlusion flag.

## Dense-Small Error Diagnosis

The dense-small error diagnosis uses seed-42 VisDrone validation detections.

- True positive: a prediction matched to a same-class ground-truth box with IoU `>= 0.5`.
- Localization error: an unmatched prediction overlapping a same-class ground-truth box with IoU in `[0.1, 0.5)`.
- False positive: an unmatched prediction not counted as a localization error.
- Missed detection: an unmatched ground-truth box.

Predictions are filtered with score `>= 0.05` and `maxDets=100`, matching the COCO-style evaluation setting used in the manuscript.
