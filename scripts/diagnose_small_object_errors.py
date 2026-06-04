#!/usr/bin/env python3
"""IoU-based small-object error diagnosis for COCO-style detection outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw * ah + bw * bh - inter, 1e-12)
    return inter / union


def is_small(ann: dict, small_area: float) -> bool:
    bbox = ann.get("bbox", [0, 0, 0, 0])
    area = float(ann.get("area", bbox[2] * bbox[3]))
    return area < small_area


def coco_small_counts_by_image(annotations: list[dict], small_area: float) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for ann in annotations:
        if is_small(ann, small_area):
            counts[int(ann["image_id"])] += 1
    return counts


def dense_threshold(counts: dict[int, int], quantile: float) -> int:
    values = sorted(counts.values())
    if not values:
        return 0
    idx = int((len(values) - 1) * quantile)
    return int(values[idx])


def select_ground_truth(
    annotations: list[dict],
    subset: str,
    dense_images: set[int],
    small_area: float,
    very_small_area: float,
) -> list[dict]:
    selected = []
    for ann in annotations:
        if subset == "dense-small":
            keep = int(ann["image_id"]) in dense_images and is_small(ann, small_area)
        elif subset == "very-small":
            keep = is_small(ann, very_small_area)
        elif subset == "occluded-small":
            keep = is_small(ann, small_area) and float(ann.get("occlusion", 0)) > 0
        else:
            keep = is_small(ann, small_area)
        if keep:
            selected.append(ann)
    return selected


def diagnose(gt_anns: list[dict], detections: list[dict], score_thr: float, max_dets: int) -> dict:
    gt_by_image_class: dict[tuple[int, int], list[dict]] = defaultdict(list)
    det_by_image: dict[int, list[dict]] = defaultdict(list)

    for ann in gt_anns:
        gt_by_image_class[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)
    gt_image_ids = {int(ann["image_id"]) for ann in gt_anns}

    for det in detections:
        if float(det.get("score", 0.0)) >= score_thr and int(det["image_id"]) in gt_image_ids:
            det_by_image[int(det["image_id"])].append(det)

    matched_gt: set[int] = set()
    tp = 0
    localization_errors = 0
    false_positives = 0

    for image_id, dets in det_by_image.items():
        dets = sorted(dets, key=lambda row: float(row.get("score", 0.0)), reverse=True)[:max_dets]
        for det in dets:
            cls = int(det["category_id"])
            candidates = gt_by_image_class.get((image_id, cls), [])
            best_iou = 0.0
            best_ann_id = None
            for ann in candidates:
                ann_id = int(ann.get("id", id(ann)))
                if ann_id in matched_gt:
                    continue
                cur_iou = iou_xywh(det["bbox"], ann["bbox"])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_ann_id = ann_id
            if best_iou >= 0.5 and best_ann_id is not None:
                tp += 1
                matched_gt.add(best_ann_id)
            elif best_iou >= 0.1:
                localization_errors += 1
            else:
                false_positives += 1

    missed = len(gt_anns) - len(matched_gt)
    error_events = missed + localization_errors + false_positives
    return {
        "tp_at_0_5": tp,
        "error_events": error_events,
        "missed_detections": missed,
        "missed_percent": 0.0 if error_events == 0 else 100.0 * missed / error_events,
        "localization_errors": localization_errors,
        "localization_percent": 0.0 if error_events == 0 else 100.0 * localization_errors / error_events,
        "false_positives": false_positives,
        "false_positive_percent": 0.0 if error_events == 0 else 100.0 * false_positives / error_events,
        "recall_at_0_5": 0.0 if len(gt_anns) == 0 else tp / len(gt_anns),
        "gt_count": len(gt_anns),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Diagnose missed detections, localization errors, and false positives.")
    parser.add_argument("--gt", type=Path, required=True, help="COCO ground-truth annotation JSON.")
    parser.add_argument("--detections", type=Path, nargs="+", required=True, help="COCO detection JSON files.")
    parser.add_argument("--model-names", nargs="+", default=None, help="Optional model names matching detections.")
    parser.add_argument("--subset", choices=["small", "very-small", "dense-small", "occluded-small"], default="dense-small")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--dense-quantile", type=float, default=0.75)
    parser.add_argument("--small-area", type=float, default=32 * 32)
    parser.add_argument("--very-small-area", type=float, default=16 * 16)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    gt = load_json(args.gt)
    annotations = gt.get("annotations", [])
    counts = coco_small_counts_by_image(annotations, args.small_area)
    threshold = dense_threshold(counts, args.dense_quantile)
    dense_images = {image_id for image_id, count in counts.items() if count >= threshold}
    gt_anns = select_ground_truth(annotations, args.subset, dense_images, args.small_area, args.very_small_area)

    names = args.model_names or [path.stem for path in args.detections]
    if len(names) != len(args.detections):
        raise ValueError("--model-names must match the number of --detections files")

    rows = []
    for name, det_path in zip(names, args.detections):
        row = {"model": name, "subset": args.subset, "dense_threshold": threshold}
        row.update(diagnose(gt_anns, load_json(det_path), args.score_thr, args.max_dets))
        rows.append(row)

    fields = [
        "model",
        "subset",
        "dense_threshold",
        "gt_count",
        "tp_at_0_5",
        "error_events",
        "missed_detections",
        "missed_percent",
        "localization_errors",
        "localization_percent",
        "false_positives",
        "false_positive_percent",
        "recall_at_0_5",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.out_csv)


if __name__ == "__main__":
    main()
