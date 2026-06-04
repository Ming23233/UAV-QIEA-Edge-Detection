#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
TOOLS_ROOT = BYTETRACK_ROOT / "tools"
if str(BYTETRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(BYTETRACK_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from yolox.exp import get_exp

import train_uav_stage1 as train_helpers


VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
    {"id": 11, "name": "others"},
]

CATEGORY_MAP = {
    "car": 4,
    "van": 5,
    "truck": 6,
    "truvk": 6,
    "bus": 9,
    "feright_car": 6,
    "feright car": 6,
    "freight_car": 6,
    "freight car": 6,
    "feright": 6,
}

SPLIT_DIRS = {
    "train": ("train/trainimg", "train/trainlabel"),
    "val": ("val/valimg", "val/vallabel"),
    "test": ("test/testimg", "test/testlabel"),
}

METHODS = [
    {
        "method": "Baseline",
        "exp_file": "exps/example/uav/yolox_nano_visdrone_det_640_baseline.py",
        "checkpoint": PROJECT_ROOT
        / "outputs"
        / "stage2_visdrone_640_baselines"
        / "baseline_640_seed42"
        / "best_ckpt.pth",
    },
    {
        "method": "+P2",
        "exp_file": "exps/example/uav/yolox_nano_visdrone_det_640_p2.py",
        "checkpoint": PROJECT_ROOT
        / "outputs"
        / "stage17_visdrone_640_100ep_correct_multiseed"
        / "p2_seed42"
        / "best_ckpt.pth",
    },
    {
        "method": "QIEA-Final",
        "exp_file": "exps/example/uav/generated_upgrade/stage6_qiea_final_visdrone_640.py",
        "checkpoint": PROJECT_ROOT
        / "outputs"
        / "stage6_qiea_final"
        / "qiea_final_seed42"
        / "best_ckpt.pth",
    },
]


def clean_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def parse_box(obj: ET.Element) -> tuple[float, float, float, float] | None:
    polygon = obj.find("polygon")
    if polygon is None:
        bndbox = obj.find("bndbox")
        if bndbox is None:
            return None
        try:
            x1 = float(bndbox.findtext("xmin"))
            y1 = float(bndbox.findtext("ymin"))
            x2 = float(bndbox.findtext("xmax"))
            y2 = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            return None
        return x1, y1, x2, y2

    xs = []
    ys = []
    for idx in range(1, 5):
        try:
            xs.append(float(polygon.findtext(f"x{idx}")))
            ys.append(float(polygon.findtext(f"y{idx}")))
        except (TypeError, ValueError):
            return None
    return min(xs), min(ys), max(xs), max(ys)


def convert_split(source_root: Path, split: str, output_root: Path, ann_start: int, image_start: int) -> tuple[dict, dict, int, int]:
    image_rel, label_rel = SPLIT_DIRS[split]
    image_dir = source_root / image_rel
    label_dir = source_root / label_rel
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    images = []
    annotations = []
    skipped_labels = 0
    skipped_objects = 0
    unknown_categories: dict[str, int] = {}
    image_id = image_start
    ann_id = ann_start

    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.xml"
        if not label_path.exists():
            skipped_labels += 1
            continue
        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
            }
        )

        tree = ET.parse(label_path)
        for obj in tree.findall(".//object"):
            raw_name = clean_name(obj.findtext("name") or "")
            category_id = CATEGORY_MAP.get(raw_name)
            if category_id is None:
                unknown_categories[raw_name] = unknown_categories.get(raw_name, 0) + 1
                skipped_objects += 1
                continue
            box = parse_box(obj)
            if box is None:
                skipped_objects += 1
                continue
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(x1, width - 1.0))
            y1 = max(0.0, min(y1, height - 1.0))
            x2 = max(0.0, min(x2, width))
            y2 = max(0.0, min(y2, height))
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 1.0 or box_h <= 1.0:
                skipped_objects += 1
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        image_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_CATEGORIES,
    }
    stats = {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "skipped_labels": skipped_labels,
        "skipped_objects": skipped_objects,
        "unknown_categories": unknown_categories,
    }
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"instances_{split}.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload, stats, ann_id, image_id


def convert_dataset(source_root: Path, output_root: Path) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    ann_id = 1
    image_id = 1
    for split in ("train", "val", "test"):
        _, stats, ann_id, image_id = convert_split(source_root, split, output_root, ann_id, image_id)
        stats_rows.append(stats)
    (output_root / "conversion_summary.json").write_text(
        json.dumps(stats_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats_rows


def evaluate_method(method: dict, dataset_root: Path, split: str, output_dir: Path, batch_size: int, device_name: str) -> dict:
    device = train_helpers.select_device(device_name)
    exp = get_exp(str(BYTETRACK_ROOT / method["exp_file"]), None)
    exp.data_dir = str(dataset_root)
    exp.val_ann = f"instances_{split}.json"
    exp.val_name = ""
    exp.data_num_workers = 0
    model = exp.get_model().to(device)
    checkpoint = Path(method["checkpoint"])
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    method_dir = output_dir / split / method["method"].replace("/", "_").replace("+", "plus").replace(" ", "_")
    method_dir.mkdir(parents=True, exist_ok=True)
    val_loader = exp.get_eval_loader(batch_size, False)
    result_json = method_dir / "detections.json"
    metrics = train_helpers.evaluate_coco(
        model,
        val_loader,
        num_classes=exp.num_classes,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        device=device,
        result_json_path=result_json,
    )
    row = {
        "method": method["method"],
        "split": split,
        "exp_file": method["exp_file"],
        "checkpoint": str(checkpoint),
        "detections": str(result_json),
        **metrics,
    }
    (method_dir / "metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def write_summary(output_dir: Path, rows: list[dict], conversion_stats: list[dict]) -> None:
    fields = [
        "split",
        "method",
        "ap50_95",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "proxy_recall50",
        "proxy_mean_best_iou",
        "checkpoint",
        "exp_file",
        "detections",
    ]
    with (output_dir / "stage22_dronevehicle_external_case.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    lines = [
        "# Stage 22 DroneVehicle External Case",
        "",
        "Setting: visible-light DroneVehicle split, oriented XML boxes converted to horizontal COCO boxes; categories are aligned to VisDrone IDs.",
        "",
        "## Conversion",
        "",
        "| Split | Images | Annotations | Skipped labels | Skipped objects |",
        "|---|---:|---:|---:|---:|",
    ]
    for stats in conversion_stats:
        lines.append(
            f"| {stats['split']} | {stats['images']} | {stats['annotations']} | "
            f"{stats['skipped_labels']} | {stats['skipped_objects']} |"
        )

    lines.extend(
        [
            "",
            "## External Validation",
            "",
            "| Split | Method | AP50:95 | AP50 | AP_small | AP_medium | AP_large | Recall50 | Mean best IoU |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['method']} | {row.get('ap50_95', 0):.6f} | "
            f"{row.get('ap50', 0):.6f} | {row.get('ap_small', 0):.6f} | "
            f"{row.get('ap_medium', 0):.6f} | {row.get('ap_large', 0):.6f} | "
            f"{row.get('proxy_recall50', 0):.6f} | {row.get('proxy_mean_best_iou', 0):.6f} |"
        )
    (output_dir / "stage22_dronevehicle_external_case.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "stage22_dronevehicle_external_case.json").write_text(
        json.dumps({"conversion": conversion_stats, "results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser("Stage 22: DroneVehicle visible-light external case validation.")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data" / "DroneVehicle")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage22_dronevehicle_external_case" / "dronevehicle_visible_coco",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage22_dronevehicle_external_case",
    )
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    if args.skip_convert and (args.dataset_root / "conversion_summary.json").exists():
        conversion_stats = json.loads((args.dataset_root / "conversion_summary.json").read_text(encoding="utf-8"))
    else:
        conversion_stats = convert_dataset(args.source_root, args.dataset_root)

    rows = []
    for split in args.splits:
        for method in METHODS:
            rows.append(evaluate_method(method, args.dataset_root, split, args.output_dir, args.batch_size, args.device))
            write_summary(args.output_dir, rows, conversion_stats)
    write_summary(args.output_dir, rows, conversion_stats)


if __name__ == "__main__":
    main()
