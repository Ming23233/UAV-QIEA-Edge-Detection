#!/usr/bin/env python3
"""Dataset statistics for COCO-style UAV detection datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


SPLITS = ("train", "val")


def scale_bin(area: float) -> str:
    if area < 16 * 16:
        return "tiny_lt_16"
    if area < 32 * 32:
        return "small_16_32"
    if area < 96 * 96:
        return "medium_32_96"
    return "large_ge_96"


def short_side_bin(w: float, h: float) -> str:
    side = min(w, h)
    if side < 8:
        return "short_lt_8"
    if side < 16:
        return "short_8_16"
    if side < 32:
        return "short_16_32"
    if side < 64:
        return "short_32_64"
    if side < 96:
        return "short_64_96"
    return "short_ge_96"


def relative_area_bin(rel_area: float) -> str:
    if rel_area < 0.001:
        return "rel_lt_0.1pct"
    if rel_area < 0.005:
        return "rel_0.1_0.5pct"
    if rel_area < 0.01:
        return "rel_0.5_1pct"
    if rel_area < 0.05:
        return "rel_1_5pct"
    return "rel_ge_5pct"


def pct(n: float, d: float) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def safe_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_coco(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_split_stats(dataset_name: str, split: str, annotation_path: Path) -> dict:
    data = load_coco(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    image_by_id = {img["id"]: img for img in images}
    cat_by_id = {cat["id"]: cat.get("name", str(cat["id"])) for cat in categories}

    class_counts: Counter[str] = Counter()
    scale_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    rel_counts: Counter[str] = Counter()
    class_scale_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_areas: dict[str, list[float]] = defaultdict(list)
    image_ann_counts: Counter[int] = Counter()
    occlusion_counts: Counter[str] = Counter()
    truncation_counts: Counter[str] = Counter()

    valid_annotations = 0
    skipped_annotations = 0
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_areas: list[float] = []
    relative_areas: list[float] = []
    aspect_ratios: list[float] = []

    for ann in annotations:
        img = image_by_id.get(ann.get("image_id"))
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if img is None or len(bbox) < 4:
            skipped_annotations += 1
            continue
        w = float(bbox[2])
        h = float(bbox[3])
        if w <= 0 or h <= 0:
            skipped_annotations += 1
            continue

        img_w = float(img.get("width", 0) or 0)
        img_h = float(img.get("height", 0) or 0)
        img_area = img_w * img_h
        area = float(ann.get("area", w * h) or w * h)
        rel_area = 0.0 if img_area <= 0 else area / img_area
        cls_name = cat_by_id.get(ann.get("category_id"), str(ann.get("category_id")))
        sbin = scale_bin(area)

        valid_annotations += 1
        image_ann_counts[ann["image_id"]] += 1
        class_counts[cls_name] += 1
        scale_counts[sbin] += 1
        side_counts[short_side_bin(w, h)] += 1
        rel_counts[relative_area_bin(rel_area)] += 1
        class_scale_counts[cls_name][sbin] += 1
        class_areas[cls_name].append(area)
        box_widths.append(w)
        box_heights.append(h)
        box_areas.append(area)
        relative_areas.append(rel_area)
        aspect_ratios.append(w / h if h > 0 else 0.0)

        if "occlusion" in ann:
            occlusion_counts[str(ann["occlusion"])] += 1
        if "truncation" in ann:
            truncation_counts[str(ann["truncation"])] += 1

    ann_counts_per_image = [image_ann_counts.get(img["id"], 0) for img in images]
    coco_small = scale_counts["tiny_lt_16"] + scale_counts["small_16_32"]
    tiny = scale_counts["tiny_lt_16"]
    medium = scale_counts["medium_32_96"]
    large = scale_counts["large_ge_96"]

    class_rows = []
    for cls_name, count in class_counts.most_common():
        areas = class_areas[cls_name]
        row = {
            "class": cls_name,
            "count": count,
            "percent": pct(count, valid_annotations),
            "area_mean": safe_stats(areas)["mean"],
            "area_median": safe_stats(areas)["median"],
            "tiny_lt_16": class_scale_counts[cls_name]["tiny_lt_16"],
            "small_16_32": class_scale_counts[cls_name]["small_16_32"],
            "medium_32_96": class_scale_counts[cls_name]["medium_32_96"],
            "large_ge_96": class_scale_counts[cls_name]["large_ge_96"],
        }
        row["coco_small_percent_in_class"] = pct(
            row["tiny_lt_16"] + row["small_16_32"], count
        )
        class_rows.append(row)

    scale_rows = []
    for name in ("tiny_lt_16", "small_16_32", "medium_32_96", "large_ge_96"):
        scale_rows.append(
            {"scale_bin": name, "count": scale_counts[name], "percent": pct(scale_counts[name], valid_annotations)}
        )

    side_rows = []
    for name in ("short_lt_8", "short_8_16", "short_16_32", "short_32_64", "short_64_96", "short_ge_96"):
        side_rows.append({"short_side_bin": name, "count": side_counts[name], "percent": pct(side_counts[name], valid_annotations)})

    rel_rows = []
    for name in ("rel_lt_0.1pct", "rel_0.1_0.5pct", "rel_0.5_1pct", "rel_1_5pct", "rel_ge_5pct"):
        rel_rows.append({"relative_area_bin": name, "count": rel_counts[name], "percent": pct(rel_counts[name], valid_annotations)})

    return {
        "dataset": dataset_name,
        "split": split,
        "annotation_path": annotation_path.as_posix(),
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_valid_annotations": valid_annotations,
        "num_skipped_annotations": skipped_annotations,
        "num_categories": len(categories),
        "annotations_per_image": safe_stats([float(v) for v in ann_counts_per_image]),
        "bbox_width": safe_stats(box_widths),
        "bbox_height": safe_stats(box_heights),
        "bbox_area": safe_stats(box_areas),
        "sqrt_area": safe_stats([math.sqrt(v) for v in box_areas]),
        "relative_area": safe_stats(relative_areas),
        "aspect_ratio": safe_stats(aspect_ratios),
        "tiny_lt_16_count": tiny,
        "tiny_lt_16_percent": pct(tiny, valid_annotations),
        "coco_small_lt_32_count": coco_small,
        "coco_small_lt_32_percent": pct(coco_small, valid_annotations),
        "medium_32_96_count": medium,
        "medium_32_96_percent": pct(medium, valid_annotations),
        "large_ge_96_count": large,
        "large_ge_96_percent": pct(large, valid_annotations),
        "class_distribution": class_rows,
        "scale_distribution": scale_rows,
        "short_side_distribution": side_rows,
        "relative_area_distribution": rel_rows,
        "occlusion_distribution": dict(sorted(occlusion_counts.items())),
        "truncation_distribution": dict(sorted(truncation_counts.items())),
    }


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(8, min(18, 0.7 * len(labels) + 3))
    plt.figure(figsize=(width, 4.8))
    plt.bar(labels, values, color="#3b82f6")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_split_outputs(out_dir: Path, stats: dict) -> None:
    prefix = f"{stats['dataset']}_{stats['split']}"
    (out_dir / f"{prefix}_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        out_dir / f"{prefix}_class_distribution.csv",
        stats["class_distribution"],
        [
            "class",
            "count",
            "percent",
            "area_mean",
            "area_median",
            "tiny_lt_16",
            "small_16_32",
            "medium_32_96",
            "large_ge_96",
            "coco_small_percent_in_class",
        ],
    )
    write_csv(
        out_dir / f"{prefix}_scale_distribution.csv",
        stats["scale_distribution"],
        ["scale_bin", "count", "percent"],
    )
    write_csv(
        out_dir / f"{prefix}_short_side_distribution.csv",
        stats["short_side_distribution"],
        ["short_side_bin", "count", "percent"],
    )
    write_csv(
        out_dir / f"{prefix}_relative_area_distribution.csv",
        stats["relative_area_distribution"],
        ["relative_area_bin", "count", "percent"],
    )

    plot_bar(
        out_dir / f"{prefix}_class_distribution.png",
        [r["class"] for r in stats["class_distribution"]],
        [r["count"] for r in stats["class_distribution"]],
        f"{stats['dataset']} {stats['split']} class distribution",
        "annotations",
    )
    plot_bar(
        out_dir / f"{prefix}_scale_distribution.png",
        [r["scale_bin"] for r in stats["scale_distribution"]],
        [r["percent"] for r in stats["scale_distribution"]],
        f"{stats['dataset']} {stats['split']} scale distribution",
        "percent (%)",
    )
    plot_bar(
        out_dir / f"{prefix}_relative_area_distribution.png",
        [r["relative_area_bin"] for r in stats["relative_area_distribution"]],
        [r["percent"] for r in stats["relative_area_distribution"]],
        f"{stats['dataset']} {stats['split']} relative area distribution",
        "percent (%)",
    )


def markdown_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def write_markdown(out_dir: Path, all_stats: list[dict], skipped: list[str]) -> None:
    lines = [
        "# UAV Dataset Statistics",
        "",
        "Scale definitions: tiny `<16^2`, COCO-small `<32^2`, medium `[32^2, 96^2)`, large `>=96^2` by bounding-box area in pixels.",
        "",
    ]
    overview_rows = []
    for s in all_stats:
        overview_rows.append(
            [
                s["dataset"],
                s["split"],
                s["num_images"],
                s["num_valid_annotations"],
                f"{s['annotations_per_image']['mean']:.2f}",
                f"{s['tiny_lt_16_percent']:.2f}%",
                f"{s['coco_small_lt_32_percent']:.2f}%",
                f"{s['medium_32_96_percent']:.2f}%",
                f"{s['large_ge_96_percent']:.2f}%",
                f"{s['sqrt_area']['median']:.2f}",
            ]
        )
    lines.append(markdown_table(
        [
            "Dataset",
            "Split",
            "Images",
            "Objects",
            "Obj/img",
            "Tiny",
            "COCO-small",
            "Medium",
            "Large",
            "Median sqrt(area)",
        ],
        overview_rows,
    ))
    lines.append("")

    for s in all_stats:
        lines.append(f"## {s['dataset']} {s['split']}")
        top_rows = []
        for row in s["class_distribution"][:12]:
            top_rows.append(
                [
                    row["class"],
                    row["count"],
                    f"{row['percent']:.2f}%",
                    f"{row['coco_small_percent_in_class']:.2f}%",
                    f"{row['area_median']:.1f}",
                ]
            )
        lines.append(markdown_table(
            ["Class", "Count", "Percent", "COCO-small in class", "Median area"],
            top_rows,
        ))
        lines.append("")

    if skipped:
        lines.append("## Skipped datasets")
        lines.extend(f"- {item}" for item in skipped)
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH, for example visdrone=data/visdrone_det_coco")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute COCO-style UAV dataset statistics.")
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_arg,
        required=True,
        help="Dataset entry as NAME=PATH. PATH should contain annotations/instances_train.json.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for statistics files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats: list[dict] = []
    skipped: list[str] = []

    for dataset_name, dataset_root in args.dataset:
        dataset_root = dataset_root.resolve()
        if not dataset_root.exists():
            skipped.append(f"{dataset_name}: missing root {dataset_root}")
            continue
        for split in SPLITS:
            ann_path = dataset_root / "annotations" / f"instances_{split}.json"
            if not ann_path.exists():
                skipped.append(f"{dataset_name} {split}: missing {ann_path}")
                continue
            stats = compute_split_stats(dataset_name, split, ann_path)
            write_split_outputs(out_dir, stats)
            all_stats.append(stats)

    write_markdown(out_dir, all_stats, skipped)
    print(f"Wrote dataset statistics to {out_dir}")
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
