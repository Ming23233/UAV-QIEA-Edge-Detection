#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage4_proxy_split"


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    image_ids = {int(img["id"]) for img in data["images"]}
    ann_image_ids = {int(ann["image_id"]) for ann in data["annotations"]}
    classes = Counter(int(ann["category_id"]) for ann in data["annotations"])
    small = 0
    tiny = 0
    for ann in data["annotations"]:
        area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
        if area < 32 * 32:
            small += 1
        if area < 16 * 16:
            tiny += 1
    return {
        "path": str(path),
        "images": len(image_ids),
        "annotations": len(data["annotations"]),
        "images_with_annotations": len(ann_image_ids),
        "empty_images": len(image_ids - ann_image_ids),
        "small_objects": small,
        "tiny_objects": tiny,
        "class_counts": dict(sorted(classes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Validate Stage 4 proxy split.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = args.dataset_root / "annotations"
    source = summarize(ann_dir / "instances_train.json")
    train = summarize(ann_dir / "search_train.json")
    val = summarize(ann_dir / "search_val.json")
    ratio = train["images"] / max(train["images"] + val["images"], 1)
    status = "completed" if 0.88 <= ratio <= 0.92 and train["images"] + val["images"] == source["images"] else "needs_regeneration"
    payload = {
        "stage": "stage4_proxy_split",
        "status": status,
        "dataset_root": str(args.dataset_root),
        "train_ratio": ratio,
        "source": source,
        "search_train": train,
        "search_val": val,
    }
    (args.output_dir / "stage4_proxy_split_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Stage 4 Proxy Split",
        "",
        f"Status: `{status}`.",
        f"Train ratio: `{ratio:.4f}`.",
        "",
        "| Split | Images | Annotations | Small objects | Tiny objects |",
        "|---|---:|---:|---:|---:|",
        f"| Source train | {source['images']} | {source['annotations']} | {source['small_objects']} | {source['tiny_objects']} |",
        f"| Search train | {train['images']} | {train['annotations']} | {train['small_objects']} | {train['tiny_objects']} |",
        f"| Search val | {val['images']} | {val['annotations']} | {val['small_objects']} | {val['tiny_objects']} |",
    ]
    (args.output_dir / "stage4_proxy_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "stage4_proxy_split_summary.md")
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
