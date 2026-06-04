#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.amp import GradScaler, autocast


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

# AU-AIR category order in annotations.json:
# ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
AU_AIR_TO_VISDRONE = {
    0: 1,   # Human -> pedestrian
    1: 4,   # Car -> car
    2: 6,   # Truck -> truck
    3: 5,   # Van -> van
    4: 10,  # Motorbike -> motor
    5: 3,   # Bicycle -> bicycle
    6: 9,   # Bus -> bus
    7: 6,   # Trailer -> truck-like vehicle
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


def safe_method_name(name: str) -> str:
    return name.replace("/", "_").replace("+", "plus").replace(" ", "_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sequence_id(image_name: str) -> str:
    match = re.match(r"^(frame_\d{14})", image_name)
    return match.group(1) if match else image_name.split("_x_")[0]


def split_sequences(sequences: list[str], val_sequence: str | None, test_sequence: str | None) -> dict[str, set[str]]:
    seqs = sorted(set(sequences))
    if len(seqs) < 3:
        raise ValueError(f"Need at least 3 sequences for train/val/test split, found {len(seqs)}.")
    test_seq = test_sequence or seqs[-1]
    val_seq = val_sequence or seqs[-2]
    if test_seq not in seqs:
        raise ValueError(f"test_sequence={test_seq} not in available sequences: {seqs}")
    if val_seq not in seqs:
        raise ValueError(f"val_sequence={val_seq} not in available sequences: {seqs}")
    if val_seq == test_seq:
        raise ValueError("Validation and test sequences must be different.")
    train = {seq for seq in seqs if seq not in {val_seq, test_seq}}
    return {"train": train, "val": {val_seq}, "test": {test_seq}}


def convert_auair(source_root: Path, output_root: Path, val_sequence: str | None, test_sequence: str | None) -> list[dict]:
    ann_path = source_root / "annotations.json"
    image_dir = source_root / "images"
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    entries = raw.get("annotations", [])
    source_categories = raw.get("categories", [])
    splits = split_sequences([sequence_id(item["image_name"]) for item in entries], val_sequence, test_sequence)

    output_root.mkdir(parents=True, exist_ok=True)
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ann_id = 1
    image_id = 1
    split_payloads = {
        split: {"images": [], "annotations": [], "categories": VISDRONE_CATEGORIES}
        for split in ("train", "val", "test")
    }
    split_source_counts = {split: Counter() for split in ("train", "val", "test")}
    split_mapped_counts = {split: Counter() for split in ("train", "val", "test")}
    skipped = Counter()

    for item in sorted(entries, key=lambda x: (sequence_id(x["image_name"]), x["image_name"])):
        seq = sequence_id(item["image_name"])
        split = next(name for name, seq_set in splits.items() if seq in seq_set)
        image_path = image_dir / item["image_name"]
        if not image_path.exists():
            skipped["missing_images"] += 1
            continue
        width = int(float(item.get("image_width:", item.get("image_width", 1920))))
        height = int(float(item.get("image_height", 1080)))
        split_payloads[split]["images"].append(
            {
                "id": image_id,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
                "auair_sequence": seq,
            }
        )

        for box in item.get("bbox", []):
            source_cls = int(box.get("class", -1))
            mapped_cls = AU_AIR_TO_VISDRONE.get(source_cls)
            source_name = source_categories[source_cls] if 0 <= source_cls < len(source_categories) else str(source_cls)
            split_source_counts[split][source_name] += 1
            if mapped_cls is None:
                skipped["unknown_classes"] += 1
                continue
            x = float(box.get("left", 0.0))
            y = float(box.get("top", 0.0))
            w = float(box.get("width", 0.0))
            h = float(box.get("height", 0.0))
            x1 = max(0.0, min(x, width - 1.0))
            y1 = max(0.0, min(y, height - 1.0))
            x2 = max(0.0, min(x + w, width))
            y2 = max(0.0, min(y + h, height))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1.0 or bh <= 1.0:
                skipped["invalid_boxes"] += 1
                continue
            split_payloads[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": mapped_cls,
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "source_category": source_name,
                }
            )
            split_mapped_counts[split][mapped_cls] += 1
            ann_id += 1
        image_id += 1

    for split, payload in split_payloads.items():
        (annotations_dir / f"instances_{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        rows.append(
            {
                "split": split,
                "sequences": sorted(splits[split]),
                "images": len(payload["images"]),
                "annotations": len(payload["annotations"]),
                "source_class_counts": dict(split_source_counts[split]),
                "mapped_category_counts": dict(split_mapped_counts[split]),
                "skipped": dict(skipped),
            }
        )

    (output_root / "conversion_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def configure_exp(
    exp,
    dataset_root: Path,
    train_ann: str,
    val_ann: str,
    epochs: int,
    batch_size: int,
    lr: float,
    data_workers: int,
    img_size: int,
):
    exp.data_dir = str(dataset_root)
    exp.train_ann = train_ann
    exp.val_ann = val_ann
    exp.train_name = ""
    exp.val_name = ""
    exp.input_size = (img_size, img_size)
    exp.test_size = (img_size, img_size)
    if getattr(exp, "random_size", None) is not None:
        exp.random_size = None
    exp.max_epoch = epochs
    exp.no_aug_epochs = min(getattr(exp, "no_aug_epochs", 2), epochs)
    exp.data_num_workers = data_workers
    exp.basic_lr_per_img = lr / float(batch_size)
    exp.print_interval = 50
    exp.max_labels = 100
    return exp


def load_model(exp, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = exp.get_model().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def set_force_cpu_assign(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "get_assignments"):
            setattr(module, "force_cpu_assign", enabled)


def make_subset(dataset_root: Path, ratio: float, seed: int) -> str:
    if ratio >= 0.999:
        return "instances_train.json"
    source = dataset_root / "annotations" / "instances_train.json"
    output = dataset_root / "annotations" / f"instances_train_{int(ratio * 100):02d}pct_seed{seed}.json"
    if output.exists():
        return output.name
    data = json.loads(source.read_text(encoding="utf-8"))
    images = data["images"]
    rng = random.Random(seed)
    sample_size = max(1, int(round(len(images) * ratio)))
    chosen = sorted(rng.sample(images, sample_size), key=lambda item: item["id"])
    chosen_ids = {item["id"] for item in chosen}
    subset = {
        "images": chosen,
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in chosen_ids],
        "categories": data["categories"],
    }
    output.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    return output.name


def evaluate_method(
    method: dict,
    dataset_root: Path,
    split: str,
    output_dir: Path,
    batch_size: int,
    device_name: str,
    data_workers: int,
    img_size: int,
) -> dict:
    device = train_helpers.select_device(device_name)
    exp = get_exp(str(BYTETRACK_ROOT / method["exp_file"]), None)
    configure_exp(
        exp,
        dataset_root,
        "instances_train.json",
        f"instances_{split}.json",
        epochs=1,
        batch_size=batch_size,
        lr=1e-5,
        data_workers=data_workers,
        img_size=img_size,
    )
    model = load_model(exp, Path(method["checkpoint"]), device)
    loader = exp.get_eval_loader(batch_size, False)
    method_dir = output_dir / "zero_shot" / safe_method_name(method["method"])
    method_dir.mkdir(parents=True, exist_ok=True)
    result_json = method_dir / f"{split}_detections.json"
    metrics = train_helpers.evaluate_coco(
        model,
        loader,
        num_classes=exp.num_classes,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        device=device,
        result_json_path=result_json,
    )
    row = {
        "setting": "zero_shot",
        "split": split,
        "method": method["method"],
        "checkpoint": str(method["checkpoint"]),
        "exp_file": method["exp_file"],
        "detections": str(result_json),
        **metrics,
    }
    (method_dir / f"{split}_metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def train_one_method(
    method: dict,
    dataset_root: Path,
    train_ann: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device_name: str,
    data_workers: int,
    amp: bool,
    img_size: int,
    force_cpu_assign: bool,
) -> dict:
    set_seed(seed)
    device = train_helpers.select_device(device_name)
    exp = get_exp(str(BYTETRACK_ROOT / method["exp_file"]), None)
    configure_exp(exp, dataset_root, train_ann, "instances_val.json", epochs, batch_size, lr, data_workers, img_size)
    model = load_model(exp, Path(method["checkpoint"]), device)
    set_force_cpu_assign(model, force_cpu_assign)
    optimizer = exp.get_optimizer(batch_size)
    use_amp = bool(amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    train_loader = exp.get_data_loader(batch_size, False)
    val_loader = exp.get_eval_loader(batch_size, False)
    lr_scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * batch_size, len(train_loader))

    method_dir = output_dir / "finetune" / safe_method_name(method["method"])
    method_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = method_dir / "best_ckpt.pth"
    last_ckpt = method_dir / "last_ckpt.pth"
    eval_results_path = method_dir / "val_detections.json"
    history = []
    best_metrics = None
    start_epoch = 0

    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        history = ckpt.get("history", [])
        best_metrics = ckpt.get("best_metrics")
        start_epoch = int(ckpt.get("epoch", 0))
        logger.info(f"{method['method']}: resumed at completed_epoch={start_epoch}")

    logger.info(
        f"{method['method']}: train_ann={train_ann}, epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, train_batches={len(train_loader)}, force_cpu_assign={force_cpu_assign}"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        for iter_i, batch in enumerate(train_loader):
            imgs, targets, _, _ = train_helpers.to_device(batch, device)
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(imgs, targets)
                loss = outputs["total_loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_now = lr_scheduler.update_lr(epoch * len(train_loader) + iter_i + 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now
            epoch_losses.append(float(loss.detach().cpu()))
            if (iter_i + 1) % exp.print_interval == 0 or (iter_i + 1) == len(train_loader):
                logger.info(
                    f"{method['method']} epoch={epoch + 1}/{epochs} iter={iter_i + 1}/{len(train_loader)} "
                    f"loss={epoch_losses[-1]:.4f} lr={lr_now:.6f}"
                )

        metrics = train_helpers.evaluate_coco(
            model,
            val_loader,
            num_classes=exp.num_classes,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            device=device,
            result_json_path=eval_results_path,
        )
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), **metrics}
        history.append(record)
        logger.info(f"{method['method']} validation: {record}")
        if best_metrics is None or train_helpers.metric_key(record) > train_helpers.metric_key(best_metrics):
            best_metrics = record
            torch.save({"model": model.state_dict(), "metrics": record}, best_ckpt)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_metrics": best_metrics,
            },
            last_ckpt,
        )
        summary = {
            "method": method["method"],
            "source_checkpoint": str(method["checkpoint"]),
            "exp_file": method["exp_file"],
            "dataset_root": str(dataset_root),
            "train_ann": train_ann,
            "val_ann": "instances_val.json",
            "status": "running",
            "history": history,
            "best": best_metrics,
            "checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
        }
        (method_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "method": method["method"],
        "source_checkpoint": str(method["checkpoint"]),
        "exp_file": method["exp_file"],
        "dataset_root": str(dataset_root),
        "train_ann": train_ann,
        "val_ann": "instances_val.json",
        "status": "completed",
        "history": history,
        "best": best_metrics,
        "checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_finetuned(
    summary: dict,
    dataset_root: Path,
    output_dir: Path,
    batch_size: int,
    device_name: str,
    data_workers: int,
    img_size: int,
) -> dict:
    device = train_helpers.select_device(device_name)
    exp = get_exp(str(BYTETRACK_ROOT / summary["exp_file"]), None)
    configure_exp(
        exp,
        dataset_root,
        summary["train_ann"],
        "instances_test.json",
        epochs=1,
        batch_size=batch_size,
        lr=1e-5,
        data_workers=data_workers,
        img_size=img_size,
    )
    model = load_model(exp, Path(summary["checkpoint"]), device)
    loader = exp.get_eval_loader(batch_size, False)
    method_dir = output_dir / "finetune" / safe_method_name(summary["method"])
    result_json = method_dir / "test_detections.json"
    metrics = train_helpers.evaluate_coco(
        model,
        loader,
        num_classes=exp.num_classes,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        device=device,
        result_json_path=result_json,
    )
    row = {
        "setting": "target_finetune",
        "method": summary["method"],
        "source_checkpoint": summary["source_checkpoint"],
        "finetuned_checkpoint": summary["checkpoint"],
        "val_best_epoch": summary["best"].get("epoch") if summary.get("best") else None,
        "val_ap50_95": summary["best"].get("ap50_95") if summary.get("best") else None,
        "val_ap50": summary["best"].get("ap50") if summary.get("best") else None,
        "val_ap_small": summary["best"].get("ap_small") if summary.get("best") else None,
        "val_recall50": summary["best"].get("proxy_recall50") if summary.get("best") else None,
        **{f"test_{key}": value for key, value in metrics.items()},
        "test_detections": str(result_json),
    }
    (method_dir / "test_metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def write_outputs(output_dir: Path, conversion_stats: list[dict], zero_rows: list[dict], finetune_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def num(value, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    (output_dir / "stage24_auair_engineering_case.json").write_text(
        json.dumps(
            {
                "conversion": conversion_stats,
                "zero_shot": zero_rows,
                "target_finetune": finetune_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    zero_fields = [
        "setting",
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
    with (output_dir / "stage24_auair_zero_shot.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=zero_fields)
        writer.writeheader()
        for row in zero_rows:
            writer.writerow({field: row.get(field) for field in zero_fields})

    ft_fields = [
        "setting",
        "method",
        "val_best_epoch",
        "val_ap50_95",
        "val_ap50",
        "val_ap_small",
        "val_recall50",
        "test_ap50_95",
        "test_ap50",
        "test_ap75",
        "test_ap_small",
        "test_ap_medium",
        "test_ap_large",
        "test_proxy_recall50",
        "test_proxy_mean_best_iou",
        "source_checkpoint",
        "finetuned_checkpoint",
        "test_detections",
    ]
    with (output_dir / "stage24_auair_target_finetune.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ft_fields)
        writer.writeheader()
        for row in finetune_rows:
            writer.writerow({field: row.get(field) for field in ft_fields})

    lines = [
        "# Stage 24 AU-AIR Engineering Case",
        "",
        "Setting: AU-AIR RGB frames are converted to COCO and category IDs are aligned to the existing VisDrone-trained 11-class detector. Splits are sequence-level to avoid adjacent-frame leakage.",
        "",
        "## Conversion",
        "",
        "| Split | Sequences | Images | Annotations |",
        "|---|---|---:|---:|",
    ]
    for row in conversion_stats:
        lines.append(
            f"| {row['split']} | {', '.join(row['sequences'])} | {row['images']} | {row['annotations']} |"
        )
    if zero_rows:
        lines.extend(
            [
                "",
                "## Zero-shot External Test",
                "",
                "| Method | AP50:95 | AP50 | AP_small | AP_medium | AP_large | Recall50 | Mean Best IoU |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in zero_rows:
            lines.append(
                f"| {row['method']} | {num(row.get('ap50_95')):.6f} | {num(row.get('ap50')):.6f} | "
                f"{num(row.get('ap_small')):.6f} | {num(row.get('ap_medium')):.6f} | "
                f"{num(row.get('ap_large')):.6f} | {num(row.get('proxy_recall50')):.6f} | "
                f"{num(row.get('proxy_mean_best_iou')):.6f} |"
            )
    if finetune_rows:
        lines.extend(
            [
                "",
                "## Target-domain Fine-tuning Test",
                "",
                "| Method | Val best epoch | Val AP50:95 | Test AP50:95 | Test AP50 | Test AP_small | Test Recall50 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in finetune_rows:
            lines.append(
                f"| {row['method']} | {row.get('val_best_epoch', '')} | {num(row.get('val_ap50_95')):.6f} | "
                f"{num(row.get('test_ap50_95')):.6f} | {num(row.get('test_ap50')):.6f} | "
                f"{num(row.get('test_ap_small')):.6f} | {num(row.get('test_proxy_recall50')):.6f} |"
            )
    (output_dir / "stage24_auair_engineering_case.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Stage 24: AU-AIR engineering case validation and fine-tuning.")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data" / "AU-AIR")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage24_auair_engineering_case" / "auair_coco",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage24_auair_engineering_case",
    )
    parser.add_argument("--val-sequence", default="frame_20190905142119")
    parser.add_argument("--test-sequence", default="frame_20190906150731")
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-cpu-assign", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_convert and (args.dataset_root / "conversion_summary.json").exists():
        conversion_stats = json.loads((args.dataset_root / "conversion_summary.json").read_text(encoding="utf-8"))
    else:
        conversion_stats = convert_auair(args.source_root, args.dataset_root, args.val_sequence, args.test_sequence)

    zero_rows: list[dict] = []
    zero_csv = args.output_dir / "stage24_auair_zero_shot.csv"
    if zero_csv.exists():
        with zero_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            zero_rows = list(csv.DictReader(handle))

    if not args.skip_zero_shot:
        completed_zero = {row.get("method") for row in zero_rows}
        for method in METHODS:
            if method["method"] in completed_zero:
                continue
            zero_rows.append(
                evaluate_method(
                    method,
                    args.dataset_root,
                    "test",
                    args.output_dir,
                    args.batch_size,
                    args.device,
                    args.data_workers,
                    args.img_size,
                )
            )
            write_outputs(args.output_dir, conversion_stats, zero_rows, [])

    finetune_rows: list[dict] = []
    ft_csv = args.output_dir / "stage24_auair_target_finetune.csv"
    if ft_csv.exists():
        with ft_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            finetune_rows = list(csv.DictReader(handle))

    if not args.skip_finetune:
        train_ann = make_subset(args.dataset_root, args.train_ratio, args.seed)
        completed_ft = {row.get("method") for row in finetune_rows}
        for method in METHODS:
            if method["method"] in completed_ft:
                continue
            summary = train_one_method(
                method,
                args.dataset_root,
                train_ann,
                args.output_dir,
                args.epochs,
                args.batch_size,
                args.lr,
                args.seed,
                args.device,
                args.data_workers,
                args.amp,
                args.img_size,
                args.force_cpu_assign,
            )
            finetune_rows.append(
                evaluate_finetuned(
                    summary,
                    args.dataset_root,
                    args.output_dir,
                    args.batch_size,
                    args.device,
                    args.data_workers,
                    args.img_size,
                )
            )
            write_outputs(args.output_dir, conversion_stats, zero_rows, finetune_rows)

    write_outputs(args.output_dir, conversion_stats, zero_rows, finetune_rows)


if __name__ == "__main__":
    main()
