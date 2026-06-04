#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_subset(dataset_root: Path, ratio: float, seed: int) -> Path:
    annotations_dir = dataset_root / "annotations"
    source = annotations_dir / "instances_train.json"
    if not source.exists():
        raise FileNotFoundError(source)
    out_name = f"instances_train_{int(ratio * 100):02d}pct_seed{seed}.json"
    output = annotations_dir / out_name
    if output.exists():
        return output

    data = json.loads(source.read_text(encoding="utf-8"))
    images = data["images"]
    rng = random.Random(seed)
    sample_size = max(1, int(round(len(images) * ratio)))
    chosen_images = sorted(rng.sample(images, sample_size), key=lambda item: item["id"])
    chosen_ids = {item["id"] for item in chosen_images}
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in chosen_ids]
    subset = {
        "images": chosen_images,
        "annotations": annotations,
        "categories": data["categories"],
    }
    output.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    return output


def load_model(exp, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = exp.get_model().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def configure_exp(exp, dataset_root: Path, train_ann: str, val_ann: str, epochs: int, batch_size: int, lr: float, data_workers: int):
    exp.data_dir = str(dataset_root)
    exp.train_ann = train_ann
    exp.val_ann = val_ann
    exp.train_name = ""
    exp.val_name = ""
    exp.max_epoch = epochs
    exp.no_aug_epochs = min(getattr(exp, "no_aug_epochs", 2), epochs)
    exp.data_num_workers = data_workers
    exp.basic_lr_per_img = lr / float(batch_size)
    exp.print_interval = 20
    return exp


def train_one_method(
    method: dict,
    dataset_root: Path,
    train_ann: str,
    val_ann: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device_name: str,
    data_workers: int,
    amp: bool,
) -> dict:
    set_seed(seed)
    device = train_helpers.select_device(device_name)
    exp = get_exp(str(BYTETRACK_ROOT / method["exp_file"]), None)
    configure_exp(exp, dataset_root, train_ann, val_ann, epochs, batch_size, lr, data_workers)
    model = load_model(exp, Path(method["checkpoint"]), device)
    optimizer = exp.get_optimizer(batch_size)
    use_amp = bool(amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    train_loader = exp.get_data_loader(batch_size, False)
    val_loader = exp.get_eval_loader(batch_size, False)
    lr_scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * batch_size, len(train_loader))

    method_dir = output_dir / method["method"].replace("/", "_").replace("+", "plus").replace(" ", "_")
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
        f"{method['method']}: train_ann={train_ann}, val_ann={val_ann}, "
        f"epochs={epochs}, batch={batch_size}, lr={lr}"
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
            "val_ann": val_ann,
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
        "val_ann": val_ann,
        "status": "completed",
        "history": history,
        "best": best_metrics,
        "checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
    }
    (method_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_test(summary: dict, dataset_root: Path, output_dir: Path, batch_size: int, device_name: str) -> dict:
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
        data_workers=0,
    )
    model = load_model(exp, Path(summary["checkpoint"]), device)
    test_loader = exp.get_eval_loader(batch_size, False)
    method_dir = output_dir / summary["method"].replace("/", "_").replace("+", "plus").replace(" ", "_")
    result_json = method_dir / "test_detections.json"
    metrics = train_helpers.evaluate_coco(
        model,
        test_loader,
        num_classes=exp.num_classes,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        device=device,
        result_json_path=result_json,
    )
    row = {
        "method": summary["method"],
        "source_checkpoint": summary["source_checkpoint"],
        "finetuned_checkpoint": summary["checkpoint"],
        "val_best_epoch": summary["best"]["epoch"] if summary.get("best") else None,
        "val_ap50_95": summary["best"].get("ap50_95") if summary.get("best") else None,
        "val_ap50": summary["best"].get("ap50") if summary.get("best") else None,
        "val_ap_small": summary["best"].get("ap_small") if summary.get("best") else None,
        "val_recall50": summary["best"].get("proxy_recall50") if summary.get("best") else None,
        **{f"test_{key}": value for key, value in metrics.items()},
        "test_detections": str(result_json),
    }
    (method_dir / "test_metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [
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
    with (output_dir / "stage23_dronevehicle_10pct_finetune.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    lines = [
        "# Stage 23 DroneVehicle 10% Fine-tuning Case",
        "",
        "Setting: 10% DroneVehicle visible-light train subset for target-domain calibration; best checkpoint selected on DroneVehicle val; final report on full test.",
        "",
        "| Method | Val best epoch | Val AP50:95 | Val AP50 | Val AP_small | Test AP50:95 | Test AP50 | Test AP_small | Test Recall50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row.get('val_best_epoch', '')} | {row.get('val_ap50_95', 0):.6f} | "
            f"{row.get('val_ap50', 0):.6f} | {row.get('val_ap_small', 0):.6f} | "
            f"{row.get('test_ap50_95', 0):.6f} | {row.get('test_ap50', 0):.6f} | "
            f"{row.get('test_ap_small', 0):.6f} | {row.get('test_proxy_recall50', 0):.6f} |"
        )
    (output_dir / "stage23_dronevehicle_10pct_finetune.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "stage23_dronevehicle_10pct_finetune.json").write_text(
        json.dumps({"results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser("Stage 23: DroneVehicle 10% target-domain fine-tuning.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage22_dronevehicle_external_case" / "dronevehicle_visible_coco",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage23_dronevehicle_10pct_finetune",
    )
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_ann = make_subset(args.dataset_root, args.ratio, args.seed)
    rows = []
    for method in METHODS:
        summary = train_one_method(
            method,
            args.dataset_root,
            subset_ann.name,
            "instances_val.json",
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.lr,
            args.seed,
            args.device,
            args.data_workers,
            args.amp,
        )
        row = evaluate_test(summary, args.dataset_root, args.output_dir, args.batch_size, args.device)
        rows.append(row)
        write_outputs(args.output_dir, rows)
    write_outputs(args.output_dir, rows)


if __name__ == "__main__":
    main()
