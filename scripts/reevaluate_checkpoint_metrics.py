#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
TOOLS_ROOT = BYTETRACK_ROOT / "tools"
if str(BYTETRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(BYTETRACK_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from yolox.exp import get_exp


def load_train_helpers():
    path = TOOLS_ROOT / "train_uav_stage1.py"
    spec = importlib.util.spec_from_file_location("train_uav_stage1_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_model(exp_file: str, checkpoint: Path, dataset_root: str, device: torch.device):
    exp = get_exp(str(BYTETRACK_ROOT / exp_file), None)
    exp.data_dir = dataset_root
    model = exp.get_model().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return exp, model


def update_summary(summary_path: Path, metrics: dict, metrics_path: Path) -> None:
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best") or {}
    best.update(metrics)
    summary["best"] = best
    summary["reevaluation"] = {
        "metrics_path": str(metrics_path),
        "note": "COCO metrics recomputed from best checkpoint with size-specific AP fields.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Re-evaluate a YOLOX checkpoint and optionally enrich summary.json.")
    parser.add_argument("-f", "--exp-file", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    helpers = load_train_helpers()
    device = helpers.select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp, model = load_model(args.exp_file, args.checkpoint, args.dataset_root, device)
    val_loader = exp.get_eval_loader(args.batch_size, False)
    result_json = args.output_dir / "reeval_results.json"
    metrics = helpers.evaluate_coco(
        model,
        val_loader,
        num_classes=exp.num_classes,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        device=device,
        result_json_path=result_json,
    )
    payload = {
        "exp_file": args.exp_file,
        "checkpoint": str(args.checkpoint),
        "dataset_root": args.dataset_root,
        "metrics": metrics,
        "result_json": str(result_json),
    }
    metrics_path = args.output_dir / "reeval_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.summary_json:
        update_summary(args.summary_json, metrics, metrics_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
