#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
DEFAULT_PYTHON = "python"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage16_visdrone_100ep_baseline_p2_multiseed"

RUNS = [
    ("baseline", "Baseline", "exps/example/uav/yolox_nano_visdrone_det_640_baseline.py"),
    ("p2", "+P2", "exps/example/uav/yolox_nano_visdrone_det_640_p2.py"),
]

METRICS = [
    "train_loss",
    "ap50_95",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "proxy_recall50",
    "proxy_mean_best_iou",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_env() -> dict[str, str]:
    env = os.environ.copy()
    path_value = env.get("Path") or env.get("PATH") or ""
    for key in list(env):
        if key.lower() == "path":
            del env[key]
    env["Path"] = path_value
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BYTETRACK_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    return env


def write_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def completed(path: Path, target_epochs: int) -> bool:
    payload = load_summary(path)
    if payload.get("status") != "completed":
        return False
    history = payload.get("history") or []
    return len(history) >= target_epochs


def read_record(method: str, label: str, seed: int, path: Path, target_epochs: int) -> dict:
    record = {
        "method": method,
        "label": label,
        "seed": seed,
        "target_epochs": target_epochs,
        "summary_path": str(path),
    }
    payload = load_summary(path)
    if not payload:
        record["status"] = "missing"
        return record
    best = payload.get("best") or {}
    history = payload.get("history") or []
    record["status"] = payload.get("status", "unknown")
    record["actual_epochs"] = len(history)
    record["best_epoch"] = best.get("epoch")
    for key in METRICS:
        record[key] = best.get(key)
    return record


def mean_std(values: list) -> tuple[float | None, float | None]:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    if not nums:
        return None, None
    return statistics.fmean(nums), statistics.stdev(nums) if len(nums) > 1 else 0.0


def fmt(value) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "NA"


def collect_records(args) -> list[dict]:
    records = []
    for seed in args.seeds:
        for method, label, _ in RUNS:
            summary = args.output_dir / f"{method}_seed{seed}" / "summary.json"
            records.append(read_record(method, label, seed, summary, args.epochs))
    return records


def write_summary(args, records: list[dict]) -> None:
    aggregate = []
    for method, label, _ in RUNS:
        method_records = [
            row
            for row in records
            if row["method"] == method and row.get("status") == "completed" and row.get("actual_epochs", 0) >= args.epochs
        ]
        row = {
            "method": method,
            "label": label,
            "completed_seeds": len(method_records),
            "seeds": ",".join(str(item["seed"]) for item in method_records),
        }
        for metric in METRICS:
            mean, std = mean_std([item.get(metric) for item in method_records])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        aggregate.append(row)

    payload = {
        "stage": "VisDrone 100-epoch Baseline/+P2 three-seed stability",
        "dataset": "VisDrone",
        "setting": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": args.seeds,
            "dataset_root": str(args.dataset_root),
        },
        "records": records,
        "aggregate": aggregate,
        "updated_at": now(),
    }
    (args.output_dir / "visdrone_100ep_baseline_p2_multiseed.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    record_fields = ["method", "label", "seed", "status", "target_epochs", "actual_epochs", "best_epoch"] + METRICS + ["summary_path"]
    with (args.output_dir / "visdrone_100ep_baseline_p2_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in record_fields})

    aggregate_fields = ["method", "label", "completed_seeds", "seeds"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    with (args.output_dir / "visdrone_100ep_baseline_p2_mean_std.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        for row in aggregate:
            writer.writerow({field: row.get(field) for field in aggregate_fields})

    lines = [
        "# VisDrone 100-epoch Baseline/+P2 Multiseed",
        "",
        f"Setting: VisDrone, 640x640, {args.epochs} epochs, seeds {','.join(str(seed) for seed in args.seeds)}.",
        "",
        "| Method | Completed seeds | Seeds | AP50:95 | AP50 | AP_small | Recall50 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {label} | {count} | {seeds} | {ap5095} +/- {ap5095s} | {ap50} +/- {ap50s} | {aps} +/- {apss} | {recall} +/- {recalls} |".format(
                label=row["label"],
                count=row["completed_seeds"],
                seeds=row["seeds"] or "NA",
                ap5095=fmt(row.get("ap50_95_mean")),
                ap5095s=fmt(row.get("ap50_95_std")),
                ap50=fmt(row.get("ap50_mean")),
                ap50s=fmt(row.get("ap50_std")),
                aps=fmt(row.get("ap_small_mean")),
                apss=fmt(row.get("ap_small_std")),
                recall=fmt(row.get("proxy_recall50_mean")),
                recalls=fmt(row.get("proxy_recall50_std")),
            )
        )
    lines.extend(
        [
            "",
            "## Run Status",
            "",
            "| Method | Seed | Status | Actual epochs | Best epoch |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in records:
        lines.append(
            f"| {row['label']} | {row['seed']} | {row.get('status', 'missing')} | {row.get('actual_epochs', 'NA')} | {row.get('best_epoch', 'NA')} |"
        )
    (args.output_dir / "visdrone_100ep_baseline_p2_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(args, method: str, label: str, exp_file: str, seed: int, runner_log: Path) -> None:
    run_dir = args.output_dir / f"{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    status_path = run_dir / "train_status.log"

    if completed(summary_path, args.epochs):
        write_log(runner_log, f"[{now()}] SKIP completed {run_dir.name}")
        return

    write_log(runner_log, f"[{now()}] START {run_dir.name} ({label}, seed={seed})")
    write_log(status_path, f"[{now()}] START {run_dir.name}")

    command = [
        args.python,
        "tools/train_uav_stage1.py",
        "-f",
        exp_file,
        "--device",
        "cuda",
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
        "--resume",
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(run_dir),
    ]
    if args.amp:
        command.append("--amp")

    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n[{now()}] COMMAND {' '.join(command)}\n")
        stdout.flush()
        proc = subprocess.run(
            command,
            cwd=str(BYTETRACK_ROOT),
            env=normalize_env(),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

    if proc.returncode != 0:
        write_log(runner_log, f"[{now()}] FAIL {run_dir.name} exit={proc.returncode}")
        write_log(status_path, f"[{now()}] FAIL {run_dir.name} exit={proc.returncode}")
        write_summary(args, collect_records(args))
        raise SystemExit(proc.returncode)

    write_log(runner_log, f"[{now()}] DONE {run_dir.name}")
    write_log(status_path, f"[{now()}] DONE {run_dir.name}")
    write_summary(args, collect_records(args))


def main() -> None:
    parser = argparse.ArgumentParser("Run VisDrone Baseline/+P2 100-epoch multiseed experiments.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_log = args.output_dir / "runner.log"
    write_log(
        runner_log,
        f"[{now()}] START pipeline epochs={args.epochs} batch_size={args.batch_size} seeds={','.join(str(seed) for seed in args.seeds)}",
    )
    write_summary(args, collect_records(args))

    for seed in args.seeds:
        for method, label, exp_file in RUNS:
            run_one(args, method, label, exp_file, seed, runner_log)

    write_summary(args, collect_records(args))
    write_log(runner_log, f"[{now()}] FINISHED pipeline")


if __name__ == "__main__":
    main()
