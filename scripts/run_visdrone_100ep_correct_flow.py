#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
DEFAULT_PYTHON = "python"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage17_visdrone_640_100ep_correct_multiseed"
DEFAULT_STAGE2 = PROJECT_ROOT / "outputs" / "stage2_visdrone_640_baselines"

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


def history_len(summary: dict) -> int:
    return len(summary.get("history") or [])


def completed(path: Path, target_epochs: int) -> bool:
    payload = load_summary(path)
    return payload.get("status") == "completed" and history_len(payload) >= target_epochs


def source_summary(args, method: str, seed: int) -> Path | None:
    if method == "baseline" and seed == 42:
        path = args.stage2_dir / "baseline_640_seed42" / "summary.json"
        if completed(path, args.epochs):
            return path
    return None


def run_dir(args, method: str, seed: int) -> Path:
    return args.output_dir / f"{method}_seed{seed}"


def active_summary_path(args, method: str, seed: int) -> Path:
    source = source_summary(args, method, seed)
    if source is not None:
        return source
    return run_dir(args, method, seed) / "summary.json"


def bootstrap_p2_seed42(args, runner_log: Path) -> None:
    destination = run_dir(args, "p2", 42)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "last_ckpt.pth").exists():
        return

    source = args.stage2_dir / "p2_640_seed42"
    source_last = source / "last_ckpt.pth"
    source_summary = source / "summary.json"
    if not source_last.exists():
        raise FileNotFoundError(f"Cannot resume +P2 seed42; missing source checkpoint: {source_last}")
    summary = load_summary(source_summary)
    if history_len(summary) < 1:
        raise RuntimeError(f"Cannot resume +P2 seed42; invalid source summary: {source_summary}")

    existing_files = [item for item in destination.iterdir() if item.name not in {"bootstrap_note.txt"}]
    if existing_files:
        raise RuntimeError(f"Refusing to overwrite non-empty bootstrap directory: {destination}")

    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    write_log(destination / "bootstrap_note.txt", f"[{now()}] Copied +P2 seed42 50-epoch state from {source}")
    write_log(runner_log, f"[{now()}] BOOTSTRAP p2_seed42 from {source}")


def read_record(args, method: str, label: str, seed: int) -> dict:
    path = active_summary_path(args, method, seed)
    record = {
        "method": method,
        "label": label,
        "seed": seed,
        "target_epochs": args.epochs,
        "summary_path": str(path),
        "source": "reused_stage2" if source_summary(args, method, seed) is not None else "stage17",
    }
    payload = load_summary(path)
    if not payload:
        record["status"] = "missing"
        return record
    best = payload.get("best") or {}
    record["status"] = payload.get("status", "unknown")
    record["actual_epochs"] = history_len(payload)
    record["best_epoch"] = best.get("epoch")
    for key in METRICS:
        record[key] = best.get(key)
    return record


def collect_records(args) -> list[dict]:
    records = []
    for seed in args.seeds:
        for method, label, _ in RUNS:
            records.append(read_record(args, method, label, seed))
    return records


def mean_std(values: list) -> tuple[float | None, float | None]:
    nums = [float(value) for value in values if isinstance(value, (int, float))]
    if not nums:
        return None, None
    return statistics.fmean(nums), statistics.stdev(nums) if len(nums) > 1 else 0.0


def fmt(value) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "NA"


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
        "stage": "stage17_visdrone_640_100ep_correct_multiseed",
        "dataset": "VisDrone",
        "policy": [
            "Reuse completed Baseline seed42 100-epoch run from stage2.",
            "Resume +P2 seed42 from copied 50-epoch state to 100 epochs.",
            "Train Baseline/+P2 seed43 and seed44 with the same 640 configuration to 100 epochs.",
        ],
        "setting": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": args.seeds,
            "dataset_root": str(args.dataset_root),
            "stage2_dir": str(args.stage2_dir),
        },
        "records": records,
        "aggregate": aggregate,
        "updated_at": now(),
    }
    (args.output_dir / "stage17_correct_flow.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    record_fields = [
        "method",
        "label",
        "seed",
        "source",
        "status",
        "target_epochs",
        "actual_epochs",
        "best_epoch",
    ] + METRICS + ["summary_path"]
    with (args.output_dir / "stage17_correct_flow_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in record_fields})

    aggregate_fields = ["method", "label", "completed_seeds", "seeds"]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
    with (args.output_dir / "stage17_correct_flow_mean_std.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        for row in aggregate:
            writer.writerow({field: row.get(field) for field in aggregate_fields})

    lines = [
        "# Stage 17 Correct Flow",
        "",
        f"Setting: VisDrone, YOLOX-nano 640, {args.epochs} epochs, seeds {','.join(str(seed) for seed in args.seeds)}.",
        "",
        "Policy: reuse completed Baseline seed42, resume +P2 seed42 from its copied 50-epoch training state, and train seed43/44 under the same 640 setting.",
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
            "| Method | Seed | Source | Status | Actual epochs | Best epoch |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for row in records:
        lines.append(
            f"| {row['label']} | {row['seed']} | {row.get('source', 'NA')} | {row.get('status', 'missing')} | {row.get('actual_epochs', 'NA')} | {row.get('best_epoch', 'NA')} |"
        )
    (args.output_dir / "stage17_correct_flow.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(args, method: str, label: str, exp_file: str, seed: int, runner_log: Path) -> None:
    reused = source_summary(args, method, seed)
    if reused is not None:
        write_log(runner_log, f"[{now()}] REUSE {method}_seed{seed} from {reused}")
        return

    if method == "p2" and seed == 42:
        bootstrap_p2_seed42(args, runner_log)

    destination = run_dir(args, method, seed)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    stdout_path = destination / "stdout.log"
    stderr_path = destination / "stderr.log"
    status_path = destination / "train_status.log"

    if completed(summary_path, args.epochs):
        write_log(runner_log, f"[{now()}] SKIP completed {destination.name}")
        return

    write_log(runner_log, f"[{now()}] START {destination.name} ({label}, seed={seed})")
    write_log(status_path, f"[{now()}] START {destination.name}")

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
        str(destination),
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
        write_log(runner_log, f"[{now()}] FAIL {destination.name} exit={proc.returncode}")
        write_log(status_path, f"[{now()}] FAIL {destination.name} exit={proc.returncode}")
        write_summary(args, collect_records(args))
        raise SystemExit(proc.returncode)

    write_log(runner_log, f"[{now()}] DONE {destination.name}")
    write_log(status_path, f"[{now()}] DONE {destination.name}")
    write_summary(args, collect_records(args))


def main() -> None:
    parser = argparse.ArgumentParser("Run the correct VisDrone 640 100-epoch Baseline/+P2 flow.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
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
        f"[{now()}] START correct flow epochs={args.epochs} batch_size={args.batch_size} seeds={','.join(str(seed) for seed in args.seeds)}",
    )
    write_summary(args, collect_records(args))

    for seed in args.seeds:
        for method, label, exp_file in RUNS:
            run_one(args, method, label, exp_file, seed, runner_log)
            write_summary(args, collect_records(args))

    write_summary(args, collect_records(args))
    write_log(runner_log, f"[{now()}] FINISHED correct flow")


if __name__ == "__main__":
    main()
