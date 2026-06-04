#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
EXP_GEN_DIR = BYTETRACK_ROOT / "exps" / "example" / "uav" / "generated_upgrade"
DEFAULT_PYTHON = "python"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage26_best_candidate_fulltrain_seed42"


CANDIDATES = [
    {
        "method": "random_best",
        "label": "Random-best",
        "candidate_id": "p2_ca1_csa0_ctx0_fus0_slw100_cr30",
        "coord_attention": True,
        "channel_spatial_attention": False,
        "tiny_context": False,
        "scale_aware_fusion": False,
        "small_obj_loss_weight": 1.0,
        "center_radius": 3.0,
    },
    {
        "method": "ga_best",
        "label": "GA-best",
        "candidate_id": "p2_ca0_csa0_ctx0_fus0_slw125_cr30",
        "coord_attention": False,
        "channel_spatial_attention": False,
        "tiny_context": False,
        "scale_aware_fusion": False,
        "small_obj_loss_weight": 1.25,
        "center_radius": 3.0,
    },
    {
        "method": "sa_qubo_best",
        "label": "SA/QUBO-best",
        "candidate_id": "p2_ca0_csa0_ctx1_fus0_slw125_cr35",
        "coord_attention": False,
        "channel_spatial_attention": False,
        "tiny_context": True,
        "scale_aware_fusion": False,
        "small_obj_loss_weight": 1.25,
        "center_radius": 3.5,
    },
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


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def exp_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def write_exp(candidate: dict, epochs: int) -> str:
    EXP_GEN_DIR.mkdir(parents=True, exist_ok=True)
    exp_name = f"stage26_{candidate['method']}_seed42_fulltrain"
    path = EXP_GEN_DIR / f"{exp_name}.py"
    lines = [
        "#!/usr/bin/env python3",
        "import os",
        "import sys",
        "",
        "sys.path.append(os.path.dirname(os.path.dirname(__file__)))",
        "from yolox_nano_visdrone_det_640_base import Exp as BaseExp",
        "",
        "",
        "class Exp(BaseExp):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.use_p2 = True",
        f"        self.coord_attention = {exp_value(candidate['coord_attention'])}",
        f"        self.channel_spatial_attention = {exp_value(candidate['channel_spatial_attention'])}",
        f"        self.tiny_context = {exp_value(candidate['tiny_context'])}",
        f"        self.scale_aware_fusion = {exp_value(candidate['scale_aware_fusion'])}",
        f"        self.small_obj_loss_weight = {exp_value(candidate['small_obj_loss_weight'])}",
        f"        self.center_radius = {exp_value(candidate['center_radius'])}",
        '        self.train_ann = "instances_train.json"',
        '        self.val_ann = "instances_val.json"',
        f"        self.max_epoch = {epochs}",
        "        self.no_aug_epochs = min(self.no_aug_epochs, self.max_epoch)",
        f'        self.exp_name = "{exp_name}"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"exps/example/uav/generated_upgrade/{path.name}"


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def history_len(summary: dict) -> int:
    return len(summary.get("history") or [])


def completed(path: Path, epochs: int) -> bool:
    payload = load_summary(path)
    return payload.get("status") == "completed" and history_len(payload) >= epochs


def run_one(args: argparse.Namespace, candidate: dict, exp_file: str, runner_log: Path) -> None:
    run_dir = args.output_dir / f"{candidate['method']}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if completed(summary_path, args.epochs):
        log(runner_log, f"SKIP completed {run_dir.name}")
        return

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
        str(args.seed),
        "--resume",
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(run_dir),
        "--data-num-workers",
        str(args.data_num_workers),
    ]
    if args.amp:
        command.append("--amp")

    log(runner_log, f"START {run_dir.name} ({candidate['label']}, {candidate['candidate_id']})")
    (run_dir / "train_status.log").write_text(f"[{now()}] START {run_dir.name}\n", encoding="utf-8")
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (run_dir / "stderr.log").open(
        "a", encoding="utf-8"
    ) as stderr:
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
        log(runner_log, f"FAIL {run_dir.name} exit={proc.returncode}")
        raise SystemExit(proc.returncode)
    log(runner_log, f"DONE {run_dir.name}")


def collect_records(args: argparse.Namespace, exp_files: dict[str, str]) -> list[dict]:
    records = []
    for candidate in CANDIDATES:
        run_dir = args.output_dir / f"{candidate['method']}_seed{args.seed}"
        summary_path = run_dir / "summary.json"
        payload = load_summary(summary_path)
        best = payload.get("best") or {}
        row = {
            "method": candidate["method"],
            "label": candidate["label"],
            "candidate_id": candidate["candidate_id"],
            "seed": args.seed,
            "status": payload.get("status", "missing"),
            "actual_epochs": history_len(payload),
            "best_epoch": best.get("epoch"),
            "summary_path": str(summary_path),
            "checkpoint": payload.get("checkpoint"),
            "exp_file": exp_files.get(candidate["method"]),
        }
        for metric in METRICS:
            row[metric] = best.get(metric)
        for key in [
            "coord_attention",
            "channel_spatial_attention",
            "tiny_context",
            "scale_aware_fusion",
            "small_obj_loss_weight",
            "center_radius",
        ]:
            row[key] = candidate[key]
        records.append(row)
    return records


def write_outputs(args: argparse.Namespace, records: list[dict]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "stage26_best_candidate_fulltrain_seed42",
        "purpose": "Full 100-epoch validation of Random-best, GA-best, and SA/QUBO-best proxy-search candidates.",
        "setting": {
            "dataset": "VisDrone",
            "epochs": args.epochs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "amp": args.amp,
            "dataset_root": str(args.dataset_root),
        },
        "records": records,
        "updated_at": now(),
    }
    (args.output_dir / "stage26_best_candidate_fulltrain_seed42.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "method",
        "label",
        "candidate_id",
        "seed",
        "status",
        "actual_epochs",
        "best_epoch",
        *METRICS,
        "coord_attention",
        "channel_spatial_attention",
        "tiny_context",
        "scale_aware_fusion",
        "small_obj_loss_weight",
        "center_radius",
        "exp_file",
        "summary_path",
        "checkpoint",
    ]
    with (args.output_dir / "stage26_best_candidate_fulltrain_seed42.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})

    lines = [
        "# Stage 26 Best-Candidate Full Training",
        "",
        "This stage validates proxy-search best candidates under the full VisDrone 100-epoch training protocol with seed42.",
        "",
        "| Method | Candidate | Status | Epochs | Best epoch | AP50:95 | AP50 | AP_small | Recall50 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {label} | `{candidate}` | {status} | {epochs} | {best_epoch} | {ap5095} | {ap50} | {aps} | {recall} |".format(
                label=row["label"],
                candidate=row["candidate_id"],
                status=row["status"],
                epochs=row.get("actual_epochs", 0),
                best_epoch=row.get("best_epoch", "NA"),
                ap5095=f"{row.get('ap50_95'):.6f}" if isinstance(row.get("ap50_95"), (int, float)) else "NA",
                ap50=f"{row.get('ap50'):.6f}" if isinstance(row.get("ap50"), (int, float)) else "NA",
                aps=f"{row.get('ap_small'):.6f}" if isinstance(row.get("ap_small"), (int, float)) else "NA",
                recall=f"{row.get('proxy_recall50'):.6f}" if isinstance(row.get("proxy_recall50"), (int, float)) else "NA",
            )
        )
    (args.output_dir / "stage26_best_candidate_fulltrain_seed42.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser("Run full 100-epoch seed42 training for proxy-search best candidates.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_log = args.output_dir / "stage26_runner.log"
    log(runner_log, f"Stage 26 started epochs={args.epochs} seed={args.seed} batch_size={args.batch_size}")

    exp_files = {}
    for candidate in CANDIDATES:
        exp_file = write_exp(candidate, args.epochs)
        exp_files[candidate["method"]] = exp_file
        run_one(args, candidate, exp_file, runner_log)
        write_outputs(args, collect_records(args, exp_files))

    write_outputs(args, collect_records(args, exp_files))
    log(runner_log, "Stage 26 finished")


if __name__ == "__main__":
    main()
