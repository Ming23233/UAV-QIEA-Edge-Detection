#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
EXP_GEN_DIR = BYTETRACK_ROOT / "exps" / "example" / "uav" / "generated_upgrade"
DEFAULT_PYTHON = "python"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_WINNER = PROJECT_ROOT / "outputs" / "stage5_proxy_search" / "winner_config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage6_qiea_final"

SEARCH_KEYS = [
    "coord_attention",
    "channel_spatial_attention",
    "tiny_context",
    "scale_aware_fusion",
    "small_obj_loss_weight",
    "center_radius",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_env() -> dict:
    env = os.environ.copy()
    path_value = env.get("Path") or env.get("PATH") or ""
    for key in list(env):
        if key.lower() == "path":
            del env[key]
    env["Path"] = path_value
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BYTETRACK_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    return env


def exp_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int) and value in (0, 1):
        return "True" if value else "False"
    return repr(value)


def load_winner(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "config" not in payload:
        raise ValueError(f"winner config missing `config`: {path}")
    return payload


def write_exp(winner: dict, dataset: str, epochs: int) -> str:
    EXP_GEN_DIR.mkdir(parents=True, exist_ok=True)
    if dataset == "visdrone":
        base = "yolox_nano_visdrone_det_640_base"
        exp_name = "stage6_qiea_final_visdrone_640"
    elif dataset == "uavdt":
        base = "yolox_nano_uavdt_base"
        exp_name = "stage9_qiea_final_uavdt_640"
    else:
        raise ValueError(dataset)

    path = EXP_GEN_DIR / f"{exp_name}.py"
    config = winner["config"]
    lines = [
        "#!/usr/bin/env python3",
        "import os",
        "import sys",
        "",
        "sys.path.append(os.path.dirname(os.path.dirname(__file__)))",
        f"from {base} import Exp as BaseExp",
        "",
        "",
        "class Exp(BaseExp):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.use_p2 = True",
    ]
    for key in SEARCH_KEYS:
        lines.append(f"        self.{key} = {exp_value(config[key])}")
    lines.extend(
        [
            f"        self.max_epoch = {epochs}",
            "        self.no_aug_epochs = min(self.no_aug_epochs, self.max_epoch)",
            f'        self.exp_name = "{exp_name}"',
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"exps/example/uav/generated_upgrade/{path.name}"


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed"


def run_training(args, exp_file: str, run_dir: Path, runner_log: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if completed(summary_path):
        with runner_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] SKIP completed {run_dir.name}\n")
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
        args.dataset_root,
        "--output-dir",
        str(run_dir),
    ]
    if args.amp:
        command.append("--amp")

    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] START {run_dir.name}\n")
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
        with runner_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] FAIL {run_dir.name} exit={proc.returncode}\n")
        raise SystemExit(proc.returncode)

    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] DONE {run_dir.name}\n")


def write_summary(output_dir: Path, winner: dict, run_dir: Path, exp_file: str) -> None:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best") or {}
    row = {
        "method": "qiea_final",
        "candidate_id": winner.get("candidate_id"),
        "exp_file": exp_file,
        "summary_path": str(summary_path),
        "checkpoint": summary.get("checkpoint"),
        "status": summary.get("status"),
        "best_epoch": best.get("epoch"),
        "train_loss": best.get("train_loss"),
        "ap50_95": best.get("ap50_95"),
        "ap50": best.get("ap50"),
        "ap75": best.get("ap75"),
        "ap_small": best.get("ap_small"),
        "ap_medium": best.get("ap_medium"),
        "ap_large": best.get("ap_large"),
        "proxy_recall50": best.get("proxy_recall50"),
        "proxy_mean_best_iou": best.get("proxy_mean_best_iou"),
        "config": winner.get("config"),
    }
    payload = {
        "stage": "stage6_qiea_final",
        "winner_from_stage5": winner,
        "record": row,
    }
    (output_dir / "stage6_qiea_final.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [key for key in row if key != "config"] + ["use_p2"] + SEARCH_KEYS
    with (output_dir / "stage6_qiea_final.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**{key: row.get(key) for key in fields}, **row["config"]})

    lines = [
        "# Stage 6 QIEA-Final Formal Training",
        "",
        f"Selected candidate: `{winner.get('candidate_id')}` from Stage 5 QIEA proxy search.",
        "",
        "| Method | Status | Best epoch | AP50:95 | AP50 | AP_small | Proxy Recall50 |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| QIEA-Final | {status} | {epoch} | {ap5095:.6f} | {ap50:.6f} | {aps} | {recall:.6f} |".format(
            status=row["status"],
            epoch=row["best_epoch"],
            ap5095=float(row.get("ap50_95") or 0.0),
            ap50=float(row.get("ap50") or 0.0),
            aps=f"{row.get('ap_small'):.6f}" if isinstance(row.get("ap_small"), (float, int)) else "NA",
            recall=float(row.get("proxy_recall50") or 0.0),
        ),
        "",
        "Config: " + ", ".join(f"{key}={row['config'][key]}" for key in SEARCH_KEYS),
    ]
    (output_dir / "stage6_qiea_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Run Stage 6 formal training for Stage 5 QIEA winner.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["visdrone", "uavdt"], default="visdrone")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    winner = load_winner(args.winner)
    exp_file = write_exp(winner, args.dataset, args.epochs)
    run_dir = args.output_dir / f"qiea_final_seed{args.seed}"
    runner_log = args.output_dir / "stage6_runner.log"
    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] Stage 6 started\n")

    run_training(args, exp_file, run_dir, runner_log)
    write_summary(args.output_dir, winner, run_dir, exp_file)

    proxy_src = PROJECT_ROOT / "outputs" / "stage5_proxy_search" / "winner_config.json"
    if proxy_src.exists():
        shutil.copy2(proxy_src, args.output_dir / "winner_config.json")

    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] Stage 6 finished\n")


if __name__ == "__main__":
    main()
