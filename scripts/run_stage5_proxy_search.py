#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
EXP_GEN_DIR = BYTETRACK_ROOT / "exps" / "example" / "uav" / "generated_upgrade"
DEFAULT_PYTHON = "python"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_SEARCH_SPACE = PROJECT_ROOT / "outputs" / "stage3_search_space" / "stage3_search_space_simplified_640.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage5_proxy_search"

SEARCH_KEYS = [
    "coord_attention",
    "channel_spatial_attention",
    "tiny_context",
    "scale_aware_fusion",
    "small_obj_loss_weight",
    "center_radius",
]

BINARY_KEYS = [
    "coord_attention",
    "channel_spatial_attention",
    "tiny_context",
    "scale_aware_fusion",
]

CATEGORICAL_VALUES = {
    "small_obj_loss_weight": [1.0, 1.25, 1.5],
    "center_radius": [2.5, 3.0, 3.5],
}


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


def config_key(config: dict) -> tuple:
    return tuple(config[k] for k in SEARCH_KEYS)


def candidate_id(config: dict) -> str:
    return (
        "p2"
        f"_ca{int(config['coord_attention'])}"
        f"_csa{int(config['channel_spatial_attention'])}"
        f"_ctx{int(config['tiny_context'])}"
        f"_fus{int(config['scale_aware_fusion'])}"
        f"_slw{int(round(float(config['small_obj_loss_weight']) * 100))}"
        f"_cr{int(round(float(config['center_radius']) * 10))}"
    )


def read_search_space(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    for row in payload["candidates"]:
        config = {key: row["config"][key] for key in SEARCH_KEYS}
        config["use_p2"] = 1
        candidates.append({"candidate_id": candidate_id(config), "config": config})
    return candidates


def exp_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int) and value in (0, 1):
        return "True" if value else "False"
    return repr(value)


def write_exp(candidate: dict, proxy_epochs: int, search_split: bool) -> str:
    EXP_GEN_DIR.mkdir(parents=True, exist_ok=True)
    exp_name = f"stage5_{candidate['candidate_id']}"
    path = EXP_GEN_DIR / f"{exp_name}.py"
    config = candidate["config"]
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
    ]
    for key in SEARCH_KEYS:
        lines.append(f"        self.{key} = {exp_value(config[key])}")
    if search_split:
        lines.extend(
            [
                '        self.train_ann = "search_train.json"',
                '        self.val_ann = "search_val.json"',
            ]
        )
    lines.extend(
        [
            f"        self.max_epoch = {proxy_epochs}",
            "        self.no_aug_epochs = min(self.no_aug_epochs, self.max_epoch)",
            f'        self.exp_name = "{exp_name}"',
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"exps/example/uav/generated_upgrade/{path.name}"


def summary_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed"


def load_summary(run_dir: Path, method: str, candidate: dict, generation: int | None = None) -> dict | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    best = summary.get("best") or {}
    record = {
        "method": method,
        "generation": generation,
        "candidate_id": candidate["candidate_id"],
        "config": candidate["config"],
        "status": summary.get("status", "unknown"),
        "summary_path": str(path),
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
    }
    record["fitness"] = compute_fitness(record)
    return record


def number(value, default=0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def complexity_penalty(config: dict) -> float:
    modules = sum(int(config[key]) for key in BINARY_KEYS)
    loss_penalty = max(0.0, float(config["small_obj_loss_weight"]) - 1.0) * 0.002
    radius_penalty = abs(float(config["center_radius"]) - 2.5) * 0.001
    return modules * 0.0015 + loss_penalty + radius_penalty


def compute_fitness(record: dict) -> float:
    config = record["config"]
    ap_small = record.get("ap_small")
    if isinstance(ap_small, (int, float)) and ap_small >= 0:
        base = float(ap_small)
    else:
        base = number(record.get("ap50_95"))
    return (
        base
        + 0.30 * number(record.get("ap50"))
        + 0.20 * number(record.get("proxy_recall50"))
        - complexity_penalty(config)
    )


def run_one(args, method: str, candidate: dict, generation: int | None, runner_log: Path) -> dict:
    run_dir = args.output_dir / method / candidate["candidate_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    status_path = run_dir / "train_status.log"

    if summary_completed(summary_path):
        with runner_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] SKIP completed {method}/{candidate['candidate_id']}\n")
        existing = load_summary(run_dir, method, candidate, generation)
        if existing:
            return existing

    exp_file = write_exp(candidate, args.proxy_epochs, search_split=True)
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
        str(args.proxy_epochs),
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
        gen = f" gen={generation}" if generation is not None else ""
        handle.write(f"[{now()}] START {method}/{candidate['candidate_id']}{gen}\n")
    status_path.write_text(f"[{now()}] START {method}/{candidate['candidate_id']}\n", encoding="utf-8")

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
        with runner_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] FAIL {method}/{candidate['candidate_id']} exit={proc.returncode}\n")
        status_path.write_text(
            status_path.read_text(encoding="utf-8") + f"[{now()}] FAIL exit={proc.returncode}\n",
            encoding="utf-8",
        )
        raise SystemExit(proc.returncode)

    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] DONE {method}/{candidate['candidate_id']}\n")
    status_path.write_text(
        status_path.read_text(encoding="utf-8") + f"[{now()}] DONE\n",
        encoding="utf-8",
    )
    record = load_summary(run_dir, method, candidate, generation)
    if record is None:
        raise RuntimeError(f"Missing summary after completed run: {run_dir}")
    return record


def mutate(parent: dict, candidates_by_key: dict, seen: set[tuple], rng: random.Random) -> dict | None:
    base = dict(parent["config"])
    for _ in range(80):
        config = dict(base)
        for key in BINARY_KEYS:
            if rng.random() < 0.35:
                config[key] = 1 - int(config[key])
        for key, values in CATEGORICAL_VALUES.items():
            if rng.random() < 0.45:
                config[key] = rng.choice(values)
        config["use_p2"] = 1
        key = config_key(config)
        if key not in seen and key in candidates_by_key:
            return candidates_by_key[key]
    return None


def choose_qiea_population(
    generation: int,
    candidates: list[dict],
    candidates_by_key: dict,
    qiea_records: list[dict],
    seen: set[tuple],
    rng: random.Random,
    population: int,
) -> list[dict]:
    remaining = [candidate for candidate in candidates if config_key(candidate["config"]) not in seen]
    if not remaining:
        return []
    if generation == 1 or not qiea_records:
        picked = rng.sample(remaining, min(population, len(remaining)))
        return picked

    parents = sorted(qiea_records, key=lambda row: row.get("fitness", -999), reverse=True)[: max(1, population // 2)]
    picked = []
    for parent in parents:
        child = mutate(parent, candidates_by_key, seen | {config_key(c["config"]) for c in picked}, rng)
        if child:
            picked.append(child)
        if len(picked) >= population:
            return picked
    remaining = [candidate for candidate in remaining if candidate["candidate_id"] not in {c["candidate_id"] for c in picked}]
    while len(picked) < population and remaining:
        child = rng.choice(remaining)
        picked.append(child)
        remaining = [candidate for candidate in remaining if candidate["candidate_id"] != child["candidate_id"]]
    return picked


def write_outputs(output_dir: Path, records: list[dict], qiea_winner: dict | None, overall_winner: dict | None) -> None:
    output = {
        "stage": "stage5_proxy_search",
        "records": records,
        "qiea_winner": qiea_winner,
        "overall_winner": overall_winner,
    }
    (output_dir / "stage5_search_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if qiea_winner:
        (output_dir / "winner_config.json").write_text(
            json.dumps(qiea_winner, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    fields = [
        "method",
        "generation",
        "candidate_id",
        "status",
        "fitness",
        "best_epoch",
        "ap50_95",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "proxy_recall50",
        "proxy_mean_best_iou",
        "summary_path",
        "use_p2",
    ] + SEARCH_KEYS
    with (output_dir / "stage5_search_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records, key=lambda item: item.get("fitness", -999), reverse=True):
            writer.writerow({**{field: row.get(field) for field in fields}, **row.get("config", {})})

    lines = [
        "# Stage 5 Proxy Search",
        "",
        "Ranking uses proxy validation on `search_val.json`; QIEA-Final for Stage 6 is selected from QIEA candidates only.",
        "",
        "| Rank | Method | Gen | Candidate | Fitness | AP50:95 | AP50 | AP_small | Recall50 | Config |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(records, key=lambda item: item.get("fitness", -999), reverse=True)
    for index, row in enumerate(ranked[:10], start=1):
        cfg = row["config"]
        cfg_text = ", ".join(f"{key}={cfg[key]}" for key in SEARCH_KEYS)
        lines.append(
            "| {rank} | {method} | {gen} | {cid} | {fitness:.6f} | {ap5095:.6f} | {ap50:.6f} | {aps} | {recall:.6f} | {cfg} |".format(
                rank=index,
                method=row["method"],
                gen=row.get("generation") or 0,
                cid=row["candidate_id"],
                fitness=float(row["fitness"]),
                ap5095=number(row.get("ap50_95")),
                ap50=number(row.get("ap50")),
                aps=f"{row.get('ap_small'):.6f}" if isinstance(row.get("ap_small"), (float, int)) else "NA",
                recall=number(row.get("proxy_recall50")),
                cfg=cfg_text,
            )
        )
    if qiea_winner:
        lines.extend(
            [
                "",
                "## QIEA Winner",
                "",
                f"Candidate: `{qiea_winner['candidate_id']}`; fitness `{qiea_winner['fitness']:.6f}`.",
            ]
        )
    if overall_winner and qiea_winner and overall_winner["candidate_id"] != qiea_winner["candidate_id"]:
        lines.extend(
            [
                "",
                "## Check",
                "",
                f"Overall best proxy candidate is `{overall_winner['candidate_id']}` from `{overall_winner['method']}`; keep this distinction in the paper.",
            ]
        )
    (output_dir / "stage5_search_top10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Run Stage 5 QIEA vs Random proxy search.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--search-space", type=Path, default=DEFAULT_SEARCH_SPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--proxy-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qiea-population", type=int, default=4)
    parser.add_argument("--qiea-generations", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_log = args.output_dir / "stage5_runner.log"
    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] Stage 5 started\n")

    candidates = read_search_space(args.search_space)
    candidates_by_key = {config_key(candidate["config"]): candidate for candidate in candidates}
    rng = random.Random(args.seed)
    qiea_records: list[dict] = []
    qiea_seen: set[tuple] = set()

    for generation in range(1, args.qiea_generations + 1):
        population = choose_qiea_population(
            generation,
            candidates,
            candidates_by_key,
            qiea_records,
            qiea_seen,
            rng,
            args.qiea_population,
        )
        for candidate in population:
            qiea_seen.add(config_key(candidate["config"]))
            record = run_one(args, "qiea", candidate, generation, runner_log)
            qiea_records = [row for row in qiea_records if row["candidate_id"] != record["candidate_id"]]
            qiea_records.append(record)
            write_outputs(args.output_dir, qiea_records, None, None)

    random_rng = random.Random(args.seed + 1000)
    remaining = [candidate for candidate in candidates if config_key(candidate["config"]) not in qiea_seen]
    random_pick = random_rng.sample(remaining, min(args.random_candidates, len(remaining)))
    random_records = []
    for candidate in random_pick:
        record = run_one(args, "random", candidate, None, runner_log)
        random_records.append(record)
        all_records = qiea_records + random_records
        qiea_winner = max(qiea_records, key=lambda row: row.get("fitness", -999)) if qiea_records else None
        overall_winner = max(all_records, key=lambda row: row.get("fitness", -999)) if all_records else None
        write_outputs(args.output_dir, all_records, qiea_winner, overall_winner)

    all_records = qiea_records + random_records
    qiea_winner = max(qiea_records, key=lambda row: row.get("fitness", -999)) if qiea_records else None
    overall_winner = max(all_records, key=lambda row: row.get("fitness", -999)) if all_records else None
    write_outputs(args.output_dir, all_records, qiea_winner, overall_winner)

    if qiea_winner:
        src = Path(qiea_winner["summary_path"]).parent
        dst = args.output_dir / "qiea_winner_proxy_run"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] Stage 5 finished\n")


if __name__ == "__main__":
    main()
