#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_stage5_proxy_search as stage5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "python"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "visdrone_det_coco"
DEFAULT_EXISTING_STAGE5 = PROJECT_ROOT / "outputs" / "stage5_proxy_search" / "stage5_search_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "stage19_search_algorithm_comparison"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")
    print(message, flush=True)


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in payload.get("records", []):
        if row.get("status") == "completed" and row.get("method") in {"qiea", "random"}:
            copied = dict(row)
            copied["source"] = "stage5_existing"
            records.append(copied)
    return records


def config_key_from_record(row: dict) -> tuple:
    return stage5.config_key(row["config"])


def candidate_by_id(candidates: list[dict]) -> dict[str, dict]:
    return {candidate["candidate_id"]: candidate for candidate in candidates}


def build_metric_cache(existing_records: list[dict]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for row in existing_records:
        cache[row["candidate_id"]] = row
    return cache


def reassign_record(cached: dict, method: str, generation: int | None, output_dir: Path) -> dict:
    record = dict(cached)
    record["method"] = method
    record["generation"] = generation
    record["source"] = cached.get("method", cached.get("source", "cached"))
    record["reused_metrics"] = True
    return record


def run_or_reuse(
    args: argparse.Namespace,
    method: str,
    candidate: dict,
    generation: int | None,
    runner_log: Path,
    metric_cache: dict[str, dict],
) -> dict:
    cached = metric_cache.get(candidate["candidate_id"])
    if cached:
        log(runner_log, f"REUSE {method}/{candidate['candidate_id']} from {cached.get('method', cached.get('source'))}")
        return reassign_record(cached, method, generation, args.output_dir)

    run_args = SimpleNamespace(
        python=args.python,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        proxy_epochs=args.proxy_epochs,
        lr=args.lr,
        seed=args.seed,
        amp=args.amp,
    )
    record = stage5.run_one(run_args, method, candidate, generation, runner_log)
    record["source"] = "stage19_new_run"
    record["reused_metrics"] = False
    metric_cache[candidate["candidate_id"]] = record
    return record


def mutate_config(config: dict, rng: random.Random, mutation_rate: float = 0.35) -> dict:
    child = dict(config)
    for key in stage5.BINARY_KEYS:
        if rng.random() < mutation_rate:
            child[key] = 1 - int(child[key])
    for key, values in stage5.CATEGORICAL_VALUES.items():
        if rng.random() < mutation_rate:
            child[key] = rng.choice(values)
    child["use_p2"] = 1
    return child


def crossover(parent_a: dict, parent_b: dict, rng: random.Random) -> dict:
    config = {}
    for key in stage5.SEARCH_KEYS:
        config[key] = parent_a["config"][key] if rng.random() < 0.5 else parent_b["config"][key]
    config["use_p2"] = 1
    return config


def lookup_candidate(config: dict, candidates_by_key: dict[tuple, dict]) -> dict | None:
    return candidates_by_key.get(stage5.config_key(config))


def unique_random_candidate(
    candidates: list[dict],
    seen_ids: set[str],
    rng: random.Random,
) -> dict | None:
    remaining = [candidate for candidate in candidates if candidate["candidate_id"] not in seen_ids]
    if not remaining:
        return None
    return rng.choice(remaining)


def choose_ga_generation(
    generation: int,
    candidates: list[dict],
    candidates_by_key: dict[tuple, dict],
    evaluated: list[dict],
    seen_ids: set[str],
    population: int,
    rng: random.Random,
) -> list[dict]:
    if generation == 1 or len(evaluated) < 2:
        picked = []
        while len(picked) < population:
            candidate = unique_random_candidate(candidates, seen_ids | {row["candidate_id"] for row in picked}, rng)
            if candidate is None:
                break
            picked.append(candidate)
        return picked

    parents = sorted(evaluated, key=lambda row: row.get("fitness", -999.0), reverse=True)[: max(2, population // 2)]
    picked: list[dict] = []
    local_seen = set(seen_ids)
    while len(picked) < population:
        pa, pb = rng.sample(parents, 2)
        config = crossover(pa, pb, rng)
        config = mutate_config(config, rng, mutation_rate=0.30)
        candidate = lookup_candidate(config, candidates_by_key)
        if candidate and candidate["candidate_id"] not in local_seen:
            picked.append(candidate)
            local_seen.add(candidate["candidate_id"])
            continue
        fallback = unique_random_candidate(candidates, local_seen, rng)
        if fallback is None:
            break
        picked.append(fallback)
        local_seen.add(fallback["candidate_id"])
    return picked


def neighbor_candidate(
    current: dict,
    candidates: list[dict],
    candidates_by_key: dict[tuple, dict],
    seen_ids: set[str],
    rng: random.Random,
) -> dict | None:
    for _ in range(80):
        config = dict(current["config"])
        key = rng.choice(stage5.SEARCH_KEYS)
        if key in stage5.BINARY_KEYS:
            config[key] = 1 - int(config[key])
        else:
            values = [value for value in stage5.CATEGORICAL_VALUES[key] if value != config[key]]
            config[key] = rng.choice(values)
        config["use_p2"] = 1
        candidate = lookup_candidate(config, candidates_by_key)
        if candidate and candidate["candidate_id"] not in seen_ids:
            return candidate
    return unique_random_candidate(candidates, seen_ids, rng)


def run_ga(
    args: argparse.Namespace,
    candidates: list[dict],
    candidates_by_key: dict[tuple, dict],
    runner_log: Path,
    metric_cache: dict[str, dict],
) -> list[dict]:
    rng = random.Random(args.seed + 2000)
    records: list[dict] = []
    seen_ids: set[str] = set()
    for generation in range(1, args.ga_generations + 1):
        population = choose_ga_generation(
            generation,
            candidates,
            candidates_by_key,
            records,
            seen_ids,
            args.ga_population,
            rng,
        )
        for candidate in population:
            seen_ids.add(candidate["candidate_id"])
            record = run_or_reuse(args, "ga", candidate, generation, runner_log, metric_cache)
            records.append(record)
            write_combined_outputs(args.output_dir, load_existing_records(args.existing_stage5) + records)
    return records


def run_sa(
    args: argparse.Namespace,
    candidates: list[dict],
    candidates_by_key: dict[tuple, dict],
    runner_log: Path,
    metric_cache: dict[str, dict],
) -> list[dict]:
    rng = random.Random(args.seed + 3000)
    records: list[dict] = []
    seen_ids: set[str] = set()
    current_candidate = unique_random_candidate(candidates, seen_ids, rng)
    current_record = None
    for step in range(1, args.sa_steps + 1):
        if current_candidate is None:
            break
        seen_ids.add(current_candidate["candidate_id"])
        proposal_record = run_or_reuse(args, "sa_qubo", current_candidate, step, runner_log, metric_cache)
        records.append(proposal_record)
        if current_record is None:
            current_record = proposal_record
        else:
            temperature = max(args.sa_min_temp, args.sa_initial_temp * (args.sa_cooling ** (step - 1)))
            delta = float(proposal_record.get("fitness", -999.0)) - float(current_record.get("fitness", -999.0))
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
                current_record = proposal_record
        current_cfg = current_record["config"] if current_record else current_candidate["config"]
        current_candidate = neighbor_candidate(
            {"config": current_cfg},
            candidates,
            candidates_by_key,
            seen_ids,
            rng,
        )
        write_combined_outputs(args.output_dir, load_existing_records(args.existing_stage5) + records)
    return records


def aggregate_rows(records: list[dict]) -> list[dict]:
    rows = []
    for method in ["qiea", "random", "ga", "sa_qubo"]:
        group = [row for row in records if row.get("method") == method and row.get("status") == "completed"]
        if not group:
            continue
        ranked = sorted(group, key=lambda row: row.get("fitness", -999.0), reverse=True)
        fitness = [float(row["fitness"]) for row in ranked if isinstance(row.get("fitness"), (int, float))]
        top3 = fitness[:3]
        best = ranked[0]
        rows.append(
            {
                "method": method,
                "n": len(group),
                "best_candidate": best.get("candidate_id"),
                "best_fitness": best.get("fitness"),
                "best_ap50_95": best.get("ap50_95"),
                "best_ap50": best.get("ap50"),
                "best_ap_small": best.get("ap_small"),
                "best_recall50": best.get("proxy_recall50"),
                "mean_fitness": sum(fitness) / len(fitness) if fitness else None,
                "top3_mean_fitness": sum(top3) / len(top3) if top3 else None,
            }
        )
    return rows


def write_combined_outputs(output_dir: Path, records: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [row for row in records if row.get("status") == "completed"]
    (output_dir / "stage19_search_algorithm_records.json").write_text(
        json.dumps({"stage": "stage19_search_algorithm_comparison", "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    flat_fields = [
        "method",
        "generation",
        "candidate_id",
        "fitness",
        "ap50_95",
        "ap50",
        "ap_small",
        "proxy_recall50",
        "reused_metrics",
        "source",
    ] + stage5.SEARCH_KEYS
    with (output_dir / "stage19_search_algorithm_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in sorted(records, key=lambda item: (item.get("method", ""), -(item.get("fitness") or -999.0))):
            out = {field: row.get(field) for field in flat_fields}
            out.update({key: row.get("config", {}).get(key) for key in stage5.SEARCH_KEYS})
            writer.writerow(out)

    summary = aggregate_rows(records)
    with (output_dir / "stage19_search_algorithm_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "method",
            "n",
            "best_candidate",
            "best_fitness",
            "best_ap50_95",
            "best_ap50",
            "best_ap_small",
            "best_recall50",
            "mean_fitness",
            "top3_mean_fitness",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Stage 19 Search Algorithm Comparison",
        "",
        "All methods use the same proxy setting where available: VisDrone search split, 640 input, 10 proxy epochs, seed 42, and the same complexity-penalized fitness.",
        "",
        "| Method | N | Best candidate | Best fitness | Best AP50:95 | Best AP50 | Best AP_small | Best Recall50 | Mean fitness | Top-3 fitness |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['n']} | `{row['best_candidate']}` | {row['best_fitness']:.6f} | "
            f"{row['best_ap50_95']:.6f} | {row['best_ap50']:.6f} | {row['best_ap_small']:.6f} | "
            f"{row['best_recall50']:.6f} | {row['mean_fitness']:.6f} | {row['top3_mean_fitness']:.6f} |"
        )
    (output_dir / "stage19_search_algorithm_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Stage 19: compare QIEA, random, GA, and SA/QUBO-inspired structure search.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--search-space", type=Path, default=stage5.DEFAULT_SEARCH_SPACE)
    parser.add_argument("--existing-stage5", type=Path, default=DEFAULT_EXISTING_STAGE5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--proxy-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ga-population", type=int, default=4)
    parser.add_argument("--ga-generations", type=int, default=4)
    parser.add_argument("--sa-steps", type=int, default=16)
    parser.add_argument("--sa-initial-temp", type=float, default=0.004)
    parser.add_argument("--sa-min-temp", type=float, default=0.0005)
    parser.add_argument("--sa-cooling", type=float, default=0.85)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner_log = args.output_dir / "stage19_runner.log"
    log(runner_log, "Stage 19 started")

    candidates = stage5.read_search_space(args.search_space)
    candidates_by_key = {stage5.config_key(candidate["config"]): candidate for candidate in candidates}
    existing_records = load_existing_records(args.existing_stage5)
    metric_cache = build_metric_cache(existing_records)
    write_combined_outputs(args.output_dir, existing_records)

    ga_records = run_ga(args, candidates, candidates_by_key, runner_log, metric_cache)
    sa_records = run_sa(args, candidates, candidates_by_key, runner_log, metric_cache)
    all_records = existing_records + ga_records + sa_records
    write_combined_outputs(args.output_dir, all_records)

    if (args.output_dir / "qiea_random_existing").exists():
        shutil.rmtree(args.output_dir / "qiea_random_existing")
    if args.existing_stage5.parent.exists():
        shutil.copytree(args.existing_stage5.parent, args.output_dir / "qiea_random_existing", ignore=shutil.ignore_patterns("*.pth"))

    log(runner_log, "Stage 19 finished")


if __name__ == "__main__":
    main()
