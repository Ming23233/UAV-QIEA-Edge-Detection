#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "stage3_search_space"

VARIABLES = [
    {
        "name": "coord_attention",
        "paper_name": "CA",
        "type": "binary",
        "values": [0, 1],
        "description": "Coordinate attention on the P2 high-resolution branch.",
    },
    {
        "name": "channel_spatial_attention",
        "paper_name": "CSA",
        "type": "binary",
        "values": [0, 1],
        "description": "Compact channel-spatial attention on the P2 branch.",
    },
    {
        "name": "tiny_context",
        "paper_name": "Context",
        "type": "binary",
        "values": [0, 1],
        "description": "Dilated context refinement for dense tiny objects.",
    },
    {
        "name": "scale_aware_fusion",
        "paper_name": "Fusion",
        "type": "binary",
        "values": [0, 1],
        "description": "Gated P3-to-P2 semantic fusion.",
    },
    {
        "name": "small_obj_loss_weight",
        "paper_name": "Small-loss weight",
        "type": "categorical_float",
        "values": [1.0, 1.25, 1.5],
        "description": "Foreground loss multiplier for boxes smaller than 32x32.",
    },
    {
        "name": "center_radius",
        "paper_name": "Center radius",
        "type": "categorical_float",
        "values": [2.5, 3.0, 3.5],
        "description": "YOLOX center-prior radius used during assignment.",
    },
]


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


def enumerate_candidates() -> list[dict]:
    names = [item["name"] for item in VARIABLES]
    values = [item["values"] for item in VARIABLES]
    rows = []
    for index, combo in enumerate(itertools.product(*values), start=1):
        config = dict(zip(names, combo))
        config["use_p2"] = 1
        rows.append(
            {
                "index": index,
                "candidate_id": candidate_id(config),
                "config": config,
            }
        )
    return rows


def write_csv(path: Path, candidates: list[dict]) -> None:
    fields = ["index", "candidate_id", "use_p2"] + [item["name"] for item in VARIABLES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            out = {
                "index": row["index"],
                "candidate_id": row["candidate_id"],
                **row["config"],
            }
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser("Generate Stage 3 simplified search space.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qiea-candidates", type=int, default=16)
    parser.add_argument("--random-candidates", type=int, default=16)
    parser.add_argument("--qiea-population", type=int, default=4)
    parser.add_argument("--qiea-generations", type=int, default=4)
    parser.add_argument("--proxy-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.qiea_population * args.qiea_generations != args.qiea_candidates:
        raise ValueError("qiea_population * qiea_generations must equal qiea_candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = enumerate_candidates()

    payload = {
        "stage": "stage3_search_space",
        "name": "one_week_simplified_qiea_640",
        "base_model": "YOLOX-nano",
        "dataset": "VisDrone",
        "input_size": [640, 640],
        "base_exp": "exps/example/uav/yolox_nano_visdrone_det_640_base.py",
        "fixed": {
            "use_p2": True,
            "train_ann": "search_train.json",
            "val_ann": "search_val.json",
            "proxy_epochs": args.proxy_epochs,
            "final_epochs": args.final_epochs,
            "seed": args.seed,
        },
        "search_budget": {
            "qiea_population": args.qiea_population,
            "qiea_generations": args.qiea_generations,
            "qiea_candidates": args.qiea_candidates,
            "random_candidates": args.random_candidates,
        },
        "variables": VARIABLES,
        "total_combinations": len(candidates),
        "fitness": {
            "primary": "AP_small + 0.30*AP50 + 0.20*proxy_recall50 - lightweight_complexity_penalty",
            "fallback": "AP50:95 + 0.30*AP50 + 0.20*proxy_recall50 - lightweight_complexity_penalty",
            "fallback_condition": "Use fallback when AP_small is unavailable for a proxy summary.",
        },
        "candidate_table": str(args.output_dir / "stage3_search_space_candidates.csv"),
        "candidates": candidates,
    }

    json_path = args.output_dir / "stage3_search_space_simplified_640.json"
    csv_path = args.output_dir / "stage3_search_space_candidates.csv"
    md_path = args.output_dir / "stage3_search_space_simplified_640.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, candidates)

    lines = [
        "# Stage 3 Search Space",
        "",
        "Setting: YOLOX-nano on VisDrone at 640x640. P2 is fixed on; QIEA searches lightweight refinements and assignment parameters.",
        "",
        f"Total combinations: {len(candidates)}.",
        f"Proxy budget: QIEA {args.qiea_candidates} candidates ({args.qiea_generations} generations x {args.qiea_population} population), Random {args.random_candidates} candidates, {args.proxy_epochs} epochs each.",
        "",
        "| Variable | Values | Role |",
        "|---|---|---|",
    ]
    for item in VARIABLES:
        lines.append(
            f"| {item['paper_name']} | {', '.join(str(v) for v in item['values'])} | {item['description']} |"
        )
    lines.extend(
        [
            "",
            "Fitness prefers small-object AP first, then AP50 and proxy recall, with a small penalty for extra modules.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
