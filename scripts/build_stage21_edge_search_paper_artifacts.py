#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "stage21_edge_search_paper_artifacts"
STAGE19 = PROJECT_ROOT / "outputs" / "stage19_search_algorithm_comparison"
STAGE20 = PROJECT_ROOT / "outputs" / "stage20_edge_efficiency"
STAGE18 = PROJECT_ROOT / "outputs" / "stage18_qiea_final_100ep_and_proxy_evidence"


BASELINE_P2_MEAN = {
    "Baseline": {
        "ap50_95": 0.06353482360615666,
        "ap50": 0.12769517430232727,
        "ap_small": 0.02952488617895882,
        "ap_large": 0.13227577610795674,
        "recall50": 0.2935911612984061,
    },
    "+P2": {
        "ap50_95": 0.06844198397977315,
        "ap50": 0.13890827365254088,
        "ap_small": 0.03870639620920202,
        "ap_large": 0.11847605114409547,
        "recall50": 0.29640876207709377,
    },
}

BASELINE_P2_STD = {
    "Baseline": {
        "ap50_95": 0.00021536158617364992,
        "ap50": 0.0013579163119608773,
        "ap_small": 0.0010259065955084883,
        "ap_large": 0.0041071647867318,
        "recall50": 0.0021658801830205165,
    },
    "+P2": {
        "ap50_95": 0.001343358086501618,
        "ap50": 0.0022239600316111773,
        "ap_small": 0.001069927445510636,
        "ap_large": 0.0026552528139930057,
        "recall50": 0.0021100910146429276,
    },
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qiea_mean_std() -> tuple[dict, dict]:
    rows = read_csv(STAGE18 / "qiea_final_100ep_multiseed_mean_std.csv")
    mean = {}
    std = {}
    for row in rows:
        metric = row["metric"]
        key = "recall50" if metric == "recall50" else metric
        mean[key] = as_float(row["mean"])
        std[key] = as_float(row["std"])
    return mean, std


def build_main_model_table() -> list[dict]:
    q_mean, q_std = qiea_mean_std()
    rows = []
    for label in ["Baseline", "+P2"]:
        row = {"model": label, "n": 3}
        for metric in ["ap50_95", "ap50", "ap_small", "ap_large", "recall50"]:
            row[f"{metric}_mean"] = BASELINE_P2_MEAN[label][metric]
            row[f"{metric}_std"] = BASELINE_P2_STD[label][metric]
        rows.append(row)
    row = {"model": "QIEA-Final", "n": 3}
    for metric in ["ap50_95", "ap50", "ap_small", "ap_large", "recall50"]:
        row[f"{metric}_mean"] = q_mean.get(metric)
        row[f"{metric}_std"] = q_std.get(metric)
    rows.append(row)
    return rows


def build_fitness_ablation(records: list[dict]) -> list[dict]:
    completed = [row for row in records if row.get("status", "completed") == "completed"]
    formulas = [
        ("AP-only", lambda r: as_float(r.get("ap50"))),
        ("Small+Recall", lambda r: as_float(r.get("ap_small")) + 0.2 * as_float(r.get("proxy_recall50"))),
        ("Small+Recall+AP", lambda r: as_float(r.get("ap_small")) + 0.3 * as_float(r.get("ap50")) + 0.2 * as_float(r.get("proxy_recall50"))),
        (
            "Small+Recall+AP-Complexity",
            lambda r: as_float(r.get("fitness")),
        ),
    ]
    rows = []
    for name, fn in formulas:
        ranked = sorted(completed, key=fn, reverse=True)
        top = ranked[0] if ranked else {}
        rows.append(
            {
                "fitness_formula": name,
                "selected_method": top.get("method", ""),
                "selected_candidate": top.get("candidate_id", ""),
                "score": fn(top) if top else math.nan,
                "ap50_95": as_float(top.get("ap50_95")),
                "ap50": as_float(top.get("ap50")),
                "ap_small": as_float(top.get("ap_small")),
                "recall50": as_float(top.get("proxy_recall50")),
                "complexity_penalized_fitness": as_float(top.get("fitness")),
            }
        )
    return rows


def build_proxy_pareto(records: list[dict], output_path: Path) -> None:
    completed = [row for row in records if row.get("status", "completed") == "completed"]
    color_map = {"qiea": "#1f77b4", "random": "#777777", "ga": "#2ca02c", "sa_qubo": "#d62728"}
    marker_map = {"qiea": "o", "random": "s", "ga": "^", "sa_qubo": "D"}
    plt.figure(figsize=(8.5, 5.5), dpi=220)
    for method in ["qiea", "random", "ga", "sa_qubo"]:
        group = [row for row in completed if row.get("method") == method]
        if not group:
            continue
        x = [as_float(row.get("fitness")) for row in group]
        y = [as_float(row.get("ap_small")) for row in group]
        plt.scatter(x, y, s=42, alpha=0.82, label=method, c=color_map.get(method), marker=marker_map.get(method, "o"))
    plt.xlabel("Complexity-penalized proxy fitness")
    plt.ylabel("Proxy AP_small")
    plt.title("Proxy Search Candidates: Fitness vs. Small-object Accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend(frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def build_efficiency_pareto(edge_rows: list[dict], output_path: Path) -> None:
    completed = [row for row in edge_rows if row.get("status") == "completed"]
    perf = {
        "Baseline": BASELINE_P2_MEAN["Baseline"]["ap_small"],
        "+P2": BASELINE_P2_MEAN["+P2"]["ap_small"],
    }
    q_mean, _ = qiea_mean_std()
    perf["QIEA-Final"] = q_mean.get("ap_small")
    perf["QIEA lightweight proxy winner"] = math.nan
    plt.figure(figsize=(8.5, 5.5), dpi=220)
    for row in completed:
        label = row["label"]
        latency = as_float(row.get("latency_ms_batch1"))
        ap_small = perf.get(label, math.nan)
        if math.isnan(ap_small):
            continue
        plt.scatter([latency], [ap_small], s=90)
        plt.annotate(label, (latency, ap_small), xytext=(6, 4), textcoords="offset points", fontsize=9)
    plt.xlabel("Batch-1 latency (ms, CUDA proxy)")
    plt.ylabel("AP_small (100-epoch multi-seed mean)")
    plt.title("Edge-oriented Accuracy-Latency Trade-off")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    search_records = read_csv(STAGE19 / "stage19_search_algorithm_records.csv")
    search_summary = read_csv(STAGE19 / "stage19_search_algorithm_summary.csv")
    edge_rows = read_csv(STAGE20 / "stage20_edge_efficiency.csv")

    main_models = build_main_model_table()
    write_csv(
        OUT_DIR / "table_main_100ep_multiseed.csv",
        main_models,
        [
            "model",
            "n",
            "ap50_95_mean",
            "ap50_95_std",
            "ap50_mean",
            "ap50_std",
            "ap_small_mean",
            "ap_small_std",
            "ap_large_mean",
            "ap_large_std",
            "recall50_mean",
            "recall50_std",
        ],
    )

    fitness_rows = build_fitness_ablation(search_records)
    write_csv(
        OUT_DIR / "table_proxy_fitness_ablation.csv",
        fitness_rows,
        [
            "fitness_formula",
            "selected_method",
            "selected_candidate",
            "score",
            "ap50_95",
            "ap50",
            "ap_small",
            "recall50",
            "complexity_penalized_fitness",
        ],
    )

    build_proxy_pareto(search_records, OUT_DIR / "figure_proxy_search_pareto.png")
    build_efficiency_pareto(edge_rows, OUT_DIR / "figure_edge_accuracy_latency.png")

    payload = {
        "stage": "stage21_edge_search_paper_artifacts",
        "inputs": {
            "stage19_records": str(STAGE19 / "stage19_search_algorithm_records.csv"),
            "stage20_efficiency": str(STAGE20 / "stage20_edge_efficiency.csv"),
            "stage18_qiea": str(STAGE18 / "qiea_final_100ep_multiseed_mean_std.csv"),
        },
        "outputs": {
            "main_model_table": str(OUT_DIR / "table_main_100ep_multiseed.csv"),
            "fitness_ablation_table": str(OUT_DIR / "table_proxy_fitness_ablation.csv"),
            "proxy_pareto": str(OUT_DIR / "figure_proxy_search_pareto.png"),
            "edge_latency_pareto": str(OUT_DIR / "figure_edge_accuracy_latency.png"),
        },
    }
    (OUT_DIR / "stage21_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Stage 21 Edge/Search Paper Artifacts",
        "",
        "Generated paper-ready tables and figures for the edge-constrained search framing.",
        "",
        "## Outputs",
        "",
        f"- Main 100-epoch multi-seed table: `{OUT_DIR / 'table_main_100ep_multiseed.csv'}`",
        f"- Proxy fitness ablation table: `{OUT_DIR / 'table_proxy_fitness_ablation.csv'}`",
        f"- Proxy search Pareto figure: `{OUT_DIR / 'figure_proxy_search_pareto.png'}`",
        f"- Edge accuracy-latency figure: `{OUT_DIR / 'figure_edge_accuracy_latency.png'}`",
        "",
        "Regenerate this stage after Stage 19 finishes to include GA and SA/QUBO-inspired results.",
    ]
    (OUT_DIR / "stage21_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
