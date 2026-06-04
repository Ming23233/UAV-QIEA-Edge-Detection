from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "outputs"
OUT_DIR = RESULTS / "stage25_submission_evidence"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value, default=math.nan) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct_delta(value: float, base: float) -> float:
    if not math.isfinite(value) or not math.isfinite(base) or base == 0:
        return math.nan
    return (value - base) / base * 100.0


def fmt(value, digits=4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: list[dict], fields: list[str], labels: list[str] | None = None) -> list[str]:
    labels = labels or fields
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def build_main_multiseed() -> list[dict]:
    source = RESULTS / "stage21_edge_search_paper_artifacts" / "table_main_100ep_multiseed.csv"
    rows = read_csv(source)
    baseline = next((r for r in rows if r.get("model") == "Baseline"), None)
    base_ap = to_float(baseline.get("ap50_95_mean")) if baseline else math.nan
    base_small = to_float(baseline.get("ap_small_mean")) if baseline else math.nan
    base_recall = to_float(baseline.get("recall50_mean")) if baseline else math.nan
    output = []
    for row in rows:
        ap = to_float(row.get("ap50_95_mean"))
        small = to_float(row.get("ap_small_mean"))
        recall = to_float(row.get("recall50_mean"))
        output.append(
            {
                "model": row.get("model"),
                "n": row.get("n"),
                "ap50_95_mean": fmt(ap, 6),
                "ap50_95_std": fmt(to_float(row.get("ap50_95_std")), 6),
                "ap50_mean": fmt(to_float(row.get("ap50_mean")), 6),
                "ap50_std": fmt(to_float(row.get("ap50_std")), 6),
                "ap_small_mean": fmt(small, 6),
                "ap_small_std": fmt(to_float(row.get("ap_small_std")), 6),
                "recall50_mean": fmt(recall, 6),
                "recall50_std": fmt(to_float(row.get("recall50_std")), 6),
                "delta_ap50_95_vs_baseline_pct": fmt(pct_delta(ap, base_ap), 2),
                "delta_ap_small_vs_baseline_pct": fmt(pct_delta(small, base_small), 2),
                "delta_recall50_vs_baseline_pct": fmt(pct_delta(recall, base_recall), 2),
            }
        )
    fields = [
        "model",
        "n",
        "ap50_95_mean",
        "ap50_95_std",
        "ap50_mean",
        "ap50_std",
        "ap_small_mean",
        "ap_small_std",
        "recall50_mean",
        "recall50_std",
        "delta_ap50_95_vs_baseline_pct",
        "delta_ap_small_vs_baseline_pct",
        "delta_recall50_vs_baseline_pct",
    ]
    write_csv(OUT_DIR / "table_1_main_multiseed_unified.csv", output, fields)
    return output


def build_ablation() -> list[dict]:
    data = read_json(RESULTS / "stage10_ablation" / "stage10_ablation.json") or {}
    records = data.get("records", [])
    baseline = next((r for r in records if r.get("label") == "Baseline"), None)
    base_small = to_float(baseline.get("ap_small")) if baseline else math.nan
    base_recall = to_float(baseline.get("proxy_recall50")) if baseline else math.nan
    output = []
    for row in records:
        output.append(
            {
                "method": row.get("label"),
                "status": row.get("status"),
                "best_epoch": row.get("best_epoch"),
                "ap50_95": fmt(to_float(row.get("ap50_95")), 6),
                "ap50": fmt(to_float(row.get("ap50")), 6),
                "ap_small": fmt(to_float(row.get("ap_small")), 6),
                "recall50": fmt(to_float(row.get("proxy_recall50")), 6),
                "delta_ap_small_abs": fmt(to_float(row.get("ap_small")) - base_small, 6),
                "delta_recall50_abs": fmt(to_float(row.get("proxy_recall50")) - base_recall, 6),
            }
        )
    fields = [
        "method",
        "status",
        "best_epoch",
        "ap50_95",
        "ap50",
        "ap_small",
        "recall50",
        "delta_ap_small_abs",
        "delta_recall50_abs",
    ]
    write_csv(OUT_DIR / "table_2_ablation_unified.csv", output, fields)
    return output


def build_efficiency() -> list[dict]:
    gpu_rows = read_csv(RESULTS / "stage20_edge_efficiency" / "stage20_edge_efficiency.csv")
    cpu_rows = read_csv(OUT_DIR / "cpu_edge_efficiency" / "stage20_edge_efficiency.csv")
    main_rows = read_csv(RESULTS / "stage21_edge_search_paper_artifacts" / "table_main_100ep_multiseed.csv")
    main_by_model = {row.get("model"): row for row in main_rows}
    cpu_by_label = {row.get("label"): row for row in cpu_rows}
    output = []
    for row in gpu_rows:
        label = row.get("label")
        if label == "QIEA lightweight proxy winner":
            model_key = "QIEA-Final"
        else:
            model_key = label
        main = main_by_model.get(model_key, {})
        cpu = cpu_by_label.get(label, {})
        output.append(
            {
                "method": label,
                "params_m": fmt(to_float(row.get("params_m")), 3),
                "flops_g": fmt(to_float(row.get("flops_g_2x_macs")), 3),
                "gpu_latency_ms_b1": fmt(to_float(row.get("latency_ms_batch1")), 3),
                "gpu_fps_b1": fmt(to_float(row.get("fps_batch1")), 2),
                "gpu_peak_mem_mb_b1": fmt(to_float(row.get("peak_mem_mb_batch1")), 1),
                "cpu_latency_ms_b1_proxy": fmt(to_float(cpu.get("latency_ms_batch1")), 3),
                "cpu_fps_b1_proxy": fmt(to_float(cpu.get("fps_batch1")), 2),
                "checkpoint_mb": fmt(to_float(row.get("checkpoint_mb")), 2),
                "ap50_95_mean_main": fmt(to_float(main.get("ap50_95_mean")), 6),
                "ap_small_mean_main": fmt(to_float(main.get("ap_small_mean")), 6),
                "recall50_mean_main": fmt(to_float(main.get("recall50_mean")), 6),
                "note": "GPU proxy + CPU-only simulated edge proxy; not measured on a physical Jetson/NPU device.",
            }
        )
    fields = [
        "method",
        "params_m",
        "flops_g",
        "gpu_latency_ms_b1",
        "gpu_fps_b1",
        "gpu_peak_mem_mb_b1",
        "cpu_latency_ms_b1_proxy",
        "cpu_fps_b1_proxy",
        "checkpoint_mb",
        "ap50_95_mean_main",
        "ap_small_mean_main",
        "recall50_mean_main",
        "note",
    ]
    write_csv(OUT_DIR / "table_3_efficiency_deployment_unified.csv", output, fields)
    return output


def build_search_tables() -> tuple[list[dict], list[dict]]:
    summary_rows = read_csv(RESULTS / "stage19_search_algorithm_comparison" / "stage19_search_algorithm_summary.csv")
    random_row = next((r for r in summary_rows if r.get("method") == "random"), None)
    rand_best = to_float(random_row.get("best_fitness")) if random_row else math.nan
    rand_mean = to_float(random_row.get("mean_fitness")) if random_row else math.nan
    rand_top3 = to_float(random_row.get("top3_mean_fitness")) if random_row else math.nan
    output = []
    for row in summary_rows:
        best = to_float(row.get("best_fitness"))
        mean = to_float(row.get("mean_fitness"))
        top3 = to_float(row.get("top3_mean_fitness"))
        output.append(
            {
                "method": row.get("method"),
                "n": row.get("n"),
                "best_candidate": row.get("best_candidate"),
                "best_fitness": fmt(best, 6),
                "best_ap50_95": fmt(to_float(row.get("best_ap50_95")), 6),
                "best_ap50": fmt(to_float(row.get("best_ap50")), 6),
                "best_ap_small": fmt(to_float(row.get("best_ap_small")), 6),
                "best_recall50": fmt(to_float(row.get("best_recall50")), 6),
                "mean_fitness": fmt(mean, 6),
                "top3_mean_fitness": fmt(top3, 6),
                "best_fitness_delta_vs_random_pct": fmt(pct_delta(best, rand_best), 2),
                "mean_fitness_delta_vs_random_pct": fmt(pct_delta(mean, rand_mean), 2),
                "top3_delta_vs_random_pct": fmt(pct_delta(top3, rand_top3), 2),
            }
        )
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
        "best_fitness_delta_vs_random_pct",
        "mean_fitness_delta_vs_random_pct",
        "top3_delta_vs_random_pct",
    ]
    write_csv(OUT_DIR / "table_4_search_algorithm_comparison.csv", output, fields)

    records = read_csv(RESULTS / "stage19_search_algorithm_comparison" / "stage19_search_algorithm_records.csv")
    records = sorted(records, key=lambda r: to_float(r.get("fitness")), reverse=True)
    top_records = []
    for rank, row in enumerate(records[:20], start=1):
        top_records.append(
            {
                "rank": rank,
                "method": row.get("method"),
                "generation": row.get("generation"),
                "candidate_id": row.get("candidate_id"),
                "fitness": fmt(to_float(row.get("fitness")), 6),
                "ap50_95": fmt(to_float(row.get("ap50_95")), 6),
                "ap50": fmt(to_float(row.get("ap50")), 6),
                "ap_small": fmt(to_float(row.get("ap_small")), 6),
                "recall50": fmt(to_float(row.get("proxy_recall50")), 6),
                "coord_attention": row.get("coord_attention"),
                "channel_spatial_attention": row.get("channel_spatial_attention"),
                "tiny_context": row.get("tiny_context"),
                "scale_aware_fusion": row.get("scale_aware_fusion"),
                "small_obj_loss_weight": row.get("small_obj_loss_weight"),
                "center_radius": row.get("center_radius"),
            }
        )
    fields_top = [
        "rank",
        "method",
        "generation",
        "candidate_id",
        "fitness",
        "ap50_95",
        "ap50",
        "ap_small",
        "recall50",
        "coord_attention",
        "channel_spatial_attention",
        "tiny_context",
        "scale_aware_fusion",
        "small_obj_loss_weight",
        "center_radius",
    ]
    write_csv(OUT_DIR / "table_5_search_top_candidates.csv", top_records, fields_top)
    build_search_curve(records)
    return output, top_records


def build_search_curve(records: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_method = defaultdict(list)
    for row in records:
        by_method[row.get("method")].append(row)
    plt.figure(figsize=(8, 4.8), dpi=180)
    for method in ["random", "qiea", "ga", "sa_qubo"]:
        rows = by_method.get(method, [])
        if not rows:
            continue
        rows = sorted(
            rows,
            key=lambda r: (to_float(r.get("generation"), 0), r.get("candidate_id") or ""),
        )
        best = -1e9
        ys = []
        xs = []
        for idx, row in enumerate(rows, start=1):
            best = max(best, to_float(row.get("fitness"), -1e9))
            xs.append(idx)
            ys.append(best)
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.2, label=method)
    plt.xlabel("Candidate evaluations")
    plt.ylabel("Cumulative best proxy fitness")
    plt.title("Proxy Search Progression")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_1_qiea_search_curve.png")
    plt.close()


def build_auair() -> list[dict]:
    base_dir = RESULTS / "stage24_auair_engineering_case_416_30pct"
    zero_rows = read_csv(base_dir / "stage24_auair_zero_shot.csv")
    finetune_rows = read_csv(base_dir / "stage24_auair_target_finetune.csv")
    zero_by_method = {row.get("method"): row for row in zero_rows}
    output = []
    for row in finetune_rows:
        method = row.get("method")
        zero = zero_by_method.get(method, {})
        zero_ap = to_float(zero.get("ap50_95"))
        test_ap = to_float(row.get("test_ap50_95"))
        output.append(
            {
                "method": method,
                "zero_shot_ap50_95": fmt(zero_ap, 6),
                "zero_shot_ap50": fmt(to_float(zero.get("ap50")), 6),
                "zero_shot_ap_small": fmt(to_float(zero.get("ap_small")), 6),
                "zero_shot_recall50": fmt(to_float(zero.get("proxy_recall50")), 6),
                "finetune_best_val_epoch": row.get("val_best_epoch"),
                "finetune_val_ap50_95": fmt(to_float(row.get("val_ap50_95")), 6),
                "finetune_test_ap50_95": fmt(test_ap, 6),
                "finetune_test_ap50": fmt(to_float(row.get("test_ap50")), 6),
                "finetune_test_ap_small": fmt(to_float(row.get("test_ap_small")), 6),
                "finetune_test_recall50": fmt(to_float(row.get("test_proxy_recall50")), 6),
                "test_ap50_95_gain_from_zero_abs": fmt(test_ap - zero_ap, 6),
                "test_ap50_95_gain_from_zero_pct": fmt(pct_delta(test_ap, zero_ap), 2),
            }
        )
    fields = [
        "method",
        "zero_shot_ap50_95",
        "zero_shot_ap50",
        "zero_shot_ap_small",
        "zero_shot_recall50",
        "finetune_best_val_epoch",
        "finetune_val_ap50_95",
        "finetune_test_ap50_95",
        "finetune_test_ap50",
        "finetune_test_ap_small",
        "finetune_test_recall50",
        "test_ap50_95_gain_from_zero_abs",
        "test_ap50_95_gain_from_zero_pct",
    ]
    write_csv(OUT_DIR / "table_6_auair_engineering_case.csv", output, fields)
    return output


def build_visual_manifest() -> list[dict]:
    candidates = []
    roots = [
        RESULTS / "stage13_visualizations" / "uav_qualitative_2x2",
        RESULTS / "stage13_visualizations" / "uav_qualitative_large",
        RESULTS / "stage13_visualizations" / "uav_qualitative_ultra_2x",
        PROJECT_ROOT / "paper_drafts" / "updated_cn_20260517",
        RESULTS / "stage21_edge_search_paper_artifacts",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            candidates.append(
                {
                    "file": str(path),
                    "name": path.name,
                    "size_mb": fmt(path.stat().st_size / (1024 * 1024), 3),
                    "purpose": infer_visual_purpose(path),
                }
            )
    fields = ["name", "purpose", "size_mb", "file"]
    write_csv(OUT_DIR / "table_7_visualization_manifest.csv", candidates, fields)
    return candidates


def infer_visual_purpose(path: Path) -> str:
    name = path.name.lower()
    if "proxy" in name or "search" in name or "pareto" in name:
        return "QIEA/search evidence figure"
    if "edge" in name or "efficiency" in name or "latency" in name:
        return "edge efficiency figure"
    if "qualitative" in str(path).lower() or "success" in name or "failure" in name or "dense" in name:
        return "UAV detection qualitative visualization"
    if "small" in name or "metrics" in name or "tradeoff" in name:
        return "main result or small-object metric figure"
    return "supporting figure"


def copy_existing_figures() -> None:
    figure_dir = OUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        RESULTS / "stage21_edge_search_paper_artifacts" / "figure_proxy_search_pareto.png",
        RESULTS / "stage21_edge_search_paper_artifacts" / "figure_edge_accuracy_latency.png",
        PROJECT_ROOT / "paper_drafts" / "updated_cn_20260517" / "figure3_stage17_main_metrics.png",
        PROJECT_ROOT / "paper_drafts" / "updated_cn_20260517" / "figure4_ap_small_stability.png",
        PROJECT_ROOT / "paper_drafts" / "updated_cn_20260517" / "figure5_metric_delta_tradeoff.png",
        PROJECT_ROOT / "paper_drafts" / "updated_cn_20260517" / "figure6_efficiency_tradeoff_updated.png",
        OUT_DIR / "figure_1_qiea_search_curve.png",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, figure_dir / src.name)


def write_summary(
    main_rows: list[dict],
    ablation_rows: list[dict],
    efficiency_rows: list[dict],
    search_rows: list[dict],
    auair_rows: list[dict],
    visuals: list[dict],
) -> None:
    lines = [
        "# Stage 25 投稿前实验证据包",
        "",
        "本目录把论文投稿前最需要补强的三类证据统一整理：效率/部署约束、QIEA 搜索过程证据、最终实验结果链条。",
        "",
        "## 1. 效率/部署实验",
        "",
        "已整理 GPU batch=1 推理代理结果，并补充 CPU-only 模拟边缘推理测试。CPU 测试用于说明部署约束和低算力环境趋势，不能写成真实 Jetson/NPU 实测。",
        "",
    ]
    eff_fields = [
        "method",
        "params_m",
        "flops_g",
        "gpu_latency_ms_b1",
        "gpu_fps_b1",
        "gpu_peak_mem_mb_b1",
        "cpu_latency_ms_b1_proxy",
        "cpu_fps_b1_proxy",
        "ap50_95_mean_main",
        "ap_small_mean_main",
        "recall50_mean_main",
    ]
    lines.extend(markdown_table(efficiency_rows, eff_fields))
    lines.extend(
        [
            "",
            "建议写法：P2 在增加少量参数的情况下提升小目标指标和召回，但带来 FLOPs 与延迟开销；QIEA-Final 用于结构搜索和权衡分析，不应写成最终精度最高模型。",
            "",
            "## 2. QIEA 搜索过程证据",
            "",
            "QIEA 与 Random Search 使用相同代理设置进行比较。结果显示 QIEA 优于随机搜索，但 GA 和 SA/QUBO-inspired 搜索也具有竞争性，因此文章结论应保持客观。",
            "",
        ]
    )
    search_fields = [
        "method",
        "n",
        "best_candidate",
        "best_fitness",
        "best_ap50_95",
        "best_ap_small",
        "best_recall50",
        "best_fitness_delta_vs_random_pct",
        "top3_delta_vs_random_pct",
    ]
    lines.extend(markdown_table(search_rows, search_fields))
    lines.extend(
        [
            "",
            "建议写法：QIEA 能够在代理搜索空间中找到优于随机搜索的候选结构，说明量子启发式更新机制参与了结构选择；但它与 GA、SA/QUBO-inspired 等启发式方法存在竞争关系。",
            "",
            "## 3. 主实验与多种子结果",
            "",
        ]
    )
    main_fields = [
        "model",
        "n",
        "ap50_95_mean",
        "ap50_95_std",
        "ap_small_mean",
        "ap_small_std",
        "recall50_mean",
        "recall50_std",
        "delta_ap_small_vs_baseline_pct",
        "delta_recall50_vs_baseline_pct",
    ]
    lines.extend(markdown_table(main_rows, main_fields))
    lines.extend(
        [
            "",
            "主结论：P2 的小目标指标提升最稳定；QIEA-Final 的召回倾向存在，但最终 AP50:95 不应被描述为优于 P2。",
            "",
            "## 4. 消融实验",
            "",
        ]
    )
    ablation_fields = ["method", "best_epoch", "ap50_95", "ap50", "ap_small", "recall50", "delta_ap_small_abs"]
    lines.extend(markdown_table(ablation_rows, ablation_fields))
    lines.extend(
        [
            "",
            "## 5. AU-AIR 工程案例",
            "",
            "AU-AIR 用作外部真实无人机数据工程案例。zero-shot 结果较低，说明跨域差异明显；30% 目标域微调后性能提升，说明方法具备一定目标域适应能力。",
            "",
        ]
    )
    auair_fields = [
        "method",
        "zero_shot_ap50_95",
        "finetune_test_ap50_95",
        "finetune_test_ap50",
        "finetune_test_ap_small",
        "finetune_test_recall50",
        "test_ap50_95_gain_from_zero_pct",
    ]
    lines.extend(markdown_table(auair_rows, auair_fields))
    lines.extend(
        [
            "",
            "AU-AIR 结论：+P2 在外部测试集上取得最高 Test AP50:95、AP50、小目标 AP 和 Recall50，适合作为工程补充证据。",
            "",
            "## 6. 可视化图清单",
            "",
            f"已整理 {len(visuals)} 个候选图像文件到 `table_7_visualization_manifest.csv`，并将关键图复制到 `figures/` 子目录。",
            "",
            "建议论文图表安排：",
            "",
            "- Fig. 1 方法流程图。",
            "- Fig. 2 数据集目标尺度分布。",
            "- Fig. 3 主结果/多种子指标图。",
            "- Fig. 4 QIEA 搜索曲线或 Pareto 图。",
            "- Fig. 5 精度-延迟权衡图。",
            "- Fig. 6 无人机检测可视化 2x2 图。",
            "- Table 1 主数据集多种子结果。",
            "- Table 2 消融实验。",
            "- Table 3 效率/部署结果。",
            "- Table 4 搜索算法对比。",
            "- Table 5 AU-AIR 工程案例。",
            "",
            "## 7. 文章中应避免的表述",
            "",
            "- 不要写“QIEA 显著提高最终检测精度”。",
            "- 不要把 CPU-only 测试写成真实边缘设备实测。",
            "- 不要只强调 AP50:95，必须同时解释小目标 AP、Recall50、延迟、FLOPs 与搜索代价。",
            "",
            "## 8. 推荐最终定位",
            "",
            "面向边缘计算约束的无人机小目标检测高分辨率增强与量子启发式轻量结构搜索方法。核心结论是：P2 稳定增强小目标检测，QIEA 提供结构搜索和精度/效率权衡证据，AU-AIR 作为外部工程案例验证目标域适应性。",
        ]
    )
    (OUT_DIR / "submission_evidence_summary_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = build_main_multiseed()
    ablation_rows = build_ablation()
    efficiency_rows = build_efficiency()
    search_rows, _ = build_search_tables()
    auair_rows = build_auair()
    visuals = build_visual_manifest()
    copy_existing_figures()
    write_summary(main_rows, ablation_rows, efficiency_rows, search_rows, auair_rows, visuals)
    manifest = {
        "output_dir": str(OUT_DIR),
        "tables": [
            "table_1_main_multiseed_unified.csv",
            "table_2_ablation_unified.csv",
            "table_3_efficiency_deployment_unified.csv",
            "table_4_search_algorithm_comparison.csv",
            "table_5_search_top_candidates.csv",
            "table_6_auair_engineering_case.csv",
            "table_7_visualization_manifest.csv",
        ],
        "figures": [p.name for p in sorted((OUT_DIR / "figures").glob("*"))] if (OUT_DIR / "figures").exists() else [],
        "summary": "submission_evidence_summary_cn.md",
    }
    (OUT_DIR / "stage25_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
