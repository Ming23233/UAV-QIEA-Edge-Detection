#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTE_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
if str(BYTE_ROOT) not in sys.path:
    sys.path.insert(0, str(BYTE_ROOT))

from yolox.exp import get_exp


METHODS = [
    {
        "method": "baseline_seed42_reused_stage2",
        "label": "Baseline",
        "exp": "exps/example/uav/yolox_nano_visdrone_det_640_baseline.py",
        "ckpt": PROJECT_ROOT / "outputs" / "stage2_visdrone_640_baselines" / "baseline_640_seed42" / "best_ckpt.pth",
    },
    {
        "method": "p2_seed42",
        "label": "+P2",
        "exp": "exps/example/uav/yolox_nano_visdrone_det_640_p2.py",
        "ckpt": PROJECT_ROOT / "outputs" / "stage17_visdrone_640_100ep_correct_multiseed" / "p2_seed42" / "best_ckpt.pth",
    },
    {
        "method": "qiea_final_seed42",
        "label": "QIEA-Final",
        "exp": "exps/example/uav/generated_upgrade/stage6_qiea_final_visdrone_640.py",
        "ckpt": PROJECT_ROOT / "outputs" / "stage6_qiea_final" / "qiea_final_seed42" / "best_ckpt.pth",
    },
    {
        "method": "qiea_lightweight_proxy_winner",
        "label": "QIEA lightweight proxy winner",
        "exp": "exps/example/uav/generated_upgrade/stage5_p2_ca1_csa0_ctx0_fus0_slw125_cr25.py",
        "ckpt": PROJECT_ROOT
        / "outputs"
        / "stage5_proxy_search"
        / "qiea"
        / "p2_ca1_csa0_ctx0_fus0_slw125_cr25"
        / "best_ckpt.pth",
    },
]


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model(exp_file: str, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    exp = get_exp(str(BYTE_ROOT / exp_file), None)
    model = exp.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def profile_flops(model: torch.nn.Module, input_size: int, device: torch.device) -> dict:
    try:
        from thop import profile

        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy,), verbose=False)
        return {
            "macs": float(macs),
            "thop_params": float(params),
            "flops_2x_macs": float(macs) * 2.0,
            "flops_error": "",
        }
    except Exception as exc:
        return {"macs": math.nan, "thop_params": math.nan, "flops_2x_macs": math.nan, "flops_error": repr(exc)}


def measure(model: torch.nn.Module, input_size: int, device: torch.device, warmup: int, iters: int) -> dict:
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "fps_batch1": iters / elapsed,
        "latency_ms_batch1": elapsed / iters * 1000.0,
        "peak_mem_mb_batch1": torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None,
    }


def try_export_onnx(model: torch.nn.Module, output_path: Path, input_size: int, device: torch.device) -> dict:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=12,
            dynamo=False,
            input_names=["images"],
            output_names=["outputs"],
            dynamic_axes=None,
            do_constant_folding=True,
        )
        return {"onnx_path": str(output_path), "onnx_mb": output_path.stat().st_size / (1024**2), "onnx_error": ""}
    except Exception as exc:
        return {"onnx_path": str(output_path), "onnx_mb": math.nan, "onnx_error": repr(exc)}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_outputs(output_dir: Path, rows: list[dict], config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage20_edge_efficiency.json").write_text(
        json.dumps({"stage": "stage20_edge_efficiency", "config": config, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fields = [
        "label",
        "status",
        "params_m",
        "trainable_params_m",
        "macs_g",
        "flops_g_2x_macs",
        "latency_ms_batch1",
        "fps_batch1",
        "peak_mem_mb_batch1",
        "checkpoint_mb",
        "onnx_mb",
        "flops_error",
        "onnx_error",
        "checkpoint",
        "onnx_path",
    ]
    with (output_dir / "stage20_edge_efficiency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    lines = [
        "# Stage 20 Edge-Oriented Efficiency Evaluation",
        "",
        f"Input size: {config['input_size']}x{config['input_size']}; batch size: 1; device: `{config['device']}`; warm-up: {config['warmup']}; measured iterations: {config['iters']}.",
        "",
        "This is an edge-oriented inference proxy evaluation unless executed on an actual edge device.",
        "",
        "| Method | Status | Params (M) | MACs (G) | FLOPs (G) | Latency (ms) | FPS | Peak mem (MB) | CKPT (MB) | ONNX (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row.get('status')} | {fmt(row.get('params_m'))} | {fmt(row.get('macs_g'))} | "
            f"{fmt(row.get('flops_g_2x_macs'))} | {fmt(row.get('latency_ms_batch1'))} | {fmt(row.get('fps_batch1'), 2)} | "
            f"{fmt(row.get('peak_mem_mb_batch1'), 1)} | {fmt(row.get('checkpoint_mb'), 2)} | {fmt(row.get('onnx_mb'), 2)} |"
        )
    (output_dir / "stage20_edge_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Stage 20: edge-oriented model efficiency evaluation.")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "stage20_edge_efficiency")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    for item in METHODS:
        if not item["ckpt"].exists():
            rows.append(
                {
                    "method": item["method"],
                    "label": item["label"],
                    "status": "missing_checkpoint",
                    "checkpoint": str(item["ckpt"]),
                }
            )
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            model = load_model(item["exp"], item["ckpt"], device)
            params, trainable = count_params(model)
            flops = profile_flops(model, args.input_size, device)
            timing = measure(model, args.input_size, device, args.warmup, args.iters)
            onnx = {"onnx_path": "", "onnx_mb": math.nan, "onnx_error": "skipped"}
            if not args.skip_onnx:
                onnx = try_export_onnx(model, args.output_dir / "onnx" / f"{item['method']}.onnx", args.input_size, device)
            rows.append(
                {
                    "method": item["method"],
                    "label": item["label"],
                    "status": "completed",
                    "exp_file": item["exp"],
                    "checkpoint": str(item["ckpt"]),
                    "checkpoint_mb": item["ckpt"].stat().st_size / (1024**2),
                    "input_size": args.input_size,
                    "device": str(device),
                    "params_m": params / 1e6,
                    "trainable_params_m": trainable / 1e6,
                    "macs_g": flops["macs"] / 1e9,
                    "flops_g_2x_macs": flops["flops_2x_macs"] / 1e9,
                    "flops_error": flops["flops_error"],
                    **timing,
                    **onnx,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method": item["method"],
                    "label": item["label"],
                    "status": "failed",
                    "checkpoint": str(item["ckpt"]),
                    "error": repr(exc),
                }
            )
        finally:
            try:
                del model
            except Exception:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_outputs(
        args.output_dir,
        rows,
        {"input_size": args.input_size, "warmup": args.warmup, "iters": args.iters, "device": str(device)},
    )


if __name__ == "__main__":
    main()
