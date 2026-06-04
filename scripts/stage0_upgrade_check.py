#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTE_ROOT = PROJECT_ROOT / "third_party" / "ByteTrack"
if str(BYTE_ROOT) not in sys.path:
    sys.path.insert(0, str(BYTE_ROOT))

from yolox.exp import get_exp


EXPS = [
    "exps/example/uav/yolox_nano_visdrone_det_640_baseline.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_p2.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_full.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_p2_ca.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_p2_csa.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_p2_context.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_p2_fusion.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_p2_loss_assignment.py",
    "exps/example/uav/yolox_nano_visdrone_det_640_qiea_final.py",
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def dataset_status(root):
    root = Path(root)
    anns = root / "annotations"
    imgs = root / "images"
    return {
        "root": str(root),
        "exists": root.exists(),
        "images_exists": imgs.exists(),
        "annotations_exists": anns.exists(),
        "instances_train": str(anns / "instances_train.json"),
        "instances_train_exists": (anns / "instances_train.json").exists(),
        "instances_val": str(anns / "instances_val.json"),
        "instances_val_exists": (anns / "instances_val.json").exists(),
    }


def run_smoke(python, dataset_root, output_dir):
    cmd = [
        python,
        str(BYTE_ROOT / "tools" / "train_uav_stage1.py"),
        "-f",
        "exps/example/uav/yolox_nano_visdrone_det_640_p2.py",
        "--device",
        "cuda",
        "--batch-size",
        "4",
        "--epochs",
        "1",
        "--lr",
        "0.0015",
        "--seed",
        "42",
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BYTE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd,
        cwd=str(BYTE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--visdrone-root", required=True)
    parser.add_argument("--uavdt-root", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "stage0_env_check"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_rows = []
    for exp_file in EXPS:
        exp = get_exp(str(BYTE_ROOT / exp_file), None)
        model = exp.get_model().to(device)
        model.eval()
        dummy = torch.zeros(1, 3, exp.test_size[0], exp.test_size[1], device=device)
        with torch.no_grad():
            out = model(dummy)
        exp_rows.append({
            "exp_file": exp_file,
            "input_size": list(exp.input_size),
            "test_size": list(exp.test_size),
            "max_epoch": exp.max_epoch,
            "use_p2": bool(getattr(exp, "use_p2", False)),
            "coord_attention": bool(getattr(exp, "coord_attention", False)),
            "channel_spatial_attention": bool(getattr(exp, "channel_spatial_attention", False)),
            "tiny_context": bool(getattr(exp, "tiny_context", False)),
            "scale_aware_fusion": bool(getattr(exp, "scale_aware_fusion", False)),
            "small_obj_loss_weight": float(getattr(exp, "small_obj_loss_weight", 1.0)),
            "center_radius": float(getattr(exp, "center_radius", 2.5)),
            "params": count_params(model),
            "forward_output_shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    smoke_dir = output_dir / "p2_640_smoke"
    smoke = run_smoke(args.python, args.visdrone_root, smoke_dir)

    report = {
        "stage": "Stage 0 upgrade environment and smoke check",
        "python": args.python,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "visdrone": dataset_status(args.visdrone_root),
        "uavdt": dataset_status(args.uavdt_root),
        "experiments": exp_rows,
        "smoke": smoke,
    }
    (output_dir / "stage0_env_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Stage 0 Upgrade Environment Check",
        "",
        f"Python: `{args.python}`",
        f"Torch: `{torch.__version__}`",
        f"CUDA available: `{torch.cuda.is_available()}`",
        f"CUDA version: `{torch.version.cuda}`",
        f"GPU: `{report['gpu']}`",
        "",
        "## Dataset Paths",
        "",
        f"- VisDrone: `{args.visdrone_root}`",
        f"- UAVDT: `{args.uavdt_root}`",
        "",
        "## 640 Experiment Configs",
        "",
        "| Exp | Input | P2 | CA | CSA | Context | Fusion | LossW | Radius | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in exp_rows:
        lines.append(
            f"| {Path(row['exp_file']).name} | {row['input_size'][0]} | {int(row['use_p2'])} | "
            f"{int(row['coord_attention'])} | {int(row['channel_spatial_attention'])} | "
            f"{int(row['tiny_context'])} | {int(row['scale_aware_fusion'])} | "
            f"{row['small_obj_loss_weight']:.2f} | {row['center_radius']:.1f} | {row['params']} |"
        )
    lines.extend([
        "",
        "## Smoke Test",
        "",
        f"Return code: `{smoke['returncode']}`",
        f"Output dir: `{smoke_dir}`",
    ])
    if smoke["returncode"] == 0:
        lines.append("")
        lines.append("Stage 0 smoke test passed.")
    else:
        lines.append("")
        lines.append("Stage 0 smoke test failed. See JSON stdout/stderr tails.")
    (output_dir / "stage0_env_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "stage0_env_check.md")
    raise SystemExit(smoke["returncode"])


if __name__ == "__main__":
    main()
