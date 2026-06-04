#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO


ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ByteTrack"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess


RGB_MEANS = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (140, 140, 140)
GT_COLOR = (60, 200, 60)
BASE_COLOR = (0, 165, 255)
P2_COLOR = (255, 120, 40)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def load_exp_and_model(exp_file: Path, ckpt_file: Path, dataset_root: Path, device: torch.device):
    exp = get_exp(str(exp_file), None)
    exp.data_dir = str(dataset_root)
    model = exp.get_model()
    checkpoint = torch.load(str(ckpt_file), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return exp, model


def load_gt_boxes(coco: COCO, image_id: int):
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    boxes = []
    labels = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(ann["category_id"]))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def preprocess_image(image: np.ndarray, test_size):
    proc, ratio = preproc(image, test_size, RGB_MEANS, STD)
    tensor = torch.from_numpy(proc).unsqueeze(0).float()
    return tensor, float(ratio)


def infer_image(model, exp, device, image: np.ndarray):
    tensor, ratio = preprocess_image(image, exp.test_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        raw_output = model(tensor)
        det_output = postprocess(raw_output.clone(), exp.num_classes, exp.test_conf, exp.nmsthre)[0]
    raw = raw_output[0].detach().cpu()
    det = None if det_output is None else det_output.detach().cpu()
    return raw, det, ratio


def cxcywh_to_xyxy(boxes: torch.Tensor):
    converted = torch.zeros_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    tl = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    return inter / (area1 + area2 - inter + 1e-12)


def raw_matches(raw_output: torch.Tensor, ratio: float, gt_boxes: np.ndarray, topk: int = 100):
    if gt_boxes.shape[0] == 0:
        return {
            "best_ious": np.zeros((0,), dtype=np.float32),
            "best_boxes": np.zeros((0, 4), dtype=np.float32),
            "best_scores": np.zeros((0,), dtype=np.float32),
            "best_classes": np.zeros((0,), dtype=np.int64),
        }

    gt = torch.from_numpy(gt_boxes).float()
    if raw_output.numel() == 0:
        zeros = np.zeros((gt.shape[0],), dtype=np.float32)
        return {
            "best_ious": zeros,
            "best_boxes": np.zeros((gt.shape[0], 4), dtype=np.float32),
            "best_scores": zeros,
            "best_classes": np.zeros((gt.shape[0],), dtype=np.int64),
        }

    class_conf, class_ids = torch.max(raw_output[:, 5:], dim=1)
    scores = raw_output[:, 4] * class_conf
    keep = torch.topk(scores, min(topk, scores.shape[0])).indices
    boxes = cxcywh_to_xyxy(raw_output[keep, :4].clone())
    boxes /= ratio
    ious = box_iou_xyxy(gt, boxes)
    max_iou, max_idx = ious.max(dim=1)
    return {
        "best_ious": max_iou.numpy(),
        "best_boxes": boxes[max_idx].numpy(),
        "best_scores": scores[keep][max_idx].numpy(),
        "best_classes": class_ids[keep][max_idx].numpy(),
    }


def extract_detections(det_output: torch.Tensor, ratio: float, score_thresh: float = 0.02, topn: int = 25):
    if det_output is None or det_output.numel() == 0:
        return []
    boxes = det_output[:, :4].clone() / ratio
    scores = (det_output[:, 4] * det_output[:, 5]).numpy()
    classes = det_output[:, 6].numpy().astype(np.int64)
    order = np.argsort(-scores)
    detections = []
    for idx in order:
        score = float(scores[idx])
        if score < score_thresh:
            continue
        detections.append(
            {
                "box": boxes[idx].numpy(),
                "score": score,
                "cls": int(classes[idx]),
            }
        )
        if len(detections) >= topn:
            break
    return detections


def draw_box(image: np.ndarray, box, color, text=None, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        text_size, baseline = cv2.getTextSize(text, font, scale, 1)
        y_text = max(y1, text_size[1] + 4)
        cv2.rectangle(
            image,
            (x1, y_text - text_size[1] - 4),
            (x1 + text_size[0] + 4, y_text + baseline - 2),
            color,
            -1,
        )
        cv2.putText(image, text, (x1 + 2, y_text - 2), font, scale, WHITE, 1, cv2.LINE_AA)


def render_detection_panel(image: np.ndarray, gt_boxes, detections, det_color, include_gt=False, title=None):
    canvas = image.copy()
    if include_gt:
        for box in gt_boxes:
            draw_box(canvas, box, GT_COLOR, thickness=2)
    for det in detections:
        draw_box(canvas, det["box"], det_color, text=f"{det['score']:.2f}", thickness=2)
    if title:
        cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, BLACK, 2, cv2.LINE_AA)
    return canvas


def fit_tile(image: np.ndarray, width: int, height: int):
    tile = np.full((height, width, 3), 255, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))))
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    tile[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return tile


def put_label(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.62, color=BLACK, thickness: int = 2):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def short_name(file_name: str):
    return Path(file_name).name.replace(".jpg", "").replace(".png", "")


def make_overview_figure(coco, dataset_root: Path, model_payload, records, out_path: Path):
    columns = ["Input", "Ground Truth", "Baseline", "P2"]
    tile_w = 360
    tile_h = 228
    margin = 24
    header_h = 52
    label_h = 34
    rows = len(records)
    canvas = np.full(
        (header_h + rows * (label_h + tile_h) + (rows + 1) * margin, margin * (len(columns) + 1) + len(columns) * tile_w, 3),
        255,
        dtype=np.uint8,
    )
    for col_idx, name in enumerate(columns):
        x = margin + col_idx * (tile_w + margin)
        put_label(canvas, name, x + 8, header_h - 18, scale=0.72)

    for row_idx, record in enumerate(records):
        y_label = header_h + margin + row_idx * (label_h + tile_h + margin)
        y_tile = y_label + label_h
        put_label(
            canvas,
            f"{short_name(record['file_name'])} | mean delta {record['mean_delta']:.3f} | max delta {record['max_delta']:.3f}",
            margin,
            y_label + 22,
            scale=0.58,
        )
        image_path = dataset_root / record["file_name"]
        image = cv2.imread(str(image_path))
        gt_boxes, _ = load_gt_boxes(coco, record["image_id"])
        baseline_raw, baseline_det, baseline_ratio = infer_image(model_payload["baseline"]["model"], model_payload["baseline"]["exp"], model_payload["device"], image)
        p2_raw, p2_det, p2_ratio = infer_image(model_payload["p2"]["model"], model_payload["p2"]["exp"], model_payload["device"], image)
        _ = baseline_raw, p2_raw
        raw_tile = fit_tile(image, tile_w, tile_h)
        gt_tile = fit_tile(render_detection_panel(image, gt_boxes, [], GT_COLOR, include_gt=True), tile_w, tile_h)
        base_tile = fit_tile(render_detection_panel(image, gt_boxes, extract_detections(baseline_det, baseline_ratio, score_thresh=0.05, topn=15), BASE_COLOR), tile_w, tile_h)
        p2_tile = fit_tile(render_detection_panel(image, gt_boxes, extract_detections(p2_det, p2_ratio, score_thresh=0.05, topn=15), P2_COLOR), tile_w, tile_h)
        for col_idx, tile in enumerate([raw_tile, gt_tile, base_tile, p2_tile]):
            x = margin + col_idx * (tile_w + margin)
            canvas[y_tile:y_tile + tile_h, x:x + tile_w] = tile
    cv2.imwrite(str(out_path), canvas)


def crop_region(image: np.ndarray, boxes: list[np.ndarray]):
    valid_boxes = [b for b in boxes if b is not None and np.any(b > 0)]
    if not valid_boxes:
        return image, (0, 0)
    all_boxes = np.vstack(valid_boxes)
    x1 = np.min(all_boxes[:, 0])
    y1 = np.min(all_boxes[:, 1])
    x2 = np.max(all_boxes[:, 2])
    y2 = np.max(all_boxes[:, 3])
    bw = max(24.0, x2 - x1)
    bh = max(24.0, y2 - y1)
    margin = max(18.0, bw * 2.3, bh * 2.3)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_x1 = max(0, int(round(cx - margin / 2.0)))
    crop_y1 = max(0, int(round(cy - margin / 2.0)))
    crop_x2 = min(image.shape[1], int(round(cx + margin / 2.0)))
    crop_y2 = min(image.shape[0], int(round(cy + margin / 2.0)))
    if crop_x2 - crop_x1 < 96:
        pad = (96 - (crop_x2 - crop_x1)) // 2 + 1
        crop_x1 = max(0, crop_x1 - pad)
        crop_x2 = min(image.shape[1], crop_x2 + pad)
    if crop_y2 - crop_y1 < 96:
        pad = (96 - (crop_y2 - crop_y1)) // 2 + 1
        crop_y1 = max(0, crop_y1 - pad)
        crop_y2 = min(image.shape[0], crop_y2 + pad)
    return image[crop_y1:crop_y2, crop_x1:crop_x2].copy(), (crop_x1, crop_y1)


def translate_box(box: np.ndarray, offset):
    if box is None:
        return None
    ox, oy = offset
    return np.asarray([box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy], dtype=np.float32)


def make_crop_figure(coco, dataset_root: Path, model_payload, records, out_path: Path):
    columns = ["Raw Crop", "Ground Truth", "Baseline Match", "P2 Match"]
    tile_w = 300
    tile_h = 260
    margin = 22
    header_h = 52
    label_h = 34
    rows = len(records)
    canvas = np.full(
        (header_h + rows * (label_h + tile_h) + (rows + 1) * margin, margin * (len(columns) + 1) + len(columns) * tile_w, 3),
        255,
        dtype=np.uint8,
    )
    for col_idx, name in enumerate(columns):
        x = margin + col_idx * (tile_w + margin)
        put_label(canvas, name, x + 8, header_h - 18, scale=0.68)

    for row_idx, record in enumerate(records):
        y_label = header_h + margin + row_idx * (label_h + tile_h + margin)
        y_tile = y_label + label_h
        image_path = dataset_root / record["file_name"]
        image = cv2.imread(str(image_path))
        gt_boxes, _ = load_gt_boxes(coco, record["image_id"])
        baseline_raw, _, baseline_ratio = infer_image(model_payload["baseline"]["model"], model_payload["baseline"]["exp"], model_payload["device"], image)
        p2_raw, _, p2_ratio = infer_image(model_payload["p2"]["model"], model_payload["p2"]["exp"], model_payload["device"], image)
        baseline_match = raw_matches(baseline_raw, baseline_ratio, gt_boxes)
        p2_match = raw_matches(p2_raw, p2_ratio, gt_boxes)
        delta = p2_match["best_ious"] - baseline_match["best_ious"]
        gt_idx = int(np.argmax(delta))
        gt_box = gt_boxes[gt_idx]
        base_box = baseline_match["best_boxes"][gt_idx]
        p2_box = p2_match["best_boxes"][gt_idx]
        crop, offset = crop_region(image, [gt_box, base_box, p2_box])
        gt_crop = crop.copy()
        base_crop = crop.copy()
        p2_crop = crop.copy()
        draw_box(gt_crop, translate_box(gt_box, offset), GT_COLOR, thickness=2)
        draw_box(base_crop, translate_box(gt_box, offset), GT_COLOR, thickness=2)
        draw_box(p2_crop, translate_box(gt_box, offset), GT_COLOR, thickness=2)
        draw_box(
            base_crop,
            translate_box(base_box, offset),
            BASE_COLOR,
            text=f"IoU {baseline_match['best_ious'][gt_idx]:.2f}",
            thickness=3,
        )
        draw_box(
            p2_crop,
            translate_box(p2_box, offset),
            P2_COLOR,
            text=f"IoU {p2_match['best_ious'][gt_idx]:.2f}",
            thickness=3,
        )
        label = f"{short_name(record['file_name'])} | target delta {delta[gt_idx]:.3f}"
        put_label(canvas, label, margin, y_label + 22, scale=0.58)
        for col_idx, tile in enumerate(
            [
                fit_tile(crop, tile_w, tile_h),
                fit_tile(gt_crop, tile_w, tile_h),
                fit_tile(base_crop, tile_w, tile_h),
                fit_tile(p2_crop, tile_w, tile_h),
            ]
        ):
            x = margin + col_idx * (tile_w + margin)
            canvas[y_tile:y_tile + tile_h, x:x + tile_w] = tile
    cv2.imwrite(str(out_path), canvas)


def make_dataset_comparison_figure(coco, dataset_root: Path, model_payload, records, out_path: Path, title_prefix: str):
    columns = ["Input", "Ground Truth", "Baseline", "P2"]
    tile_w = 250
    tile_h = 250
    margin = 20
    header_h = 52
    label_h = 32
    rows = len(records)
    canvas = np.full(
        (header_h + rows * (label_h + tile_h) + (rows + 1) * margin, margin * (len(columns) + 1) + len(columns) * tile_w, 3),
        255,
        dtype=np.uint8,
    )
    for col_idx, name in enumerate(columns):
        x = margin + col_idx * (tile_w + margin)
        put_label(canvas, name, x + 8, header_h - 18, scale=0.68)

    for row_idx, record in enumerate(records):
        y_label = header_h + margin + row_idx * (label_h + tile_h + margin)
        y_tile = y_label + label_h
        image_path = dataset_root / record["file_name"]
        image = cv2.imread(str(image_path))
        gt_boxes, _ = load_gt_boxes(coco, record["image_id"])
        baseline_raw, baseline_det, baseline_ratio = infer_image(model_payload["baseline"]["model"], model_payload["baseline"]["exp"], model_payload["device"], image)
        p2_raw, p2_det, p2_ratio = infer_image(model_payload["p2"]["model"], model_payload["p2"]["exp"], model_payload["device"], image)
        _ = baseline_raw, p2_raw
        put_label(canvas, f"{title_prefix} {short_name(record['file_name'])}", margin, y_label + 22, scale=0.58)
        tiles = [
            fit_tile(image, tile_w, tile_h),
            fit_tile(render_detection_panel(image, gt_boxes, [], GT_COLOR, include_gt=True), tile_w, tile_h),
            fit_tile(render_detection_panel(image, gt_boxes, extract_detections(baseline_det, baseline_ratio, score_thresh=0.05, topn=10), BASE_COLOR), tile_w, tile_h),
            fit_tile(render_detection_panel(image, gt_boxes, extract_detections(p2_det, p2_ratio, score_thresh=0.05, topn=10), P2_COLOR), tile_w, tile_h),
        ]
        for col_idx, tile in enumerate(tiles):
            x = margin + col_idx * (tile_w + margin)
            canvas[y_tile:y_tile + tile_h, x:x + tile_w] = tile
    cv2.imwrite(str(out_path), canvas)


def summarize_images(coco, dataset_root: Path, model_payload):
    records = []
    image_ids = sorted(coco.imgs.keys())
    total = len(image_ids)
    for idx, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs([image_id])[0]
        image = cv2.imread(str(dataset_root / info["file_name"]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {dataset_root / info['file_name']}")
        gt_boxes, _ = load_gt_boxes(coco, image_id)
        if gt_boxes.shape[0] == 0:
            continue
        baseline_raw, _, baseline_ratio = infer_image(model_payload["baseline"]["model"], model_payload["baseline"]["exp"], model_payload["device"], image)
        p2_raw, _, p2_ratio = infer_image(model_payload["p2"]["model"], model_payload["p2"]["exp"], model_payload["device"], image)
        baseline_match = raw_matches(baseline_raw, baseline_ratio, gt_boxes)
        p2_match = raw_matches(p2_raw, p2_ratio, gt_boxes)
        gt_wh = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]
        gt_area = gt_wh[:, 0] * gt_wh[:, 1]
        delta = p2_match["best_ious"] - baseline_match["best_ious"]
        records.append(
            {
                "image_id": int(image_id),
                "file_name": info["file_name"],
                "n_gt": int(gt_boxes.shape[0]),
                "small_ratio": float(np.mean(gt_area <= (32.0 * 32.0))),
                "baseline_mean_iou": float(np.mean(baseline_match["best_ious"])),
                "p2_mean_iou": float(np.mean(p2_match["best_ious"])),
                "mean_delta": float(np.mean(delta)),
                "max_delta": float(np.max(delta)),
            }
        )
        if idx % 50 == 0 or idx == total:
            print(f"[progress] processed {idx}/{total} images from {dataset_root}")
    return records


def choose_records(records, min_gt: int, count: int, key: str):
    filtered = [r for r in records if r["n_gt"] >= min_gt and r[key] > 0]
    if len(filtered) < count:
        filtered = [r for r in records if r[key] > 0]
    filtered.sort(key=lambda x: (x[key], x["mean_delta"], x["small_ratio"], x["n_gt"]), reverse=True)
    return filtered[:count]


def main():
    parser = argparse.ArgumentParser("Generate qualitative detection figures for the LaTeX paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--paper-fig-dir", default="paper_latex/figures")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1].parent
    out_dir = workspace / args.paper_fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    vis_dataset = workspace / "uav_tdmnet" / "data" / "visdrone_det_coco"
    vis_coco = COCO(str(vis_dataset / "annotations" / "instances_val.json"))
    vis_baseline_exp, vis_baseline_model = load_exp_and_model(
        workspace / "uav_tdmnet" / "third_party" / "ByteTrack" / "exps" / "example" / "uav" / "yolox_nano_visdrone_det_fast_baseline.py",
        workspace / "uav_tdmnet" / "runs_visdrone_full_fast" / "baseline" / "best_ckpt.pth",
        vis_dataset,
        device,
    )
    vis_p2_exp, vis_p2_model = load_exp_and_model(
        workspace / "uav_tdmnet" / "third_party" / "ByteTrack" / "exps" / "example" / "uav" / "yolox_nano_visdrone_det_fast_p2.py",
        workspace / "uav_tdmnet" / "runs_visdrone_full_fast" / "p2" / "best_ckpt.pth",
        vis_dataset,
        device,
    )
    vis_models = {
        "device": device,
        "baseline": {"exp": vis_baseline_exp, "model": vis_baseline_model},
        "p2": {"exp": vis_p2_exp, "model": vis_p2_model},
    }

    vis_records = summarize_images(vis_coco, vis_dataset / "images", vis_models)
    vis_overview = choose_records(vis_records, min_gt=10, count=3, key="mean_delta")
    vis_crops = choose_records(vis_records, min_gt=6, count=4, key="max_delta")

    tiny_dataset = workspace / "uav_tdmnet" / "data" / "tiny_uav_coco"
    tiny_coco = COCO(str(tiny_dataset / "annotations" / "instances_val.json"))
    tiny_baseline_exp, tiny_baseline_model = load_exp_and_model(
        workspace / "uav_tdmnet" / "third_party" / "ByteTrack" / "exps" / "example" / "uav" / "yolox_nano_uav_tiny_baseline.py",
        workspace / "uav_tdmnet" / "runs_long" / "baseline_20ep" / "best_ckpt.pth",
        tiny_dataset,
        device,
    )
    tiny_p2_exp, tiny_p2_model = load_exp_and_model(
        workspace / "uav_tdmnet" / "third_party" / "ByteTrack" / "exps" / "example" / "uav" / "yolox_nano_uav_tiny_p2.py",
        workspace / "uav_tdmnet" / "runs_long" / "p2_20ep" / "best_ckpt.pth",
        tiny_dataset,
        device,
    )
    tiny_models = {
        "device": device,
        "baseline": {"exp": tiny_baseline_exp, "model": tiny_baseline_model},
        "p2": {"exp": tiny_p2_exp, "model": tiny_p2_model},
    }
    tiny_records = summarize_images(tiny_coco, tiny_dataset / "val", tiny_models)
    tiny_selected = choose_records(tiny_records, min_gt=1, count=3, key="mean_delta")

    make_overview_figure(vis_coco, vis_dataset / "images", vis_models, vis_overview, out_dir / "visdrone_detection_overview.png")
    make_crop_figure(vis_coco, vis_dataset / "images", vis_models, vis_crops, out_dir / "visdrone_detection_crops.png")
    make_dataset_comparison_figure(tiny_coco, tiny_dataset / "val", tiny_models, tiny_selected, out_dir / "synthetic_detection_compare.png", "Synthetic")

    payload = {
        "visdrone_overview": vis_overview,
        "visdrone_crops": vis_crops,
        "synthetic_selected": tiny_selected,
    }
    (out_dir / "qualitative_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
