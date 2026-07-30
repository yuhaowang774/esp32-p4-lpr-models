"""
在 CCPD2020 倾斜子集上评估 YOLO 模型的倾斜车牌检测能力
输出指标：召回率、覆盖率、误检率、按倾角分组的精度

使用方法：
  # 直接送原图推理
  python eval_tilted.py --weights phase2_qat/weights/best.pt

  # 模拟设备端 PPA 中心裁切+缩放到 256×256（与设备端实际流程一致）
  python eval_tilted.py --weights phase2_qat/weights/best.pt --simulate-ppa

  # 指定外扩比例
  python eval_tilted.py --weights phase2_qat/weights/best.pt --expand-ratio 1.08
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='Path to YOLO weights')
    p.add_argument('--data-dir', default=os.environ.get("EVAL_OUTPUT_DIR", "./outputs/tilted_eval"))
    p.add_argument('--labels', default=os.path.join(os.environ.get("EVAL_OUTPUT_DIR", "./outputs/tilted_eval"), "eval_labels.txt"))
    p.add_argument('--out', default='eval_results.json')
    p.add_argument('--expand-ratio', type=float, default=1.08, help='bbox 外扩比例')
    p.add_argument('--score-thr', type=float, default=0.55, help='score 阈值')
    p.add_argument('--iou-thr', type=float, default=0.5, help='判定 TP 的 IoU 阈值')
    p.add_argument('--simulate-ppa', action='store_true',
                   help='模拟设备端 PPA 中心裁切+缩放到 256×256（与设备端实际流程一致）')
    return p.parse_args()


def simulate_ppa_center_crop(img):
    """模拟 app_gate.c:603-655 的 PPA 中心裁切+缩放到 MODEL_INPUT_SIZE(320)"""
    h, w = img.shape[:2]
    crop_size = min(w, h)
    crop_x = (w - crop_size) // 2
    crop_y = (h - crop_size) // 2
    cropped = img[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    # 模拟 1/16 量化精度回退（floor((320/crop_size)*16)/16）
    ppa_quant_scale = math.floor((320.0 / crop_size) * 16.0) / 16.0
    actual_out = int(crop_size * ppa_quant_scale)
    scale = crop_size / actual_out  # 反变换时的缩放比
    # 简化：直接 resize 到 320×320
    resized = cv2.resize(cropped, (320, 320))
    # 返回 crop_x, crop_y, scale 供 GT 坐标反向变换使用
    return resized, crop_x, crop_y, scale


def load_gt_labels(labels_path):
    """加载 GT 标签：filename -> (angle, 4 points)"""
    gt = {}
    with open(labels_path, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp, delimiter='\t')
        for row in reader:
            gt[row['filename']] = {
                'angle': int(row['angle']),
                'points': [
                    [int(row['x1']), int(row['y1'])],
                    [int(row['x2']), int(row['y2'])],
                    [int(row['x3']), int(row['y3'])],
                    [int(row['x4']), int(row['y4'])],
                ],
            }
    return gt


def compute_coverage(pred_bbox, gt_4points):
    """覆盖率 = (pred ∩ gt) / gt"""
    gt_x1 = min(p[0] for p in gt_4points)
    gt_y1 = min(p[1] for p in gt_4points)
    gt_x2 = max(p[0] for p in gt_4points)
    gt_y2 = max(p[1] for p in gt_4points)
    inter_x1 = max(pred_bbox[0], gt_x1)
    inter_y1 = max(pred_bbox[1], gt_y1)
    inter_x2 = min(pred_bbox[2], gt_x2)
    inter_y2 = min(pred_bbox[3], gt_y2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    return inter_area / gt_area if gt_area > 0 else 0.0


def compute_iou(pred_bbox, gt_4points):
    """IoU = (pred ∩ gt) / (pred ∪ gt)"""
    gt_x1 = min(p[0] for p in gt_4points)
    gt_y1 = min(p[1] for p in gt_4points)
    gt_x2 = max(p[0] for p in gt_4points)
    gt_y2 = max(p[1] for p in gt_4points)
    inter_x1 = max(pred_bbox[0], gt_x1)
    inter_y1 = max(pred_bbox[1], gt_y1)
    inter_x2 = min(pred_bbox[2], gt_x2)
    inter_y2 = min(pred_bbox[3], gt_y2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union = pred_area + gt_area - inter_area
    return inter_area / union if union > 0 else 0.0


def expand_bbox(bbox, ratio=1.08):
    """对 bbox 进行外扩"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * ratio
    h = (y2 - y1) * ratio
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def angle_bucket(angle):
    a = abs(angle)
    if a < 15:
        return "[10, 15)"
    elif a < 30:
        return "[15, 30)"
    else:
        return "[30, 45]"


def main():
    args = parse_args()
    model = YOLO(args.weights)
    gt_labels = load_gt_labels(args.labels)

    stats = defaultdict(lambda: {'total': 0, 'detected': 0,
                                  'coverage_sum': 0.0, 'iou_sum': 0.0})
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in sorted(Path(args.data_dir).glob('*.jpg')):
        fname = img_path.name
        if fname not in gt_labels:
            continue
        gt = gt_labels[fname]
        bucket = angle_bucket(gt['angle'])
        stats[bucket]['total'] += 1

        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 模拟设备端 PPA 中心裁切（如果启用）
        if args.simulate_ppa:
            resized, crop_x, crop_y, scale = simulate_ppa_center_crop(img)
            result = model(resized, verbose=False)[0]
            # 反向变换 bbox 到原图坐标系（clone 后修改，避免原地修改推理 tensor）
            if len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.clone()
                xyxy[:, 0] += crop_x
                xyxy[:, 1] += crop_y
                xyxy[:, 2] += crop_x
                xyxy[:, 3] += crop_y
                # 存到 result 上的自定义属性供后续使用
                result._xyxy_orig = xyxy
        else:
            result = model(str(img_path), verbose=False)[0]
            if len(result.boxes) > 0:
                result._xyxy_orig = result.boxes.xyxy.clone()

        # 取 score 最高的预测
        best_pred = None
        if len(result.boxes) > 0:
            best_idx = result.boxes.conf.argmax().item()
            if result.boxes.conf[best_idx].item() >= args.score_thr:
                xyxy = result._xyxy_orig[best_idx].tolist()
                best_pred = expand_bbox(xyxy, args.expand_ratio)

        if best_pred is not None:
            iou = compute_iou(best_pred, gt['points'])
            if iou >= args.iou_thr:
                stats[bucket]['detected'] += 1
                total_tp += 1
                stats[bucket]['coverage_sum'] += compute_coverage(best_pred, gt['points'])
                stats[bucket]['iou_sum'] += iou
            else:
                total_fp += 1
        else:
            total_fn += 1

    summary = {'buckets': {}, 'overall': {}}
    coverage_tp_sum = 0.0
    tp_count = 0
    for bucket, s in stats.items():
        if s['total'] == 0:
            continue
        recall = s['detected'] / s['total']
        avg_cov = s['coverage_sum'] / s['detected'] if s['detected'] > 0 else 0
        avg_iou = s['iou_sum'] / s['detected'] if s['detected'] > 0 else 0
        summary['buckets'][bucket] = {
            'total': s['total'],
            'detected': s['detected'],
            'recall': recall,
            'avg_coverage': avg_cov,
            'avg_iou': avg_iou,
        }
        coverage_tp_sum += s['coverage_sum']
        tp_count += s['detected']

    total_det = total_tp + total_fp
    summary['overall'] = {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'recall': total_tp / max(total_tp + total_fn, 1),
        'avg_coverage': coverage_tp_sum / max(tp_count, 1),
        'false_positive_rate': total_fp / max(total_det, 1),
    }

    with open(args.out, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
