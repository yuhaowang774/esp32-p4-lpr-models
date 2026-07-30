"""
统计蓝牌和绿牌的bbox尺寸分布、面积分布、宽高比分布
用于分析蓝牌mAP50-95低的原因
"""
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

TRAIN_LBL_DIR = str(Path(CCPD_AUG_DIR) / "labels" / "train")
VAL_LBL_DIR = str(Path(CCPD_AUG_DIR) / "labels" / "val")


def parse_label(lbl_path):
    """读取YOLO标签，返回 [(cls, cx, cy, w, h), ...]"""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls, cx, cy, w, h))
    return boxes


def analyze_dir(lbl_dir, name):
    print(f"\n{'='*60}")
    print(f"Analyzing {name}: {lbl_dir}")
    print(f"{'='*60}")

    # 排除增强样本（_x2, _x3, _x4后缀），只统计原始样本
    blue_areas = []
    green_areas = []
    blue_aspect = []
    green_aspect = []
    blue_w = []
    green_w = []
    blue_h = []
    green_h = []
    blue_count = 0
    green_count = 0
    blue_aug_count = 0
    green_aug_count = 0

    lbl_files = list(Path(lbl_dir).glob("*.txt"))

    for lbl_path in lbl_files:
        fname = lbl_path.name
        # 判断是否为增强样本
        is_aug = '_x2' in fname or '_x3' in fname or '_x4' in fname

        boxes = parse_label(str(lbl_path))
        for cls, cx, cy, w, h in boxes:
            area = w * h
            aspect = w / h if h > 0 else 0
            if cls == 0:  # blue_plate
                if is_aug:
                    blue_aug_count += 1
                else:
                    blue_count += 1
                    blue_areas.append(area)
                    blue_aspect.append(aspect)
                    blue_w.append(w)
                    blue_h.append(h)
            else:  # green_plate
                if is_aug:
                    green_aug_count += 1
                else:
                    green_count += 1
                    green_areas.append(area)
                    green_aspect.append(aspect)
                    green_w.append(w)
                    green_h.append(h)

    print(f"\n样本数量（原始 / 增强x2_x3_x4）:")
    print(f"  blue_plate: {blue_count} / {blue_aug_count}")
    print(f"  green_plate: {green_count} / {green_aug_count}")

    if blue_count > 0 and green_count > 0:
        print(f"\n原始样本bbox统计:")
        print(f"  {'指标':<20} {'blue_plate':<25} {'green_plate':<25}")
        print(f"  {'-'*70}")
        print(f"  {'面积均值':<20} {np.mean(blue_areas):<25.6f} {np.mean(green_areas):<25.6f}")
        print(f"  {'面积中位数':<20} {np.median(blue_areas):<25.6f} {np.median(green_areas):<25.6f}")
        print(f"  {'面积std':<20} {np.std(blue_areas):<25.6f} {np.std(green_areas):<25.6f}")
        print(f"  {'宽均值':<20} {np.mean(blue_w):<25.6f} {np.mean(green_w):<25.6f}")
        print(f"  {'宽中位数':<20} {np.median(blue_w):<25.6f} {np.median(green_w):<25.6f}")
        print(f"  {'高均值':<20} {np.mean(blue_h):<25.6f} {np.mean(green_h):<25.6f}")
        print(f"  {'高中位数':<20} {np.median(blue_h):<25.6f} {np.median(green_h):<25.6f}")
        print(f"  {'宽高比均值':<20} {np.mean(blue_aspect):<25.4f} {np.mean(green_aspect):<25.4f}")
        print(f"  {'宽高比中位数':<20} {np.median(blue_aspect):<25.4f} {np.median(green_aspect):<25.4f}")

        # 面积分桶
        print(f"\n面积分桶（归一化面积）:")
        bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        for i in range(len(bins)-1):
            lo, hi = bins[i], bins[i+1]
            b_cnt = sum(1 for a in blue_areas if lo <= a < hi)
            g_cnt = sum(1 for a in green_areas if lo <= a < hi)
            print(f"  [{lo:.3f}, {hi:.3f}): blue={b_cnt:<8} green={g_cnt:<8}")

        # 小目标占比
        small_thresh = 0.01  # 面积<1%算小目标
        b_small = sum(1 for a in blue_areas if a < small_thresh)
        g_small = sum(1 for a in green_areas if a < small_thresh)
        print(f"\n小目标占比（面积<{small_thresh}）:")
        print(f"  blue_plate: {b_small}/{blue_count} = {b_small/blue_count*100:.2f}%")
        print(f"  green_plate: {g_small}/{green_count} = {g_small/green_count*100:.2f}%")


if __name__ == '__main__':
    analyze_dir(TRAIN_LBL_DIR, "TRAIN")
    analyze_dir(VAL_LBL_DIR, "VAL")
