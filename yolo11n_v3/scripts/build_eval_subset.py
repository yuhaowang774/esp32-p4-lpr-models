"""
从 CCPD2020 中筛选 |水平倾角| ∈ [10°, 45°] 的样本构建评估子集
目标样本数：5000-10000 张
同时保存 GT 4 点 bbox 信息供覆盖率计算

CCPD 文件名格式（按 - 分隔）：
  颜色_编号 - 水平倾角_垂直倾角 - bbox左上右下2点 - bbox4点 - 字符位置 - 倾斜角 - 亮度
示例：
  blue_00230363984674-89_91-389&508_481&561-484&551_382&545_385&505_487&511-...-58-6.jpg

输出：
  - 图片复制到 OUTPUT_DIR
  - 标签写入 OUTPUT_LABELS（TSV 格式，含 filename/angle/4点 bbox 8个数字）
"""
import os
import re
import random
from shutil import copy2

CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "./outputs/tilted_eval")
OUTPUT_LABELS = os.path.join(OUTPUT_DIR, "eval_labels.txt")
TARGET_COUNT = 8000  # 目标 8000 张


def parse_ccpd_fields(filename):
    """解析 CCPD 文件名，返回 (实际水平倾角, GT 4 点 bbox)"""
    parts = filename.split('-')
    if len(parts) < 4:
        return None, None
    # 第2段：水平倾角_垂直倾角
    angles = parts[1].split('_')
    if len(angles) < 2:
        return None, None
    try:
        h_angle_encoded = int(angles[0])
    except ValueError:
        return None, None
    actual_h_angle = h_angle_encoded - 90  # 实际水平倾角（带符号）
    # 第4段：bbox 4 点，格式 x1&y1_x2&y2_x3&y3_x4&y4
    points_str = parts[3]
    gt_4points = []
    for pt_str in points_str.split('_'):
        try:
            x, y = pt_str.split('&')
            gt_4points.append([int(x), int(y)])
        except (ValueError, IndexError):
            return None, None
    if len(gt_4points) != 4:
        return None, None
    return actual_h_angle, gt_4points


def build_subset():
    if not os.path.isdir(CCPD_DIR):
        raise FileNotFoundError(f"CCPD_DIR not found: {CCPD_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidates = []
    scanned = 0
    for root, _, files in os.walk(CCPD_DIR):
        for f in files:
            if not f.endswith('.jpg'):
                continue
            scanned += 1
            angle, gt_4points = parse_ccpd_fields(f)
            if angle is None or gt_4points is None:
                continue
            if 10 <= abs(angle) <= 45:
                candidates.append((f, os.path.join(root, f), angle, gt_4points))

    print(f"[scan] total_jpg={scanned}, in_10_45_range={len(candidates)}")

    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[:TARGET_COUNT]

    with open(OUTPUT_LABELS, 'w', encoding='utf-8') as fp:
        # 写入表头：文件名 \t 倾角 \t 4点 bbox（8个数字）
        fp.write("filename\tangle\tx1\ty1\tx2\ty2\tx3\ty3\tx4\ty4\n")
        for f, src_path, angle, gt_4points in selected:
            dst_path = os.path.join(OUTPUT_DIR, f)
            copy2(src_path, dst_path)
            pts_flat = "\t".join(str(v) for pt in gt_4points for v in pt)
            fp.write(f"{f}\t{angle}\t{pts_flat}\n")

    print(f"Selected: {len(selected)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Labels file: {OUTPUT_LABELS}")


if __name__ == '__main__':
    build_subset()
