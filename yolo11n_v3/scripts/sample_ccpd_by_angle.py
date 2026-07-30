"""
从 CCPD2020 中按水平倾角分层采样训练子集
倾角区间：[0°, 15°) / [15°, 30°) / [30°, 45°]
采样比例：50% / 30% / 20%

CCPD 文件名格式：
  颜色_编号-水平倾角_垂直倾角-bbox4点-字符位置-倾斜角-亮度.jpg
  水平倾角编码为 90±实际角度，即编码值 89 = 实际 -1°，91 = 实际 +1°

输出：train_stratified.txt（每行一个图片绝对路径）
"""
import os
import re
from collections import defaultdict

CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
OUTPUT_LIST = "train_stratified.txt"

# 倾角区间与采样比例（绝对值角度）
BUCKETS = [
    (0, 15, 0.50),
    (15, 30, 0.30),
    (30, 46, 0.20),
]


def parse_tilt_angle(filename):
    """从 CCPD 文件名解析水平倾角（绝对值）"""
    # 格式：blue_xxx-89_91-...  其中 89 表示水平倾角（90-1=89，实际倾角 1°）
    match = re.match(r'\w+_\d+-(-?\d+)_(-?\d+)-', filename)
    if not match:
        return None
    h_angle_encoded = int(match.group(1))
    return abs(h_angle_encoded - 90)  # 实际水平倾角绝对值


def stratified_sample():
    if not os.path.isdir(CCPD_DIR):
        raise FileNotFoundError(f"CCPD_DIR not found: {CCPD_DIR}")

    buckets = defaultdict(list)
    total_files = 0
    skipped = 0
    for root, _, files in os.walk(CCPD_DIR):
        for f in files:
            if not f.endswith('.jpg'):
                continue
            total_files += 1
            angle = parse_tilt_angle(f)
            if angle is None:
                skipped += 1
                continue
            for lo, hi, _ in BUCKETS:
                if lo <= angle < hi:
                    buckets[(lo, hi)].append((f, os.path.join(root, f)))
                    break

    print(f"[scan] total_jpg={total_files}, skipped(no_angle)={skipped}")

    sampled = []
    for lo, hi, ratio in BUCKETS:
        items = buckets.get((lo, hi), [])
        n = int(len(items) * ratio)
        sampled.extend(items[:n])
        print(f"[{lo}°, {hi}°): total={len(items)}, sampled={n}")

    with open(OUTPUT_LIST, 'w', encoding='utf-8') as fp:
        for _, path in sampled:
            fp.write(path + '\n')
    print(f"Total sampled: {len(sampled)} -> {OUTPUT_LIST}")


if __name__ == '__main__':
    stratified_sample()
