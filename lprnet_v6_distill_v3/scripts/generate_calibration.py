"""V6 蒸馏 V2 量化校准数据生成

功能：
生成 ESP-PPQ INT8 量化校准所需的 calibration_data.npz 文件。

校准数据分布：
- 50% clean 样本（直接 resize 到 128x32）
- 50% letterbox 样本（V9 两步法 letterbox）
匹配实际推理时的输入分布。

校准样本数：2048（从 V6 蒸馏 V1 的 1024 翻倍，提升量化参数估计）
"""

import sys
import os
import numpy as np
import torch

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_v2 import CalibrationDatasetV2
from config import *


def generate_calibration_dataset(output_path=None, num_samples=None,
                                  letterbox_ratio=None, seed=None):
    """生成量化校准数据集并保存为 .npz 文件

    Args:
        output_path: 输出文件路径，默认为 OUTPUT_DIR/calibration_letterbox/calibration_data.npz
        num_samples: 校准样本数，默认 CALIB_NUM_SAMPLES=2048
        letterbox_ratio: letterbox 样本比例，默认 CALIB_LETTERBOX_RATIO=0.5
        seed: 随机种子，默认 CALIB_SEED=42
    """
    if num_samples is None:
        num_samples = CALIB_NUM_SAMPLES
    if letterbox_ratio is None:
        letterbox_ratio = CALIB_LETTERBOX_RATIO
    if seed is None:
        seed = CALIB_SEED

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "calibration_letterbox",
                                    "calibration_data.npz")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating calibration dataset:")
    print(f"  num_samples: {num_samples}")
    print(f"  letterbox_ratio: {letterbox_ratio}")
    print(f"  seed: {seed}")
    print(f"  output: {output_path}")

    # 使用 val.txt 作为校准数据来源（与 V6 蒸馏 V1 一致）
    dataset = CalibrationDatasetV2(
        txt_path=VAL_TXT,
        img_dir=os.path.join(DATA_DIR, "val"),
        num_samples=num_samples,
        letterbox_ratio=letterbox_ratio,
        seed=seed,
    )

    # 收集所有样本为 numpy 数组
    samples = []
    for i in range(len(dataset)):
        samples.append(dataset[i].numpy())
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples")

    # 堆叠为 [N, 3, 32, 128]
    samples_array = np.stack(samples, axis=0).astype(np.float32)
    print(f"  Calibration data shape: {samples_array.shape}")

    # ESP-PPQ 要求 CPU tensor
    np.savez(output_path, images=samples_array)
    print(f"  Saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    generate_calibration_dataset()


if __name__ == "__main__":
    main()
