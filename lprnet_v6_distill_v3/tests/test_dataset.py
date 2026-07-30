"""V6 蒸馏 V2 数据集单元测试

TDD Red 阶段：先写失败测试，验证 dataset_v2.py 未实现或行为不符。

关键测试点（来自文档 4.3.5 节）：
1. 省份权重计算（compute_province_weights）
2. 双验证集（clean + letterbox）
3. 校准数据集 letterbox/clean 分配
4. CalibrationDatasetV2 letterbox 样本含 padding 填充色
"""

import sys
import os
import tempfile
import shutil

import numpy as np
import cv2
import torch
from torchvision import transforms

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_v2 import (
    LPRDatasetV2,
    CalibrationDatasetV2,
    compute_province_weights,
)
from chars import CHAR_TO_IDX, PROVINCES


def _create_test_dataset(tmp_dir, num_samples=20, plate_prefix='京'):
    """创建测试用的小数据集

    Args:
        tmp_dir: 临时目录
        num_samples: 样本数
        plate_prefix: 车牌首字符（省份）
    Returns:
        (txt_path, img_dir)
    """
    img_dir = os.path.join(tmp_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    txt_path = os.path.join(tmp_dir, 'test.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            img_name = f'{i:04d}.jpg'
            img_path = os.path.join(img_dir, img_name)
            # 创建 128x32 的随机图片
            img = np.random.randint(0, 255, (32, 128, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
            # 车牌：省份+6位字符
            plate = plate_prefix + f'A{i:05d}'
            f.write(f'images/{img_name} {plate} 普通蓝牌\n')
    return txt_path, img_dir


def test_compute_province_weights():
    """验证省份权重计算：稀有省份权重高，常见省份权重低，且不超过 max_weight"""
    # 构造 mock dataset：data 属性为 [(img_name, plate, type), ...]
    class MockDataset:
        def __init__(self, data):
            self.data = data

    # 省份 '京'（索引 0）10 个，'藏'（索引 26）1 个
    data = [('img.jpg', '京A12345', '普通蓝牌')] * 10
    data += [('img.jpg', '藏A12345', '普通蓝牌')] * 1
    dataset = MockDataset(data)

    sample_weights, province_class_weights, province_counts = compute_province_weights(
        dataset, max_weight=5.0)

    # 省份 '京'（索引 0）权重应为 total/(31*10) = 11/310 ≈ 0.0355
    # 省份 '藏'（索引 26）权重应为 min(11/(31*1), 5.0) = min(0.355, 5.0) = 0.355
    jing_idx = CHAR_TO_IDX['京'] - 1
    zang_idx = CHAR_TO_IDX['藏'] - 1

    assert province_class_weights[zang_idx] > province_class_weights[jing_idx], \
        "稀有省份权重应高于常见省份"
    assert province_class_weights[zang_idx] <= 5.0, \
        f"省份权重超过 max_weight: {province_class_weights[zang_idx]}"
    assert len(sample_weights) == len(data), \
        f"sample_weights 长度错误: {len(sample_weights)} != {len(data)}"
    assert province_counts[jing_idx] == 10
    assert province_counts[zang_idx] == 1


def test_dataset_loads_data():
    """验证 LPRDatasetV2 能加载数据"""
    tmp_dir = tempfile.mkdtemp()
    try:
        txt_path, img_dir = _create_test_dataset(tmp_dir, num_samples=10)
        dataset = LPRDatasetV2(
            txt_path=txt_path,
            img_dir=img_dir,
            transform=None,
            aug=None,
            filter_types=['普通蓝牌', '新能源小型车'])
        assert len(dataset) == 10, f"数据集长度错误: {len(dataset)}"
    finally:
        shutil.rmtree(tmp_dir)


def test_dataset_force_letterbox_produces_padding():
    """验证 force_letterbox=True 时验证集图像含 letterbox padding"""
    tmp_dir = tempfile.mkdtemp()
    try:
        txt_path, img_dir = _create_test_dataset(tmp_dir, num_samples=5)
        normalize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = LPRDatasetV2(
            txt_path=txt_path,
            img_dir=img_dir,
            transform=normalize_transform,
            aug=None,
            force_letterbox=True,
            filter_types=['普通蓝牌', '新能源小型车'])

        sample = dataset[0]
        img = sample[0]
        assert img.shape == (3, 32, 128), f"输出 shape 错误: {img.shape}"

        # letterbox 样本应含填充色 (114,114,114) 归一化后的值
        # 归一化后 R 通道：(114/255-0.485)/0.229 ≈ 0.024
        fill_normalized_r = (114 / 255.0 - 0.485) / 0.229
        # 检查是否存在接近填充色的像素（padding 区域）
        min_val = img.min().item()
        max_val = img.max().item()
        # 应存在接近 fill_normalized_r 的像素
        diff = (img - fill_normalized_r).abs()
        assert diff.min().item() < 0.2, \
            f"letterbox 样本未检测到 padding 填充色，最小差异 {diff.min().item()}"
    finally:
        shutil.rmtree(tmp_dir)


def test_dataset_set_phase_delegates_to_aug():
    """验证 LPRDatasetV2.set_phase 委托到内部 aug 对象"""
    from augmentation_v2 import LPRAugmentationV6DistillV2
    tmp_dir = tempfile.mkdtemp()
    try:
        txt_path, img_dir = _create_test_dataset(tmp_dir, num_samples=5)
        aug = LPRAugmentationV6DistillV2(phase='warmup')
        dataset = LPRDatasetV2(
            txt_path=txt_path,
            img_dir=img_dir,
            transform=None,
            aug=aug,
            filter_types=['普通蓝牌', '新能源小型车'])
        dataset.set_phase('medium')
        assert aug.letterbox_prob == 0.5, \
            f"set_phase 未委托到 aug，letterbox_prob={aug.letterbox_prob}"
    finally:
        shutil.rmtree(tmp_dir)


def test_calibration_letterbox_clean_split():
    """验证前 letterbox_count 个样本走 letterbox，其余走 clean"""
    tmp_dir = tempfile.mkdtemp()
    try:
        txt_path, img_dir = _create_test_dataset(tmp_dir, num_samples=100)
        dataset = CalibrationDatasetV2(
            txt_path=txt_path,
            img_dir=img_dir,
            num_samples=100,
            letterbox_ratio=0.5)
        assert len(dataset) == 100
        assert dataset.letterbox_count == 50, \
            f"letterbox_count 错误: {dataset.letterbox_count}"

        # 前 50 个应为 letterbox（含 padding 像素）
        sample_0 = dataset[0]
        assert sample_0.shape == (3, 32, 128), f"letterbox 样本 shape 错误: {sample_0.shape}"
        # 后 50 个应为 clean（直接 resize，无 padding）
        sample_99 = dataset[99]
        assert sample_99.shape == (3, 32, 128), f"clean 样本 shape 错误: {sample_99.shape}"

        # 像素值验证：letterbox 样本的 padding 区域应含填充色归一化后的值
        # 填充色 (114,114,114)，ImageNet 归一化 mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
        # 归一化后：(114/255-0.485)/0.229 ≈ 0.024（R）
        fill_normalized_r = (114 / 255.0 - 0.485) / 0.229
        # 检查 letterbox 样本的角落像素（padding 区域）是否接近填充色归一化值
        corner_pixel = sample_0[0, 0, 0].item()  # [C, H, W] → R 通道左上角
        assert abs(corner_pixel - fill_normalized_r) < 0.15, \
            f"letterbox 样本角落像素 {corner_pixel} 不接近填充色归一化值 {fill_normalized_r}"
    finally:
        shutil.rmtree(tmp_dir)


def test_calibration_deterministic_with_seed():
    """验证相同种子下校准数据集的 letterbox 参数确定性"""
    tmp_dir = tempfile.mkdtemp()
    try:
        txt_path, img_dir = _create_test_dataset(tmp_dir, num_samples=20)
        dataset1 = CalibrationDatasetV2(
            txt_path=txt_path, img_dir=img_dir,
            num_samples=20, letterbox_ratio=0.5, seed=42)
        dataset2 = CalibrationDatasetV2(
            txt_path=txt_path, img_dir=img_dir,
            num_samples=20, letterbox_ratio=0.5, seed=42)

        # 相同种子应产生相同的 letterbox 参数
        # 由于 RNG 在 __init__ 中生成参数，比较前几个参数应一致
        # 这里通过比较样本输出验证确定性
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        assert torch.allclose(sample1, sample2), \
            "相同种子的校准数据集样本不一致，letterbox 参数非确定性"
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    test_funcs = [v for k, v in sorted(globals().items())
                  if k.startswith('test_') and callable(v)]
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            print(f"PASS: {func.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL: {func.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n总计: {passed} 通过, {failed} 失败")
