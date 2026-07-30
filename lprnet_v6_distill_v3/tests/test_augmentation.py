"""V6 蒸馏 V2 数据增强单元测试

TDD Red 阶段：先写失败测试，验证 augmentation_v2.py 未实现或行为不符。
"""

import sys
import os
import numpy as np

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation_v2 import (
    letterbox_resize,
    right_side_padding_augment,
    LPRAugmentationV6DistillV2,
    simulate_yolo_crop_letterbox,
)


def test_letterbox_fill_color_and_size():
    """验证 letterbox 输出尺寸 128x32 且填充色为 (114,114,114)"""
    img = np.zeros((40, 100, 3), dtype=np.uint8)  # 非标准比例输入（宽高比 2.5）
    result = letterbox_resize(img, target_w=128, target_h=32,
                              fill_color=(114, 114, 114))
    assert result.shape == (32, 128, 3), f"输出尺寸错误: {result.shape}"
    # 检查填充区域像素值（左上角应为填充色）
    assert result[0, 0, 0] == 114, f"R 通道填充色错误: {result[0, 0, 0]}"
    assert result[0, 0, 1] == 114, f"G 通道填充色错误: {result[0, 0, 1]}"
    assert result[0, 0, 2] == 114, f"B 通道填充色错误: {result[0, 0, 2]}"


def test_letterbox_produces_padding_for_nonstandard_aspect():
    """验证 letterbox 对非标准宽高比输入确实产生 padding（非均匀缩放退化）"""
    # 宽高比 2.5（< 4.0），letterbox 后高度填满，宽度方向有左右 padding
    img = np.full((40, 100, 3), 200, dtype=np.uint8)
    result = letterbox_resize(img, target_w=128, target_h=32,
                              fill_color=(114, 114, 114))
    # scale=min(128/100, 32/40)=0.8，new_w=80, new_h=32，左右各有 24px padding
    left_col = result[:, 0, :]
    assert np.all(left_col == 114), "宽高比 2.5 输入应产生左侧 padding"
    right_col = result[:, -1, :]
    assert np.all(right_col == 114), "宽高比 2.5 输入应产生右侧 padding"

    # 宽高比 5.0（> 4.0），letterbox 后宽度填满，高度方向有上下 padding
    img2 = np.full((20, 100, 3), 200, dtype=np.uint8)
    result2 = letterbox_resize(img2, target_w=128, target_h=32,
                               fill_color=(114, 114, 114))
    top_row = result2[0, :, :]
    assert np.all(top_row == 114), "宽高比 5.0 输入应产生顶部 padding"


def test_right_side_padding_max_ratio():
    """验证右侧 padding 不超过 max_pad_ratio 比例"""
    img = np.zeros((32, 128, 3), dtype=np.uint8)
    result = right_side_padding_augment(img, prob=1.0, max_pad_ratio=0.15)
    pad_width = result.shape[1] - 128
    assert pad_width > 0, "padding 未生效，pad_width=0"
    assert pad_width <= int(128 * 0.15), f"padding 超过上限: {pad_width} > {int(128 * 0.15)}"


def test_right_side_padding_fill_color():
    """验证右侧 padding 填充色为 (114,114,114)"""
    img = np.zeros((32, 128, 3), dtype=np.uint8)
    result = right_side_padding_augment(img, prob=1.0, max_pad_ratio=0.15,
                                        fill_color=(114, 114, 114))
    pad_width = result.shape[1] - 128
    # padding 区域应全为填充色
    pad_region = result[:, 128:, :]
    assert np.all(pad_region == 114), f"padding 区域填充色错误: {pad_region[0, 0]}"


def test_right_side_padding_prob_zero():
    """验证 prob=0 时右侧 padding 不生效"""
    img = np.zeros((32, 128, 3), dtype=np.uint8)
    result = right_side_padding_augment(img, prob=0.0, max_pad_ratio=0.15)
    assert result.shape == (32, 128, 3), "prob=0 时不应产生 padding"


def test_simulate_yolo_crop_letterbox_produces_padding():
    """验证 V9 两步法 letterbox 对 128x32 输入产生 padding（关键：均匀缩放会退化）"""
    # 128x32 输入，宽高比 4.0，简单 letterbox scale=1.0 不产生 padding
    # V9 两步法：先 resize 到随机宽高比，再 letterbox 回 128x32，必然产生 padding
    img = np.full((32, 128, 3), 200, dtype=np.uint8)
    # 固定随机参数验证
    result = simulate_yolo_crop_letterbox(
        img, aspect_ratio=3.0, target_h=32,
        fill_color=(114, 114, 114))
    assert result.shape == (32, 128, 3), f"输出尺寸错误: {result.shape}"
    # aspect_ratio=3.0 < 4.0，letterbox 后应有 padding
    # 检查是否存在填充色像素
    has_padding = np.any(result == 114)
    assert has_padding, "V9 两步法 letterbox 未产生 padding（可能退化为均匀缩放）"


def test_augmentation_set_phase_changes_letterbox_prob():
    """验证 set_phase 切换 letterbox_prob"""
    aug = LPRAugmentationV6DistillV2(phase='warmup')
    assert aug.letterbox_prob == 0.3, f"warmup letterbox_prob 应为 0.3, 实际 {aug.letterbox_prob}"

    aug.set_phase('medium')
    assert aug.letterbox_prob == 0.5, f"medium letterbox_prob 应为 0.5, 实际 {aug.letterbox_prob}"

    aug.set_phase('full')
    assert aug.letterbox_prob == 0.5, f"full letterbox_prob 应为 0.5, 实际 {aug.letterbox_prob}"


def test_augmentation_output_size():
    """验证增强流水线输出尺寸始终为 128x32"""
    img = np.zeros((32, 128, 3), dtype=np.uint8)
    label = [1, 2, 3, 4, 5, 6, 7]
    aug = LPRAugmentationV6DistillV2(phase='full')
    # 强制走 letterbox 路径
    aug.letterbox_prob = 1.0
    result, _ = aug(img, label)
    assert result.shape == (32, 128, 3), f"full 阶段 letterbox 输出尺寸错误: {result.shape}"

    # 强制走直接 resize 路径
    aug.letterbox_prob = 0.0
    result, _ = aug(img, label)
    assert result.shape == (32, 128, 3), f"full 阶段直接 resize 输出尺寸错误: {result.shape}"


def test_augmentation_warmup_no_geometric_aug():
    """验证 warmup 阶段不启用几何/裁切/遮挡/噪声增强（仅 letterbox）"""
    img = np.full((32, 128, 3), 200, dtype=np.uint8)
    label = [1, 2, 3, 4, 5, 6, 7]
    aug = LPRAugmentationV6DistillV2(phase='warmup')
    aug.letterbox_prob = 0.0  # 关闭 letterbox，验证其他增强不改变图像
    result, _ = aug(img, label)
    # warmup 阶段无增强 + letterbox 关闭，应返回原始图像（仅可能 resize）
    assert result.shape == (32, 128, 3)
    # 像素值应保持不变（无几何/光照/噪声增强）
    assert np.all(result == 200), "warmup 阶段不应启用其他增强"


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
            print(f"FAIL: {func.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n总计: {passed} 通过, {failed} 失败")
