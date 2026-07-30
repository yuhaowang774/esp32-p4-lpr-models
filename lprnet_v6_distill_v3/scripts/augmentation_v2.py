"""V6 蒸馏 V2 数据增强

包含：
- letterbox_resize: letterbox 到目标尺寸（填充色 114,114,114）
- simulate_yolo_crop_letterbox: V9 两步法（先 resize 到随机宽高比，再 letterbox 回 128x32）
- right_side_padding_augment: 右侧 padding 增强
- LPRAugmentationV6DistillV2: 5 组增强 + letterbox + 渐进增强

关键约束：
- letterbox 必须在流水线最后执行
- 填充色统一为 (114, 114, 114)，与 ESP32 端侧推理一致
- 128x32 输入的 letterbox 必须采用 V9 两步法（均匀缩放保持宽高比 4.0 不产生 padding）
"""

import random
import cv2
import numpy as np


def letterbox_resize(img, target_w=128, target_h=32, fill_color=(114, 114, 114)):
    """letterbox 到目标尺寸，保持宽高比，不足部分用 fill_color 填充

    Args:
        img: [H, W, 3] uint8 BGR/RGB
        target_w: 目标宽度
        target_h: 目标高度
        fill_color: 填充色 (R, G, B)
    Returns:
        [target_h, target_w, 3] uint8
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale + 0.5)
    new_h = int(h * scale + 0.5)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    result = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
    return result


def simulate_yolo_crop_letterbox(img, aspect_ratio=None, target_h=None,
                                  target_w_out=128, target_h_out=32,
                                  fill_color=(114, 114, 114), align=8):
    """V9 两步法 letterbox：先 resize 到随机宽高比尺寸，再 letterbox 回 128x32

    解决 128x32 输入简单 letterbox scale=1.0 不产生 padding 的问题。
    必须用随机宽高比（2.5-6.5）×随机高度（16-72）才能产生显著 padding。

    Args:
        img: [H, W, 3] uint8
        aspect_ratio: 随机宽高比，None 则随机生成
        target_h: 随机高度，None 则随机生成
        target_w_out: letterbox 目标宽度（默认 128）
        target_h_out: letterbox 目标高度（默认 32）
        fill_color: 填充色
        align: 尺寸对齐（默认 8）
    Returns:
        [target_h_out, target_w_out, 3] uint8
    """
    if aspect_ratio is None:
        aspect_ratio = random.uniform(2.5, 6.5)
    if target_h is None:
        target_h = random.randint(16, 72)

    target_w = int(target_h * aspect_ratio)
    # 8 对齐（与 V9 simulate_yolo_crop_letterbox 一致）
    target_w = (target_w + align - 1) // align * align
    target_h = (target_h + align - 1) // align * align

    # 步骤 1：resize 到随机宽高比尺寸
    sim_crop = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # 步骤 2：letterbox 回 128x32
    return letterbox_resize(sim_crop, target_w=target_w_out, target_h=target_h_out,
                            fill_color=fill_color)


def right_side_padding_augment(img, prob=0.3, max_pad_ratio=0.15,
                                fill_color=(114, 114, 114)):
    """右侧随机 padding，模拟末尾字符靠近 padding 边界

    V2 独立函数版本（V9 中为 LPRAugmentationV5 类方法，参数硬编码）。

    Args:
        img: [H, W, 3] uint8
        prob: 触发概率
        max_pad_ratio: 最大 padding 比例（相对宽度）
        fill_color: 填充色
    Returns:
        [H, W + pad_right, 3] uint8
    """
    if random.random() >= prob:
        return img
    h, w = img.shape[:2]
    pad_right = random.randint(1, max(1, int(w * max_pad_ratio)))
    result = np.full((h, w + pad_right, 3), fill_color, dtype=np.uint8)
    result[:, :w, :] = img
    return result


# ===== 5 组增强函数（简化实现） =====

def random_crop(img, max_crop_ratio=0.1):
    """第1组：随机裁切（模拟检测器裁切偏差）"""
    h, w = img.shape[:2]
    crop_h = int(h * random.uniform(0, max_crop_ratio))
    crop_w = int(w * random.uniform(0, max_crop_ratio))
    top = random.randint(0, crop_h) if crop_h > 0 else 0
    left = random.randint(0, crop_w) if crop_w > 0 else 0
    return img[top:h - (crop_h - top), left:w - (crop_w - left)]


def random_geometric(img, max_angle=5, max_shear=3):
    """第2组：几何变换（旋转、仿射）"""
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_occlusion(img, max_occlusion_ratio=0.15):
    """第3组：遮挡模拟（模拟污损、遮挡）"""
    h, w = img.shape[:2]
    occ_h = int(h * random.uniform(0.05, max_occlusion_ratio))
    occ_w = int(w * random.uniform(0.05, max_occlusion_ratio))
    top = random.randint(0, h - occ_h)
    left = random.randint(0, w - occ_w)
    fill_val = random.randint(0, 255)
    img = img.copy()
    img[top:top + occ_h, left:left + occ_w] = fill_val
    return img


def random_lighting(img, brightness=0.2, contrast=0.2):
    """第4组：光照变化（亮度、对比度）"""
    img = img.astype(np.float32)
    # 亮度
    b = random.uniform(-brightness, brightness) * 255
    img = img + b
    # 对比度
    c = random.uniform(1 - contrast, 1 + contrast)
    mean = img.mean()
    img = (img - mean) * c + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def random_noise(img, gaussian_std=0.02, blur_prob=0.3):
    """第5组：噪声注入（高斯噪声、模糊）"""
    if random.random() < blur_prob:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    noise = np.random.normal(0, gaussian_std * 255, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


class LPRAugmentationV6DistillV2:
    """V6 蒸馏 V2 数据增强：5组增强 + letterbox + 右侧 padding

    渐进增强阶段：
    - warmup (epoch 1-10): 无增强 + letterbox(0.3)
    - medium (epoch 11-40): 裁切+几何+光照 + letterbox(0.5) + 右侧padding(0.3)
    - full (epoch 41+): 全部5组 + letterbox(0.5) + 右侧padding(0.3)

    letterbox 必须在流水线最后执行，否则 padding 区域被后续增强破坏。
    """

    def __init__(self, phase='warmup', letterbox_prob=0.3,
                 letterbox_fill_color=(114, 114, 114),
                 right_pad_prob=0.3, right_pad_max_ratio=0.15,
                 disable_letterbox=False):
        self.phase = phase
        self.letterbox_fill_color = letterbox_fill_color
        self.right_pad_prob = right_pad_prob
        self.right_pad_max_ratio = right_pad_max_ratio
        self.disable_letterbox = disable_letterbox
        self.set_phase(phase)

    def set_phase(self, phase):
        """渐进增强阶段切换，与 V9 LPRAugmentationV5.set_phase 一致"""
        self.phase = phase
        if phase == 'warmup':
            self.letterbox_prob = 0.3
        elif phase == 'medium':
            self.letterbox_prob = 0.5
        else:  # full
            self.letterbox_prob = 0.5
        # 消融实验：强制禁用 letterbox
        if self.disable_letterbox:
            self.letterbox_prob = 0.0

    def __call__(self, img, label):
        """执行增强流水线

        Args:
            img: [H, W, 3] uint8 RGB
            label: 标签（不修改，仅透传）
        Returns:
            img: [32, 128, 3] uint8 RGB
            label: 原标签
        """
        # 第1组：随机裁切（模拟检测器裁切偏差）
        if self.phase in ('medium', 'full'):
            img = random_crop(img, max_crop_ratio=0.1)

        # 第2组：几何变换（旋转、仿射）
        if self.phase in ('medium', 'full'):
            img = random_geometric(img, max_angle=5, max_shear=3)

        # 第3组：遮挡模拟（模拟污损、遮挡）
        if self.phase == 'full':
            img = random_occlusion(img, max_occlusion_ratio=0.15)

        # 第4组：光照变化（亮度、对比度、色调）
        if self.phase in ('medium', 'full'):
            img = random_lighting(img, brightness=0.2, contrast=0.2)

        # 第5组：噪声注入（高斯噪声、模糊）
        if self.phase == 'full':
            img = random_noise(img, gaussian_std=0.02, blur_prob=0.3)

        # 第6组：右侧 padding 增强（模拟末尾字符靠近边界）
        # 注意：右侧 padding 在 letterbox 之前，因为 letterbox 会将图像缩放到 128x32
        # 右侧 padding 后图像变宽，letterbox 时会产生上下 padding
        if self.phase in ('medium', 'full'):
            img = right_side_padding_augment(
                img, prob=self.right_pad_prob,
                max_pad_ratio=self.right_pad_max_ratio,
                fill_color=self.letterbox_fill_color)

        # 第7组：letterbox 增强（必须最后执行！）
        # 模拟 YOLO 裁切→letterbox 到 128×32 的完整推理流程
        # 填充色 (114, 114, 114) 与 ESP32 端推理一致
        # 注意：训练图像来自 bg128.32（已 128×32），简单 letterbox scale=1.0 不产生 padding
        # 必须采用 V9 simulate_yolo_crop_letterbox 两步法：先 resize 到随机宽高比，再 letterbox 回 128×32
        if random.random() < self.letterbox_prob:
            img = simulate_yolo_crop_letterbox(
                img, fill_color=self.letterbox_fill_color)
        else:
            img = cv2.resize(img, (128, 32))

        return img, label
