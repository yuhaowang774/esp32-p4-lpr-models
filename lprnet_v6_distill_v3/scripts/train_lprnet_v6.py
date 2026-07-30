import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'
ALPHABETS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
DIGITS = '0123456789'
ALL_CHARS = PROVINCES + ALPHABETS + DIGITS

CHAR_TO_IDX = {'blank': 0}
for i, char in enumerate(ALL_CHARS, start=1):
    CHAR_TO_IDX[char] = i
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHAR_TO_IDX)


def get_fill_color():
    r = random.random()
    if r < 0.30:
        return 114
    elif r < 0.55:
        return random.randint(30, 80)
    elif r < 0.80:
        return random.randint(160, 220)
    else:
        return None


def apply_fill(img, fill_color):
    if fill_color is None:
        h, w = img.shape[:2]
        border_size = 3
        top = img[:border_size, :, :].mean(axis=(0, 1))
        bottom = img[-border_size:, :, :].mean(axis=(0, 1))
        left = img[:, :border_size, :].mean(axis=(0, 1))
        right = img[:, -border_size:, :].mean(axis=(0, 1))
        fill_color = int((top.mean() + bottom.mean() + left.mean() + right.mean()) / 4)
    return fill_color


class LPRAugmentationV5:
    def __init__(self, target_w=128, target_h=32, phase='warmup'):
        self.target_w = target_w
        self.target_h = target_h
        self.phase = phase
        self.applied = []

    def set_phase(self, phase):
        self.phase = phase

    def _aspect_ratio_resize_with_pad(self, img):
        h, w = img.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        result = np.full((self.target_h, self.target_w, 3), fill, dtype=np.uint8)
        x_offset = (self.target_w - new_w) // 2
        y_offset = (self.target_h - new_h) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized
        return result

    def random_edge_crop(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        h, w = img.shape[:2]
        num_sides = random.choice([1, 2])
        sides = random.sample(['left', 'right', 'top', 'bottom'], num_sides)
        for side in sides:
            if side == 'left':
                crop_w = int(w * random.uniform(0.05, 0.25))
                img[:h, :crop_w] = fill
            elif side == 'right':
                crop_w = int(w * random.uniform(0.05, 0.25))
                img[:h, w - crop_w:] = fill
            elif side == 'top':
                crop_h = int(h * random.uniform(0.05, 0.20))
                img[:crop_h, :] = fill
            elif side == 'bottom':
                crop_h = int(h * random.uniform(0.05, 0.20))
                img[h - crop_h:, :] = fill
        self.applied.append(f'边缘裁切({",".join(sides)})')
        return img

    def random_severe_crop(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        h, w = img.shape[:2]
        side = random.choice(['left', 'right', 'top', 'bottom'])
        if side == 'left':
            crop_w = int(w * random.uniform(0.20, 0.35))
            img[:h, :crop_w] = fill
        elif side == 'right':
            crop_w = int(w * random.uniform(0.20, 0.35))
            img[:h, w - crop_w:] = fill
        elif side == 'top':
            crop_h = int(h * random.uniform(0.20, 0.35))
            img[:crop_h, :] = fill
        elif side == 'bottom':
            crop_h = int(h * random.uniform(0.20, 0.35))
            img[h - crop_h:, :] = fill
        self.applied.append(f'严重裁切({side})')
        return img

    def random_extra_background(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        h, w = img.shape[:2]
        max_extra_w = int(w * random.uniform(0.05, 0.25))
        max_extra_h = int(h * random.uniform(0.05, 0.25))
        left = random.randint(0, max_extra_w)
        right = random.randint(0, max_extra_w)
        top = random.randint(0, max_extra_h)
        bottom = random.randint(0, max_extra_h)
        new_w = w + left + right
        new_h = h + top + bottom
        result = np.full((new_h, new_w, 3), fill, dtype=np.uint8)
        result[top:top + h, left:left + w] = img
        self.applied.append(f'额外背景(左{left}右{right}上{top}下{bottom})')
        return result

    def apply_crop_group(self, img):
        r = random.random()
        if r < 0.45:
            return self.random_edge_crop(img)
        elif r < 0.50:
            return self.random_severe_crop(img)
        elif r < 0.80:
            return self.random_extra_background(img)
        return img

    def random_rotation(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        if random.random() < 0.45:
            angle = random.uniform(-10, 10)
        elif random.random() < 0.7:
            angle = random.uniform(-25, 25)
        else:
            angle = random.uniform(-45, 45)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        result = cv2.warpAffine(img, M, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(fill, fill, fill))
        self.applied.append(f'旋转({angle:.1f}°)')
        return result

    def random_shear(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        h, w = img.shape[:2]
        shear_x = random.uniform(-0.20, 0.20)
        shear_y = random.uniform(-0.08, 0.08)
        M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        new_w = int(w + abs(h * shear_x))
        new_h = int(h + abs(w * shear_y))
        result = cv2.warpAffine(img, M, (max(new_w, w), max(new_h, h)),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(fill, fill, fill))
        self.applied.append(f'剪切(X:{shear_x:.2f},Y:{shear_y:.2f})')
        return result

    def random_perspective(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        h, w = img.shape[:2]
        margin_x = int(w * random.uniform(0.05, 0.25))
        margin_y = int(h * random.uniform(0.05, 0.25))
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [random.randint(0, margin_x), random.randint(0, margin_y)],
            [w - random.randint(0, margin_x), random.randint(0, margin_y)],
            [w - random.randint(0, margin_x), h - random.randint(0, margin_y)],
            [random.randint(0, margin_x), h - random.randint(0, margin_y)]
        ])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(img, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(fill, fill, fill))
        self.applied.append('透视变换')
        return result

    def random_tilt(self, img):
        fill = get_fill_color()
        fill = apply_fill(img, fill)
        h, w = img.shape[:2]
        tilt_type = random.choice(['left_up', 'right_up', 'top_left', 'top_right'])
        margin = int(w * random.uniform(0.05, 0.20))
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if tilt_type == 'left_up':
            dst_points = np.float32([[margin, 0], [w, 0], [w, h], [0, h]])
        elif tilt_type == 'right_up':
            dst_points = np.float32([[0, 0], [w - margin, 0], [w, h], [0, h]])
        elif tilt_type == 'top_left':
            dst_points = np.float32([[0, margin], [w, 0], [w, h], [0, h - margin]])
        else:
            dst_points = np.float32([[0, 0], [w, margin], [w, h - margin], [0, h]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(img, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(fill, fill, fill))
        self.applied.append(f'单边倾斜({tilt_type})')
        return result

    def apply_geo_group(self, img):
        geo_ops = [
            (0.50, self.random_rotation),
            (0.25, self.random_shear),
            (0.30, self.random_perspective),
            (0.20, self.random_tilt),
        ]
        max_ops = 2
        count = 0
        for prob, op in geo_ops:
            if count >= max_ops:
                break
            if random.random() < prob:
                img = self._safe_aug(img, op)
                count += 1
        return img

    def random_char_occlusion(self, img):
        h, w = img.shape[:2]
        num_blocks = random.randint(1, 2)
        for _ in range(num_blocks):
            occ_w = random.randint(int(w * 0.03), int(w * 0.10))
            occ_h = random.randint(int(h * 0.30), int(h * 0.80))
            x = random.randint(int(w * 0.05), max(int(w * 0.05) + 1, w - occ_w - int(w * 0.05)))
            y = random.randint(0, max(1, h - occ_h))
            color = random.choice([
                random.randint(30, 80),
                random.randint(100, 150),
                random.randint(180, 240)
            ])
            img[y:y + occ_h, x:x + occ_w] = color
        self.applied.append(f'字符遮挡({num_blocks}块)')
        return img

    def random_province_occlusion(self, img):
        h, w = img.shape[:2]
        occ_w = random.randint(int(w * 0.04), int(w * 0.12))
        occ_h = random.randint(int(h * 0.20), int(h * 0.60))
        x = random.randint(0, max(1, int(w * 0.15) - occ_w))
        y = random.randint(0, max(1, h - occ_h))
        color = random.choice([
            random.randint(30, 80),
            random.randint(100, 150),
            random.randint(180, 240)
        ])
        img[y:y + occ_h, x:x + occ_w] = color
        self.applied.append('省份遮挡')
        return img

    def random_partial_occlusion(self, img):
        h, w = img.shape[:2]
        occ_w = random.randint(int(w * 0.02), int(w * 0.08))
        occ_h = random.randint(int(h * 0.1), int(h * 0.4))
        x = random.randint(0, max(1, w - occ_w))
        y = random.randint(0, max(1, h - occ_h))
        color = random.randint(0, 255)
        img[y:y + occ_h, x:x + occ_w] = color
        self.applied.append('边缘遮挡')
        return img

    def apply_occlusion_group(self, img):
        r = random.random()
        if r < 0.15:
            return self.random_char_occlusion(img)
        elif r < 0.25:
            return self.random_province_occlusion(img)
        elif r < 0.33:
            return self.random_partial_occlusion(img)
        return img

    def random_brightness(self, img):
        factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Brightness(Image.fromarray(img))
        self.applied.append(f'亮度({factor:.2f})')
        return np.array(enhancer.enhance(factor))

    def random_contrast(self, img):
        factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Contrast(Image.fromarray(img))
        self.applied.append(f'对比度({factor:.2f})')
        return np.array(enhancer.enhance(factor))

    def random_saturation(self, img):
        factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Color(Image.fromarray(img))
        self.applied.append(f'饱和度({factor:.2f})')
        return np.array(enhancer.enhance(factor))

    def random_color_shift(self, img):
        shift = random.randint(-15, 15)
        result = img.astype(np.int16)
        result[:, :, 0] += shift
        result[:, :, 1] += random.randint(-15, 15)
        result[:, :, 2] += random.randint(-15, 15)
        result = np.clip(result, 0, 255).astype(np.uint8)
        self.applied.append(f'色偏(R:{shift})')
        return result

    def random_shadow(self, img):
        h, w = img.shape[:2]
        x1, x2 = random.randint(0, w), random.randint(0, w)
        shadow = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            shadow[i, :] = np.linspace(x1, x2, w) / w
        shadow = np.clip(shadow, 0.3, 1.0)
        shadow = np.stack([shadow] * 3, axis=2)
        self.applied.append('阴影')
        return (img * shadow).astype(np.uint8)

    def random_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        self.applied.append('灰度化')
        return result

    def random_rain(self, img):
        h, w = img.shape[:2]
        rain = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(random.randint(30, 100)):
            x, y = random.randint(0, w), random.randint(0, h)
            length = random.randint(3, 10)
            cv2.line(rain, (x, y), (x, y + length), (200, 200, 200), 1)
        self.applied.append('雨滴')
        return cv2.addWeighted(img, 0.85, rain, 0.15, 0)

    def random_fog(self, img):
        h, w = img.shape[:2]
        fog = np.random.randint(180, 220, (h, w, 3), dtype=np.uint8)
        alpha = random.uniform(0.1, 0.25)
        self.applied.append('雾')
        return cv2.addWeighted(img, 1 - alpha, fog, alpha, 0)

    def apply_light_group(self, img):
        light_ops = [
            (0.40, self.random_brightness),
            (0.40, self.random_contrast),
            (0.20, self.random_saturation),
            (0.15, self.random_color_shift),
            (0.15, self.random_shadow),
            (0.03, self.random_grayscale),
            (0.05, self.random_rain),
            (0.03, self.random_fog),
        ]
        max_ops = 2
        count = 0
        for prob, op in light_ops:
            if count >= max_ops:
                break
            if random.random() < prob:
                img = self._safe_aug(img, op)
                count += 1
        return img

    def add_gaussian_noise(self, img):
        sigma = random.randint(5, 20)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        self.applied.append(f'高斯噪声(σ={sigma})')
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def random_blur(self, img):
        k = random.choice([3, 5])
        self.applied.append(f'高斯模糊(k={k})')
        return cv2.GaussianBlur(img, (k, k), 0)

    def random_motion_blur(self, img):
        degree = random.randint(3, 8)
        angle = random.randint(0, 180)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        kernel = np.diag(np.ones(degree))
        kernel = cv2.warpAffine(kernel, M, (degree, degree))
        kernel = kernel / degree
        self.applied.append(f'运动模糊(角度={angle})')
        return cv2.filter2D(img, -1, kernel)

    def random_jpeg_compression(self, img):
        quality = random.randint(30, 70)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.applied.append(f'JPEG压缩(Q={quality})')
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def random_resolution(self, img):
        h, w = img.shape[:2]
        scale = random.uniform(0.5, 0.8)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        self.applied.append(f'低分辨率({scale:.2f})')
        return result

    def apply_noise_group(self, img):
        noise_ops = [
            (0.25, self.add_gaussian_noise),
            (0.20, self.random_blur),
            (0.15, self.random_motion_blur),
            (0.15, self.random_jpeg_compression),
            (0.10, self.random_resolution),
        ]
        max_ops = 2
        count = 0
        for prob, op in noise_ops:
            if count >= max_ops:
                break
            if random.random() < prob:
                img = self._safe_aug(img, op)
                count += 1
        return img

    def get_phase_probs(self):
        if self.phase == 'warmup':
            return {'crop': 0.30, 'geo': 0.20, 'occlusion': 0.0, 'light': 0.20, 'noise': 0.0}
        elif self.phase == 'medium':
            return {'crop': 0.60, 'geo': 0.45, 'occlusion': 0.20, 'light': 0.35, 'noise': 0.15}
        else:
            return {'crop': 0.80, 'geo': 0.55, 'occlusion': 0.33, 'light': 0.50, 'noise': 0.25}

    def _safe_aug(self, img, aug_fn):
        try:
            if img is None or img.size == 0 or img.shape[0] < 4 or img.shape[1] < 4:
                return img
            result = aug_fn(img)
            if result is None or result.size == 0:
                return img
            return result
        except Exception:
            return img

    def __call__(self, img):
        self.applied = [f'保持宽高比+随机填充({self.target_w}x{self.target_h})']
        probs = self.get_phase_probs()

        if random.random() < probs['crop']:
            img = self._safe_aug(img, self.apply_crop_group)
        if random.random() < probs['geo']:
            img = self._safe_aug(img, self.apply_geo_group)
        if random.random() < probs['occlusion']:
            img = self._safe_aug(img, self.apply_occlusion_group)
        if random.random() < probs['light']:
            img = self._safe_aug(img, self.apply_light_group)
        if random.random() < probs['noise']:
            img = self._safe_aug(img, self.apply_noise_group)

        return self._aspect_ratio_resize_with_pad(img)


class SmallBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class LPRNetV6(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            SmallBasicBlock(32, 64),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            SmallBasicBlock(64, 128),
            nn.MaxPool2d((2, 2), stride=(2, 1)),

            SmallBasicBlock(128, 128),
            SpatialAttention(kernel_size=3),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(128, 128, (2, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(128, num_classes, 1, bias=False),
        )

        self.container = nn.Sequential(
            nn.Conv2d(num_classes + 128, num_classes, 1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

        self._init_container()

    def _init_container(self):
        with torch.no_grad():
            w = self.container[0].weight.data
            nn.init.xavier_normal_(w[:, :self.container[0].in_channels - 128, :, :])
            nn.init.zeros_(w[:, self.container[0].in_channels - 128:, :, :])

    def forward(self, x):
        skip = None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 8:
                skip = x

        skip = skip[:, :, :x.size(2), :]

        x = torch.cat([x, skip], dim=1)
        x = self.container(x)

        logits = torch.mean(x, dim=2, keepdim=False)
        logits = logits.permute(2, 0, 1)
        return logits


class WeightedCTCLoss(nn.Module):
    def __init__(self, blank=0, blank_weight=2.0, zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.blank_weight = blank_weight
        self.zero_infinity = zero_infinity
        self.base_ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.base_ctc(log_probs, targets, input_lengths, target_lengths)
        if self.blank_weight != 1.0:
            with torch.no_grad():
                pred = log_probs.argmax(dim=2)
                blank_ratio = (pred == self.blank).float().mean()
                weight = 1.0 + (self.blank_weight - 1.0) * (1.0 - blank_ratio)
            loss = loss * weight
        return loss.mean()


class ProvinceAuxLoss(nn.Module):
    def __init__(self, num_provinces=31, aux_weight=0.5):
        super().__init__()
        self.num_provinces = num_provinces
        self.aux_weight = aux_weight

    def forward(self, logits, labels, label_lengths, ctc_loss):
        T = logits.size(0)
        B = logits.size(1)

        early_steps = min(4, T)
        province_logits = logits[:early_steps, :, 1:self.num_provinces + 1]
        province_pooled = province_logits.max(dim=0)[0]

        province_labels = []
        label_idx = 0
        for i in range(B):
            length = label_lengths[i].item()
            if length > 0:
                province_labels.append(labels[label_idx].item() - 1)
            else:
                province_labels.append(0)
            label_idx += length

        province_labels = torch.tensor(province_labels, device=logits.device, dtype=torch.long)
        aux_loss = F.cross_entropy(province_pooled, province_labels)

        return ctc_loss + self.aux_weight * aux_loss


class LPRDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None, aug=None,
                 filter_types=None):
        self.img_dir = img_dir
        self.transform = transform
        self.aug = aug
        self.filter_types = filter_types or ['普通蓝牌', '新能源小型车']

        self.data = []
        self._load_data(txt_path)

        print(f'加载数据: {len(self.data)} 张图片')
        self._print_stats()

    def _load_data(self, txt_path):
        missing_count = 0
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    img_name = os.path.basename(img_path)
                    plate = parts[1]
                    plate_type = parts[2] if len(parts) > 2 else '普通蓝牌'

                    if plate_type in self.filter_types:
                        if self._validate_plate(plate):
                            full_path = os.path.join(self.img_dir, img_name)
                            if os.path.exists(full_path):
                                self.data.append((img_name, plate, plate_type))
                            else:
                                missing_count += 1
        if missing_count > 0:
            print(f'跳过缺失图片: {missing_count} 张')

    def _validate_plate(self, plate):
        if len(plate) < 7 or len(plate) > 8:
            return False
        for c in plate:
            if c not in CHAR_TO_IDX:
                return False
        return True

    def _print_stats(self):
        type_counter = Counter([d[2] for d in self.data])
        print('车牌类型分布:')
        for t, c in type_counter.items():
            print(f'  {t}: {c}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, plate, _ = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img_arr = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f'无法解码图片: {img_path}')
        except Exception as e:
            raise FileNotFoundError(f'无法读取图片: {img_path}, 错误: {e}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.aug:
            img = self.aug(img)

        if self.transform:
            img = self.transform(img)

        label = [CHAR_TO_IDX[c] for c in plate]
        label = torch.tensor(label, dtype=torch.long)

        return img, label, len(label)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


def decode_predictions(logits, max_len=8):
    pred = logits.argmax(dim=2).permute(1, 0)
    results = []
    for seq in pred:
        chars = []
        prev = -1
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:
                chars.append(IDX_TO_CHAR.get(idx, ''))
                if len(chars) >= max_len:
                    break
            prev = idx
        results.append(''.join(chars))
    return results


def calculate_accuracy(logits, labels, lengths):
    preds = decode_predictions(logits)
    correct = 0
    total = 0
    label_idx = 0
    for i, length in enumerate(lengths):
        pred = preds[i]
        true_label = ''.join([IDX_TO_CHAR[idx.item()] for idx in labels[label_idx:label_idx + length]])
        label_idx += length
        if pred == true_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(save_dir, exist_ok=True)

        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=config['betas']
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=config['warmup_pct'],
            anneal_strategy='cos',
            final_div_factor=config['final_div_factor']
        )

        self.ctc_criterion = WeightedCTCLoss(blank=0, blank_weight=config.get('blank_weight', 2.0), zero_infinity=True)
        self.province_aux = ProvinceAuxLoss(
            num_provinces=31,
            aux_weight=config.get('province_aux_weight', 0.5)
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['use_amp'])

        self.best_acc = 0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def get_phase(self, epoch):
        if epoch < 10:
            return 'warmup'
        elif epoch < 40:
            return 'medium'
        else:
            return 'full'

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        phase = self.get_phase(epoch)
        underlying = self.train_loader.dataset
        while hasattr(underlying, 'dataset'):
            underlying = underlying.dataset
        if hasattr(underlying, 'aug') and underlying.aug:
            underlying.aug.set_phase(phase)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]} [{phase}]')

        for images, labels, lengths in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.config['use_amp']):
                logits = self.model(images)
                log_probs = F.log_softmax(logits, dim=2)

                batch_size = images.size(0)
                input_lengths = torch.full(
                    (batch_size,), logits.size(0),
                    dtype=torch.long, device=self.device
                )

                ctc_loss = self.ctc_criterion(log_probs, labels, input_lengths, lengths)
                loss = self.province_aux(logits, labels, lengths, ctc_loss)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['grad_clip'])

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            with torch.no_grad():
                acc = calculate_accuracy(logits, labels, lengths)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        return total_loss / num_batches, total_acc / num_batches

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for images, labels, lengths in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.config['use_amp']):
                logits = self.model(images)
                log_probs = F.log_softmax(logits, dim=2)

                batch_size = images.size(0)
                input_lengths = torch.full(
                    (batch_size,), logits.size(0),
                    dtype=torch.long, device=self.device
                )

                loss = self.ctc_criterion(log_probs, labels, input_lengths, lengths)

            acc = calculate_accuracy(logits, labels, lengths)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config,
            'char_to_idx': CHAR_TO_IDX,
            'idx_to_char': IDX_TO_CHAR,
        }

        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'保存最佳模型: acc={self.best_acc:.4f}')

    def train(self, start_epoch=0):
        print(f'设备: {self.device}')
        print(f'训练集: {len(self.train_loader.dataset)} 样本')
        print(f'验证集: {len(self.val_loader.dataset)} 样本')
        print(f'早停: {self.config["early_stop_patience"]}轮无提升停止')
        print(f'渐进增强: warmup(1-10) → medium(11-40) → full(41+)')
        print(f'V6改进: 通道减半 [3,32,64,128,128] | ESP32-P4 PSRAM带宽优化')
        print('-' * 50)

        for epoch in range(start_epoch, self.config['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            phase = self.get_phase(epoch)
            print(f'\nEpoch {epoch + 1}/{self.config["epochs"]} [{phase}]:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.config['early_stop_patience']:
                print(f'\n早停: {self.config["early_stop_patience"]}轮无提升，停止训练')
                break

        print(f'\n训练完成! 最佳准确率: {self.best_acc:.4f}')
        return self.history


def main():
    import os
    _data_dir = os.environ.get("DATA_DIR", "./data/CBLPRD")
    _save_dir = os.environ.get("V6_OUTPUT_DIR", "./outputs/lprnet_v6")
    config = {
        'train_txt_path': os.path.join(_data_dir, 'train.txt'),
        'val_txt_path': os.path.join(_data_dir, 'val.txt'),
        'train_img_dir': os.path.join(_data_dir, 'train'),
        'val_img_dir': os.path.join(_data_dir, 'val'),
        'save_dir': _save_dir,
        'resume_from': None,

        'img_height': 32,
        'img_width': 128,
        'batch_size': 1024,
        'num_workers': 8,
        'epochs': 120,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'dropout_rate': 0.3,
        'use_amp': True,
        'early_stop_patience': 15,
        'warmup_pct': 0.1,
        'final_div_factor': 100,
        'betas': (0.9, 0.98),
        'grad_clip': 5.0,
        'province_aux_weight': 0.5,
        'blank_weight': 2.0,
    }

    print('=' * 50)
    print('LPRNet V6 训练配置 (方案A: 通道减半)')
    print('=' * 50)
    for k, v in config.items():
        print(f'  {k}: {v}')
    print('=' * 50)

    augmentation = LPRAugmentationV5(
        target_w=config['img_width'],
        target_h=config['img_height'],
        phase='warmup'
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LPRDataset(
        txt_path=config['train_txt_path'],
        img_dir=config['train_img_dir'],
        transform=transform,
        aug=augmentation,
        filter_types=['普通蓝牌', '新能源小型车']
    )

    val_dataset = LPRDataset(
        txt_path=config['val_txt_path'],
        img_dir=config['val_img_dir'],
        transform=transform,
        aug=None,
        filter_types=['普通蓝牌', '新能源小型车']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = LPRNetV6(num_classes=NUM_CLASSES, dropout_rate=config['dropout_rate'])

    start_epoch = 0
    if config.get('resume_from') and os.path.exists(config['resume_from']):
        checkpoint = torch.load(config['resume_from'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        print(f'断点续训: {config["resume_from"]}')
        print(f'从 Epoch {start_epoch + 1} 继续, 历史最佳: {checkpoint.get("best_acc", 0):.4f}')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}')

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=config['save_dir']
    )

    history = trainer.train(start_epoch=start_epoch)

    history_path = os.path.join(config['save_dir'], 'training_history.pth')
    torch.save(history, history_path)
    print(f'训练历史已保存: {history_path}')


if __name__ == '__main__':
    main()
