"""V6 蒸馏 V2 数据集

包含：
- compute_province_weights: 省份逆频率权重（用于 WeightedRandomSampler 和 ProvinceAuxLoss）
- LPRDatasetV2: 训练/验证数据集（支持省份均衡采样、双验证集、渐进增强）
- CalibrationDatasetV2: 量化校准数据集（50% clean + 50% letterbox 混合分布）

关键设计：
1. 省份均衡采样：WeightedRandomSampler + compute_province_weights
2. 双验证集：force_letterbox=True 启用确定性 letterbox 两步变换（独立 RNG）
3. 校准数据确定性：独立 random.Random(seed) 实例，保证可复现
"""

import os
import cv2
import random
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from chars import CHAR_TO_IDX, IDX_TO_CHAR, PROVINCES
from augmentation_v2 import letterbox_resize, simulate_yolo_crop_letterbox


def compute_province_weights(dataset, max_weight=5.0, green_weight=1.0):
    """计算省份逆频率权重，返回 sample_weights 和 province_class_weights

    与 V9 train_lprnet_v9_balanced.py 第 708-728 行一致。
    V3 新增：green_weight 参数，对绿牌样本（新能源小型车）额外加权。

    Args:
        dataset: 数据集对象，需有 data 属性 [(img_name, plate, type), ...]
        max_weight: 省份权重上限，避免过拟合稀有省份
        green_weight: 绿牌样本额外权重倍数（V3 新增，1.0=不加权）
    Returns:
        sample_weights: list[float]，per-sample 权重，用于 WeightedRandomSampler
        province_class_weights: torch.Tensor [31]，per-province 权重，用于 ProvinceAuxLoss
        province_counts: Counter，省份索引→样本数
    """
    province_counts = Counter()
    for _, plate, _ in dataset.data:
        province_idx = CHAR_TO_IDX[plate[0]] - 1
        province_counts[province_idx] += 1

    total = sum(province_counts.values())
    num_provinces = len(PROVINCES)

    # per-province 权重张量 [31]，用于 ProvinceAuxLoss 的 F.cross_entropy(weight=...)
    province_class_weights = torch.zeros(num_provinces)
    for idx in range(num_provinces):
        count = province_counts.get(idx, 1)
        province_class_weights[idx] = min(total / (num_provinces * count), max_weight)

    # per-sample 权重 list，用于 WeightedRandomSampler
    # V3: 绿牌样本（新能源小型车）额外乘以 green_weight
    sample_weights = []
    green_count = 0
    for _, plate, plate_type in dataset.data:
        province_idx = CHAR_TO_IDX[plate[0]] - 1
        w = province_class_weights[province_idx].item()
        if plate_type == '新能源小型车' and green_weight > 1.0:
            w *= green_weight
            green_count += 1
        sample_weights.append(w)

    if green_weight > 1.0:
        print(f'  绿牌加权: {green_count} 个绿牌样本 x{green_weight}')

    return sample_weights, province_class_weights, province_counts


class LPRDatasetV2(Dataset):
    """V6 蒸馏 V2 数据集

    支持三种模式：
    1. 训练模式（aug != None）：渐进增强 + letterbox
    2. clean 验证模式（aug=None, force_letterbox=False）：直接 resize 到 128x32
    3. letterbox 验证模式（aug=None, force_letterbox=True）：确定性 letterbox 两步变换

    省份均衡采样通过外部 WeightedRandomSampler 实现，本类仅提供 sample_weights 接口。
    """

    def __init__(self, txt_path, img_dir, transform=None, aug=None,
                 filter_types=None, force_letterbox=False,
                 letterbox_seed=42):
        """
        Args:
            txt_path: 数据列表文件路径
            img_dir: 图片目录
            transform: torchvision transform（ToTensor + Normalize）
            aug: LPRAugmentationV6DistillV2 实例，None 表示无增强
            filter_types: 车牌类型过滤
            force_letterbox: 是否强制 letterbox（用于 val_letterbox 验证集）
            letterbox_seed: 确定性 letterbox 的 RNG 种子
        """
        self.img_dir = img_dir
        self.transform = transform
        self.aug = aug
        self.filter_types = filter_types or ['普通蓝牌', '新能源小型车']
        self.force_letterbox = force_letterbox

        if force_letterbox:
            # 独立 RNG 实例，保证 letterbox 参数可复现（与 V9 _letterbox_rng 一致）
            self._letterbox_rng = random.Random(letterbox_seed)
            self._letterbox_params = None

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

    def set_phase(self, phase):
        """渐进增强阶段切换，委托到内部 aug 对象"""
        if self.aug is not None:
            self.aug.set_phase(phase)

    def _generate_letterbox_params(self):
        """生成确定性 letterbox 参数（V9 两步法）"""
        rng = self._letterbox_rng
        self._letterbox_params = []
        for _ in range(len(self.data)):
            if rng.random() < 0.5:
                aspect_ratio = rng.uniform(2.5, 5.5)
            else:
                aspect_ratio = rng.uniform(3.5, 6.5)
            target_h = rng.randint(16, 72)
            target_w = int(target_h * aspect_ratio)
            target_w = (target_w + 7) // 8 * 8
            target_h = (target_h + 7) // 8 * 8
            self._letterbox_params.append((target_w, target_h))

    def _deterministic_letterbox(self, img, idx):
        """确定性 letterbox 两步变换（与 V9 _deterministic_letterbox 一致）

        128x32 输入简单 letterbox scale=1.0 不产生 padding，
        必须采用 V9 两步法：先缩放到随机小尺寸，再 letterbox 回 128x32。
        """
        if self._letterbox_params is None:
            self._generate_letterbox_params()

        target_w, target_h = self._letterbox_params[idx]
        # 步骤 1：resize 到随机宽高比尺寸（模拟 YOLO 裁切）
        sim_crop = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # 步骤 2：letterbox 回 128x32
        return letterbox_resize(sim_crop, target_w=128, target_h=32,
                                fill_color=(114, 114, 114))

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
            img, _ = self.aug(img, None)

        if self.force_letterbox:
            img = self._deterministic_letterbox(img, idx)
        elif not self.aug:
            # clean 验证模式：直接 resize 到 128x32
            img = cv2.resize(img, (128, 32))

        if self.transform:
            img = self.transform(img)

        label = [CHAR_TO_IDX[c] for c in plate]
        label = torch.tensor(label, dtype=torch.long)

        return img, label, len(label)


def collate_fn(batch):
    """CTC collate：处理变长标签"""
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


class CalibrationDatasetV2(Dataset):
    """V2 校准集：混合 clean 和 letterbox 分布

    量化校准数据分布应匹配实际推理时的输入分布。
    ESP32 端侧推理时约 50% 输入来自 YOLO 裁切（letterbox），50% 为 clean。
    """

    def __init__(self, txt_path, img_dir, num_samples=2048, letterbox_ratio=0.5,
                 seed=42):
        """
        Args:
            txt_path: 数据列表文件路径
            img_dir: 图片目录
            num_samples: 校准样本数
            letterbox_ratio: letterbox 样本比例
            seed: 随机种子（保证可复现）
        """
        self.img_dir = img_dir
        self.letterbox_ratio = letterbox_ratio

        self.transform_clean = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_letterbox = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载样本列表
        self.samples = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append(os.path.basename(parts[0]))

        # 确定性采样：独立 RNG 实例
        sample_rng = random.Random(seed)
        sample_rng.shuffle(self.samples)
        self.samples = self.samples[:num_samples]
        # 确定性分配：前 letterbox_ratio 比例走 letterbox，其余走 clean
        self.letterbox_count = int(len(self.samples) * letterbox_ratio)
        # 独立 RNG 实例，保证 __getitem__ 中的随机缩放可复现（与 V9 _letterbox_rng 一致）
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.samples[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 确定性分配：前 letterbox_count 个走 letterbox，其余走 clean
        if idx < self.letterbox_count:
            # V9 两步法：先 resize 到随机宽高比尺寸，再 letterbox 回 128x32
            # 均匀缩放保持宽高比≈4.0，AspectRatioResize 会缩回 128x32 无 padding（0-1px）
            # 必须用随机宽高比（2.5-6.5）×随机高度（16-72）才能产生显著 padding
            aspect_ratio = self.rng.uniform(2.5, 6.5)
            target_h = self.rng.randint(16, 72)
            target_w = int(target_h * aspect_ratio)
            # 8 对齐（与 V9 simulate_yolo_crop_letterbox 一致）
            target_w = (target_w + 7) // 8 * 8
            target_h = (target_h + 7) // 8 * 8
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img = letterbox_resize(img, target_w=128, target_h=32,
                                    fill_color=(114, 114, 114))
            img = self.transform_letterbox(img)
        else:
            img = cv2.resize(img, (128, 32))
            img = self.transform_clean(img)

        return img
