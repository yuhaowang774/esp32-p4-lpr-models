"""V6 蒸馏 V3 训练脚本

基于 V6 蒸馏 V2 成功方案，针对绿牌末尾字符丢失问题优化：
- blank_logit_reduction: 0.3→0.1
- letterbox 宽高比范围: (2.5,6.5)→(2.5,8.0)
- 绿牌样本加权 1.5x（GREEN_PLATE_WEIGHT）
- letterbox_prob: 0.5→0.6
- 学习率: 5e-4→3e-4（从 V2 best 权重微调）
- STUDENT_WEIGHTS 从 V2 best 权重初始化

核心流程：
1. 加载 V5 教师模型（冻结参数）
2. 加载 V6 学生模型（从 V6 蒸馏 V2 best 权重初始化）
3. 省份均衡采样 + 绿牌加权（WeightedRandomSampler）
4. 渐进增强（warmup→medium→full）+ 渐进 alpha 调度（0.7→0.5→0.3）
5. 双验证集评估（clean + letterbox），最优模型按 val_balanced_acc 选择
6. AMP 混合精度训练
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 启用 stdout 行缓冲，确保日志实时写入文件
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ===== 日志 tee：同时输出到 stdout 和日志文件 =====
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.txt")
_log_fp = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.stdout, _log_fp)
sys.stderr = _Tee(sys.stderr, _log_fp)

from model_v5 import LPRNetV5
from model_v6 import LPRNetV6
from chars import IDX_TO_CHAR, CHAR_TO_IDX, PROVINCES
from distill_loss_v2 import DistillCTCLossV2
from dataset_v2 import (
    LPRDatasetV2,
    CalibrationDatasetV2,
    compute_province_weights,
    collate_fn,
)
from augmentation_v2 import LPRAugmentationV6DistillV2
from config import *


def load_teacher_model():
    """加载 V5 教师模型并冻结参数"""
    model = LPRNetV5(num_classes=NUM_CLASSES)
    checkpoint = torch.load(TEACHER_WEIGHTS, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_student_model():
    """加载 V6 学生模型（方案A：从 V6 蒸馏 V1 best 权重初始化）"""
    model = LPRNetV6(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    checkpoint = torch.load(STUDENT_WEIGHTS, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.train()
    return model


def get_phase(epoch):
    """渐进增强阶段切换

    warmup (1-10): 无增强 + letterbox(0.3)
    medium (11-40): 裁切+几何+光照 + letterbox(0.5) + 右侧padding(0.3)
    full (41+): 全部5组 + letterbox(0.5) + 右侧padding(0.3)
    """
    if epoch <= PHASE_BOUNDARIES["warmup"]:
        return "warmup"
    elif epoch <= PHASE_BOUNDARIES["medium"]:
        return "medium"
    else:
        return "full"


def get_alpha(epoch):
    """渐进 alpha 调度

    初期 (1-5): alpha=0.7，侧重蒸馏
    中期 (6-20): alpha=0.5，蒸馏与 CTC 平衡
    后期 (21+): alpha=0.3，侧重 CTC
    """
    for boundary, alpha in ALPHA_SCHEDULE:
        if epoch <= boundary:
            return alpha
    return ALPHA_SCHEDULE[-1][1]


def decode_predictions(logits, max_len=8):
    """CTC greedy 解码"""
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


def train_one_epoch(teacher, student, loss_fn, dataloader, optimizer, scheduler,
                    scaler, use_amp, device, epoch):
    """训练一个 epoch"""
    teacher.eval()
    student.train()

    # 渐进增强：根据 epoch 切换增强阶段
    phase = get_phase(epoch)
    underlying = dataloader.dataset
    while hasattr(underlying, "dataset"):
        underlying = underlying.dataset
    if hasattr(underlying, "set_phase"):
        underlying.set_phase(phase)

    # 渐进 alpha 调度
    alpha = get_alpha(epoch)
    loss_fn.alpha = alpha

    total_loss = 0
    total_distill = 0
    total_ctc = 0
    total_aux = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            # 教师前向（no_grad）
            with torch.no_grad():
                teacher_logits = teacher(images)  # [T, B, C]

            # 学生前向
            student_logits = student(images)  # [T, B, C]

            input_lengths = torch.full(
                (images.size(0),), student_logits.size(0), dtype=torch.long, device=device
            )

            # 计算损失
            loss, metrics = loss_fn(
                student_logits, teacher_logits, labels, input_lengths, label_lengths
            )

        # 反向传播（AMP）
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_distill += metrics['distill']
        total_ctc += metrics['ctc']
        total_aux += metrics['aux']

        # 计算训练精度
        with torch.no_grad():
            preds = decode_predictions(student_logits)
            offset = 0
            for i, length in enumerate(label_lengths):
                length = length.item()
                target_chars = "".join(
                    IDX_TO_CHAR.get(labels[offset + j].item(), "")
                    for j in range(length)
                )
                if preds[i] == target_chars:
                    correct += 1
                total += 1
                offset += length

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_distill = total_distill / (batch_idx + 1)
            avg_ctc = total_ctc / (batch_idx + 1)
            avg_aux = total_aux / (batch_idx + 1)
            acc = correct / max(total, 1) * 100
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"phase={phase} alpha={alpha:.1f} "
                f"Loss: {avg_loss:.4f} (D:{avg_distill:.4f} C:{avg_ctc:.4f} A:{avg_aux:.4f}) "
                f"Acc: {acc:.1f}%"
            )

    return total_loss / len(dataloader), correct / max(total, 1)


@torch.no_grad()
def validate(student, dataloader, device, use_amp=False):
    """验证：返回序列级和字符级准确率"""
    student.eval()

    correct = 0
    total = 0
    char_correct = 0
    char_total = 0

    for images, labels, label_lengths in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            student_logits = student(images)
        preds = decode_predictions(student_logits)

        offset = 0
        for i, length in enumerate(label_lengths):
            length = length.item()
            target_chars = "".join(
                IDX_TO_CHAR.get(labels[offset + j].item(), "")
                for j in range(length)
            )
            if preds[i] == target_chars:
                correct += 1
            # 字符级精度
            for c1, c2 in zip(preds[i], target_chars):
                if c1 == c2:
                    char_correct += 1
            char_total += max(len(preds[i]), len(target_chars))
            total += 1
            offset += length

    seq_acc = correct / max(total, 1) * 100
    char_acc = char_correct / max(char_total, 1) * 100
    return seq_acc, char_acc


@torch.no_grad()
def validate_balanced(student, dataloader, device, use_amp=False):
    """省份均衡准确率：每个省份单独计算准确率后取平均"""
    student.eval()

    # 每个省份的 correct/total
    province_correct = {}
    province_total = {}

    for images, labels, label_lengths in dataloader:
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            student_logits = student(images)
        preds = decode_predictions(student_logits)

        offset = 0
        for i, length in enumerate(label_lengths):
            length = length.item()
            target_chars = "".join(
                IDX_TO_CHAR.get(labels[offset + j].item(), "")
                for j in range(length)
            )
            # 省份为首字符
            if len(target_chars) > 0:
                prov = target_chars[0]
                if prov not in province_correct:
                    province_correct[prov] = 0
                    province_total[prov] = 0
                if preds[i] == target_chars:
                    province_correct[prov] += 1
                province_total[prov] += 1
            offset += length

    # 计算各省份准确率的平均
    if not province_total:
        return 0.0
    accs = [province_correct[p] / max(province_total[p], 1)
            for p in province_total]
    return sum(accs) / len(accs) * 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print("Loading teacher model (V5)...")
    teacher = load_teacher_model().to(device)
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    print("Loading student model (V6, from V6 distill V1 weights)...")
    student = load_student_model().to(device)
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    # 数据增强和 transform
    augmentation = LPRAugmentationV6DistillV2(phase='warmup')
    normalize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 训练集（省份均衡采样）
    train_dataset = LPRDatasetV2(
        txt_path=TRAIN_TXT,
        img_dir=os.path.join(DATA_DIR, "train"),
        transform=normalize_transform,
        aug=augmentation,
        filter_types=FILTER_TYPES,
    )
    sample_weights, province_class_weights, _ = compute_province_weights(
        train_dataset, max_weight=PROVINCE_WEIGHT_CAP,
        green_weight=GREEN_PLATE_WEIGHT)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # 双验证集：clean + letterbox
    val_clean_dataset = LPRDatasetV2(
        txt_path=VAL_TXT,
        img_dir=os.path.join(DATA_DIR, "val"),
        transform=normalize_transform,
        aug=None,
        force_letterbox=False,
        filter_types=FILTER_TYPES,
    )
    val_letterbox_dataset = LPRDatasetV2(
        txt_path=VAL_TXT,
        img_dir=os.path.join(DATA_DIR, "val"),
        transform=normalize_transform,
        aug=None,
        force_letterbox=True,
        letterbox_seed=VAL_LETTERBOX_SEED,
        filter_types=FILTER_TYPES,
    )
    val_clean_loader = DataLoader(
        val_clean_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    val_letterbox_loader = DataLoader(
        val_letterbox_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    # 损失函数（province_class_weights 来自上一步，无需单独计算）
    loss_fn = DistillCTCLossV2(
        temperature=DISTILL_TEMPERATURE,
        alpha=DISTILL_ALPHA,
        blank_weight=BLANK_WEIGHT,
        blank_logit_reduction=BLANK_LOGIT_REDUCTION,
        province_aux_weight=PROVINCE_AUX_WEIGHT,
        province_weights=province_class_weights,
    )

    # 优化器
    optimizer = AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )

    # AMP 配置
    use_amp = torch.cuda.is_available() and USE_AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_PCT,
        final_div_factor=100,
    )

    # 训练循环
    best_balanced_acc = 0
    patience_counter = 0

    print(f"\nStarting V6 distill V2 training")
    print(f"  T={DISTILL_TEMPERATURE}, blank_reduction={BLANK_LOGIT_REDUCTION}")
    print(f"  alpha schedule: 0.7(1-5)→0.5(6-20)→0.3(21+)")
    print(f"  Train: {len(train_dataset)} samples, Val: {len(val_clean_dataset)} samples")
    print(f"  Batch: {BATCH_SIZE}, Epochs: {EPOCHS}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            teacher, student, loss_fn, train_loader, optimizer, scheduler,
            scaler, use_amp, device, epoch
        )

        # 双验证集评估
        clean_seq_acc, clean_char_acc = validate(student, val_clean_loader, device, use_amp)
        lb_seq_acc, lb_char_acc = validate(student, val_letterbox_loader, device, use_amp)
        balanced_acc = (VAL_CLEAN_WEIGHT * clean_seq_acc +
                        VAL_LETTERBOX_WEIGHT * lb_seq_acc)
        province_acc = validate_balanced(student, val_clean_loader, device, use_amp)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{EPOCHS} ({elapsed:.0f}s) "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Clean: {clean_seq_acc:.1f}%/{clean_char_acc:.1f}% "
            f"LB: {lb_seq_acc:.1f}%/{lb_char_acc:.1f}% "
            f"Balanced: {balanced_acc:.1f}% "
            f"Prov: {province_acc:.1f}%"
        )

        # 最优模型保存（按 balanced_acc）
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            patience_counter = 0
            save_path = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_clean_acc": clean_seq_acc,
                    "val_clean_char_acc": clean_char_acc,
                    "val_letterbox_acc": lb_seq_acc,
                    "val_letterbox_char_acc": lb_char_acc,
                    "val_balanced_acc": balanced_acc,
                    "val_province_acc": province_acc,
                    "distill_config": {
                        "temperature": DISTILL_TEMPERATURE,
                        "alpha_schedule": ALPHA_SCHEDULE,
                        "blank_logit_reduction": BLANK_LOGIT_REDUCTION,
                        "teacher": "V5",
                        "student_init": "V6_distill_v1",
                    },
                },
                save_path,
            )
            print(f"  -> Saved best model: balanced_acc={balanced_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 保存最终模型
    final_path = os.path.join(OUTPUT_DIR, f"final_{MODEL_NAME}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "val_clean_acc": clean_seq_acc,
            "val_letterbox_acc": lb_seq_acc,
            "val_balanced_acc": balanced_acc,
        },
        final_path,
    )
    print(f"\nTraining complete. Best Val Balanced Acc: {best_balanced_acc:.1f}%")


if __name__ == "__main__":
    main()
