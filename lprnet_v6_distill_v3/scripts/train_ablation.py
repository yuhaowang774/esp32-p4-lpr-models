"""V6 蒸馏 V2 消融实验训练脚本

通过命令行参数控制各改进项的开关，训练消融变体。

消融变体设计：
- A0: V2 完整版（baseline，已训练）
- A1: 移除 letterbox 增强（letterbox_prob=0）
- A2: 移除省份均衡采样（普通 RandomSampler，province_weights=None）
- A3: 移除 blank_logit_reduction（设为 0.0）
- A4: 移除右侧 padding（right_pad_prob=0）
- A5: 固定 alpha=0.5（不渐进调度）

用法：
  python train_ablation.py --variant A1 --epochs 20
  python train_ablation.py --variant A2 --epochs 20
  ...
"""

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from model_v5 import LPRNetV5
from model_v6 import LPRNetV6
from chars import IDX_TO_CHAR, CHAR_TO_IDX, PROVINCES
from distill_loss_v2 import DistillCTCLossV2
from dataset_v2 import LPRDatasetV2, compute_province_weights, collate_fn
from augmentation_v2 import LPRAugmentationV6DistillV2
from train_lprnet_v6_distill_v2 import (
    load_teacher_model, load_student_model, get_phase,
    decode_predictions, validate, validate_balanced,
)
from config import *


def get_alpha_ablation(epoch, fixed_alpha=None):
    """消融实验的 alpha 调度：fixed_alpha 不为 None 时固定 alpha"""
    if fixed_alpha is not None:
        return fixed_alpha
    return train_lprnet_v6_distill_v2_get_alpha(epoch)


def train_lprnet_v6_distill_v2_get_alpha(epoch):
    """导入原 get_alpha 函数"""
    from train_lprnet_v6_distill_v2 import get_alpha
    return get_alpha(epoch)


def train_one_epoch_ablation(teacher, student, loss_fn, dataloader, optimizer,
                              scheduler, scaler, use_amp, device, epoch,
                              fixed_alpha=None):
    """消融实验训练一个 epoch"""
    import train_lprnet_v6_distill_v2 as base
    teacher.eval()
    student.train()

    phase = get_phase(epoch)
    underlying = dataloader.dataset
    while hasattr(underlying, "dataset"):
        underlying = underlying.dataset
    if hasattr(underlying, "set_phase"):
        underlying.set_phase(phase)

    # alpha 调度：fixed_alpha 不为 None 时固定
    if fixed_alpha is not None:
        alpha = fixed_alpha
    else:
        alpha = train_lprnet_v6_distill_v2_get_alpha(epoch)
    loss_fn.alpha = alpha

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            input_lengths = torch.full(
                (images.size(0),), student_logits.size(0),
                dtype=torch.long, device=device)
            loss, metrics = loss_fn(
                student_logits, teacher_logits, labels,
                input_lengths, label_lengths)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = decode_predictions(student_logits)
            offset = 0
            for i, length in enumerate(label_lengths):
                length = length.item()
                target_chars = "".join(
                    IDX_TO_CHAR.get(labels[offset + j].item(), "")
                    for j in range(length))
                if preds[i] == target_chars:
                    correct += 1
                total += 1
                offset += length

        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = correct / max(total, 1) * 100
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                  f"phase={phase} alpha={alpha:.1f} "
                  f"Loss: {avg_loss:.4f} Acc: {acc:.1f}%")

    return total_loss / len(dataloader), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="V6 蒸馏 V2 消融实验")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["A1", "A2", "A3", "A4", "A5"],
                        help="消融变体名称")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练 epochs 数（默认 20）")
    args = parser.parse_args()

    # 根据变体设置消融参数
    no_letterbox = args.variant == "A1"
    no_province_balance = args.variant == "A2"
    no_blank_reduction = args.variant == "A3"
    no_right_padding = args.variant == "A4"
    fixed_alpha = 0.5 if args.variant == "A5" else None

    variant_name = f"ablation_{args.variant}"
    output_dir = os.path.join(OUTPUT_DIR, variant_name)
    os.makedirs(output_dir, exist_ok=True)

    # 日志 tee
    log_path = os.path.join(output_dir, "training_log.txt")
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)

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

    sys.stdout = _Tee(sys.stdout, log_fp)
    sys.stderr = _Tee(sys.stderr, log_fp)

    print(f"\n{'='*60}")
    print(f"消融实验变体: {args.variant}")
    print(f"  no_letterbox: {no_letterbox}")
    print(f"  no_province_balance: {no_province_balance}")
    print(f"  no_blank_reduction: {no_blank_reduction}")
    print(f"  no_right_padding: {no_right_padding}")
    print(f"  fixed_alpha: {fixed_alpha}")
    print(f"  epochs: {args.epochs}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("Loading teacher model (V5)...")
    teacher = load_teacher_model().to(device)
    print("Loading student model (V6, from V6 distill V1 weights)...")
    student = load_student_model().to(device)

    # 数据增强配置
    if no_letterbox:
        # A1: 禁用 letterbox，仅保留其他增强
        augmentation = LPRAugmentationV6DistillV2(phase='warmup',
                                                   disable_letterbox=True)
    else:
        augmentation = LPRAugmentationV6DistillV2(phase='warmup')

    if no_right_padding:
        # A4: 禁用右侧 padding
        augmentation.right_pad_prob = 0.0

    normalize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 训练集
    train_dataset = LPRDatasetV2(
        txt_path=TRAIN_TXT,
        img_dir=os.path.join(DATA_DIR, "train"),
        transform=normalize_transform,
        aug=augmentation,
        filter_types=FILTER_TYPES,
    )

    # 采样器配置
    if no_province_balance:
        # A2: 普通随机采样
        sampler = RandomSampler(train_dataset)
        province_class_weights = None
        print("使用普通 RandomSampler（无省份均衡）")
    else:
        sample_weights, province_class_weights, _ = compute_province_weights(
            train_dataset, max_weight=PROVINCE_WEIGHT_CAP)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        print("使用 WeightedRandomSampler（省份均衡）")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
        drop_last=True, persistent_workers=True)

    # 双验证集
    val_clean_dataset = LPRDatasetV2(
        txt_path=VAL_TXT, img_dir=os.path.join(DATA_DIR, "val"),
        transform=normalize_transform, aug=None, force_letterbox=False,
        filter_types=FILTER_TYPES)
    val_letterbox_dataset = LPRDatasetV2(
        txt_path=VAL_TXT, img_dir=os.path.join(DATA_DIR, "val"),
        transform=normalize_transform, aug=None, force_letterbox=True,
        letterbox_seed=VAL_LETTERBOX_SEED, filter_types=FILTER_TYPES)
    val_clean_loader = DataLoader(
        val_clean_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True)
    val_letterbox_loader = DataLoader(
        val_letterbox_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True)

    # 损失函数配置
    blank_reduction = 0.0 if no_blank_reduction else BLANK_LOGIT_REDUCTION
    loss_fn = DistillCTCLossV2(
        temperature=DISTILL_TEMPERATURE,
        alpha=DISTILL_ALPHA,
        blank_weight=BLANK_WEIGHT,
        blank_logit_reduction=blank_reduction,
        province_aux_weight=PROVINCE_AUX_WEIGHT,
        province_weights=province_class_weights,
    )
    print(f"blank_logit_reduction: {blank_reduction}")

    # 优化器
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    use_amp = torch.cuda.is_available() and USE_AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=WARMUP_PCT,
        final_div_factor=100)

    # 训练循环
    best_balanced_acc = 0
    patience_counter = 0
    ablation_epochs = args.epochs

    print(f"\nStarting ablation {args.variant} training ({ablation_epochs} epochs)\n")

    for epoch in range(1, ablation_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch_ablation(
            teacher, student, loss_fn, train_loader, optimizer, scheduler,
            scaler, use_amp, device, epoch, fixed_alpha)

        clean_seq, clean_char = validate(student, val_clean_loader, device, use_amp)
        lb_seq, lb_char = validate(student, val_letterbox_loader, device, use_amp)
        balanced_acc = VAL_CLEAN_WEIGHT * clean_seq + VAL_LETTERBOX_WEIGHT * lb_seq
        province_acc = validate_balanced(student, val_clean_loader, device, use_amp)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{ablation_epochs} ({elapsed:.0f}s) "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Clean: {clean_seq:.1f}%/{clean_char:.1f}% "
              f"LB: {lb_seq:.1f}%/{lb_char:.1f}% "
              f"Balanced: {balanced_acc:.1f}% Prov: {province_acc:.1f}%")

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            patience_counter = 0
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "variant": args.variant,
                "model_state_dict": student.state_dict(),
                "val_clean_acc": clean_seq,
                "val_letterbox_acc": lb_seq,
                "val_balanced_acc": balanced_acc,
                "val_province_acc": province_acc,
                "ablation_config": {
                    "no_letterbox": no_letterbox,
                    "no_province_balance": no_province_balance,
                    "blank_logit_reduction": blank_reduction,
                    "no_right_padding": no_right_padding,
                    "fixed_alpha": fixed_alpha,
                    "epochs": ablation_epochs,
                },
            }, save_path)
            print(f"  -> Saved best: balanced_acc={balanced_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nAblation {args.variant} complete. Best Val Balanced Acc: {best_balanced_acc:.1f}%")


if __name__ == "__main__":
    main()
