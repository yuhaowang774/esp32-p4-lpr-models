"""V6 蒸馏 V3 test 集评估

加载 final 模型（V3 因从 V2 best 初始化导致早停，best 仅 epoch 1 权重，
final 为 epoch 16 权重，包含更多 V3 训练），在 test 集上评估：
- Clean 准确率（序列/字符）
- Letterbox 准确率（序列/字符）
- Balanced 准确率
- 省份均衡准确率
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_v6 import LPRNetV6
from dataset_v2 import LPRDatasetV2, collate_fn
from train_lprnet_v6_distill_v3 import validate, validate_balanced
from config import *


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 final 模型（V3 改用 final，原因见 export_onnx.py 注释）
    best_path = os.path.join(OUTPUT_DIR, f"final_{MODEL_NAME}.pth")
    print(f"Loading V3 final model from {best_path}...")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model = LPRNetV6(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 打印 checkpoint 中的训练信息
    if "epoch" in checkpoint:
        print(f"  Best epoch: {checkpoint['epoch']}")
    if "val_balanced_acc" in checkpoint:
        print(f"  Val balanced acc: {checkpoint['val_balanced_acc']:.1f}%")

    # transform
    normalize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # test 集：clean + letterbox
    test_clean = LPRDatasetV2(
        txt_path=TEST_TXT,
        img_dir=os.path.join(DATA_DIR, "test"),
        transform=normalize_transform,
        aug=None,
        force_letterbox=False,
        filter_types=FILTER_TYPES,
    )
    test_letterbox = LPRDatasetV2(
        txt_path=TEST_TXT,
        img_dir=os.path.join(DATA_DIR, "test"),
        transform=normalize_transform,
        aug=None,
        force_letterbox=True,
        letterbox_seed=VAL_LETTERBOX_SEED,
        filter_types=FILTER_TYPES,
    )

    use_amp = torch.cuda.is_available() and USE_AMP

    clean_loader = DataLoader(
        test_clean, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    lb_loader = DataLoader(
        test_letterbox, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    print(f"\nEvaluating on test set ({len(test_clean)} samples)...\n")

    # Clean 评估
    clean_seq, clean_char = validate(model, clean_loader, device, use_amp)
    print(f"Test Clean:    Seq={clean_seq:.2f}%  Char={clean_char:.2f}%")

    # Letterbox 评估
    lb_seq, lb_char = validate(model, lb_loader, device, use_amp)
    print(f"Test Letterbox: Seq={lb_seq:.2f}%  Char={lb_char:.2f}%")

    # Balanced
    balanced = VAL_CLEAN_WEIGHT * clean_seq + VAL_LETTERBOX_WEIGHT * lb_seq
    print(f"Test Balanced: {balanced:.2f}%")

    # 省份均衡
    prov_acc = validate_balanced(model, clean_loader, device, use_amp)
    print(f"Test Province: {prov_acc:.2f}%")

    print(f"\n===== V6 蒸馏 V3 Test 评估完成 =====")
    print(f"  Clean:     {clean_seq:.2f}% / {clean_char:.2f}% (seq/char)")
    print(f"  Letterbox: {lb_seq:.2f}% / {lb_char:.2f}% (seq/char)")
    print(f"  Balanced:  {balanced:.2f}%")
    print(f"  Province:  {prov_acc:.2f}%")


if __name__ == "__main__":
    main()
