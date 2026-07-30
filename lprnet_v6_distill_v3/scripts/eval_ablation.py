"""消融实验 test 集评估

评估所有消融变体在 test 集上的表现，生成对比表格。
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


def eval_variant(variant_name, model_path, device, use_amp):
    """评估单个变体"""
    print(f"\n--- {variant_name} ---")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LPRNetV6(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_clean = LPRDatasetV2(
        txt_path=TEST_TXT, img_dir=os.path.join(DATA_DIR, "test"),
        transform=normalize_transform, aug=None, force_letterbox=False,
        filter_types=FILTER_TYPES)
    test_letterbox = LPRDatasetV2(
        txt_path=TEST_TXT, img_dir=os.path.join(DATA_DIR, "test"),
        transform=normalize_transform, aug=None, force_letterbox=True,
        letterbox_seed=VAL_LETTERBOX_SEED, filter_types=FILTER_TYPES)

    clean_loader = DataLoader(
        test_clean, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    lb_loader = DataLoader(
        test_letterbox, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    clean_seq, clean_char = validate(model, clean_loader, device, use_amp)
    lb_seq, lb_char = validate(model, lb_loader, device, use_amp)
    balanced = VAL_CLEAN_WEIGHT * clean_seq + VAL_LETTERBOX_WEIGHT * lb_seq
    prov_acc = validate_balanced(model, clean_loader, device, use_amp)

    print(f"  Clean: {clean_seq:.2f}%/{clean_char:.2f}%  "
          f"LB: {lb_seq:.2f}%/{lb_char:.2f}%  "
          f"Balanced: {balanced:.2f}%  Prov: {prov_acc:.2f}%")

    return {
        "variant": variant_name,
        "clean_seq": clean_seq, "clean_char": clean_char,
        "lb_seq": lb_seq, "lb_char": lb_char,
        "balanced": balanced, "province": prov_acc,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and USE_AMP
    print(f"Using device: {device}")

    results = []

    # A0: V2 完整版（baseline）
    v2_path = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
    if os.path.exists(v2_path):
        results.append(eval_variant("A0 (V2 完整版)", v2_path, device, use_amp))

    # A1-A5: 消融变体
    variants = {
        "A1 (移除 letterbox)": "ablation_A1",
        "A2 (移除省份均衡)": "ablation_A2",
        "A3 (移除 blank_reduction)": "ablation_A3",
        "A4 (移除右侧 padding)": "ablation_A4",
        "A5 (固定 alpha=0.5)": "ablation_A5",
    }

    for name, dirname in variants.items():
        model_path = os.path.join(OUTPUT_DIR, dirname, "best_model.pth")
        if os.path.exists(model_path):
            results.append(eval_variant(name, model_path, device, use_amp))
        else:
            print(f"\n--- {name} --- 跳过（模型不存在: {model_path}）")

    # 汇总表格
    print(f"\n{'='*90}")
    print(f"{'变体':<28} {'Clean':>10} {'Letterbox':>10} {'Balanced':>10} {'Province':>10}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['variant']:<28} {r['clean_seq']:>9.2f}% {r['lb_seq']:>9.2f}% "
              f"{r['balanced']:>9.2f}% {r['province']:>9.2f}%")
    print(f"{'='*90}")

    # 计算各改进项的贡献（与 A0 的差距）
    if len(results) > 1:
        baseline = results[0]
        print(f"\n各改进项贡献（与 V2 完整版的差距）：")
        print(f"{'改进项':<28} {'Balanced Δ':>12} {'LB Δ':>12} {'Prov Δ':>12}")
        print(f"{'-'*64}")
        for r in results[1:]:
            d_bal = r["balanced"] - baseline["balanced"]
            d_lb = r["lb_seq"] - baseline["lb_seq"]
            d_prov = r["province"] - baseline["province"]
            print(f"{r['variant']:<28} {d_bal:>+11.2f}% {d_lb:>+11.2f}% {d_prov:>+11.2f}%")


if __name__ == "__main__":
    main()
