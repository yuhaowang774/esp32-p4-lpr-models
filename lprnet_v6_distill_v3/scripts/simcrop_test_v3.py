"""V6 蒸馏 V3 PC 端 simcrop 测试

复现 ESP32-P4 端侧推理流程：
1. 读取 simcrop 图片（已模拟 YOLO 裁切）
2. letterbox 到 128x32（等比缩放，填充 114,114,114）
3. 归一化（mean/std）
4. V3 PyTorch 模型推理
5. CTC 解码（贪心解码 + V5 后处理救援，复现 lp_char_recognize.cpp 逻辑）
6. 保存结果为 batch_results.json 格式
7. 分析绿牌末尾字符丢失率

用途：在 PC 上验证 V3 模型是否解决绿牌末尾字符丢失问题，
无需烧录 ESP32-P4 即可快速评估。
"""

import sys
import os
import json
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_v6 import LPRNetV6
from chars import IDX_TO_CHAR, CHAR_TO_IDX
from config import *


# ===== ESP32 端 letterbox 逻辑复现 =====
def letterbox_esp32(img, target_w=128, target_h=32, fill_color=(114, 114, 114)):
    """等比缩放 + 灰色填充，复现 ESP32 app_image_scale_with_aspect_ratio"""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


# ===== ESP32 端 CTC 解码逻辑复现 =====
PLATE_PROVINCE_CODES = [
    " ", "京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘",
    "皖", "鲁", "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋",
    "蒙", "陕", "吉", "闽", "贵", "粤", "青", "藏", "川", "宁", "琼"
]
PLATE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
PLATE_DIGITS = "0123456789"
CHAR_NUM = 66
PROVINCE_NUM = 32
LETTER_START = 32
DIGIT_START = 56


def idx_to_char(idx):
    """索引转字符，复现 lp_char_recognize.cpp 逻辑"""
    if idx >= 1 and idx < PROVINCE_NUM:
        return PLATE_PROVINCE_CODES[idx]
    elif idx >= LETTER_START and idx < DIGIT_START:
        return PLATE_LETTERS[idx - LETTER_START]
    elif idx >= DIGIT_START and idx < CHAR_NUM:
        return PLATE_DIGITS[idx - DIGIT_START]
    return ""


def ctc_decode_esp32(output, seq_len, num_classes, max_chars, layout_batch_classes_seq=False,
                     enable_rescue=True):
    """CTC 解码，复现 lp_char_recognize.cpp 逻辑

    Args:
        output: numpy array [seq_len, num_classes] 或 [num_classes, seq_len]
        seq_len: 时间步长度
        num_classes: 类别数
        max_chars: 最大字符数（蓝牌7，绿牌8）
        layout_batch_classes_seq: True=output[c][t], False=output[t][c]
        enable_rescue: 是否启用 V5 后处理救援
    Returns:
        result: 解码字符串
    """
    # 统一为 2D 数组索引
    result = ""
    last_char = -1
    char_count = 0
    blank_bias = 0.0  # 原始值，与 lp_char_recognize.cpp 一致

    for t in range(seq_len):
        max_idx = 0
        max_val = -1e9
        for c in range(num_classes):
            if layout_batch_classes_seq:
                val = float(output[c, t])
            else:
                val = float(output[t, c])
            if c == 0:
                val += blank_bias
            if val > max_val:
                max_val = val
                max_idx = c

        if max_idx != last_char and max_idx != 0 and max_idx < CHAR_NUM and max_idx >= 0:
            result += idx_to_char(max_idx)
            char_count += 1
            if char_count >= max_chars:
                break
        last_char = max_idx

    # V5 后处理救援
    if enable_rescue and char_count < max_chars and char_count > 0:
        best_logit = -1e9
        best_idx = 0
        search_start = seq_len - 5
        if search_start < 0:
            search_start = 0
        for t in range(search_start, seq_len):
            for c in range(1, num_classes):
                if layout_batch_classes_seq:
                    val = float(output[c, t])
                else:
                    val = float(output[t, c])
                if val > best_logit:
                    best_logit = val
                    best_idx = c
        if best_idx != 0 and best_idx != last_char and best_idx < CHAR_NUM:
            result += idx_to_char(best_idx)
            char_count += 1

    return result


# ===== 主测试流程 =====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 V3 final 模型
    model_path = os.path.join(OUTPUT_DIR, f"final_{MODEL_NAME}.pth")
    print(f"Loading V3 final model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LPRNetV6(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 加载标签
    simcrop_dir = os.environ.get("SIMCROP_DIR", "./data/simcrop_test")
    labels_file = os.path.join(simcrop_dir, "labels.json")
    images_dir = os.path.join(simcrop_dir, "images")
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    print(f"Labels: {len(labels_data)} samples")

    # 归一化 transform
    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = []
    total = len(labels_data)
    correct = 0
    green_total = 0
    green_correct = 0
    green_missing_last = 0
    blue_total = 0
    blue_correct = 0

    # 绿牌末尾字符丢失统计
    green_errors = []

    print(f"\nRunning simcrop test on {total} samples...")
    t_start = time.time()

    with torch.no_grad():
        for i, label in enumerate(labels_data):
            fname = label["file_name"]
            gt = label["plate_number"]
            color = label.get("plate_color", "unknown")
            img_path = os.path.join(images_dir, fname)

            if not os.path.exists(img_path):
                continue

            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # letterbox 到 128x32
            img_lb = letterbox_esp32(img, 128, 32)

            # 归一化
            img_tensor = normalize(img_lb).unsqueeze(0).to(device)

            # 推理
            t0 = time.time()
            output = model(img_tensor)
            inference_ms = (time.time() - t0) * 1000

            # CTC 解码
            # 模型输出形状 [T, B, C] = [31, 1, 66]，取 batch 0
            output_np = output[:, 0, :].cpu().numpy()  # [T, C] = [31, 66]
            seq_len = output_np.shape[0]
            num_classes = output_np.shape[1]

            # max_chars: 绿牌8, 蓝牌7
            max_chars = 8 if color == "green" else 7
            pred = ctc_decode_esp32(output_np, seq_len, num_classes, max_chars,
                                    layout_batch_classes_seq=False, enable_rescue=True)

            # 统计
            is_correct = (pred == gt)
            if is_correct:
                correct += 1

            if color == "green":
                green_total += 1
                if is_correct:
                    green_correct += 1
                else:
                    # 检查是否为末尾字符丢失
                    if len(pred) == len(gt) - 1 and pred == gt[:-1]:
                        green_missing_last += 1
                    green_errors.append({
                        "file": fname,
                        "gt": gt,
                        "pred": pred,
                        "aspect_ratio": label.get("aspect_ratio", 0),
                        "padding": label.get("padding_pixels", 0),
                    })
            elif color == "blue":
                blue_total += 1
                if is_correct:
                    blue_correct += 1

            results.append({
                "filename": fname,
                "plate_number": pred,
                "inference_ms": inference_ms,
                "img_h": label.get("sim_crop_h", 80),
                "aspect_ratio": label.get("aspect_ratio", 0),
                "padding_pixels": label.get("padding_pixels", 0),
            })

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{total}] elapsed={elapsed:.0f}s acc={correct/(i+1)*100:.2f}%")

    elapsed = time.time() - t_start
    print(f"\n===== V6 蒸馏 V3 PC simcrop 测试完成 =====")
    print(f"耗时: {elapsed:.1f}s")
    print(f"")
    print(f"总体:")
    print(f"  样本数: {len(results)}")
    print(f"  完全匹配: {correct} ({correct/len(results)*100:.2f}%)")
    print(f"")
    print(f"蓝牌:")
    print(f"  样本数: {blue_total}")
    print(f"  完全匹配: {blue_correct} ({blue_correct/blue_total*100 if blue_total else 0:.2f}%)")
    print(f"")
    print(f"绿牌:")
    print(f"  样本数: {green_total}")
    print(f"  完全匹配: {green_correct} ({green_correct/green_total*100 if green_total else 0:.2f}%)")
    print(f"  末尾字符丢失: {green_missing_last} ({green_missing_last/green_total*100 if green_total else 0:.2f}%)")
    print(f"  其他错误: {green_total - green_correct - green_missing_last}")

    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, "simcrop_test_v3_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "V6_distill_V3_final",
            "total_images": len(results),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_file}")

    # 保存绿牌错误详情
    if green_errors:
        errors_file = os.path.join(OUTPUT_DIR, "simcrop_test_v3_green_errors.json")
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump(green_errors, f, ensure_ascii=False, indent=2)
        print(f"绿牌错误详情: {errors_file}")

    # V2 对比基准
    print(f"\n===== V2 vs V3 对比 =====")
    print(f"  V2 simcrop 完全匹配率: 95.48% (ESP32 实测)")
    print(f"  V2 绿牌末尾字符丢失: 267 个 (8.6%)")
    print(f"  V3 simcrop 完全匹配率: {correct/len(results)*100:.2f}% (PC float)")
    print(f"  V3 绿牌末尾字符丢失: {green_missing_last} ({green_missing_last/green_total*100 if green_total else 0:.2f}%)")


if __name__ == "__main__":
    main()
