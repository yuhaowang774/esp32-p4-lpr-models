"""
对 phase2_qat best.pt 在 CCPD2020 验证集上做 per-class 评估
输出蓝牌/绿牌各自的 P/R/mAP50/mAP50-95
"""
import os
from ultralytics import YOLO
from pathlib import Path

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

WEIGHTS = str(OUTPUT_DIR / "phase2_qat" / "weights" / "best.pt")
DATA = str(Path(CCPD_AUG_DIR) / "data.yaml")


def main():
    print("=" * 60)
    print("Per-class Evaluation (phase2_qat best.pt)")
    print("=" * 60)
    print(f"Weights: {WEIGHTS}")
    print(f"Data: {DATA}")
    print("=" * 60)

    model = YOLO(WEIGHTS)
    metrics = model.val(data=DATA, split="val", verbose=True, plots=True, workers=0)

    print("\n" + "=" * 60)
    print("Per-class Results")
    print("=" * 60)
    names = metrics.names
    print(f"Class names: {names}")

    # per-class metrics: class_result(i) 返回 [P, R, mAP50, mAP50-95]
    for i, name in names.items():
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"\nClass {i} ({name}):")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  mAP50:     {ap50:.4f}")
        print(f"  mAP50-95:  {ap:.4f}")

    print("\n" + "=" * 60)
    print(f"Overall mAP50: {metrics.box.map50:.4f}")
    print(f"Overall mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

