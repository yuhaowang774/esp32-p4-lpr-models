import os
import sys
from pathlib import Path

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

sys.path.insert(0, os.environ.get("TRAIN_SCRIPTS_DIR", "./scripts/train"))
# NOTE: train_yolo11n_lp.py 定义了 DSConv 和 replace_conv_with_dsconv，用于评估
# 使用深度可分离卷积的轻量化模型。此模块不包含在本仓库中，需要单独提供。
# 如需运行此脚本，请通过 TRAIN_SCRIPTS_DIR 环境变量指向包含该模块的目录。
from train_yolo11n_lp import DSConv, replace_conv_with_dsconv
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

tasks.DSConv = DSConv

DATA_YAML = str(OUTPUT_DIR.parent / "train_lp" / "dataset_lp.yaml")
PHASE1_WEIGHT = str(OUTPUT_DIR.parent / "train_lp" / "phase1_best.pt")
PHASE2_WEIGHT = str(OUTPUT_DIR.parent / "train_lp" / "phase2_best.pt")


def evaluate(weight_path, label):
    print("\n" + "=" * 60)
    print(f"Evaluating: {label}")
    print(f"Weight: {weight_path}")
    print("=" * 60)

    model = YOLO(weight_path)
    replace_conv_with_dsconv(model.model)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=320,
        batch=8,
        device="0",
        single_cls=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"{label} Test Results:")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    m1 = evaluate(PHASE1_WEIGHT, "Phase1 (150ep)")
    m2 = evaluate(PHASE2_WEIGHT, "Phase2 QAT (30ep)")

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<15} {'Phase1':>10} {'Phase2 QAT':>10}")
    print("-" * 40)
    print(f"{'Precision':<15} {m1.box.mp:>10.4f} {m2.box.mp:>10.4f}")
    print(f"{'Recall':<15} {m1.box.mr:>10.4f} {m2.box.mr:>10.4f}")
    print(f"{'mAP50':<15} {m1.box.map50:>10.4f} {m2.box.map50:>10.4f}")
    print(f"{'mAP50-95':<15} {m1.box.map:>10.4f} {m2.box.map:>10.4f}")
    print("=" * 60)
