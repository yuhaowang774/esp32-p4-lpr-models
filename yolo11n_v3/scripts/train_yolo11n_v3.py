import os
from pathlib import Path
from ultralytics import YOLO

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

DATA_PATH = CCPD_AUG_DIR
MODEL_NAME = PRETRAINED_WEIGHTS
IMG_SIZE = 320
EPOCHS_PHASE1 = 100
EPOCHS_PHASE2 = 30
BATCH_SIZE = 16  # RTX 3050 4GB: batch=32 + degrees=30 会 OOM
DEVICE = "0"

SAVE_DIR = OUTPUT_DIR
PHASE1_WEIGHT = SAVE_DIR / "phase1_best.pt"
PHASE2_WEIGHT = SAVE_DIR / "phase2_best.pt"
ONNX_OUTPUT = SAVE_DIR / "yolo11n_320x320_fp32.onnx"


def check_dataset():
    data_yaml_path = Path(DATA_PATH) / "data.yaml"
    if data_yaml_path.exists():
        print(f"[INFO] Dataset config: {data_yaml_path}")
        return str(data_yaml_path)

    train_path = Path(DATA_PATH) / "images" / "train"
    if not train_path.exists():
        print(f"[ERROR] Train dir not found: {train_path}")
        return None

    yaml_content = f"""path: {DATA_PATH}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: blue_plate
  1: green_plate
"""
    yaml_path = SAVE_DIR / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] Dataset config saved: {yaml_path}")
    return str(yaml_path)


def phase1_train(data_yaml):
    print("\n" + "=" * 50)
    print("Phase 1: YOLO11n Normal Training (Augmented Data)")
    print("=" * 50)

    # 如果已有 last.pt，从断点恢复训练
    last_pt = SAVE_DIR / "phase1" / "weights" / "last.pt"
    if last_pt.exists():
        print(f"[INFO] Resuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        model = YOLO(MODEL_NAME)

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS_PHASE1,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        patience=30,  # 启用旋转增强后收敛变慢，patience 从 20 提高到 30
        resume=last_pt.exists(),  # 有 last.pt 时自动续训
        mosaic=0.0,
        scale=0.9,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # 旋转/透视增强（新增，倾斜车牌优化）
        degrees=30.0,        # ±30° 随机旋转
        shear=5.0,            # ±5° 剪切
        perspective=0.0005,  # 轻微透视变换
        save=True,
        save_period=5,
        project=str(SAVE_DIR),
        name="phase1",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        amp=False,
        val=True,
        plots=True,
    )

    best_pt = SAVE_DIR / "phase1" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(PHASE1_WEIGHT))
        print(f"[INFO] Phase1 best weight saved: {PHASE1_WEIGHT}")

    return PHASE1_WEIGHT.exists()


def phase2_qat(data_yaml, weight_path):
    print("\n" + "=" * 50)
    print("Phase 2: YOLO11n QAT Fine-tune")
    print("=" * 50)

    model = YOLO(str(weight_path))

    results = model.train(
        data=data_yaml,
        epochs=EPOCHS_PHASE2,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE // 2,
        device=DEVICE,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1,
        cos_lr=True,
        patience=0,
        mosaic=0.0,
        scale=0.9,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # QAT 阶段减小旋转幅度防止量化退化（新增，倾斜车牌优化）
        degrees=15.0,
        shear=3.0,
        perspective=0.0003,
        save=True,
        save_period=5,
        project=str(SAVE_DIR),
        name="phase2_qat",
        exist_ok=True,
        pretrained=False,
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        amp=False,
        val=True,
        plots=True,
    )

    best_pt = SAVE_DIR / "phase2_qat" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        shutil.copy2(str(best_pt), str(PHASE2_WEIGHT))
        print(f"[INFO] QAT best weight saved: {PHASE2_WEIGHT}")

    return PHASE2_WEIGHT.exists()


def export_onnx(weight_path):
    print("\n" + "=" * 50)
    print("Phase 3: Export ONNX (DetectESPDL)")
    print("=" * 50)

    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from export_onnx_for_espdl import export_yolo11_for_espdl

    export_yolo11_for_espdl(str(weight_path), str(ONNX_OUTPUT), imgsz=IMG_SIZE)
    print(f"[INFO] ONNX exported: {ONNX_OUTPUT}")
    return ONNX_OUTPUT.exists()


def main():
    print("=" * 60)
    print("YOLO11n Training Pipeline v3 (Large-Target Augmented)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Phase1 epochs: {EPOCHS_PHASE1}")
    print(f"Phase2 QAT epochs: {EPOCHS_PHASE2}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Output: {SAVE_DIR}")
    print("=" * 60)

    data_yaml = check_dataset()
    if data_yaml is None:
        print("[ERROR] Dataset config failed")
        return

    if PHASE1_WEIGHT.exists():
        print(f"[INFO] Phase1 weight found, skip training: {PHASE1_WEIGHT}")
    else:
        if not phase1_train(data_yaml):
            print("[ERROR] Phase1 training failed")
            return

    if PHASE2_WEIGHT.exists():
        print(f"[INFO] QAT weight found, skip fine-tune: {PHASE2_WEIGHT}")
    else:
        if not phase2_qat(data_yaml, PHASE1_WEIGHT):
            print("[ERROR] QAT fine-tune failed")
            return

    if ONNX_OUTPUT.exists():
        print(f"[INFO] ONNX already exists, skip export: {ONNX_OUTPUT}")
    else:
        export_onnx(PHASE2_WEIGHT)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)
    print(f"Phase1 weight: {PHASE1_WEIGHT}")
    print(f"QAT weight: {PHASE2_WEIGHT}")
    print(f"ONNX model: {ONNX_OUTPUT}")
    print("=" * 60)
    print("\n[Next] ESP-PPQ quantize ONNX -> .espdl")
    print("       Target: esp32p4, Precision: INT8")


if __name__ == "__main__":
    main()
