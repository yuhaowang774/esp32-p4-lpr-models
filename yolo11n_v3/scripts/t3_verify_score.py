import os
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
import cv2

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

ONNX_PATH = str(OUTPUT_DIR / "yolo11n_256x256_v3_fp32.onnx")
VAL_IMAGE_DIR = str(Path(CCPD_DIR) / "images" / "val")
IMG_SIZE = 256
NUM_TEST_IMAGES = 5
SCORE_THRESH = 0.35
STRIDES = [8, 16, 32]


def preprocess(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(IMG_SIZE / h, IMG_SIZE / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    pad_h = (IMG_SIZE - new_h) // 2
    pad_w = (IMG_SIZE - new_w) // 2
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
    img = padded.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


def main():
    print("=" * 60)
    print("T3: YOLO11n 256x256 FP32 Score Verification")
    print("=" * 60)

    val_dir = Path(VAL_IMAGE_DIR)
    all_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    if not all_images:
        print(f"[ERROR] No validation images found: {VAL_IMAGE_DIR}")
        return False

    np.random.seed(42)
    selected = np.random.choice(len(all_images), min(NUM_TEST_IMAGES, len(all_images)), replace=False)
    test_images = [all_images[i] for i in selected]
    print(f"Test images: {len(test_images)}")

    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"Input: {input_name}")
    print(f"Outputs: {output_names}")

    pass_count = 0
    for idx, img_path in enumerate(test_images):
        img_np = preprocess(img_path)
        input_np = np.expand_dims(img_np, axis=0).astype(np.float32)
        outputs = session.run(None, {input_name: input_np})

        score_outputs = [(i, outputs[i]) for i in range(len(output_names)) if output_names[i].startswith("score")]
        has_detection = False
        for i, score in score_outputs:
            score_sig = 1.0 / (1.0 + np.exp(-score))
            max_score = float(np.max(score_sig))
            candidates = int(np.sum(score_sig > SCORE_THRESH))
            print(f"  Image {idx+1} {output_names[i]}: max_score={max_score:.4f}, candidates>{SCORE_THRESH}={candidates}")
            if candidates > 0:
                has_detection = True

        status = "PASS" if has_detection else "FAIL"
        print(f"  Image {idx+1}: {status}")
        if has_detection:
            pass_count += 1

    print(f"\nT3 Result: {pass_count}/{NUM_TEST_IMAGES} images have score output")
    overall = "PASS" if pass_count == NUM_TEST_IMAGES else "FAIL"
    print(f"T3 overall: {overall}")
    return overall == "PASS"


if __name__ == "__main__":
    main()
