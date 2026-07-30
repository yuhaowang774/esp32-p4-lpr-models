import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from patch_esp_ppq import apply_patch

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

ONNX_PATH = str(OUTPUT_DIR / "yolo11n_320x320_fp32.onnx")
ESPDL_PATH = str(OUTPUT_DIR / "yolo11n_320x320_v3_int8.espdl")
CALIBRATION_IMAGES_DIR = str(Path(CCPD_AUG_DIR) / "images" / "train")
NUM_CALIB_IMAGES = 200
IMG_SIZE = 320


class CalibrationDataset:
    def __init__(self, image_dir, img_size, num_samples):
        self.img_size = img_size
        self.image_files = []

        image_dir = Path(image_dir)
        all_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        np.random.seed(42)
        selected = np.random.choice(
            len(all_images), min(num_samples, len(all_images)), replace=False
        )
        self.image_files = [all_images[i] for i in selected]

        print("[INFO] Selected %d calibration images" % len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img
        img = padded.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)


def collate_fn(batch):
    if isinstance(batch, torch.Tensor):
        return batch.to("cpu")
    return torch.stack(batch).to("cpu")


def main():
    print("=" * 60)
    print("YOLO11n 320x320 ESP-PPQ INT8 Quantization")
    print("=" * 60)
    print("Input ONNX: %s" % ONNX_PATH)
    print("Output ESPDL: %s" % ESPDL_PATH)
    print("Input size: %dx%d" % (IMG_SIZE, IMG_SIZE))
    print("Calibration images: %d" % NUM_CALIB_IMAGES)
    print("Target: esp32p4")
    print("=" * 60)

    if not os.path.exists(ONNX_PATH):
        print("[ERROR] ONNX file not found: %s" % ONNX_PATH)
        return False

    apply_patch()

    print("\n[1/3] Preparing calibration data...")
    calib_dataset = CalibrationDataset(
        CALIBRATION_IMAGES_DIR, IMG_SIZE, NUM_CALIB_IMAGES
    )
    calib_dataloader = DataLoader(
        dataset=calib_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("\n[2/3] Configuring quantization...")
    quant_setting = QuantizationSettingFactory.espdl_setting()

    print("\n[3/3] Running INT8 quantization...")
    try:
        quant_ppq_graph = espdl_quantize_onnx(
            onnx_import_file=ONNX_PATH,
            espdl_export_file=ESPDL_PATH,
            calib_dataloader=calib_dataloader,
            calib_steps=NUM_CALIB_IMAGES,
            input_shape=[1, 3, IMG_SIZE, IMG_SIZE],
            target="esp32p4",
            num_of_bits=8,
            collate_fn=collate_fn,
            setting=quant_setting,
            device="cpu",
            error_report=True,
            skip_export=False,
            export_test_values=True,
            verbose=1,
        )

        print("\n" + "=" * 60)
        print("INT8 Quantization Success!")
        print("=" * 60)
        print("Output: %s" % ESPDL_PATH)
        if os.path.exists(ESPDL_PATH):
            size_mb = os.path.getsize(ESPDL_PATH) / 1024 / 1024
            print("File size: %.2f MB" % size_mb)
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("INT8 Quantization Failed!")
        print("=" * 60)
        print("Error: %s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
