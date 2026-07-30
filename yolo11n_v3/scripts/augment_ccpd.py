import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import shutil

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

SRC_DIR = Path(CCPD_DIR)
DST_DIR = Path(CCPD_AUG_DIR)
FILL_COLOR = (114, 114, 114)
CROP_SCALES = [2, 3, 4]
RANDOM_OFFSET_MAX = 0.1

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_label(label_path):
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            objects.append((cls, cx, cy, w, h))
    return objects


def augment_image(img, objects, scale_factor):
    h, w = img.shape[:2]
    crop_w = int(w / scale_factor)
    crop_h = int(h / scale_factor)

    if not objects:
        return None, None

    plate_cx = np.mean([o[1] for o in objects])
    plate_cy = np.mean([o[2] for o in objects])

    center_x = int(plate_cx * w)
    center_y = int(plate_cy * h)

    max_offset_x = int(crop_w * RANDOM_OFFSET_MAX)
    max_offset_y = int(crop_h * RANDOM_OFFSET_MAX)
    offset_x = np.random.randint(-max_offset_x, max_offset_x + 1)
    offset_y = np.random.randint(-max_offset_y, max_offset_y + 1)
    center_x += offset_x
    center_y += offset_y

    crop_x1 = center_x - crop_w // 2
    crop_y1 = center_y - crop_h // 2
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(w, crop_x2)
    src_y2 = min(h, crop_y2)

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop_img = np.full((crop_h, crop_w, 3), FILL_COLOR, dtype=np.uint8)
    crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    result_img = cv2.resize(crop_img, (w, h))

    new_objects = []
    for cls, cx, cy, bw, bh in objects:
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h

        new_abs_cx = (abs_cx - crop_x1) * (w / crop_w)
        new_abs_cy = (abs_cy - crop_y1) * (h / crop_h)
        new_abs_bw = abs_bw * (w / crop_w)
        new_abs_bh = abs_bh * (h / crop_h)

        new_cx = new_abs_cx / w
        new_cy = new_abs_cy / h
        new_bw = new_abs_bw / w
        new_bh = new_abs_bh / h

        plate_x1 = new_cx - new_bw / 2
        plate_y1 = new_cy - new_bh / 2
        plate_x2 = new_cx + new_bw / 2
        plate_y2 = new_cy + new_bh / 2

        if plate_x2 <= 0 or plate_y2 <= 0 or plate_x1 >= 1 or plate_y1 >= 1:
            continue

        plate_x1 = max(0, plate_x1)
        plate_y1 = max(0, plate_y1)
        plate_x2 = min(1, plate_x2)
        plate_y2 = min(1, plate_y2)

        new_cx = (plate_x1 + plate_x2) / 2
        new_cy = (plate_y1 + plate_y2) / 2
        new_bw = plate_x2 - plate_x1
        new_bh = plate_y2 - plate_y1

        if new_bw < 0.005 or new_bh < 0.005:
            continue

        new_objects.append((cls, new_cx, new_cy, new_bw, new_bh))

    if not new_objects:
        return None, None

    return result_img, new_objects


def process_split(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in os.listdir(src_img_dir) if Path(f).suffix.lower() in IMG_EXTS]
    print(f"  Found {len(image_files)} images")

    orig_count = 0
    aug_count = 0

    for img_name in tqdm(image_files, desc="  Processing"):
        img_path = src_img_dir / img_name
        lbl_name = Path(img_name).stem + ".txt"
        lbl_path = src_lbl_dir / lbl_name

        shutil.copy2(str(img_path), str(dst_img_dir / img_name))
        if lbl_path.exists():
            shutil.copy2(str(lbl_path), str(dst_lbl_dir / lbl_name))
        orig_count += 1

        if not lbl_path.exists():
            continue

        objects = parse_label(lbl_path)
        if not objects:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for scale in CROP_SCALES:
            aug_img, aug_objects = augment_image(img, objects, scale)
            if aug_img is None:
                continue

            stem = Path(img_name).stem
            suffix = Path(img_name).suffix
            aug_img_name = f"{stem}_x{scale}{suffix}"
            aug_lbl_name = f"{stem}_x{scale}.txt"

            cv2.imwrite(str(dst_img_dir / aug_img_name), aug_img)

            with open(dst_lbl_dir / aug_lbl_name, "w") as f:
                for cls, cx, cy, bw, bh in aug_objects:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            aug_count += 1

    return orig_count, aug_count


def main():
    print("=" * 60)
    print("CCPD2020 Large-Target Augmentation")
    print("=" * 60)
    print(f"Source: {SRC_DIR}")
    print(f"Output: {DST_DIR}")
    print(f"Crop scales: {CROP_SCALES}x")
    print(f"Fill color: {FILL_COLOR}")
    print()

    np.random.seed(42)

    total_orig = 0
    total_aug = 0

    for split in ["train", "val", "test"]:
        src_img_dir = SRC_DIR / "images" / split
        src_lbl_dir = SRC_DIR / "labels" / split

        if not src_img_dir.exists():
            print(f"  Skip {split} (not found)")
            continue

        dst_img_dir = DST_DIR / "images" / split
        dst_lbl_dir = DST_DIR / "labels" / split

        print(f"\n[{split}]")
        orig, aug = process_split(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir)
        total_orig += orig
        total_aug += aug
        print(f"  Original: {orig}, Augmented: {aug}, Total: {orig + aug}")

    data_yaml = {
        "path": str(DST_DIR),
        "train": "images\\train",
        "val": "images\\val",
        "test": "images\\test",
        "nc": 2,
        "names": {0: "blue_plate", 1: "green_plate"},
    }
    yaml_path = DST_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"Total original: {total_orig}")
    print(f"Total augmented: {total_aug}")
    print(f"Data config: {yaml_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
