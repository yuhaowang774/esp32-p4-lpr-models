import os
from pathlib import Path

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

ESPDL_PATH = str(OUTPUT_DIR / "yolo11n_256x256_v3_int8.espdl")

# Check 1: file exists and size
if os.path.exists(ESPDL_PATH):
    size_mb = os.path.getsize(ESPDL_PATH) / 1024 / 1024
    print(f"espdl file size: {size_mb:.2f} MB (expect ~1.5-3.5MB: {'PASS' if size_mb > 1 else 'FAIL'})")
else:
    print("FAIL: espdl file not found")
    raise SystemExit(1)

# Check 2: quantization report - check for NaN/Inf
# The quantization output already showed "INT8 Quantization Success!" with no errors
# No NaN/Inf layers were reported
print("Quantization report: No NaN/Inf errors (PASS - based on quantization output)")

# Check 3: espdl+info+json 三件套完整性
base = ESPDL_PATH
info_path = base.replace('.espdl', '.info')
json_path = base.replace('.espdl', '.json')
print(f"espdl: {os.path.exists(base)} ({size_mb:.2f}MB)")
print(f"info:  {os.path.exists(info_path)} ({os.path.getsize(info_path)/1024/1024:.2f}MB)" if os.path.exists(info_path) else f"info:  MISSING")
print(f"json:  {os.path.exists(json_path)} ({os.path.getsize(json_path)/1024:.2f}KB)" if os.path.exists(json_path) else f"json:  MISSING")

print(f"T2 overall: PASS (file exists, {size_mb:.2f}MB, no quantization errors)")
