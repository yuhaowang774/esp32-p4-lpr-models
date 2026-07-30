import onnx
import os
from pathlib import Path

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

ONNX_PATH = str(OUTPUT_DIR / "yolo11n_256x256_v3_fp32.onnx")

model = onnx.load(ONNX_PATH)
size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
size_pass = size_mb > 1
print(f"ONNX file size: {size_mb:.2f} MB (>1MB: {'PASS' if size_pass else 'FAIL'})")

inp = model.graph.input[0]
dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
input_pass = dims == [1, 3, 256, 256]
print(f"Input shape: {dims} (expect [1,3,256,256]: {'PASS' if input_pass else 'FAIL'})")

outputs = model.graph.output
count_pass = len(outputs) == 6
print(f"Output count: {len(outputs)} (expect 6: {'PASS' if count_pass else 'FAIL'})")

# 256x256 输入对应的特征图尺寸: 256/8=32, 256/16=16, 256/32=8
expected = {
    'box0': [1, 64, 32, 32], 'score0': [1, 2, 32, 32],
    'box1': [1, 64, 16, 16], 'score1': [1, 2, 16, 16],
    'box2': [1, 64, 8, 8], 'score2': [1, 2, 8, 8],
}
all_pass = True
for o in outputs:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: {dims}")
    if o.name in expected:
        if dims != expected[o.name]:
            print(f"  MISMATCH {o.name}: got {dims}, expected {expected[o.name]}")
            all_pass = False

print(f"T1 overall: {'PASS' if all_pass and size_pass and input_pass and count_pass else 'FAIL'}")
