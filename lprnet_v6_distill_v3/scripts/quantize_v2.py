"""V6 蒸馏 V2 INT8 量化脚本

使用 ESP-PPQ 对 V2 ONNX 模型进行 INT8 量化，部署到 ESP32-P4。

与 V6 蒸馏 V1 量化的区别：
1. 校准数据 2048 样本（V1 为 1024），50% clean + 50% letterbox 混合分布
2. 校准数据已预处理保存为 .npz，直接加载（V1 实时读取图片）
3. KL 散度校准算法（与 V1 一致）

混合分布校准数据匹配实际推理场景：
- ESP32 端侧推理时约 50% 输入来自 YOLO 裁切（letterbox），50% 为 clean
- 纯 clean 校准会导致 letterbox 输入的量化误差增大
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from config import OUTPUT_DIR, ONNX_NAME, CALIB_NUM_SAMPLES


class NpzCalibrationDataset(Dataset):
    """从 .npz 文件加载校准数据

    校准数据已预处理（resize/letterbox + normalize），直接加载即可。
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data['images']  # [N, 3, 32, 128] float32
        print(f"加载校准数据: {self.images.shape} from {npz_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx])


def main():
    # 输入输出路径
    onnx_path = os.path.join(OUTPUT_DIR, ONNX_NAME)
    calib_npz = os.path.join(OUTPUT_DIR, "calibration_letterbox", "calibration_data.npz")
    output_dir = os.path.join(OUTPUT_DIR, "quantize_output")
    os.makedirs(output_dir, exist_ok=True)
    espdl_path = os.path.join(output_dir, "lprnet_v6_distilled_v3_int8")

    print(f"ONNX 模型: {onnx_path}")
    print(f"校准数据: {calib_npz}")
    print(f"输出路径: {espdl_path}")

    # 加载校准数据
    print("\n加载混合分布校准数据（50% clean + 50% letterbox）...")
    calib_dataset = NpzCalibrationDataset(calib_npz)
    calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

    def collate_fn(batch):
        if isinstance(batch, torch.Tensor):
            return batch.to('cpu')
        return torch.stack(batch).to('cpu')

    # ESP-PPQ INT8 量化
    print(f"\n开始 ESP-PPQ INT8 量化（KL 散度校准，{CALIB_NUM_SAMPLES} 样本）...")
    quant_setting = QuantizationSettingFactory.espdl_setting()
    # 校准算法使用 KL 散度（与 V1 一致）
    quant_setting.quantize_activation_setting.calib_algorithm = "kl"

    espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=calib_dataloader,
        calib_steps=len(calib_dataset),
        input_shape=[1, 3, 32, 128],
        target="esp32p4",
        num_of_bits=8,
        collate_fn=collate_fn,
        setting=quant_setting,
        device='cpu',
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    print(f"\nINT8 量化成功! 输出: {espdl_path}")


if __name__ == '__main__':
    main()
