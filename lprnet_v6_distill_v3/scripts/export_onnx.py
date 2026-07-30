"""V6 蒸馏 V2 ONNX 导出

功能：
1. remove_dropout: 移除模型中的 Dropout 层（确保推理一致性）
2. export_onnx: 导出 ONNX 模型，移除 Dropout，onnxsim 简化

关键点：
- Dropout 在训练时启用，推理时关闭。导出 ONNX 时应移除 Dropout 节点，确保推理一致性。
- onnxsim 简化图结构，减少冗余节点。
"""

import sys
import os
import copy
import torch
import torch.nn as nn

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_v6 import LPRNetV6
from config import *


def remove_dropout(model):
    """移除模型中的 Dropout 层（递归遍历所有子模块）

    Dropout 在训练时启用，推理时关闭。导出 ONNX 时移除 Dropout 节点，
    确保推理结果确定性（不受随机种子影响）。
    """
    for name, child in model.named_children():
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            setattr(model, name, nn.Identity())
        else:
            remove_dropout(child)


def export_onnx(model, output_path):
    """导出 ONNX，移除 Dropout 确保推理一致性

    Args:
        model: LPRNetV6 模型（训练或 eval 模式均可）
        output_path: ONNX 文件输出路径
    """
    # 深拷贝避免修改原模型
    model_eval = copy.deepcopy(model)
    remove_dropout(model_eval)
    model_eval.eval()

    dummy = torch.randn(1, 3, 32, 128)
    torch.onnx.export(
        model=model_eval,
        args=dummy,
        f=output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
    )

    # onnxsim 简化（可选，失败则保留原模型）
    try:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(output_path)
        model_sim, check = simplify(model_onnx)
        if check:
            onnx.save(model_sim, output_path)
            print(f"  ONNX 简化成功: {output_path}")
        else:
            print(f"  ONNX 简化验证失败，保留原模型: {output_path}")
    except ImportError:
        print(f"  onnxsim 未安装，跳过简化: {output_path}")
    except Exception as e:
        print(f"  ONNX 简化失败（{e}），保留原模型: {output_path}")


def main():
    """从 final 模型导出 ONNX

    V3 说明：由于从 V2 best 权重初始化，验证集 balanced_acc 在 epoch 1 就达 100%，
    早停基于 balanced_acc 的严格大于判断，导致 best 模型只是 epoch 1 权重，
    V3 改进（绿牌加权、letterbox 8.0、blank_reduction 0.1）几乎未应用。
    final 模型（epoch 16，medium 阶段 6 epochs）包含更多 V3 训练，更适合
    本任务目标（改善 simcrop 绿牌末尾字符丢失，而非验证集准确率）。
    """
    best_path = os.path.join(OUTPUT_DIR, f"final_{MODEL_NAME}.pth")
    print(f"Loading V3 final model from {best_path}...")
    model = LPRNetV6(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    checkpoint = torch.load(
        best_path,
        map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    onnx_path = os.path.join(OUTPUT_DIR, ONNX_NAME)
    print(f"Exporting ONNX to {onnx_path}...")
    export_onnx(model, onnx_path)
    print(f"Done. ONNX size: {os.path.getsize(onnx_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
