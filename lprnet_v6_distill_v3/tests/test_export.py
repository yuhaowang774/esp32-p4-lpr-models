"""V6 蒸馏 V2 ONNX 导出单元测试

TDD Red 阶段：先写失败测试，验证 export_onnx.py 未实现或行为不符。

关键测试点（来自文档 4.3.4 节）：
1. 导出的 ONNX 模型无 Dropout 节点
2. ONNX 输出 shape 为 [T, B, 66]（T=31）
"""

import sys
import os
import tempfile

import torch
import numpy as np

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v6 import LPRNetV6
from export_onnx import export_onnx, remove_dropout


def test_remove_dropout():
    """验证 remove_dropout 移除模型中的 Dropout 层"""
    model = LPRNetV6(num_classes=66, dropout_rate=0.3)
    # 原始模型应含 Dropout
    has_dropout_before = any(isinstance(m, torch.nn.Dropout2d)
                              for m in model.modules())
    assert has_dropout_before, "测试模型应含 Dropout 层"

    remove_dropout(model)
    has_dropout_after = any(isinstance(m, torch.nn.Dropout2d)
                             for m in model.modules())
    assert not has_dropout_after, "remove_dropout 未移除所有 Dropout 层"


def test_onnx_no_dropout_nodes():
    """验证导出的 ONNX 模型无 Dropout 节点"""
    try:
        import onnx
    except ImportError:
        import pytest
        pytest.skip("onnx 未安装")

    model = LPRNetV6(num_classes=66, dropout_rate=0.3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, 'test.onnx')
        # 不外部调用 remove_dropout，验证 export_onnx 内部的 Dropout 移除
        export_onnx(model, onnx_path)

        onnx_model = onnx.load(onnx_path)
        dropout_nodes = [node for node in onnx_model.graph.node
                         if 'Dropout' in node.op_type]
        assert len(dropout_nodes) == 0, \
            f"发现 Dropout 节点: {[n.name for n in dropout_nodes]}"


def test_onnx_output_shape():
    """验证 ONNX 输出 shape 为 [T, B, 66]"""
    try:
        import onnxruntime as ort
    except ImportError:
        import pytest
        pytest.skip("onnxruntime 未安装")

    model = LPRNetV6(num_classes=66, dropout_rate=0.3)
    model.eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, 'test.onnx')
        export_onnx(model, onnx_path)

        sess = ort.InferenceSession(onnx_path)
        dummy = np.random.randn(1, 3, 32, 128).astype(np.float32)
        output = sess.run(None, {'input': dummy})[0]
        # T=31（MaxPool2d((2,2),stride=(2,1)) 使 W: 32→31）
        assert output.shape == (31, 1, 66), \
            f"ONNX 输出 shape 错误: {output.shape}，期望 (31, 1, 66)"


def test_onnx_export_eval_mode():
    """验证导出 ONNX 时模型处于 eval 模式（影响 BN 和 Dropout）"""
    model = LPRNetV6(num_classes=66, dropout_rate=0.3)
    model.train()  # 初始为 train 模式
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, 'test.onnx')
        export_onnx(model, onnx_path)
        # export_onnx 内部应将模型设为 eval 模式
        # 验证方式：导出后模型应处于 eval 模式（或导出过程不影响原模型）
        # 这里主要验证导出不报错
        assert os.path.exists(onnx_path), "ONNX 文件未生成"


if __name__ == '__main__':
    test_funcs = [v for k, v in sorted(globals().items())
                  if k.startswith('test_') and callable(v)]
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            print(f"PASS: {func.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL: {func.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n总计: {passed} 通过, {failed} 失败")
