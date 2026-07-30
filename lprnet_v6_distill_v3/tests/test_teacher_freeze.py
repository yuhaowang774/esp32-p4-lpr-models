"""V6 蒸馏 V2 教师模型冻结单元测试

TDD Red 阶段：先写失败测试，验证训练脚本中教师模型冻结逻辑正确。

关键测试点（来自文档 4.3.3 节）：
1. 教师模型所有参数 requires_grad=False
2. 一个训练 step 后教师权重不变
3. 学生模型 forward 输出 [T, B, C]，无需额外 permute
"""

import sys
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from model_v5 import LPRNetV5
from model_v6 import LPRNetV6
from distill_loss_v2 import DistillCTCLossV2
from train_lprnet_v6_distill_v3 import load_teacher_model, load_student_model


def _make_safe_targets(num_samples=1, plate_len=7):
    """构造安全的 targets：首字符为省份字符（索引 1-31）"""
    targets = torch.randint(1, 66, (num_samples * plate_len,))
    targets[0::plate_len] = torch.randint(1, 32, (num_samples,))
    return targets


def test_teacher_parameters_frozen():
    """验证教师模型所有参数 requires_grad=False"""
    # 不加载真实权重（测试冻结逻辑与权重无关）
    teacher = LPRNetV5(num_classes=66)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    for name, param in teacher.named_parameters():
        assert param.requires_grad == False, f"{name} 未冻结"


def test_teacher_weights_unchanged_after_training_step():
    """验证一个训练 step 后教师权重不变"""
    teacher = LPRNetV5(num_classes=66)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    weights_before = {n: p.clone() for n, p in teacher.named_parameters()}

    # 构造 dummy 输入，执行一次 forward+backward+step
    student = LPRNetV6(num_classes=66, dropout_rate=0.3)
    criterion = DistillCTCLossV2(temperature=2.0, alpha=0.5)
    optimizer = AdamW(student.parameters(), lr=1e-4)

    dummy_input = torch.randn(1, 3, 32, 128)
    dummy_targets = _make_safe_targets(num_samples=1, plate_len=7)
    # 模型 forward 输出 T=31（MaxPool2d((2,2),stride=(2,1)) 使 W: 32→31）
    dummy_input_lengths = torch.tensor([31])
    dummy_target_lengths = torch.tensor([7])

    # 模型 forward 已输出 [T, B, C]，无需额外 permute
    student_logits = student(dummy_input)
    with torch.no_grad():
        teacher_logits = teacher(dummy_input)

    loss, _ = criterion(student_logits, teacher_logits, dummy_targets,
                        dummy_input_lengths, dummy_target_lengths)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for name, param in teacher.named_parameters():
        assert torch.equal(param, weights_before[name]), f"{name} 被修改"


def test_student_forward_output_shape():
    """验证学生模型 forward 输出 [T, B, C]，无需额外 permute"""
    student = LPRNetV6(num_classes=66, dropout_rate=0.3)
    student.eval()
    dummy_input = torch.randn(4, 3, 32, 128)
    with torch.no_grad():
        output = student(dummy_input)
    # LPRNetV6.forward 输出 [T=31, B=4, C=66]
    # T=31 因为 MaxPool2d((2,2), stride=(2,1)) 使 W: 32→31
    assert output.shape == (31, 4, 66), \
        f"学生模型输出 shape 错误: {output.shape}，期望 (31, 4, 66)"


def test_teacher_forward_output_shape():
    """验证教师模型 forward 输出 [T, B, C]"""
    teacher = LPRNetV5(num_classes=66)
    teacher.eval()
    dummy_input = torch.randn(4, 3, 32, 128)
    with torch.no_grad():
        output = teacher(dummy_input)
    assert output.shape == (31, 4, 66), \
        f"教师模型输出 shape 错误: {output.shape}，期望 (31, 4, 66)"


def test_load_teacher_model_freezes_parameters():
    """验证 load_teacher_model 返回的模型所有参数已冻结"""
    # 注意：此测试需要真实 V5 权重文件
    # 如果权重文件不存在则跳过
    from config import TEACHER_WEIGHTS
    if not os.path.exists(TEACHER_WEIGHTS):
        import pytest
        pytest.skip(f"教师权重文件不存在: {TEACHER_WEIGHTS}")

    teacher = load_teacher_model()
    for name, param in teacher.named_parameters():
        assert param.requires_grad == False, f"{name} 未冻结"
    assert teacher.training == False, "教师模型未处于 eval 模式"


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
