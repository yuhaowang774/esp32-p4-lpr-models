"""V6 蒸馏 V2 蒸馏损失单元测试

TDD Red 阶段：先写失败测试，验证 distill_loss_v2.py 未实现或行为不符。

关键测试点（来自文档 4.3.1 节）：
1. forward 签名：接受 5 个参数，ProvinceAuxLoss 传 4 个参数
2. CTC 不被双重计数：total = alpha*distill + (1-alpha)*ctc_with_aux
3. province_weights 加权路径生效
4. blank_logit_reduction 生效
5. teacher_logits.detach() 防御性
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_loss_v2 import (
    DistillCTCLossV2,
    KLDivDistillLoss,
    WeightedCTCLoss,
    ProvinceAuxLoss,
)


def _make_safe_targets(num_samples=4, plate_len=7):
    """构造安全的 targets：首字符为省份字符（索引 1-31），其余为任意字符（1-65）

    ProvinceAuxLoss 期望省份标签 [0, 30]（first_char_idx - 1），
    若首字符索引 > 31（字母/数字），减 1 后为 31-64，超出 31 类范围会导致越界。
    """
    targets = torch.randint(1, 66, (num_samples * plate_len,))
    # 每个车牌首字符必须是省份字符（索引 1-31），否则 ProvinceAuxLoss 会越界
    targets[0::plate_len] = torch.randint(1, 32, (num_samples,))
    return targets


def test_distill_ctc_loss_v2_forward_signature():
    """验证 DistillCTCLossV2.forward 接受 5 个参数且 ProvinceAuxLoss 传 4 个参数"""
    criterion = DistillCTCLossV2(temperature=2.0, alpha=0.5,
                                  blank_logit_reduction=0.3)
    student_logits = torch.randn(32, 4, 66)  # raw logits [T, B, C]
    teacher_logits = torch.randn(32, 4, 66)
    targets = _make_safe_targets(num_samples=4, plate_len=7)
    input_lengths = torch.full((4,), 32, dtype=torch.long)
    target_lengths = torch.full((4,), 7, dtype=torch.long)

    total, metrics = criterion(student_logits, teacher_logits,
                                targets, input_lengths, target_lengths)
    assert total.item() > 0
    assert 'distill' in metrics and 'ctc' in metrics and 'aux' in metrics


def test_distill_ctc_loss_v2_no_double_counting():
    """验证 CTC 不被双重计数：独立计算各分量后验证 total 匹配"""
    alpha = 0.5
    criterion = DistillCTCLossV2(temperature=2.0, alpha=alpha,
                                  blank_logit_reduction=0.3)
    student_logits = torch.randn(32, 4, 66)  # raw logits，不做预 log_softmax
    teacher_logits = torch.randn(32, 4, 66)
    targets = _make_safe_targets(num_samples=4, plate_len=7)
    input_lengths = torch.full((4,), 32, dtype=torch.long)
    target_lengths = torch.full((4,), 7, dtype=torch.long)

    # 1. 调用 V2 loss 获取 total
    total, metrics = criterion(student_logits, teacher_logits,
                                targets, input_lengths, target_lengths)

    # 2. 独立调用各组件 loss 函数（不依赖 metrics dict）
    with torch.no_grad():
        distill_indep = criterion.distill_loss(student_logits, teacher_logits.detach())
        log_probs = F.log_softmax(student_logits, dim=-1)
        ctc_indep = criterion.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        ctc_with_aux_indep = criterion.province_aux(
            student_logits, targets, target_lengths, ctc_indep)

    # 3. 正确公式：total = alpha*distill + (1-alpha)*ctc_with_aux
    expected = alpha * distill_indep + (1 - alpha) * ctc_with_aux_indep
    # 错误公式（双重计数）：alpha*distill + (1-alpha)*(ctc + ctc_with_aux)
    wrong = alpha * distill_indep + (1 - alpha) * (ctc_indep + ctc_with_aux_indep)

    assert abs(total.item() - expected.item()) < 1e-4, \
        f"CTC 可能被双重计数: total={total.item()}, expected={expected.item()}, wrong={wrong.item()}"
    assert abs(total.item() - wrong.item()) > 1e-4, \
        f"测试无效：正确值和错误值相同，无法区分双重计数"


def test_province_weighted_loss():
    """验证 province_weights 不为 None 时走加权 cross_entropy 路径"""
    # 构造不均匀的省份权重：省份 0 权重 5.0，其余 1.0
    province_weights = torch.ones(31)
    province_weights[0] = 5.0

    criterion_weighted = DistillCTCLossV2(
        temperature=2.0, alpha=0.5, blank_logit_reduction=0.3,
        province_weights=province_weights)
    criterion_unweighted = DistillCTCLossV2(
        temperature=2.0, alpha=0.5, blank_logit_reduction=0.3,
        province_weights=None)

    student_logits = torch.randn(32, 4, 66)
    teacher_logits = torch.randn(32, 4, 66)
    targets = _make_safe_targets(num_samples=4, plate_len=7)
    # 强制第一个样本省份为索引 1（标签 0，权重 5.0），
    # 否则当所有标签非 0 时加权/不加权 loss 相同（概率 87.6%），断言会失败
    targets[0] = 1
    input_lengths = torch.full((4,), 32, dtype=torch.long)
    target_lengths = torch.full((4,), 7, dtype=torch.long)

    _, metrics_w = criterion_weighted(student_logits, teacher_logits,
                                       targets, input_lengths, target_lengths)
    _, metrics_uw = criterion_unweighted(student_logits, teacher_logits,
                                          targets, input_lengths, target_lengths)

    # 加权和不加权的 aux loss 应不同
    assert abs(metrics_w['aux'] - metrics_uw['aux']) > 1e-6, \
        "province_weights 未生效：加权和不加权 aux loss 相同"


def test_blank_logit_reduction_reduces_blank_logit():
    """验证 blank_logit_reduction 降低 blank 通道的 log_prob"""
    criterion = WeightedCTCLoss(blank=0, blank_weight=1.0,
                                 blank_logit_reduction=0.5)
    # 构造 log_probs，blank 通道为 0
    log_probs = torch.zeros(32, 4, 66)
    original_blank = log_probs[:, :, 0].clone()
    # WeightedCTCLoss.forward 内部会 clone 并减去 blank_logit_reduction
    # 通过替换 base_ctc 为捕获 log_probs 的 nn.Module 包装验证内部逻辑
    captured = {}

    class MockCTC(nn.Module):
        def forward(self, log_probs, targets, input_lengths, target_lengths):
            captured['log_probs'] = log_probs.clone()
            return torch.tensor(1.0)

    criterion.base_ctc = MockCTC()
    targets = _make_safe_targets(num_samples=4, plate_len=7)
    input_lengths = torch.full((4,), 32, dtype=torch.long)
    target_lengths = torch.full((4,), 7, dtype=torch.long)
    criterion(log_probs, targets, input_lengths, target_lengths)

    # blank 通道应被减去 0.5
    assert torch.allclose(captured['log_probs'][:, :, 0], original_blank - 0.5), \
        "blank_logit_reduction 未生效：blank 通道未被降低"


def test_teacher_logits_detached():
    """验证蒸馏损失中 teacher_logits 被 detach（防御性，即使 teacher 在 no_grad 下）"""
    criterion = DistillCTCLossV2(temperature=2.0, alpha=0.5)
    student_logits = torch.randn(32, 4, 66, requires_grad=True)
    teacher_logits = torch.randn(32, 4, 66, requires_grad=True)
    targets = _make_safe_targets(num_samples=4, plate_len=7)
    input_lengths = torch.full((4,), 32, dtype=torch.long)
    target_lengths = torch.full((4,), 7, dtype=torch.long)

    total, _ = criterion(student_logits, teacher_logits,
                          targets, input_lengths, target_lengths)
    total.backward()

    # teacher_logits 不应有梯度（被 detach）
    assert teacher_logits.grad is None or torch.all(teacher_logits.grad == 0), \
        "teacher_logits 未被 detach，存在梯度"


def test_alpha_schedule_changes_loss_weight():
    """验证 alpha 参数影响蒸馏和 CTC 的权重"""
    student_logits = torch.randn(32, 4, 66)
    teacher_logits = torch.randn(32, 4, 66)
    targets = _make_safe_targets(num_samples=4, plate_len=7)
    input_lengths = torch.full((4,), 32, dtype=torch.long)
    target_lengths = torch.full((4,), 7, dtype=torch.long)

    # alpha=1.0：纯蒸馏
    criterion_distill = DistillCTCLossV2(temperature=2.0, alpha=1.0)
    total_distill, _ = criterion_distill(student_logits, teacher_logits,
                                          targets, input_lengths, target_lengths)

    # alpha=0.0：纯 CTC
    criterion_ctc = DistillCTCLossV2(temperature=2.0, alpha=0.0)
    total_ctc, _ = criterion_ctc(student_logits, teacher_logits,
                                  targets, input_lengths, target_lengths)

    # 两者应不同
    assert abs(total_distill.item() - total_ctc.item()) > 1e-6, \
        "alpha=1.0 和 alpha=0.0 的 loss 相同，alpha 未生效"


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
            print(f"FAIL: {func.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n总计: {passed} 通过, {failed} 失败")
