"""V6 蒸馏 V2 损失函数

包含：
- KLDivDistillLoss: KL 散度蒸馏损失（与 V6 蒸馏 V1 一致）
- WeightedCTCLoss: 加权 CTC 损失（移植 V9 的 blank_logit_reduction）
- ProvinceAuxLoss: 省份辅助损失（移植 V9 的 province_weights 加权）
- DistillCTCLossV2: 联合损失（alpha*distill + (1-alpha)*ctc_with_aux）

关键修复（来自文档审查）：
1. CTC 接收 log-probabilities（先 log_softmax），非 raw logits
2. CTC 不被双重计数：total = alpha*distill + (1-alpha)*ctc_with_aux
3. ProvinceAuxLoss.forward 接受 4 参数 (logits, labels, label_lengths, ctc_loss)
4. teacher_logits.detach() 防御性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivDistillLoss(nn.Module):
    """Logits 蒸馏损失：KL 散度衡量教师和学生输出分布的差异

    教师和学生都输出 CTC 序列 [T, B, C]，需要 reshape 为 [T*B, C] 后计算 KL 散度，
    使 batchmean 正确除以 T*B（per-sample 平均），而非仅除以 T。
    """

    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: [T, B, C] 学生模型原始 logits
            teacher_logits: [T, B, C] 教师模型原始 logits（已 detach）
        Returns:
            KL 散度损失（标量）
        """
        T = self.temperature
        C = student_logits.size(-1)
        # reshape 为 [T*B, C]，使 batchmean 除以 T*B（per-sample 平均）
        student_log_soft = F.log_softmax(student_logits / T, dim=-1).reshape(-1, C)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1).reshape(-1, C)
        # KL 散度 * T^2（标准蒸馏缩放）
        loss = self.kl_div(student_log_soft, teacher_soft) * (T * T)
        return loss


class WeightedCTCLoss(nn.Module):
    """加权 CTC 损失（移植 V9 的 blank_logit_reduction）

    与 V6 蒸馏 V1 的区别：新增 blank_logit_reduction 参数，降低 blank 通道的 log_prob，
    减少 CTC 解码时末尾字符丢失问题（V10 验证有效）。

    blank_logit_reduction 选择 0.3 而非 V10 的 0.5：
    - V10 使用 0.5 导致绿牌退化 3.26%（78.74%→75.48%）
    - 0.3 是更保守的值，预期在减少末尾丢失的同时不引入绿牌多字符问题
    """

    def __init__(self, blank=0, blank_weight=2.0, blank_logit_reduction=0.0,
                 zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.blank_weight = blank_weight
        self.blank_logit_reduction = blank_logit_reduction
        self.zero_infinity = zero_infinity
        self.base_ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity,
                                    reduction='none')

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: [T, B, C] log-probabilities（调用方需先 log_softmax）
            targets: 1D 扁平化目标序列
            input_lengths: [B] 输入序列长度
            target_lengths: [B] 目标序列长度
        """
        if self.blank_logit_reduction > 0:
            log_probs = log_probs.clone()
            log_probs[:, :, self.blank] -= self.blank_logit_reduction
        loss = self.base_ctc(log_probs, targets, input_lengths, target_lengths)
        if self.blank_weight != 1.0:
            with torch.no_grad():
                pred = log_probs.argmax(dim=2)
                blank_ratio = (pred == self.blank).float().mean()
                weight = 1.0 + (self.blank_weight - 1.0) * (1.0 - blank_ratio)
            loss = loss * weight
        return loss.mean()


class ProvinceAuxLoss(nn.Module):
    """省份辅助分类损失（移植 V9 的 province_weights 加权）

    从车牌首字符提取省份标签（索引 1-31 → 标签 0-30），
    在前 4 个时间步的省份字符 logits 上做分类。

    与 V6 蒸馏 V1 的区别：新增 province_weights 参数，对稀有省份加权。
    """

    def __init__(self, num_provinces=31, aux_weight=0.5, province_weights=None):
        super().__init__()
        self.num_provinces = num_provinces
        self.aux_weight = aux_weight
        self.province_weights = province_weights

    def forward(self, logits, labels, label_lengths, ctc_loss):
        """
        Args:
            logits: [T, B, C] raw logits（F.cross_entropy 内部会 log_softmax）
            labels: 1D 扁平化目标序列
            label_lengths: [B] 每个样本的目标长度
            ctc_loss: 已计算的 CTC 损失（标量）
        Returns:
            ctc_loss + aux_weight * aux_loss
        """
        T = logits.size(0)
        B = logits.size(1)

        early_steps = min(4, T)
        province_logits = logits[:early_steps, :, 1:self.num_provinces + 1]
        province_pooled = province_logits.max(dim=0)[0]

        province_labels = []
        label_idx = 0
        for i in range(B):
            length = label_lengths[i].item()
            if length > 0:
                # 首字符索引 1-31 → 标签 0-30
                province_labels.append(labels[label_idx].item() - 1)
            else:
                province_labels.append(0)
            label_idx += length

        province_labels = torch.tensor(province_labels, device=logits.device,
                                        dtype=torch.long)

        if self.province_weights is not None:
            weights = self.province_weights.to(logits.device)
            aux_loss = F.cross_entropy(province_pooled, province_labels, weight=weights)
        else:
            aux_loss = F.cross_entropy(province_pooled, province_labels)

        return ctc_loss + self.aux_weight * aux_loss


class DistillCTCLossV2(nn.Module):
    """V6 蒸馏 V2 联合损失

    total = alpha * distill + (1 - alpha) * ctc_with_aux

    其中 ctc_with_aux = ctc + aux_weight * aux_loss（由 ProvinceAuxLoss 返回）。
    注意：ctc_with_aux 已包含 ctc，不可再单独加 ctc（否则双重计数）。

    渐进 alpha 调度：
    - 初期 (epoch 1-5): alpha=0.7，侧重蒸馏，让学生快速学习教师的输出分布
    - 中期 (epoch 6-20): alpha=0.5，蒸馏与 CTC 平衡
    - 后期 (epoch 21+): alpha=0.3，侧重 CTC，让学生在真实标签上精调
    """

    def __init__(self, temperature=2.0, alpha=0.5,
                 blank_weight=2.0, blank_logit_reduction=0.3,
                 province_aux_weight=0.5, province_weights=None,
                 num_provinces=31):
        super().__init__()
        self.alpha = alpha
        self.distill_loss = KLDivDistillLoss(temperature=temperature)
        # WeightedCTCLoss 和 ProvinceAuxLoss 从 V9 移植
        # V6 蒸馏 V1 原版不支持 blank_logit_reduction 和 province_weights
        self.ctc_loss = WeightedCTCLoss(blank=0, blank_weight=blank_weight,
                                         blank_logit_reduction=blank_logit_reduction)
        self.province_aux = ProvinceAuxLoss(num_provinces=num_provinces,
                                             aux_weight=province_aux_weight,
                                             province_weights=province_weights)

    def forward(self, student_logits, teacher_logits, targets, input_lengths, target_lengths):
        """
        Args:
            student_logits: [T, B, C] 学生模型 raw logits
            teacher_logits: [T, B, C] 教师模型 raw logits（no_grad 下计算）
            targets: 1D 扁平化目标序列
            input_lengths: [B] 输入序列长度
            target_lengths: [B] 目标序列长度
        Returns:
            total_loss: 标量
            metrics: dict 包含 'distill', 'ctc', 'aux', 'total'
        """
        # 蒸馏损失：学生 logits 分布逼近教师 logits 分布
        # KLDivDistillLoss 内部自带 log_softmax + 温度缩放，传 raw logits
        # detach 与原版第 64 行一致（防御性，即使 teacher 在 no_grad 下）
        distill = self.distill_loss(student_logits, teacher_logits.detach())

        # CTC 损失：nn.CTCLoss 要求输入为 log-probabilities，必须先 log_softmax
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        ctc = self.ctc_loss(student_log_probs, targets, input_lengths, target_lengths)

        # 省份辅助损失：ProvinceAuxLoss 内部用 F.cross_entropy（自带 log_softmax），传 raw logits
        # 返回值已包含 ctc，即 ctc_with_aux = ctc + aux_weight * aux_loss
        ctc_with_aux = self.province_aux(student_logits, targets, target_lengths, ctc)

        # 联合损失：注意 ctc_with_aux 已包含 ctc，不可再单独加 ctc（否则双重计数）
        total = self.alpha * distill + (1 - self.alpha) * ctc_with_aux
        return total, {'distill': distill.item(), 'ctc': ctc.item(),
                       'aux': (ctc_with_aux - ctc).item(), 'total': total.item()}
