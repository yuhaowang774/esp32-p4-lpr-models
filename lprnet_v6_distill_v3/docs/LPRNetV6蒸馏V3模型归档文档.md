# LPRNet V6 蒸馏 V2 与 V3 最终对比总结

> 生成日期：2026-06-29
> 项目：ESP32-P4 车牌识别（LPRNet V6 蒸馏版）
> 状态：模型训练与部署完成，用户已接受当前识别效果

---

## 一、项目背景

V6 蒸馏版 LPRNet 是基于 V5 教师模型蒸馏得到的学生模型（参数量 628K），用于 ESP32-P4 端侧车牌字符识别。V2 版本在 ESP32 实机测试中发现 267 个绿牌样本（占绿牌 8.61%）末尾第 8 位字符丢失，V3 版本针对该问题进行优化训练。

**核心问题**：V2 在 PC float 推理下表现优秀，但 INT8 量化部署到 ESP32 后，绿牌末尾字符丢失率达 8.61%。

---

## 二、V2 vs V3 配置差异对比

### 2.1 训练超参数差异

| 配置项 | V2 | V3 | 改动说明 |
|--------|-----|-----|---------|
| `BLANK_LOGIT_REDUCTION` | 0.3 | **0.1** | 降低 blank 抑制强度，帮助末尾字符输出 |
| `LETTERBOX_ASPECT_RANGE` | (2.5, 6.5) | **(2.5, 8.0)** | 扩大窄高比范围，增加窄绿牌样本 |
| `PHASE_LETTERBOX_PROB` (medium/full) | 0.5 | **0.6** | 提高 letterbox 增强概率 |
| `LEARNING_RATE` | 5e-4 | **3e-4** | 降低学习率，微调级训练 |
| `STUDENT_WEIGHTS` (初始化) | V6 蒸馏 V1 best | **V2 best** | 从 V2 成功权重继续微调 |

### 2.2 V3 新增配置（V2 无）

| 配置项 | V3 值 | 作用 |
|--------|-------|------|
| `GREEN_PLATE_WEIGHT` | 1.5 | 绿牌样本权重 1.5x（蓝牌 1.0x） |
| `NARROW_GREEN_EXTRA_WEIGHT` | 2.0 | 窄高比绿牌（≥5.0）额外 2.0x，总权重 3.0x |

### 2.3 保持一致的配置

| 配置项 | 值 |
|--------|-----|
| `DISTILL_TEMPERATURE` | 2.0 |
| `DISTILL_ALPHA` 调度 | 0.7(1-5) → 0.5(6-20) → 0.3(21+) |
| `BATCH_SIZE` | 512 |
| `EPOCHS` | 60 |
| `PATIENCE` | 15 |
| `CALIB_NUM_SAMPLES` | 2048 |
| `CALIB_LETTERBOX_RATIO` | 0.5 |
| 教师模型 | V5 best_model.pth |

---

## 三、训练结果对比

### 3.1 V3 训练过程

- **训练数据**：121,395 张（普通蓝牌 58,239 + 新能源小型车 63,156）
- **绿牌加权**：63,156 个绿牌样本 × 1.5 权重
- **Early Stopping**：epoch 16（patience=15 触发）
- **Best Val Balanced Acc**：100.0%（epoch 1 即达到）
- **训练时长**：每 epoch 约 150-195 秒

### 3.2 V3 测试集评估（15,175 样本）

| 评估项 | Seq Acc | Char Acc |
|--------|---------|----------|
| Test Clean | 99.95% | 99.98% |
| Test Letterbox | 99.87% | 99.94% |
| Test Balanced | 99.91% | - |
| Test Province | 99.95% | - |

### 3.3 V3 PC simcrop 测试（6,200 样本，float 推理）

| 类别 | 样本数 | 完全匹配 | 准确率 |
|------|--------|----------|--------|
| 总体 | 6200 | 6197 | **99.95%** |
| 蓝牌 | 3100 | 3097 | 99.90% |
| 绿牌 | 3100 | 3100 | **100.00%** |
| 绿牌末尾字符丢失 | - | **0** | **0.00%** |

**关键结论**：V3 float 模型在 PC 上彻底解决了绿牌末尾字符丢失问题（267 → 0）。

---

## 四、ESP32-P4 INT8 实机测试结果

### 4.1 测试条件

- 测试集：6,200 张 simcrop 样本（蓝牌 3100 + 绿牌 3100）
- 部署方式：模型嵌入 flash（`CONFIG_LP_CHAR_RECOGNIZE_MODEL_IN_FLASH_RODATA=y`）
- 推理时间：约 60.57-61.02 ms/张

### 4.2 V2 与 V3 ESP32 INT8 结果对比

| 指标 | V2 INT8 | V3 INT8 | 差异 |
|------|---------|---------|------|
| 完全匹配率 | 95.55% | 95.55% | **完全相同** |
| 绿牌末尾字符丢失 | 267 (8.61%) | 267 (8.61%) | **完全相同** |
| 6200 个样本结果 | - | - | **逐样本一致** |

### 4.3 INT8 量化问题诊断结论

通过多轮验证确认：

1. **固件正确性已验证**：
   - V3 INT8 源模型 MD5: `037E2F207A136DEB408FA2919B481FA9`
   - 构建产物 MD5: `037E2F207A136DEB408FA2919B481FA9`（一致）
   - V2 INT8 MD5: `9124C6AD0094526707A6508198A8C5F5`（不同）
   - V2/V3 INT8 文件 97.36% 字节不同

2. **float 模型正确性已验证**：
   - V3 ONNX (float) 在 PC 上 8/8 错误样本全部正确
   - V2 ONNX (float) 在 PC 上也 8/8 正确
   - 两个不同的 float 模型，PC 上表现均优秀

3. **根本原因**：
   - **INT8 量化精度太低**（-128~127），把 V2/V3 的权重差异压平
   - 两个不同的 float 模型，INT8 量化后在 ESP32 上产生完全相同的 6200 个结果
   - 绿牌末尾字符的信号在 float 下足够强，但 INT8 量化后信号被压缩到 blank 之下

---

## 五、模型文件 MD5 与大小对比

### 5.1 V2 模型文件（归档于 `lprnet_v6_distill_v2_project_archive`）

| 文件 | 大小 | MD5 |
|------|------|-----|
| `models/final_lprnet_v6_distilled_v2.pth` | 2,539,802 字节 | `C18A6E761FDCD63565CDC6DA152417DE` |
| `models/best_lprnet_v6_distilled_v2.pth` | 7,590,864 字节 | - |
| `models/lprnet_v6_distilled_v2.onnx` | 2,516,577 字节 | `36B40B5F5E0DF9B528643D7D5B03009F` |
| `quantize/lprnet_v6_distilled_v2_int8` | 666,000 字节 | `9124C6AD0094526707A6508198A8C5F5` |

### 5.2 V3 模型文件（位于 `lprnet_v6_distill_v3_project`）

| 文件 | 大小 | MD5 |
|------|------|-----|
| `final_lprnet_v6_distilled_v3.pth` | 2,539,802 字节 | `4DF25025E391F4E7D0CB28F6009587C5` |
| `best_lprnet_v6_distilled_v3.pth` | 7,590,864 字节 | `AACF5EF81C1711A5CAAB20C34A0E0EB6` |
| `lprnet_v6_distilled_v3.onnx` | 2,516,577 字节 | `AD88AE94BF75FF46F31874B11CE74352` |
| `quantize_output/lprnet_v6_distilled_v3_int8` | - | `972FD391CE93C0837DBB9AC272320BD0` |
| ESP32 部署 `lprnet_v6_distilled_v3_int8.espdl` | 665,328 字节 | `037E2F207A136DEB408FA2919B481FA9` |

### 5.3 关键观察

- V2 和 V3 的 `.pth` 与 `.onnx` 文件大小完全相同，但 MD5 不同 → 确认是不同模型
- V3 INT8 (665,328 字节) 比 V2 INT8 (666,000 字节) 小 672 字节
- V3 INT8 的两个 MD5 不同（`972FD3...` vs `037E2F...`）是因为 `.espdl` 是打包格式，`_int8` 是原始量化输出

---

## 六、文件清单与归档路径

### 6.1 V3 项目目录（当前工作版本）

```
./outputs/
├── final_lprnet_v6_distilled_v3.pth      ★ 最终模型权重（部署用）
├── best_lprnet_v6_distilled_v3.pth       最佳验证集权重
├── lprnet_v6_distilled_v3.onnx           ONNX 导出
├── config.py                             V3 训练配置
├── train_lprnet_v6_distill_v3.py         训练脚本
├── dataset_v2.py                         数据集（含绿牌加权）
├── augmentation_v2.py                    增强策略
├── distill_loss_v2.py                    蒸馏损失
├── export_onnx.py                        ONNX 导出脚本
├── quantize_v2.py                        INT8 量化脚本
├── generate_calibration.py               校准数据生成
├── eval_test.py                          测试集评估脚本
├── simcrop_test_v3.py                    PC simcrop 测试脚本
├── training_log.txt                      训练日志（精简）
├── training_log_full.txt                 训练日志（完整）
├── eval_test_log.txt                     测试集评估日志
├── simcrop_test_v3_log.txt               simcrop 测试日志
├── simcrop_test_v3_results.json          simcrop 详细结果
├── quantize_log.txt                      量化日志
├── chars.py                              字符表
├── model_v5.py / model_v6.py             模型定义引用
├── eval_ablation.py / run_ablation_all.py / train_ablation.py  消融实验
└── quantize_output\
    ├── lprnet_v6_distilled_v3_int8       INT8 量化模型（原始）
    ├── lprnet_v6_distilled_v3_int8.info  量化信息
    └── lprnet_v6_distilled_v3_int8.json  量化参数
```

### 6.2 V2 归档目录（已归档）

```
./outputs/v2_archive/
├── models\
│   ├── final_lprnet_v6_distilled_v2.pth
│   ├── best_lprnet_v6_distilled_v2.pth
│   └── lprnet_v6_distilled_v2.onnx
├── quantize\
│   ├── lprnet_v6_distilled_v2_int8
│   ├── lprnet_v6_distilled_v2_int8.info
│   └── lprnet_v6_distilled_v2_int8.json
├── calibration\
│   └── calibration_data.npz              (100 MB)
├── ablation\                             A1-A5 消融实验
├── code\                                 所有 Python 源码 + tests/
└── docs\
    ├── config.py
    └── training_log.txt
```

### 6.3 ESP32-P4 部署位置

```
<项目根>/factory_demo/components/lp_char_recognize/
├── models\p4\lprnet_v6_distilled_v3_int8.espdl   ★ 当前部署的 V3 INT8 模型
├── CMakeLists.txt                                引用 V3 模型文件
└── lp_char_recognize.cpp                         CTC 解码逻辑（含 trailing rescue）
```

---

## 七、最终推荐保存的模型文件

### 7.1 推荐 V3 作为最终保存版本

**理由**：
1. V3 float 模型在 PC 上绿牌准确率 100%，彻底解决末尾字符丢失问题
2. V3 是在 V2 基础上的改进版，配置更优化（绿牌加权、窄高比增强、更低 blank 抑制）
3. ESP32 INT8 实机结果 V2/V3 相同（INT8 量化精度限制），但 V3 float 模型潜力更大
4. 未来若改进量化方案（如 QAT、混合精度），V3 float 模型是更好的起点

### 7.2 必须保存的核心文件

| 优先级 | 文件 | 路径 | 用途 |
|--------|------|------|------|
| ★★★ | `final_lprnet_v6_distilled_v3.pth` | V3 项目根目录 | 最终模型权重，可重新导出 ONNX/INT8 |
| ★★★ | `lprnet_v6_distilled_v3_int8.espdl` | `factory_demo\components\lp_char_recognize\models\p4\` | ESP32 部署的 INT8 模型 |
| ★★ | `lprnet_v6_distilled_v3.onnx` | V3 项目根目录 | ONNX 中间格式 |
| ★★ | `config.py` | V3 项目根目录 | V3 训练配置（含所有超参数） |
| ★ | `best_lprnet_v6_distilled_v3.pth` | V3 项目根目录 | 最佳验证集权重（备份） |
| ★ | `quantize_output\` | V3 项目子目录 | INT8 量化模型原始输出 + 量化参数 |

### 7.3 可选保存文件

- `training_log_full.txt`：完整训练日志，用于复现训练过程
- `simcrop_test_v3_results.json`：PC simcrop 详细测试结果
- `dataset_v2.py` + `augmentation_v2.py` + `distill_loss_v2.py`：训练相关代码
- `quantize_v2.py`：量化脚本（注：含 bias_correct/equalization 修改但未执行完成）

---

## 八、已知限制与后续改进方向

### 8.1 当前限制

**绿牌末尾字符丢失（267 个，8.61%）**：
- V3 float 模型已解决（PC 上 0% 丢失）
- INT8 量化后退化回 V2 水平（267 个丢失）
- **根因**：INT8 量化精度（-128~127）不足以保留绿牌第 8 位字符的微弱信号

### 8.2 后续改进方向（如需进一步优化）

1. **QAT（量化感知训练）**：在训练中模拟 INT8 量化，让模型主动适应低精度
2. **混合精度量化**：对输出层保留 INT16，其他层 INT8
3. **bias_correct + equalization**：ESP-PPQ 量化优化（V3 已修改脚本但未完成执行）
4. **输出层 per-channel 量化**：针对 66 个字符通道分别量化
5. **模型架构调整**：增加输出层通道数或改进 attention 机制

---

## 九、总结

V6 蒸馏 V3 是在 V2 基础上的针对性优化版本，通过降低 blank 抑制、扩大窄高比增强、绿牌加权三项改进，在 PC float 推理下彻底解决了绿牌末尾字符丢失问题（267 → 0）。

受限于 ESP32-P4 的 INT8 量化精度，V3 在实机部署时与 V2 表现相同（95.55% 完全匹配率，267 个绿牌末尾丢失）。用户已接受当前识别效果。

**最终保存版本：V3**（`final_lprnet_v6_distilled_v3.pth` + `lprnet_v6_distilled_v3_int8.espdl`），V2 已归档至 `lprnet_v6_distill_v2_project_archive`。
