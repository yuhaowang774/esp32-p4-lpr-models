# LPRNet V6 蒸馏 V3 字符识别模型

## 模型简介

LPRNet V6 蒸馏 V3 是面向 ESP32-P4-EYE 端侧部署的中国车牌字符识别模型：

- 输入分辨率：128×32
- 字符类别数：66（CTC 解码，含 blank）
- 量化方式：INT8（esp-ppq 0.2.4）
- 部署格式：.espdl（约 0.65 MB）
- 指标：PC float 99.95%，ESP32 INT8 95.55%

## 脚本说明

| 脚本 | 说明 |
|------|------|
| `train_lprnet_v6_distill_v3.py` | V3 蒸馏训练入口（V5 教师 → V6 学生） |
| `config.py` | V3 训练配置（路径、蒸馏、CTC、letterbox、绿牌加权等） |
| `dataset_v2.py` | 数据集加载（普通蓝牌/新能源小型车） |
| `augmentation_v2.py` | letterbox、右侧 padding 等数据增强 |
| `distill_loss_v2.py` | 蒸馏损失（KL 散度 + CTC + 省份辅助损失） |
| `export_onnx.py` | 导出 ONNX 中间格式 |
| `quantize_v2.py` | 基于 esp-ppq 进行 INT8 量化并导出 .espdl |
| `eval_test.py` | 在测试集上评估识别准确率 |
| `simcrop_test_v3.py` | simcrop 合成裁剪测试集评估（含绿牌末尾字符） |
| `generate_calibration.py` | 生成 INT8 量化校准数据集 |
| `train_lprnet_v5.py` | V5 教师模型训练脚本（蒸馏依赖） |
| `train_lprnet_v6.py` | V6 学生模型训练脚本（蒸馏依赖） |
| `model_v5.py` | V5 模型结构定义 |
| `model_v6.py` | V6 模型结构定义 |
| `chars.py` | 66 类字符表定义 |
| `train_ablation.py` | 消融实验训练脚本 |
| `run_ablation_all.py` | 批量运行消融实验 |
| `eval_ablation.py` | 消融实验结果评估 |

## 蒸馏架构

- **教师模型**：LPRNet V5（容量较大，PC float 精度高）
- **学生模型**：LPRNet V6（轻量化，端侧可部署）
- **V3 在 V2 基础上的优化**（针对绿牌末尾字符丢失问题）：
  - 降低 blank 抑制：`blank_logit_reduction` 0.3 → 0.1，减少 blank 输出，帮助末尾字符解码
  - 扩大窄高比增强：letterbox 宽高比范围 (2.5, 6.5) → (2.5, 8.0)，增加窄高比绿牌样本
  - 绿牌样本加权：1.5×（窄高比绿牌额外 2.0×，合计 3.0×），强化 8 位绿牌学习

## 训练流程

1. **权重初始化**：从 V6 蒸馏 V2 的 best 权重初始化学生模型（在 V2 成功基础上微调）。
2. **蒸馏训练**：V5 教师指导 V6 学生，60 epochs，学习率 3e-4，温度 2.0，alpha 渐进调度（1-5=0.7, 6-20=0.5, 21+=0.3）。
3. **导出 ONNX**：训练完成导出 `lprnet_v6_distilled_v3.onnx`。
4. **INT8 量化**：使用 esp-ppq 进行 INT8 量化，校准样本 2048，导出 .espdl。

## 环境变量

配置集中在 `config.py`，通过环境变量覆盖默认值：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DATA_DIR` | `./data/CBLPRD` | CBLPRD 数据集目录（含 train/val/test.txt） |
| `OUTPUT_DIR` | `./outputs` | 训练/导出/量化输出目录 |
| `TEACHER_WEIGHTS` | `./pretrained/lprnet_v5_best_model.pth` | V5 教师模型权重 |
| `STUDENT_WEIGHTS` | `./pretrained/best_lprnet_v6_distilled_v2.pth` | V6 蒸馏 V2 学生权重（V3 初始化用） |

V5/V6 教师与学生单独训练时另使用 `V5_OUTPUT_DIR`、`V6_OUTPUT_DIR` 控制各自输出目录；`simcrop_test_v3.py` 使用 `SIMCROP_DIR` 指定合成测试集目录。

示例：

```bash
export DATA_DIR=/data/CBLPRD
export OUTPUT_DIR=/data/outputs
export TEACHER_WEIGHTS=/data/pretrained/lprnet_v5_best_model.pth
export STUDENT_WEIGHTS=/data/pretrained/best_lprnet_v6_distilled_v2.pth
python scripts/train_lprnet_v6_distill_v3.py
```

## 已知限制

ESP32 INT8 量化后，绿牌末尾字符丢失 8.61%（267/3100），完全匹配率 95.55%。PC float 模型无此问题（绿牌准确率 100%）。根因是窄高比绿牌（宽高比 ≥ 4.0）在 INT8 量化后第 8 位字符信号过弱，V3 已通过降低 blank 抑制、扩大窄高比增强、绿牌加权缓解，但端侧量化精度限制下仍存在残留丢失。

## 权重文件

模型权重通过 GitHub Release 托管，详见 [../RELEASE.md](../RELEASE.md)。下载后放入 `pretrained/`（训练权重）与 `outputs/`（ONNX/espdl 产物）目录。
