# YOLO11n v3 车牌检测模型

## 模型简介

YOLO11n v3 是面向 ESP32-P4-EYE 端侧部署的车牌检测模型：

- 输入分辨率：256×256
- 类别数：1（车牌）
- 量化方式：INT8（esp-ppq 0.2.4）
- 部署格式：.espdl（约 2.98 MB）
- 指标：mAP50=0.9950，mAP50-95=0.8048

## 脚本说明

| 脚本 | 说明 |
|------|------|
| `train_yolo11n_v3.py` | 训练入口，含 Phase1 正常训练与 Phase2 QAT 量化感知训练 |
| `export_onnx_for_espdl.py` | 导出 FP32 ONNX 中间格式，供 PC 端验证与量化 |
| `quantize_yolo11n_v3_256.py` | 基于 esp-ppq 进行 INT8 量化并导出 .espdl |
| `t1_verify_onnx.py` | 验证 ONNX 模型推理结果与 PyTorch 一致性 |
| `t2_verify_espdl.py` | 验证 INT8 espdl 模型量化精度损失 |
| `t3_verify_score.py` | 在评估集上验证最终量化模型 mAP 分数 |
| `eval_test.py` | 在 CCPD 测试集上评估模型精度 |
| `eval_per_class.py` | 按蓝牌/绿牌分类别评估 mAP |
| `eval_tilted.py` | 评估倾斜车牌的检测精度 |
| `augment_ccpd.py` | CCPD 数据增强（旋转、模糊、亮度等极端场景） |
| `analyze_bbox_stats.py` | 统计标注框尺寸/长宽比分布，辅助调参 |
| `build_eval_subset.py` | 构建评估子集（按角度/类型抽样） |
| `sample_ccpd_by_angle.py` | 按车牌倾斜角度抽样分析 |
| `gen_256_from_320.py` | 从 320×320 Phase1 权重重新导出 256×256 输入 |
| `patch_esp_ppq.py` | 修补 esp-ppq 在导出 espdl 时的兼容性问题 |

## 训练流程

采用两阶段训练：

1. **Phase1 正常训练（320×320）**：在 CCPD 增强数据集上以 320×320 大分辨率训练，学习率较大，配合强增强（mosaic、mixup、旋转、透视等）学习鲁棒特征。详见 `configs/phase1_args.yaml`。
2. **Phase2 QAT 量化感知训练（256×256）**：加载 Phase1 权重，切换到 256×256 输入做量化感知训练，使用极小学习率（1e-4），减弱数据增强以防止量化退化，关闭 amp（FP32 训练）。详见 `configs/phase2_qat_args.yaml`。
3. **导出与量化**：Phase2 完成后导出 256×256 ONNX，再通过 esp-ppq 进行 INT8 量化生成 .espdl。

## 环境变量

脚本默认使用相对路径，可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `CCPD_DIR` | `./data/CCPD2020` | CCPD2020 原始数据集目录 |
| `CCPD_AUG_DIR` | `./data/CCPD2020_augmented` | 增强后数据集目录 |
| `OUTPUT_DIR` | `./outputs` | 训练/导出/量化输出目录 |
| `PRETRAINED_WEIGHTS` | `./pretrained/yolo11n.pt` | 预训练权重路径 |

示例：

```bash
export CCPD_DIR=/data/CCPD2020
export CCPD_AUG_DIR=/data/CCPD2020_augmented
export OUTPUT_DIR=/data/outputs
export PRETRAINED_WEIGHTS=/data/pretrained/yolo11n.pt
python scripts/train_yolo11n_v3.py
```

## 配置文件

| 配置文件 | 说明 |
|---------|------|
| `configs/data.yaml` | 数据集路径与类别定义（蓝牌/绿牌）模板 |
| `configs/phase1_args.yaml` | Phase1 正常训练超参（epochs、imgsz、增强、损失权重等） |
| `configs/phase2_qat_args.yaml` | Phase2 QAT 量化感知训练超参（弱化增强、关闭 amp、极小 lr） |

> 注：配置文件为参考模板，实际超参以脚本内常量区为准。

## 权重文件

模型权重通过 GitHub Release 托管，详见 [../RELEASE.md](../RELEASE.md)。下载后放入 `pretrained/`（训练权重）与 `outputs/`（ONNX/espdl 产物）目录。
