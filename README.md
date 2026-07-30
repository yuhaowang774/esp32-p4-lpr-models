# ESP32-P4 车牌识别模型仓库

## 简介

本仓库开源了在 ESP32-P4-EYE 开发板（双核 RISC-V 400MHz）上部署的车牌识别模型，包含：

- **YOLO11n v3 256×256 车牌检测模型**（INT8 量化）
- **LPRNet V6 蒸馏 V3 字符识别模型**（INT8 量化）

两个模型已在 ESP32-P4 上实际部署验证，端到端识别延迟约 474ms。

## 性能指标

| 模型 | 输入 | mAP50 / 准确率 | 推理时间 | espdl 大小 |
|------|------|---------------|---------|-----------|
| YOLO11n v3 | 256×256 | mAP50=0.9950, mAP50-95=0.8048 | ~400ms | 2.98 MB |
| LPRNet V6蒸馏V3 | 128×32 | PC float 99.95%, ESP32 INT8 95.55% | ~61ms | 0.65 MB |

## 目录结构

```
opensource/
├── yolo11n_v3/                 # YOLO11n 车牌检测模型
│   ├── scripts/                # 训练、量化、评估脚本（15个）
│   ├── configs/                # 训练配置文件
│   ├── models/                 # 模型元信息（.info/.json）
│   ├── docs/                   # 模型归档文档
│   └── README.md
├── lprnet_v6_distill_v3/       # LPRNet 字符识别模型
│   ├── scripts/                # 训练、量化、评估脚本（18个，含V5/V6依赖）
│   ├── tests/                  # 单元测试
│   ├── models/                 # 模型元信息
│   ├── docs/                   # 模型归档文档
│   └── README.md
├── docs/                       # 通用技术文档
├── LICENSE                     # MIT（代码）
├── LICENSE-WEIGHTS.md          # CC BY-NC-SA 4.0（权重）
├── NOTICE.md                   # 第三方数据集声明
└── RELEASE.md                  # 权重文件下载清单
```

## 快速开始

权重文件通过 GitHub Release 托管，未直接提交到 Git 仓库。使用前需先按 [RELEASE.md](./RELEASE.md) 中的清单和 MD5 校验，将权重下载到对应目录（`yolo11n_v3/pretrained/`、`yolo11n_v3/outputs/`、`lprnet_v6_distill_v3/pretrained/`、`lprnet_v6_distill_v3/outputs/`）后再运行训练或量化脚本。

## 训练环境

- Python 3.11
- PyTorch 2.8.0+cu129
- ultralytics（YOLO11n）
- esp-ppq 0.2.4（INT8 量化）
- CUDA 12.9 + cuDNN 9.7+

## 数据集

- **CCPD2020**（CC BY-NC-SA 4.0）：https://github.com/detectRecog/CCPD
- **CBLPRD-330k**：学术开源

数据集协议详见 [NOTICE.md](./NOTICE.md)。

## 许可证

- 代码：MIT（见 [LICENSE](./LICENSE)）
- 模型权重：CC BY-NC-SA 4.0（见 [LICENSE-WEIGHTS.md](./LICENSE-WEIGHTS.md)）
- 第三方数据集与依赖声明：见 [NOTICE.md](./NOTICE.md)
