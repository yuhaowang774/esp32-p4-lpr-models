# 第三方数据集声明

本项目模型基于以下开源数据集训练，使用时需遵守各自协议：

## CCPD2020 (Chinese City Parking Dataset)

- **协议**：CC BY-NC-SA 4.0（署名-非商用-相同方式共享）
- **用途**：YOLO11n 车牌检测模型训练
- **规模**：约 25 万张中国车牌图片（蓝牌/绿牌）
- **来源**：https://github.com/detectRecog/CCPD
- **引用**：
  ```
  @article{CCPD,
    title={CCPD: a diverse and well-annotated dataset for license plate detection},
    author={Xu, Zhenbo and Yang, Wei and Meng, Aidi and Lu, Nanxue and Huang, Huan and Yin, Changchun and Li, Rixiang},
    journal={arXiv preprint arXiv:1812.04519},
    year={2018}
  }
  ```

## CBLPRD-330k (Chinese Big License Plate Recognition Dataset)

- **协议**：学术开源，限非商用
- **用途**：YOLO11n 检测 + LPRNet 字符识别训练
- **规模**：约 33 万张中国车牌图片
- **说明**：含 `bg128.32` 子集（128×32 裁剪车牌，用于 LPRNet 训练）

## Ultralytics YOLO11n 预训练权重

- **协议**：AGPL-3.0
- **用途**：YOLO11n 训练初始化权重
- **来源**：https://github.com/ultralytics/ultralytics
- **注意**：AGPL-3.0 要求基于此权重训练的模型，若通过网络提供服务，需开源全部代码。本项目模型权重已叠加 CC BY-NC-SA 4.0（非商用），与 AGPL-3.0 的开源要求叠加适用。

## ESP-DL / ESP-PPQ

- **协议**：Apache 2.0
- **用途**：ESP32-P4 端侧 INT8 量化推理框架
- **来源**：https://github.com/espressif/esp-dl

## PyTorch / Ultralytics

- **PyTorch**：BSD-style
- **Ultralytics**：AGPL-3.0
