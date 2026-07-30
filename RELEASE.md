# Release 权重文件清单

所有模型权重文件通过 GitHub Release 托管，不直接提交到 Git 仓库。下载后放入对应目录即可使用。

## YOLO11n v3 256×256

### 下载地址

Release v1.0: `https://github.com/<your-username>/esp32-p4-lpr-models/releases/tag/v1.0`

### 文件清单

| 文件 | 放置目录 | 大小 | MD5 | 说明 |
|------|---------|------|-----|------|
| `yolo11n_256x256_v3_phase2_best.pt` | `yolo11n_v3/pretrained/` | 5,423,443 字节 (5.2 MB) | `312134AFE15D21D6536524BA5A6DA971` | V3 QAT 训练最佳权重，可重新导出任意分辨率 ONNX/INT8 |
| `yolo11n_256x256_v3_fp32.onnx` | `yolo11n_v3/outputs/` | 10,407,968 字节 (9.9 MB) | `971D03B98FFE8B68117AFDD3CDB3EB0F` | FP32 ONNX 中间格式，可用于 PC 端验证或重新量化 |
| `yolo11n_256x256_v3_int8.espdl` | `yolo11n_v3/outputs/` | 3,126,048 字节 (2.98 MB) | `8157CE76714B66E1FCE5C5B2E5B181AB` | ESP32-P4 部署的 INT8 量化模型 |
| `yolo11n.pt` | `yolo11n_v3/pretrained/` | — | — | Ultralytics 官方预训练权重（从 ultralytics 下载） |

### 性能指标

| 指标 | 值 |
|------|-----|
| mAP50 | 0.9950 |
| mAP50-95 | 0.8048 |
| 推理时间 (ESP32-P4) | ~400ms（估算） |
| 等效帧率 | ~2.5 FPS |

---

## LPRNet V6 蒸馏 V3

### 文件清单

| 文件 | 放置目录 | 大小 | MD5 | 说明 |
|------|---------|------|-----|------|
| `final_lprnet_v6_distilled_v3.pth` | `lprnet_v6_distill_v3/pretrained/` | 2,539,802 字节 (2.4 MB) | `4DF25025E391F4E7D0CB28F6009587C5` | 最终模型权重，可重新导出 ONNX/INT8 |
| `best_lprnet_v6_distilled_v3.pth` | `lprnet_v6_distill_v3/pretrained/` | 7,590,864 字节 (7.2 MB) | `AACF5EF81C1711A5CAAB20C34A0E0EB6` | 最佳验证集权重（备份） |
| `lprnet_v6_distilled_v3.onnx` | `lprnet_v6_distill_v3/outputs/` | 2,516,577 字节 (2.4 MB) | `AD88AE94BF75FF46F31874B11CE74352` | ONNX 中间格式 |
| `lprnet_v6_distilled_v3_int8.espdl` | `lprnet_v6_distill_v3/outputs/` | 665,328 字节 (0.65 MB) | `037E2F207A136DEB408FA2919B481FA9` | ESP32-P4 部署的 INT8 量化模型 |
| `lprnet_v5_best_model.pth` | `lprnet_v6_distill_v3/pretrained/` | — | — | V5 教师模型权重（蒸馏用） |
| `best_lprnet_v6_distilled_v2.pth` | `lprnet_v6_distill_v3/pretrained/` | — | — | V6 蒸馏 V2 学生权重（V3 初始化用） |

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| PC float Seq Acc | 99.95% | 6200 样本测试 |
| PC float 绿牌准确率 | 100.00% | 末尾字符丢失 0 |
| ESP32 INT8 完全匹配率 | 95.55% | 6200 样本测试 |
| ESP32 INT8 绿牌末尾丢失 | 8.61% (267/3100) | INT8 量化精度限制 |
| 推理时间 (ESP32-P4) | ~61ms | — |

---

## 目录结构对照

下载权重后，仓库目录应如下：

```
opensource/
├── yolo11n_v3/
│   ├── pretrained/
│   │   ├── yolo11n_256x256_v3_phase2_best.pt    ← 从 Release 下载
│   │   └── yolo11n.pt                           ← 从 ultralytics 下载
│   └── outputs/
│       ├── yolo11n_256x256_v3_fp32.onnx         ← 从 Release 下载
│       └── yolo11n_256x256_v3_int8.espdl        ← 从 Release 下载
└── lprnet_v6_distill_v3/
    ├── pretrained/
    │   ├── final_lprnet_v6_distilled_v3.pth      ← 从 Release 下载
    │   ├── best_lprnet_v6_distilled_v3.pth       ← 从 Release 下载
    │   ├── lprnet_v5_best_model.pth              ← 从 Release 下载
    │   └── best_lprnet_v6_distilled_v2.pth       ← 从 Release 下载
    └── outputs/
        ├── lprnet_v6_distilled_v3.onnx           ← 从 Release 下载
        └── lprnet_v6_distilled_v3_int8.espdl     ← 从 Release 下载
```

## MD5 校验

下载后请校验 MD5，确保文件完整性：

```bash
# Linux/macOS
md5sum yolo11n_256x256_v3_int8.espdl
# 应输出: 8157ce76714b66e1fce5c5b2e5b181ab

# Windows PowerShell
Get-FileHash yolo11n_256x256_v3_int8.espdl -Algorithm MD5
# 应输出: 8157CE76714B66E1FCE5C5B2E5B181AB
```
