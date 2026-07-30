"""V6 蒸馏 V3 训练配置

基于 V6 蒸馏 V2 成功方案，针对绿牌末尾字符丢失问题优化：
- blank_logit_reduction: 0.3 → 0.1（减少 blank 产生，帮助末尾字符输出）
- letterbox 宽高比范围: (2.5, 6.5) → (2.5, 8.0)（增加窄高比样本）
- 绿牌样本加权: 1.5x（增强 8 位绿牌学习）
- letterbox_prob: 0.5 → 0.6（更多 letterbox 场景训练）
- 从 V2 best 权重初始化（在 V2 成功基础上微调）

V2 → V3 改进总结：
  问题：V2 在 simcrop 测试中 267 个绿牌缺末尾字符（8.6%）
  根因：模型在窄高比绿牌（>=4.0）上对第 8 位字符信号太弱
  方案：降低 blank_logit_reduction + 扩大窄高比增强 + 绿牌加权
"""

import os

# 路径配置（通过环境变量覆盖，默认使用相对路径）
DATA_DIR = os.environ.get("DATA_DIR", "./data/CBLPRD")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
TEACHER_WEIGHTS = os.environ.get("TEACHER_WEIGHTS", "./pretrained/lprnet_v5_best_model.pth")
# V3：从 V2 best 权重初始化（在 V2 成功基础上微调）
STUDENT_WEIGHTS = os.environ.get("STUDENT_WEIGHTS", "./pretrained/best_lprnet_v6_distilled_v2.pth")

# ===== 模型配置 =====
NUM_CLASSES = 66

# ===== 蒸馏配置 =====
DISTILL_TEMPERATURE = 2.0       # 温度参数（与 V6 蒸馏 V1/V2 一致）
# alpha 渐进调度：epoch 1-5=0.7, 6-20=0.5, 21+=0.3
DISTILL_ALPHA = 0.5
BLANK_LOGIT_REDUCTION = 0.1     # V3: 0.3→0.1，减少 blank 产生，帮助绿牌末尾字符
PROVINCE_AUX_WEIGHT = 0.5
PROVINCE_WEIGHT_CAP = 5.0       # 限制稀有省份权重上限

# ===== 训练配置 =====
BATCH_SIZE = 512
LEARNING_RATE = 3e-4            # V3: 5e-4→3e-4，微调级学习率（从 V2 权重继续训练）
WEIGHT_DECAY = 1e-4
EPOCHS = 60
WARMUP_PCT = 0.1
GRAD_CLIP = 5.0
PATIENCE = 15
DROPOUT_RATE = 0.3
USE_AMP = True

# ===== CTC 配置 =====
BLANK_WEIGHT = 2.0
BLANK_IDX = 0

# ===== 数据配置 =====
# 数据集文件路径
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
TEST_TXT = os.path.join(DATA_DIR, "test.txt")
FILTER_TYPES = ["普通蓝牌", "新能源小型车"]

# ===== letterbox 增强配置 =====
LETTERBOX_FILL_COLOR = (114, 114, 114)  # 与 ESP32 端侧推理一致
LETTERBOX_ASPECT_RANGE = (2.5, 8.0)     # V3: (2.5,6.5)→(2.5,8.0)，增加窄高比样本
LETTERBOX_HEIGHT_RANGE = (16, 72)       # 随机高度范围
LETTERBOX_ALIGN = 8                     # 尺寸 8 对齐

# ===== 绿牌样本加权配置（V3 新增）=====
GREEN_PLATE_WEIGHT = 1.5    # 绿牌样本权重 1.5x（蓝牌 1.0x）
# 窄高比绿牌额外加权（宽高比 >= 5.0 的绿牌）
NARROW_GREEN_EXTRA_WEIGHT = 2.0  # 窄高比绿牌总权重 = 1.5 * 2.0 = 3.0x

# ===== 渐进增强阶段配置 =====
PHASE_BOUNDARIES = {"warmup": 10, "medium": 40, "full": 999}
# V3: letterbox_prob 从 0.5 提高到 0.6
PHASE_LETTERBOX_PROB = {"warmup": 0.3, "medium": 0.6, "full": 0.6}

# ===== 渐进 alpha 调度 =====
ALPHA_SCHEDULE = [(5, 0.7), (20, 0.5), (999, 0.3)]

# ===== 右侧 padding 增强 =====
RIGHT_PADDING_PROB = 0.3
RIGHT_PADDING_MAX_RATIO = 0.15

# ===== 双验证集配置 =====
VAL_CLEAN_WEIGHT = 0.5
VAL_LETTERBOX_WEIGHT = 0.5
VAL_LETTERBOX_SEED = 42

# ===== 量化校准配置 =====
CALIB_NUM_SAMPLES = 2048
CALIB_LETTERBOX_RATIO = 0.5
CALIB_SEED = 42

# ===== 输出配置 =====
MODEL_NAME = "lprnet_v6_distilled_v3"
ONNX_NAME = "lprnet_v6_distilled_v3.onnx"
