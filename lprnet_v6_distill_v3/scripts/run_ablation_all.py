"""批量运行所有消融实验变体"""

import subprocess
import sys
import os

VARIANTS = ["A1", "A2", "A3", "A4", "A5"]
EPOCHS = 20

for variant in VARIANTS:
    print(f"\n{'='*60}")
    print(f"开始训练消融变体 {variant}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, "-u", "train_ablation.py",
         "--variant", variant, "--epochs", str(EPOCHS)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode != 0:
        print(f"变体 {variant} 训练失败，返回码 {result.returncode}")
        # 继续下一个变体
    else:
        print(f"变体 {variant} 训练完成")

print(f"\n{'='*60}")
print("所有消融实验变体训练完成")
print(f"{'='*60}")
