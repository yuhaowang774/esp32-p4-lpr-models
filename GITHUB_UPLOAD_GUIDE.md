# GitHub 开源上传指南

本指南详细介绍如何将本仓库发布到 GitHub，包括代码推送和权重文件上传。

## 前置条件

1. 已安装 Git（`git --version` 可正常输出）
2. 已注册 GitHub 账号
3. 已配置 Git 用户信息：
   ```bash
   git config --global user.name "你的名字"
   git config --global user.email "你的邮箱"
   ```
4. 已配置 GitHub 认证（SSH 密钥或 Personal Access Token）

## 第一步：创建 GitHub 仓库

1. 登录 GitHub，点击右上角 **+** → **New repository**
2. 填写仓库信息：
   - **Repository name**: `esp32-p4-lpr-models`
   - **Description**: `ESP32-P4 车牌识别模型：YOLO11n 车牌检测 + LPRNet V6 蒸馏字符识别（INT8 量化）`
   - **Visibility**: Public
   - **Initialize this repository**: **不要勾选** 任何选项（README/.gitignore/license 都不选，因为本地已有）
3. 点击 **Create repository**

## 第二步：推送代码

```bash
cd opensource

# 添加远程仓库（替换 <your-username> 为你的 GitHub 用户名）
git remote add origin https://github.com/<your-username>/esp32-p4-lpr-models.git

# 推送代码
git push -u origin main
```

如果使用 SSH：
```bash
git remote add origin git@github.com:<your-username>/esp32-p4-lpr-models.git
git push -u origin main
```

## 第三步：收集权重文件

此步骤仅维护者需要。普通用户应直接从 GitHub Release 下载权重（见第四步）。

```bash
# 回到项目根目录
cd ..

# 设置 LPRNet 训练目录环境变量（替换为你的实际路径）
$env:LPRNET_TRAIN_DIR = "你的LPRNet训练根目录"

# 执行收集脚本
.\opensource\scripts\collect_weights.ps1
```

脚本会自动：
- 从项目内和 LPRNet 训练目录收集所有 9 个权重文件
- 校验每个文件的 MD5
- 输出到 `opensource/release_assets/` 目录

预期输出：
```
[OK]   yolo11n_256x256_v3_phase2_best.pt (5.17 MB)
[OK]   yolo11n_256x256_v3_fp32.onnx (9.92 MB)
[OK]   yolo11n_256x256_v3_int8.espdl (2.98 MB)
[OK]   final_lprnet_v6_distilled_v3.pth (2.42 MB)
[OK]   best_lprnet_v6_distilled_v3.pth (7.24 MB)
[OK]   lprnet_v6_distilled_v3.onnx (2.34 MB)
[OK]   lprnet_v6_int8.espdl (0.63 MB)
[OK]   lprnet_v5_best_model.pth (28.2 MB)
[OK]   best_lprnet_v6_distilled_v2.pth (7.24 MB)
========================================
收集完成: 9 个成功, 0 个失败
总大小: 65.34 MB
```

## 第四步：创建 GitHub Release

1. 在 GitHub 仓库页面，点击右侧 **Releases** → **Create a new release**
2. 填写 Release 信息：
   - **Choose a tag**: 输入 `v1.0.0`，然后点击 **Create new tag: v1.0.0 on publish**
   - **Release title**: `v1.0.0 - 首次发布`
   - **Description**: 填写以下内容：

```markdown
## ESP32-P4 车牌识别模型 v1.0.0

### 模型清单

#### YOLO11n v3 256x256（车牌检测）
- mAP50: 0.9950
- mAP50-95: 0.8048
- INT8 量化，ESP32-P4 推理 ~400ms

#### LPRNet V6 蒸馏 V3（字符识别）
- PC float 准确率: 99.95%
- ESP32 INT8 完全匹配率: 95.55%
- INT8 量化，ESP32-P4 推理 ~61ms

### 权重文件

详见 [RELEASE.md](./RELEASE.md) 获取完整的文件清单、MD5 校验和放置目录。

### 许可证
- 代码: MIT
- 模型权重: CC BY-NC-SA 4.0（基于 CCPD/CBLPRD 数据集）
```

3. **Attach binaries**: 将 `opensource/release_assets/` 目录下的所有 9 个文件拖拽到附件区域
4. 点击 **Publish release**

## 第五步：验证

1. **检查仓库页面**：确认代码已推送，README 正常显示
2. **检查 Release**：确认 9 个权重文件已上传，文件大小正确
3. **检查 CITATION.cff**：GitHub 会在仓库页面右侧显示 "Cite this repository" 按钮
4. **检查 LICENSE**：GitHub 会自动识别 MIT 许可证并在仓库页面显示

## 第六步：更新 README 中的仓库地址

推送代码后，需要把 README.md 和 CITATION.cff 中的 `<your-username>` 替换为实际 GitHub 用户名：

```bash
cd opensource

# 替换 <your-username> 为你的实际用户名
# 例如用户名为 zhangsan，则执行：
# 在 PowerShell 中手动编辑，或用以下命令（替换 zhangsan 为你的用户名）

# 编辑 README.md
# 编辑 CITATION.cff
# 编辑 RELEASE.md（Release 下载链接）

git add README.md CITATION.cff RELEASE.md
git commit -m "docs: update repository URL with actual GitHub username"
git push
```

## 常见问题

### Q: 推送代码时提示认证失败？

A: GitHub 已不支持密码认证，需要使用以下方式之一：
1. **Personal Access Token (PAT)**：在 GitHub Settings → Developer settings → Personal access tokens 创建，用 token 代替密码
2. **SSH 密钥**：生成 SSH 密钥并添加到 GitHub，改用 SSH 地址

### Q: Release 附件上传失败或超时？

A: GitHub Release 单文件限制 2GB，总附件无限制。如果网络不稳定，可以：
1. 分批上传
2. 使用 `gh` CLI 工具：`gh release create v1.0.0 --title "v1.0.0" release_assets/*`

### Q: 权重文件收集脚本提示文件不存在？

A: 检查以下几点：
1. 脚本必须在 `esp32-p4-eye-project` 目录下执行
2. 需设置 `LPRNET_TRAIN_DIR` 环境变量指向你的 LPRNet 训练根目录
3. 如果训练目录结构不同，需要修改 `collect_weights.ps1` 中的子路径

### Q: Git 提交时出现 LF/CRLF 警告？

A: 这是 Windows 上的正常现象，不影响功能。如果需要消除警告：
```bash
git config core.autocrlf false
```

## 仓库维护

后续如果需要更新模型：
1. 修改代码后提交：`git add . && git commit -m "描述" && git push`
2. 发布新版本：创建新的 Release（如 v1.1.0），上传更新后的权重文件
3. 更新 RELEASE.md 中的版本号和文件清单
