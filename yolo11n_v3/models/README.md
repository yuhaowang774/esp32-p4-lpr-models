# YOLO11n v3 模型元信息

本目录存放模型量化元信息文件（`.info` / `.json`），用于描述 INT8 量化模型的算子、张量与量化参数等元数据。

实际权重文件（`.pt` / `.onnx` / `.espdl`）通过 GitHub Release 托管，不提交到 Git 仓库，下载方式见 [/RELEASE.md](../../RELEASE.md)。

## 文件清单

| 文件 | 说明 |
|------|------|
| `yolo11n_256x256_v3_int8.info` | INT8 量化模型元信息 |
| `yolo11n_256x256_v3_int8.json` | INT8 量化模型元信息（JSON 格式） |
