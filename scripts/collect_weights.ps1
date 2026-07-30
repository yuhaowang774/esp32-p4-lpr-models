# 权重文件收集脚本
# 从项目内和 G 盘收集所有权重文件到 release_assets/ 目录
# 用法：在 esp32-p4-eye-project 目录下执行 .\opensource\scripts\collect_weights.ps1

$ErrorActionPreference = "Stop"

# 项目根目录（脚本所在目录的上三级）
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "opensource\release_assets"

# 创建输出目录
if (Test-Path $OUTPUT_DIR) {
    Remove-Item $OUTPUT_DIR -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
Write-Host "输出目录: $OUTPUT_DIR" -ForegroundColor Cyan

# 权重文件清单：源路径 -> 目标文件名
$files = @(
    # YOLO11n v3
    @{
        src = "factory_demo\model\yolo_v4_package\03_baseline\phase2_best_v3.pt"
        dst = "yolo11n_256x256_v3_phase2_best.pt"
        expected_md5 = "312134AFE15D21D6536524BA5A6DA971"
    },
    @{
        src = "factory_demo\model\yolo\yolo11n\yolo11n_256x256_v3_fp32.onnx"
        dst = "yolo11n_256x256_v3_fp32.onnx"
        expected_md5 = "971D03B98FFE8B68117AFDD3CDB3EB0F"
    },
    @{
        src = "factory_demo\model\yolo\yolo11n\yolo11n_256x256_v3_int8.espdl"
        dst = "yolo11n_256x256_v3_int8.espdl"
        expected_md5 = "8157CE76714B66E1FCE5C5B2E5B181AB"
    },
    # LPRNet V6 蒸馏 V3
    @{
        src = "g:\BaiduNetdiskDownload\CBLPRD-330k_v1\lprnet_v6_distill_v3_project\final_lprnet_v6_distilled_v3.pth"
        dst = "final_lprnet_v6_distilled_v3.pth"
        expected_md5 = "4DF25025E391F4E7D0CB28F6009587C5"
    },
    @{
        src = "g:\BaiduNetdiskDownload\CBLPRD-330k_v1\lprnet_v6_distill_v3_project\best_lprnet_v6_distilled_v3.pth"
        dst = "best_lprnet_v6_distilled_v3.pth"
        expected_md5 = "AACF5EF81C1711A5CAAB20C34A0E0EB6"
    },
    @{
        src = "g:\BaiduNetdiskDownload\CBLPRD-330k_v1\lprnet_v6_distill_v3_project\lprnet_v6_distilled_v3.onnx"
        dst = "lprnet_v6_distilled_v3.onnx"
        expected_md5 = "AD88AE94BF75FF46F31874B11CE74352"
    },
    @{
        src = "factory_demo\model\lprnet\v6\lprnet_v6_int8.espdl"
        dst = "lprnet_v6_int8.espdl"
        expected_md5 = "98C70FC078384903B083D0935606A660"
    },
    # LPRNet V5 教师 + V2 学生（蒸馏依赖）
    @{
        src = "g:\BaiduNetdiskDownload\CBLPRD-330k_v1\lprnet_v5_project\lpenet_v5\best_model.pth"
        dst = "lprnet_v5_best_model.pth"
        expected_md5 = "95BDE52EFF653ABF3CC6E31C5062F81F"
    },
    @{
        src = "g:\BaiduNetdiskDownload\CBLPRD-330k_v1\lprnet_v6_distill_v2_project\best_lprnet_v6_distilled_v2.pth"
        dst = "best_lprnet_v6_distilled_v2.pth"
        expected_md5 = "A16FDB3025189E643B805FD1A0800D86"
    }
)

$success = 0
$failed = 0

foreach ($f in $files) {
    $srcPath = Join-Path $PROJECT_ROOT $f.src
    if (-not (Test-Path $f.src)) {
        $srcPath = $f.src
    }
    if (-not (Test-Path $srcPath)) {
        Write-Host "[FAIL] $($f.dst): 源文件不存在 ($($f.src))" -ForegroundColor Red
        $failed++
        continue
    }

    $dstPath = Join-Path $OUTPUT_DIR $f.dst
    Copy-Item $srcPath $dstPath -Force

    # MD5 校验
    $actual_md5 = (Get-FileHash $dstPath -Algorithm MD5).Hash
    if ($actual_md5 -eq $f.expected_md5) {
        $sizeMB = [math]::Round((Get-Item $dstPath).Length / 1MB, 2)
        Write-Host "[OK]   $($f.dst) ($sizeMB MB)" -ForegroundColor Green
        $success++
    } else {
        Write-Host "[WARN] $($f.dst): MD5 不匹配 (期望: $($f.expected_md5), 实际: $actual_md5)" -ForegroundColor Yellow
        $failed++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "收集完成: $success 个成功, $failed 个失败" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "输出目录: $OUTPUT_DIR" -ForegroundColor Cyan

if ($failed -eq 0) {
    $totalSize = (Get-ChildItem $OUTPUT_DIR | Measure-Object -Property Length -Sum).Sum
    $totalMB = [math]::Round($totalSize / 1MB, 2)
    Write-Host "总大小: $totalMB MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "下一步: 将 release_assets/ 目录下的所有文件上传到 GitHub Release" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "请检查失败项，确认源文件存在且 MD5 正确" -ForegroundColor Red
}
