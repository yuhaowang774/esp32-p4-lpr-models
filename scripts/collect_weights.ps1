# Weight files collection script (maintainer-only)
# Collects all weight files from project and external training directories to release_assets/
# Usage: run from esp32-p4-eye-project directory: .\opensource\scripts\collect_weights.ps1
#
# NOTE: This script is for project maintainers to collect weights from local training
# directories. Other users should download weights from GitHub Release instead.
# Configure LPRNET_TRAIN_DIR environment variable to point to your local LPRNet training root.

$ErrorActionPreference = "Stop"

# Project root (3 levels up from this script)
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "opensource\release_assets"

# External training directory (maintainer's local path, override via environment variable)
$LPRNET_TRAIN_DIR = $env:LPRNET_TRAIN_DIR
if (-not $LPRNET_TRAIN_DIR) {
    $LPRNET_TRAIN_DIR = ".\data\lprnet_train"
}

# Create output directory
if (Test-Path $OUTPUT_DIR) {
    Remove-Item $OUTPUT_DIR -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
Write-Host "Output: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "LPRNet train dir: $LPRNET_TRAIN_DIR" -ForegroundColor Cyan

# Weight files: source path -> destination filename
# For absolute paths (G:\), use directly; for relative paths, prepend PROJECT_ROOT
$files = @(
    # YOLO11n v3 (from project directory)
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
    # LPRNet V6 distill V3 (from external training directory)
    @{
        src = Join-Path $LPRNET_TRAIN_DIR "lprnet_v6_distill_v3_project\final_lprnet_v6_distilled_v3.pth"
        dst = "final_lprnet_v6_distilled_v3.pth"
        expected_md5 = "4DF25025E391F4E7D0CB28F6009587C5"
    },
    @{
        src = Join-Path $LPRNET_TRAIN_DIR "lprnet_v6_distill_v3_project\best_lprnet_v6_distilled_v3.pth"
        dst = "best_lprnet_v6_distilled_v3.pth"
        expected_md5 = "AACF5EF81C1711A5CAAB20C34A0E0EB6"
    },
    @{
        src = Join-Path $LPRNET_TRAIN_DIR "lprnet_v6_distill_v3_project\lprnet_v6_distilled_v3.onnx"
        dst = "lprnet_v6_distilled_v3.onnx"
        expected_md5 = "AD88AE94BF75FF46F31874B11CE74352"
    },
    @{
        src = "factory_demo\model\lprnet\v6\lprnet_v6_int8.espdl"
        dst = "lprnet_v6_int8.espdl"
        expected_md5 = "98C70FC078384903B083D0935606A660"
    },
    # LPRNet V5 teacher + V2 student (distillation dependencies)
    @{
        src = Join-Path $LPRNET_TRAIN_DIR "lprnet_v5_project\lpenet_v5\best_model.pth"
        dst = "lprnet_v5_best_model.pth"
        expected_md5 = "95BDE52EFF653ABF3CC6E31C5062F81F"
    },
    @{
        src = Join-Path $LPRNET_TRAIN_DIR "lprnet_v6_distill_v2_project\best_lprnet_v6_distilled_v2.pth"
        dst = "best_lprnet_v6_distilled_v2.pth"
        expected_md5 = "A16FDB3025189E643B805FD1A0800D86"
    }
)

$success = 0
$failed = 0

foreach ($f in $files) {
    # For absolute paths (e.g. G:\), use directly; for relative paths, prepend PROJECT_ROOT
    if ([System.IO.Path]::IsPathRooted($f.src)) {
        $srcPath = $f.src
    } else {
        $srcPath = Join-Path $PROJECT_ROOT $f.src
    }
    if (-not (Test-Path $srcPath)) {
        Write-Host "[FAIL] $($f.dst): source not found ($srcPath)" -ForegroundColor Red
        $failed++
        continue
    }

    $dstPath = Join-Path $OUTPUT_DIR $f.dst
    Copy-Item $srcPath $dstPath -Force

    # MD5 verification
    $actual_md5 = (Get-FileHash $dstPath -Algorithm MD5).Hash
    if ($actual_md5 -eq $f.expected_md5) {
        $sizeMB = [math]::Round((Get-Item $dstPath).Length / 1MB, 2)
        Write-Host "[OK]   $($f.dst) ($sizeMB MB)" -ForegroundColor Green
        $success++
    } else {
        Write-Host "[WARN] $($f.dst): MD5 mismatch (expected: $($f.expected_md5), actual: $actual_md5)" -ForegroundColor Yellow
        $failed++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Done: $success succeeded, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "Output: $OUTPUT_DIR" -ForegroundColor Cyan

if ($failed -eq 0) {
    $totalSize = (Get-ChildItem $OUTPUT_DIR | Measure-Object -Property Length -Sum).Sum
    $totalMB = [math]::Round($totalSize / 1MB, 2)
    Write-Host "Total: $totalMB MB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next: upload all files in release_assets/ to GitHub Release" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Please check failed items" -ForegroundColor Red
}
