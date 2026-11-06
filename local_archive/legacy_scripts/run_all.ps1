<#
DeLELSTM 改进版 · 一键跑重复实验

用法:
    .\run_all.ps1 -Mode B -NumExp 5                # 方案B：每次重新划分（推荐主实验）
    .\run_all.ps1 -Mode A -NumExp 5                # 方案A：固定划分（对齐论文口径）
    .\run_all.ps1 -Mode both -NumExp 5             # 两个都跑
    .\run_all.ps1 -Mode B -NumExp 3 -Epochs 2      # 冒烟测试：3 次 × 2 轮
    .\run_all.ps1 -Mode B -Models Delelstm,Retain  # 只跑部分模型

产出:
    results_A/  方案A 结果
    results_B/  方案B 结果
    跑完会自动调用 summarize.py 输出 mean±std
#>
param(
    [ValidateSet('A', 'B', 'both')]
    [string]$Mode = 'B',
    [int]$NumExp = 5,
    [string]$Models = '',
    [int]$Epochs = 0
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$code = Join-Path $root 'DeLELSTM-main'

if (-not (Test-Path $code)) { throw "找不到 $code" }

if ($Models -ne '') { $env:DELE_MODELS = $Models }
if ($Epochs -gt 0) { $env:DELE_EPOCHS = "$Epochs" }

Write-Host "models   : $(if ($Models -ne '') { $Models } else { '全部 5 个' })"
Write-Host "epochs   : $(if ($Epochs -gt 0) { $Epochs } else { '各数据集默认 (350/300/300)' })"
Write-Host "num_exp  : $NumExp"
Write-Host ''

$scripts = @('Electricity.py', 'PM.py', 'Exchange.py')

function Invoke-RunSet {
    param([string]$Tag, [bool]$Resample)

    $outDir = "..\results_$Tag"
    $label = if ($Resample) { '方案B：每次重新划分' } else { '方案A：固定划分' }
    Write-Host "==================== $label -> $outDir ===================="

    foreach ($s in $scripts) {
        $argv = @($s, '--num_exp', "$NumExp", '--save_dirs', $outDir)
        if ($Resample) { $argv += '--resample_split' }

        Write-Host "---- $s  $(Get-Date -Format 'HH:mm:ss') ----"
        Push-Location $code
        try {
            & python @argv
            if ($LASTEXITCODE -ne 0) { throw "$s 退出码 $LASTEXITCODE" }
        }
        finally { Pop-Location }
    }
    Write-Host ''
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()

if ($Mode -eq 'A' -or $Mode -eq 'both') { Invoke-RunSet -Tag 'A' -Resample $false }
if ($Mode -eq 'B' -or $Mode -eq 'both') { Invoke-RunSet -Tag 'B' -Resample $true }

$sw.Stop()
Write-Host ("总耗时 {0:N1} 小时" -f $sw.Elapsed.TotalHours)
Write-Host ''

Write-Host '==================== 汇总 ===================='
Push-Location $root
try {
    if ($Mode -eq 'A' -or $Mode -eq 'both') { & python 'summarize.py' 'results_A' }
    if ($Mode -eq 'B' -or $Mode -eq 'both') { & python 'summarize.py' 'results_B' }
}
finally { Pop-Location }
