param(
    [string]$PythonExe = $env:PYTHON_EXE
)

$ErrorActionPreference = "Stop"
if (-not $PythonExe) {
    $PythonExe = "python"
}

$srcDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $srcDir
Push-Location $srcDir
try {
    & $PythonExe .\02_build_base_cache.py
    & $PythonExe .\03_build_daily_atomic.py
    & $PythonExe .\04_build_daily_factor.py
    & $PythonExe .\05_build_monthly_panel.py
    & $PythonExe .\06_process_factors.py
    & $PythonExe .\07_backtest_ic_groups.py --output-root (Join-Path $projectDir "result\backtest_v4")
    & $PythonExe .\08_build_style_exposure.py
    & $PythonExe .\09_style_test.py
    & $PythonExe .\11_factor_decay.py --output-root (Join-Path $projectDir "result\decay_v4")
    & $PythonExe .\12_build_report_figures.py --backtest-root (Join-Path $projectDir "result\backtest_v4") --style-root (Join-Path $projectDir "result\style_test") --output-dir (Join-Path $srcDir "report\figs_v21")
}
finally {
    Pop-Location
}
