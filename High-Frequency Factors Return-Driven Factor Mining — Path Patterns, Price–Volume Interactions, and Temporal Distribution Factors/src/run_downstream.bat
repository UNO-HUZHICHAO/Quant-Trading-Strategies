@echo off
rem 03 全量跑完后，一键重跑下游四步（04 滚动 -> 05 面板 -> 06 中性化 -> 07 回测）。
rem 用法：双击或在 src 目录下执行 run_downstream.bat
cd /d "%~dp0"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

echo ===== 04 20日滚动因子 =====
"%PYTHON_EXE%" 04_build_daily_factor.py || goto :err
echo ===== 05 月末回测面板 =====
"%PYTHON_EXE%" 05_build_monthly_panel.py || goto :err
echo ===== 06 MAD+施密特中性化 =====
"%PYTHON_EXE%" 06_process_factors.py || goto :err
echo ===== 07 分组回测+IC+图表 =====
"%PYTHON_EXE%" 07_backtest_ic_groups.py --output-root ..\result\backtest_v4 || goto :err

echo 全部完成：result\backtest_v4\master_summary.csv
exit /b 0

:err
echo 下游流水线中断，请检查上方报错。
exit /b 1
