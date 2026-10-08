@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\run_pipeline.py %*
if errorlevel 1 echo 请保留上方报错，检查 Python 依赖、数据来源及网络。
pause
