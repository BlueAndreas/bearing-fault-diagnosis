@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\lesson01_read_signal.py
if errorlevel 1 (
    echo.
    echo 运行失败，请保留上面的报错信息。
) else (
    echo.
    echo 运行成功。请打开 outputs\lesson01 查看三张图片和学习报告。
)
pause
