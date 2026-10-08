@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\lesson03_build_dataset.py
if errorlevel 1 (
    echo.
    echo 运行失败，请保留上面的报错信息。
) else (
    echo.
    echo 运行成功。数据在 data\processed\lesson03，图片与报告在 outputs\lesson03。
)
pause
