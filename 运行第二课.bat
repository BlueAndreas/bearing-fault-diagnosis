@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\lesson02_compare_states.py
if errorlevel 1 (
    echo.
    echo 运行失败，请保留上面的报错信息。
) else (
    echo.
    echo 运行成功。请打开 outputs\lesson02 查看四种状态的图片、统计表和报告。
)
pause
