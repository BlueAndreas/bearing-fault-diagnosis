@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\lesson07_evaluate.py
if errorlevel 1 (
    echo.
    echo 运行失败，请保留上面的报错信息。
) else (
    echo.
    echo 运行成功。请打开 outputs\lesson07\第七课-测试评估与模型对比结果.md。
)
pause
