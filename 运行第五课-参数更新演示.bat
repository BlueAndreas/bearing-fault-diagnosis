@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\lesson05_beginner_walkthrough.py --training-demo
if errorlevel 1 echo 请保留上面的错误提示。
pause
