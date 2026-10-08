@echo off
chcp 65001 >nul
cd /d "%~dp0"
"C:\Users\qsy12\AppData\Local\Python\pythoncore-3.14-64\python.exe" -X utf8 app\server.py --open-browser
if errorlevel 1 echo 启动失败，请查看上面的错误提示。
pause
