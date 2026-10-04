@echo off
cd /d "%~dp0"
python download_real_model.py
if errorlevel 1 pause & exit /b 1
python app.py
pause
