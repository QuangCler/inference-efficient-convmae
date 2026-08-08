@echo off
REM Launch the demo (Windows). Assumes requirements installed + checkpoints fetched.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python app.py
