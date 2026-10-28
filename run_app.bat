@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Streamlit could not start. Install the project dependencies with:
    echo %PYTHON% -m pip install -r requirements.txt
    pause
)
