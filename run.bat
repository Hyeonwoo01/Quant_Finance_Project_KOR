@echo off
rem Keep this file ASCII-only: cmd.exe misparses non-ASCII text inside ( ) blocks.
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
    echo [setup] Creating virtual environment and installing packages...
    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python -m streamlit run app.py
