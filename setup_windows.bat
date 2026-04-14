@echo off
REM ============================================================================
REM Setup script for Proactive Handover Management Project
REM Run this batch file to automatically set up the environment on Windows
REM ============================================================================

echo.
echo ============================================================================
echo  Proactive Handover Management - Environment Setup
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version

REM Create virtual environment
echo.
echo [1/4] Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded

REM Install dependencies
echo.
echo [4/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] All dependencies installed

echo.
echo ============================================================================
echo  Setup Complete!
echo ============================================================================
echo.
echo Next steps:
echo   1. Run the quick start guide: python quick_start.py
echo   2. Run the main simulation: python main.py
echo.
echo The virtual environment will remain active in this terminal.
echo To deactivate it later, type: deactivate
echo.
pause
