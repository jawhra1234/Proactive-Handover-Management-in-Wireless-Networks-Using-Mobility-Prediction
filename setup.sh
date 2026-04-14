#!/bin/bash

# ============================================================================
# Setup script for Proactive Handover Management Project
# Run this script to automatically set up the environment on Linux/Mac
# ============================================================================

echo ""
echo "============================================================================"
echo "  Proactive Handover Management - Environment Setup"
echo "============================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.7+ from https://www.python.org"
    exit 1
fi

echo "[OK] Python found"
python3 --version

# Create virtual environment
echo ""
echo "[1/4] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment already exists"
else
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"

# Upgrade pip
echo ""
echo "[3/4] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "[OK] pip upgraded"

# Install dependencies
echo ""
echo "[4/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi
echo "[OK] All dependencies installed"

echo ""
echo "============================================================================"
echo "  Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Run the quick start guide: python quick_start.py"
echo "  2. Run the main simulation: python main.py"
echo ""
echo "The virtual environment will remain active in this terminal."
echo "To deactivate it later, type: deactivate"
echo ""
