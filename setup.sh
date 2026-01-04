#!/bin/bash
#
# MEDC: Medical Image Codec - Environment Setup Script
# This script sets up a clean Python environment with all dependencies
#

set -e  # Exit on error

echo "============================================================"
echo "MEDC: Medical Image Codec - Environment Setup"
echo "============================================================"

# Configuration
ENV_NAME="medcodec"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "[1/4] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo "--------------------------------------------------------------"

# Remove existing environment if exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment..."
    conda env remove -n $ENV_NAME -y
fi

# Create new environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "[2/4] Activating environment"
echo "--------------------------------------------------------------"

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "[3/4] Installing dependencies"
echo "--------------------------------------------------------------"

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[4/4] Verifying installation"
echo "--------------------------------------------------------------"

python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python -c "import pydicom; print(f'  pydicom: {pydicom.__version__}')"
python -c "import PIL; print(f'  pillow: {PIL.__version__}')"
python -c "import matplotlib; print(f'  matplotlib: {matplotlib.__version__}')"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run tests:"
echo "  ./run_tests.sh"
echo ""
echo "To run experiments:"
echo "  ./run_experiments.sh"
echo ""
