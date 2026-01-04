#!/bin/bash
#
# MEDC: Medical Image Codec - Experiment Runner Script
# Runs compression experiments and generates results
#

set -e  # Exit on error

echo "============================================================"
echo "MEDC: Medical Image Codec - Running Experiments"
echo "============================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if running in correct environment
if ! python -c "import numpy" 2>/dev/null; then
    echo "[ERROR] Python environment not set up correctly"
    echo "Please run: conda activate medcodec"
    exit 1
fi

# Check if data exists
if [ ! -d "data/2_skull_ct/DICOM" ]; then
    echo "[WARNING] DICOM data not found at data/2_skull_ct/DICOM"
    echo "Please download sample DICOM files from:"
    echo "  https://medimodel.com/sample-dicom-files/"
    echo ""
    echo "Expected structure:"
    echo "  data/2_skull_ct/DICOM/I0"
    echo "  data/2_skull_ct/DICOM/I1"
    echo "  ..."
    echo ""
fi

# Create results directory
mkdir -p results/images

echo "[1/3] Running main experiment suite"
echo "--------------------------------------------------------------"
python scripts/run_experiments.py

echo ""
echo "[2/3] Generating report figures"
echo "--------------------------------------------------------------"
if [ -d "report" ]; then
    cd report
    python generate_figures.py 2>/dev/null || echo "[SKIP] Figure generation (matplotlib backend issue)"
    cd ..
fi

echo ""
echo "[3/3] Displaying results"
echo "--------------------------------------------------------------"

if [ -f "results/metrics.json" ]; then
    echo "Results saved to: results/metrics.json"
    echo ""
    python -c "
import json
with open('results/metrics.json', 'r') as f:
    data = json.load(f)

print('Rate-Distortion Results:')
print('-' * 70)
print(f'{\"Quality\":>8} {\"RMSE\":>10} {\"PSNR (dB)\":>12} {\"BPP\":>10} {\"CR\":>10}')
print('-' * 70)

for r in data.get('results', []):
    q = r.get('quality', 'N/A')
    rmse = r.get('rmse', 0)
    psnr = r.get('psnr', 0)
    bpp = r.get('bpp', 0)
    cr = r.get('compression_ratio', 0)
    print(f'{q:>8} {rmse:>10.2f} {psnr:>12.2f} {bpp:>10.4f} {cr:>10.2f}x')

print('-' * 70)
"
else
    echo "[WARNING] No results found. Run experiments first."
fi

echo ""
echo "============================================================"
echo "Experiment Complete!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - results/metrics.json"
echo "  - results/images/original.png"
echo "  - results/images/reconstructed_q*.png"
echo "  - results/images/error_map_q*.png"
echo ""
