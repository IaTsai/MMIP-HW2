# MEDC: Medical Image Codec - Makefile
#
# Usage:
#   make setup      - Set up conda environment
#   make test       - Run all checkpoint tests
#   make experiment - Run compression experiments
#   make demo       - Run a quick demo encode/decode
#   make clean      - Clean temporary files
#   make all        - Run everything (setup + test + experiment)

.PHONY: setup test experiment demo clean all help

# Default target
help:
	@echo "MEDC: Medical Image Codec - Available Commands"
	@echo "============================================================"
	@echo "  make setup      - Set up conda environment"
	@echo "  make test       - Run all checkpoint tests"
	@echo "  make experiment - Run compression experiments"
	@echo "  make demo       - Run a quick demo encode/decode"
	@echo "  make clean      - Clean temporary files"
	@echo "  make all        - Run everything (setup + test + experiment)"
	@echo "============================================================"

# Setup environment
setup:
	@chmod +x setup.sh
	@./setup.sh

# Run all tests
test:
	@chmod +x run_tests.sh
	@./run_tests.sh

# Run experiments
experiment:
	@chmod +x run_experiments.sh
	@./run_experiments.sh

# Quick demo
demo:
	@echo "============================================================"
	@echo "MEDC Demo: Encode/Decode Roundtrip"
	@echo "============================================================"
	@echo ""
	@if [ -f "data/2_skull_ct/DICOM/I0" ]; then \
		echo "[1/3] Encoding DICOM file at Q=75..."; \
		python encode.py --input data/2_skull_ct/DICOM/I0 --output demo_output.mcd --quality 75; \
		echo ""; \
		echo "[2/3] Decoding compressed file..."; \
		python decode.py --input demo_output.mcd --output demo_recovered.npy; \
		echo ""; \
		echo "[3/3] File sizes:"; \
		ls -lh demo_output.mcd demo_recovered.npy; \
		rm -f demo_output.mcd demo_recovered.npy; \
		echo ""; \
		echo "Demo completed successfully!"; \
	else \
		echo "[ERROR] DICOM data not found."; \
		echo "Please download sample files to: data/2_skull_ct/DICOM/"; \
	fi

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@rm -f *.mcd *.npy
	@rm -rf __pycache__ medcodec/__pycache__ medcodec/*/__pycache__ tests/__pycache__
	@rm -rf .pytest_cache
	@rm -f results/images/*.png
	@rm -f results/metrics.json
	@echo "Done."

# Run everything
all: test experiment
	@echo ""
	@echo "============================================================"
	@echo "All tasks completed!"
	@echo "============================================================"
