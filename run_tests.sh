#!/bin/bash
#
# MEDC: Medical Image Codec - Test Runner Script
# Runs all checkpoint tests to verify the codec implementation
#

set -e  # Exit on error

echo "============================================================"
echo "MEDC: Medical Image Codec - Running All Tests"
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

# Counter for test results
PASSED=0
FAILED=0

run_test() {
    local test_name=$1
    local test_file=$2

    echo "--------------------------------------------------------------"
    echo "Running: $test_name"
    echo "--------------------------------------------------------------"

    if python "$test_file"; then
        echo "[PASS] $test_name"
        ((PASSED++))
    else
        echo "[FAIL] $test_name"
        ((FAILED++))
    fi
    echo ""
}

# Run all checkpoint tests
run_test "Checkpoint 1: I/O and Level Shift" "tests/test_checkpoint_1.py"
run_test "Checkpoint 2: DCT Transform" "tests/test_checkpoint_2.py"
run_test "Checkpoint 3: Quantization" "tests/test_checkpoint_3.py"
run_test "Checkpoint 4: Entropy Coding" "tests/test_checkpoint_4.py"
run_test "Checkpoint 5: Full Codec" "tests/test_checkpoint_5.py"
run_test "Checkpoint 6: CLI Interface" "tests/test_checkpoint_6.py"
run_test "Checkpoint 7: Evaluation Metrics" "tests/test_checkpoint_7.py"

# Summary
echo "============================================================"
echo "TEST SUMMARY"
echo "============================================================"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "  Total:  $((PASSED + FAILED))"
echo "============================================================"

if [ $FAILED -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
