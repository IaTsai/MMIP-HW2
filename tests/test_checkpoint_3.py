"""Checkpoint 3: Quantization Verification."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.transform import level_shift, forward_dct_block, inverse_dct_block
from medcodec.transform import split_into_blocks, merge_blocks
from medcodec.quantization import get_quantization_step, quantize, dequantize
from medcodec.metrics import calculate_psnr, calculate_rmse


def test_quantization_step_mapping():
    """Test quality to step mapping."""
    print("=" * 60)
    print("Test 1: Quality to Step Mapping")
    print("=" * 60)

    try:
        # Test boundary values
        step_q1 = get_quantization_step(1, bit_depth=16)
        step_q50 = get_quantization_step(50, bit_depth=16)
        step_q100 = get_quantization_step(100, bit_depth=16)

        print(f"   Q=1   → step = {step_q1}")
        print(f"   Q=50  → step = {step_q50}")
        print(f"   Q=100 → step = {step_q100}")

        # Verify monotonicity: higher quality = smaller step
        assert step_q1 > step_q50 > step_q100, \
            "Step should decrease as quality increases"

        # Verify Q=100 gives step=1 (lossless)
        assert step_q100 == 1, f"Q=100 should give step=1, got {step_q100}"

        # Test a range of quality values
        print("\n   Quality → Step mapping:")
        for q in [1, 10, 25, 50, 75, 85, 95, 100]:
            step = get_quantization_step(q, bit_depth=16)
            print(f"     Q={q:3d} → step = {step:6d}")

        print("\n   ✓ Quality to step mapping is monotonic")
        print("✅ Quantization step mapping test passed")
        return True

    except Exception as e:
        print(f"❌ Quantization step mapping test failed: {e}")
        return False


def test_quantize_dequantize():
    """Test quantize/dequantize operations."""
    print("\n" + "=" * 60)
    print("Test 2: Quantize/Dequantize Operations")
    print("=" * 60)

    try:
        # Create test DCT coefficients
        coeffs = np.array([
            [1000, 50, -20, 10, 5, -2, 1, 0],
            [80, 40, -15, 8, 3, -1, 0, 0],
            [-30, 20, 10, 5, 2, 0, 0, 0],
            [15, 8, 5, 2, 1, 0, 0, 0],
            [6, 3, 2, 1, 0, 0, 0, 0],
            [-2, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float64)

        # Test with different steps
        for step in [1, 10, 100, 1000]:
            quant = quantize(coeffs, step)
            dequant = dequantize(quant, step)

            # Check dtype (int32 for 16-bit medical images to avoid overflow)
            assert quant.dtype == np.int32, f"Quantized should be int32, got {quant.dtype}"
            assert dequant.dtype == np.float64, f"Dequantized should be float64"

            # Calculate error
            error = np.abs(coeffs - dequant).max()
            expected_max_error = step / 2  # Maximum rounding error

            print(f"   Step={step:4d}: max error = {error:.1f} (expected < {expected_max_error:.1f})")

            # Error should be bounded by step/2
            assert error <= expected_max_error + 1e-10, \
                f"Error {error} exceeds expected {expected_max_error}"

        print("\n   ✓ Quantize/dequantize error is bounded")
        print("✅ Quantize/dequantize test passed")
        return True

    except Exception as e:
        print(f"❌ Quantize/dequantize test failed: {e}")
        return False


def test_full_pipeline_psnr():
    """Test full pipeline: Level Shift → DCT → Quantize → Dequantize → IDCT → Unshift."""
    print("\n" + "=" * 60)
    print("Test 3: Full Pipeline PSNR")
    print("=" * 60)

    try:
        # Create synthetic 16-bit image (CT-like range)
        np.random.seed(42)
        h, w = 256, 256
        img = np.random.randint(-1024, 3071, (h, w), dtype=np.int16)

        print(f"   Image: {h}x{w}, range [{img.min()}, {img.max()}]")

        psnr_results = {}
        qualities = [25, 50, 85]

        for quality in qualities:
            step = get_quantization_step(quality, bit_depth=16)

            # Level shift
            img_shifted = level_shift(img, forward=True).astype(np.float64)

            # Split into blocks
            blocks, pad_info = split_into_blocks(img_shifted, 8)

            # DCT → Quantize → Dequantize → IDCT
            processed_blocks = []
            for block in blocks:
                dct = forward_dct_block(block)
                quant = quantize(dct, step)
                dequant = dequantize(quant, step)
                idct = inverse_dct_block(dequant)
                processed_blocks.append(idct)

            # Merge
            recovered = merge_blocks(processed_blocks, img_shifted.shape, pad_info)

            # Level unshift
            recovered_int16 = level_shift(recovered, forward=False)

            # Calculate metrics
            psnr = calculate_psnr(img, recovered_int16, bit_depth=16)
            rmse = calculate_rmse(img, recovered_int16)

            psnr_results[quality] = psnr
            print(f"   Q={quality:2d} (step={step:5d}): PSNR = {psnr:.2f} dB, RMSE = {rmse:.2f}")

        # Verify PSNR increases with quality
        assert psnr_results[85] > psnr_results[50] > psnr_results[25], \
            "PSNR should increase with quality"

        print("\n   ✓ PSNR increases with quality: Q85 > Q50 > Q25")
        print("✅ Full pipeline PSNR test passed")
        return True

    except Exception as e:
        print(f"❌ Full pipeline PSNR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_dicom():
    """Test quantization with real DICOM image."""
    print("\n" + "=" * 60)
    print("Test 4: Real DICOM Quantization")
    print("=" * 60)

    dicom_path = "data/2_skull_ct/DICOM/I0"

    if not os.path.exists(dicom_path):
        print("   ⚠️ DICOM file not found, skipping...")
        return True

    try:
        import pydicom

        # Read DICOM
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

        img = img.astype(np.int16)
        print(f"   Image: {img.shape}, range [{img.min()}, {img.max()}]")

        psnr_results = {}
        qualities = [25, 50, 85]

        for quality in qualities:
            step = get_quantization_step(quality, bit_depth=16)

            # Full pipeline
            img_shifted = level_shift(img, forward=True).astype(np.float64)
            blocks, pad_info = split_into_blocks(img_shifted, 8)

            processed_blocks = []
            for block in blocks:
                dct = forward_dct_block(block)
                quant = quantize(dct, step)
                dequant = dequantize(quant, step)
                idct = inverse_dct_block(dequant)
                processed_blocks.append(idct)

            recovered = merge_blocks(processed_blocks, img_shifted.shape, pad_info)
            recovered_int16 = level_shift(recovered, forward=False)

            psnr = calculate_psnr(img, recovered_int16, bit_depth=16)
            rmse = calculate_rmse(img, recovered_int16)

            psnr_results[quality] = psnr
            print(f"   Q={quality:2d} (step={step:5d}): PSNR = {psnr:.2f} dB, RMSE = {rmse:.2f}")

        # Verify PSNR increases with quality
        assert psnr_results[85] > psnr_results[50] > psnr_results[25], \
            "PSNR should increase with quality"

        print("\n   ✓ PSNR increases with quality on real DICOM")
        print("✅ Real DICOM quantization test passed")
        return True

    except Exception as e:
        print(f"❌ Real DICOM quantization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lossless_at_max_quality():
    """Test that Q=100 gives lossless compression."""
    print("\n" + "=" * 60)
    print("Test 5: Lossless at Maximum Quality (Q=100)")
    print("=" * 60)

    try:
        # Create test image
        img = np.random.randint(-1024, 3071, (64, 64), dtype=np.int16)

        step = get_quantization_step(100, bit_depth=16)
        assert step == 1, f"Q=100 should give step=1, got {step}"

        # Full pipeline with Q=100
        img_shifted = level_shift(img, forward=True).astype(np.float64)
        blocks, pad_info = split_into_blocks(img_shifted, 8)

        processed_blocks = []
        for block in blocks:
            dct = forward_dct_block(block)
            quant = quantize(dct, step)  # step=1
            dequant = dequantize(quant, step)
            idct = inverse_dct_block(dequant)
            processed_blocks.append(idct)

        recovered = merge_blocks(processed_blocks, img_shifted.shape, pad_info)
        recovered_int16 = level_shift(recovered, forward=False)

        # Check near-lossless (may have small floating point errors)
        max_error = np.abs(img - recovered_int16).max()
        print(f"   Max error at Q=100: {max_error}")

        # Allow for rounding errors
        assert max_error <= 1, f"Q=100 should be near-lossless, got max error {max_error}"

        psnr = calculate_psnr(img, recovered_int16, bit_depth=16)
        print(f"   PSNR at Q=100: {psnr:.2f} dB")

        print("   ✓ Q=100 achieves near-lossless compression")
        print("✅ Lossless at max quality test passed")
        return True

    except Exception as e:
        print(f"❌ Lossless at max quality test failed: {e}")
        return False


def main():
    """Run all Checkpoint 3 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 3: QUANTIZATION VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Step mapping
    results.append(("Step Mapping", test_quantization_step_mapping()))

    # Test 2: Quantize/dequantize
    results.append(("Quantize/Dequantize", test_quantize_dequantize()))

    # Test 3: Full pipeline PSNR
    results.append(("Full Pipeline PSNR", test_full_pipeline_psnr()))

    # Test 4: Real DICOM
    results.append(("Real DICOM Quantization", test_with_real_dicom()))

    # Test 5: Lossless at max quality
    results.append(("Lossless at Q=100", test_lossless_at_max_quality()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 3 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 3 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 3 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
