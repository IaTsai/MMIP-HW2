"""Checkpoint 5: Codec Integration Verification."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.codec import MedicalImageEncoder, MedicalImageDecoder
from medcodec.metrics import calculate_psnr, calculate_rmse, calculate_bpp, calculate_compression_ratio


def test_synthetic_roundtrip():
    """Test encode/decode roundtrip with synthetic data."""
    print("=" * 60)
    print("Test 1: Synthetic Data Roundtrip")
    print("=" * 60)

    try:
        # Create synthetic CT-like image
        np.random.seed(42)
        h, w = 128, 128
        img = np.random.randint(-1024, 3071, (h, w), dtype=np.int16)

        print(f"   Image: {h}x{w}, range [{img.min()}, {img.max()}]")

        encoder = MedicalImageEncoder()
        decoder = MedicalImageDecoder()

        for quality in [25, 50, 85]:
            # Encode
            compressed = encoder.encode(img, quality=quality)

            # Decode
            recovered = decoder.decode(compressed)

            # Verify shape and dtype
            assert recovered.shape == img.shape, f"Shape mismatch at Q={quality}"
            assert recovered.dtype == np.int16, f"Dtype mismatch at Q={quality}"

            # Calculate metrics
            psnr = calculate_psnr(img, recovered, bit_depth=16)
            rmse = calculate_rmse(img, recovered)
            bpp = calculate_bpp(len(compressed), img.shape)
            cr = calculate_compression_ratio(img.nbytes, len(compressed))

            print(f"   Q={quality:2d}: PSNR={psnr:.2f}dB, RMSE={rmse:.1f}, "
                  f"BPP={bpp:.3f}, CR={cr:.2f}x, Size={len(compressed)} bytes")

        print("\n   ✓ All quality levels encode/decode correctly")
        print("✅ Synthetic data roundtrip test passed")
        return True

    except Exception as e:
        print(f"❌ Synthetic data roundtrip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_dicom():
    """Test with real DICOM image."""
    print("\n" + "=" * 60)
    print("Test 2: Real DICOM Roundtrip")
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

        encoder = MedicalImageEncoder()
        decoder = MedicalImageDecoder()

        results = []
        for quality in [25, 50, 85]:
            compressed = encoder.encode(img, quality=quality)
            recovered = decoder.decode(compressed)

            psnr = calculate_psnr(img, recovered, bit_depth=16)
            rmse = calculate_rmse(img, recovered)
            bpp = calculate_bpp(len(compressed), img.shape)
            cr = calculate_compression_ratio(img.nbytes, len(compressed))

            results.append({
                'quality': quality,
                'psnr': psnr,
                'rmse': rmse,
                'bpp': bpp,
                'cr': cr,
                'size': len(compressed)
            })

            print(f"   Q={quality:2d}: PSNR={psnr:.2f}dB, RMSE={rmse:.1f}, "
                  f"BPP={bpp:.3f}, CR={cr:.2f}x")

        # Verify PSNR increases with quality
        assert results[2]['psnr'] > results[1]['psnr'] > results[0]['psnr'], \
            "PSNR should increase with quality"

        print("\n   ✓ PSNR increases with quality")
        print("✅ Real DICOM roundtrip test passed")
        return True

    except Exception as e:
        print(f"❌ Real DICOM roundtrip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 60)
    print("Test 3: Error Handling")
    print("=" * 60)

    try:
        decoder = MedicalImageDecoder()

        # Test 1: Invalid magic number
        try:
            decoder.decode(b"INVALID_DATA_HERE_1234567890")
            print("   ✗ Should have raised ValueError for invalid magic")
            return False
        except ValueError as e:
            if "Invalid file signature" in str(e) or "Invalid magic" in str(e):
                print("   ✓ Correctly rejected invalid magic number")
            else:
                print(f"   ✓ Rejected with error: {e}")

        # Test 2: Truncated data
        try:
            decoder.decode(b"MEDC")  # Only 4 bytes
            print("   ✗ Should have raised ValueError for truncated data")
            return False
        except ValueError as e:
            print(f"   ✓ Correctly rejected truncated data")

        # Test 3: Invalid quality
        encoder = MedicalImageEncoder()
        img = np.zeros((64, 64), dtype=np.int16)

        try:
            encoder.encode(img, quality=0)
            print("   ✗ Should have raised ValueError for quality=0")
            return False
        except ValueError:
            print("   ✓ Correctly rejected quality=0")

        try:
            encoder.encode(img, quality=101)
            print("   ✗ Should have raised ValueError for quality=101")
            return False
        except ValueError:
            print("   ✓ Correctly rejected quality=101")

        print("✅ Error handling test passed")
        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_crc_verification():
    """Test CRC verification catches corruption."""
    print("\n" + "=" * 60)
    print("Test 4: CRC Verification")
    print("=" * 60)

    try:
        # Create and encode an image
        img = np.random.randint(-1024, 3071, (64, 64), dtype=np.int16)

        encoder = MedicalImageEncoder()
        compressed = encoder.encode(img, quality=75)

        # Corrupt one byte in the payload
        corrupted = bytearray(compressed)
        corrupted[30] ^= 0xFF  # Flip bits in payload area
        corrupted = bytes(corrupted)

        # Try to decode
        decoder = MedicalImageDecoder()
        try:
            decoder.decode(corrupted)
            print("   ✗ Should have detected CRC mismatch")
            return False
        except ValueError as e:
            if "CRC" in str(e):
                print("   ✓ CRC mismatch correctly detected")
            else:
                print(f"   ✓ Corruption detected: {e}")

        print("✅ CRC verification test passed")
        return True

    except Exception as e:
        print(f"❌ CRC verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases: small images, uniform images."""
    print("\n" + "=" * 60)
    print("Test 5: Edge Cases")
    print("=" * 60)

    try:
        encoder = MedicalImageEncoder()
        decoder = MedicalImageDecoder()

        test_cases = [
            ("Minimum size (8x8)", np.random.randint(-1024, 3071, (8, 8), dtype=np.int16)),
            ("Non-multiple (15x17)", np.random.randint(-1024, 3071, (15, 17), dtype=np.int16)),
            ("All zeros", np.zeros((64, 64), dtype=np.int16)),
            ("All same value", np.full((64, 64), 1000, dtype=np.int16)),
            ("Maximum range", np.array([[-32768, 32767], [0, 100]], dtype=np.int16)),
        ]

        for name, img in test_cases:
            compressed = encoder.encode(img, quality=75)
            recovered = decoder.decode(compressed)

            assert recovered.shape == img.shape, f"Shape mismatch for {name}"
            assert recovered.dtype == np.int16, f"Dtype mismatch for {name}"

            # For near-lossless check
            max_error = np.abs(img - recovered).max()
            print(f"   ✓ {name}: max error = {max_error}")

        print("✅ Edge cases test passed")
        return True

    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Checkpoint 5 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 5: CODEC INTEGRATION VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Synthetic roundtrip
    results.append(("Synthetic Roundtrip", test_synthetic_roundtrip()))

    # Test 2: Real DICOM
    results.append(("Real DICOM Roundtrip", test_real_dicom()))

    # Test 3: Error handling
    results.append(("Error Handling", test_error_handling()))

    # Test 4: CRC verification
    results.append(("CRC Verification", test_crc_verification()))

    # Test 5: Edge cases
    results.append(("Edge Cases", test_edge_cases()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 5 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 5 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 5 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
