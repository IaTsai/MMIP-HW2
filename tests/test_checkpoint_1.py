"""Checkpoint 1: I/O and Level Shift Verification."""

import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.io import read_medical_image, write_medical_image
from medcodec.transform import level_shift


def test_dicom_read():
    """Test DICOM file reading."""
    print("=" * 60)
    print("Test 1: DICOM Reading")
    print("=" * 60)

    # Read DICOM file (without .dcm extension)
    dicom_path = "data/2_skull_ct/DICOM/I0"

    if not os.path.exists(dicom_path):
        print(f"⚠️  DICOM file not found: {dicom_path}")
        print("   Please download test data from Medimodel")
        return False

    try:
        # Read as DICOM by forcing the format
        import pydicom
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        # Apply rescale if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept

        img = img.astype(np.int16)

        print(f"   Shape: {img.shape}")
        print(f"   Dtype: {img.dtype}")
        print(f"   Min: {img.min()}, Max: {img.max()}")
        print(f"   Bits Allocated: {ds.BitsAllocated}")

        # Verify dtype
        assert img.dtype == np.int16, f"Expected int16, got {img.dtype}"
        print("✅ DICOM read successful")
        return img

    except Exception as e:
        print(f"❌ DICOM read failed: {e}")
        return None


def test_level_shift(img):
    """Test level shift reversibility."""
    print("\n" + "=" * 60)
    print("Test 2: Level Shift Reversibility")
    print("=" * 60)

    try:
        # Forward shift (signed → unsigned)
        shifted = level_shift(img, forward=True)

        print(f"   Original range: [{img.min()}, {img.max()}]")
        print(f"   Shifted range:  [{shifted.min()}, {shifted.max()}]")

        # Verify unsigned (all values >= 0)
        assert np.all(shifted >= 0), "Shifted values should be non-negative"
        print("   ✓ All shifted values are non-negative")

        # Inverse shift (unsigned → signed)
        restored = level_shift(shifted, forward=False)

        # Verify reversibility
        assert np.array_equal(img, restored), "Level shift not reversible!"
        print("   ✓ Level shift is perfectly reversible")

        print("✅ Level shift test passed")
        return True

    except Exception as e:
        print(f"❌ Level shift test failed: {e}")
        return False


def test_read_write_roundtrip(img):
    """Test read/write roundtrip for npy format."""
    print("\n" + "=" * 60)
    print("Test 3: Read/Write Roundtrip")
    print("=" * 60)

    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_path = f.name

        # Write
        write_medical_image(img, temp_path)
        print(f"   Written to: {temp_path}")

        # Read back
        img_read = read_medical_image(temp_path)

        # Verify
        assert np.array_equal(img, img_read), "Read/write roundtrip failed!"
        print("   ✓ Read/write roundtrip is lossless")

        # Cleanup
        os.unlink(temp_path)

        print("✅ Read/write roundtrip test passed")
        return True

    except Exception as e:
        print(f"❌ Read/write roundtrip failed: {e}")
        return False


def test_synthetic_data():
    """Test with synthetic data covering edge cases."""
    print("\n" + "=" * 60)
    print("Test 4: Synthetic Data (Edge Cases)")
    print("=" * 60)

    try:
        # Create synthetic data with full int16 range
        test_cases = [
            ("Min value", np.array([[-32768]], dtype=np.int16)),
            ("Max value", np.array([[32767]], dtype=np.int16)),
            ("Zero", np.array([[0]], dtype=np.int16)),
            ("Random", np.random.randint(-10000, 10000, (64, 64), dtype=np.int16)),
            ("CT-like", np.random.randint(-1024, 3071, (512, 512), dtype=np.int16)),
        ]

        for name, data in test_cases:
            # Test level shift
            shifted = level_shift(data, forward=True)
            restored = level_shift(shifted, forward=False)

            assert np.array_equal(data, restored), f"Failed for {name}"
            print(f"   ✓ {name}: OK")

        print("✅ Synthetic data test passed")
        return True

    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False


def main():
    """Run all Checkpoint 1 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 1: I/O AND LEVEL SHIFT VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: DICOM reading
    img = test_dicom_read()
    results.append(("DICOM Read", img is not None))

    if img is not None:
        # Test 2: Level shift
        results.append(("Level Shift", test_level_shift(img)))

        # Test 3: Read/write roundtrip
        results.append(("Read/Write Roundtrip", test_read_write_roundtrip(img)))

    # Test 4: Synthetic data
    results.append(("Synthetic Data", test_synthetic_data()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 1 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 1 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 1 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
