"""Checkpoint 2: DCT Transform Verification."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.transform.dct import create_dct_matrix, forward_dct_block, inverse_dct_block, DCT_MATRIX_8
from medcodec.transform.block_utils import split_into_blocks, merge_blocks


def test_dct_matrix_orthogonality():
    """Test that DCT matrix is orthogonal: T @ T' = I."""
    print("=" * 60)
    print("Test 1: DCT Matrix Orthogonality")
    print("=" * 60)

    try:
        T = create_dct_matrix(8)

        # Check orthogonality: T @ T' should equal identity matrix
        identity = T @ T.T
        expected = np.eye(8)

        max_error = np.abs(identity - expected).max()
        print(f"   Max error from identity: {max_error:.2e}")

        assert np.allclose(identity, expected, atol=1e-10), \
            f"DCT matrix not orthogonal! Max error: {max_error}"

        # Also check T' @ T = I
        identity2 = T.T @ T
        max_error2 = np.abs(identity2 - expected).max()
        print(f"   Max error (T' @ T): {max_error2:.2e}")

        assert np.allclose(identity2, expected, atol=1e-10), \
            f"DCT matrix not orthogonal (T' @ T)! Max error: {max_error2}"

        print("   ✓ T @ T' = I verified")
        print("   ✓ T' @ T = I verified")
        print("✅ DCT matrix orthogonality test passed")
        return True

    except Exception as e:
        print(f"❌ DCT matrix orthogonality test failed: {e}")
        return False


def test_single_block_reversibility():
    """Test DCT/IDCT reversibility on single 8x8 block."""
    print("\n" + "=" * 60)
    print("Test 2: Single Block DCT/IDCT Reversibility")
    print("=" * 60)

    try:
        test_cases = [
            ("Random float", np.random.randn(8, 8) * 100),
            ("Random int", np.random.randint(-1000, 1000, (8, 8)).astype(float)),
            ("All zeros", np.zeros((8, 8))),
            ("All ones", np.ones((8, 8)) * 500),
            ("Gradient", np.arange(64).reshape(8, 8).astype(float)),
            ("Checkerboard", np.tile([[0, 255], [255, 0]], (4, 4)).astype(float)),
        ]

        for name, block in test_cases:
            # Forward DCT
            dct_block = forward_dct_block(block)

            # Inverse DCT
            recovered = inverse_dct_block(dct_block)

            # Check reversibility
            max_error = np.abs(block - recovered).max()

            assert np.allclose(block, recovered, atol=1e-10), \
                f"DCT not reversible for {name}! Max error: {max_error}"

            print(f"   ✓ {name}: max error = {max_error:.2e}")

        print("✅ Single block reversibility test passed")
        return True

    except Exception as e:
        print(f"❌ Single block reversibility test failed: {e}")
        return False


def test_dct_energy_compaction():
    """Test that DCT concentrates energy in low frequencies."""
    print("\n" + "=" * 60)
    print("Test 3: DCT Energy Compaction")
    print("=" * 60)

    try:
        # Create a smooth block (typical image content)
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(x, y)
        smooth_block = (X + Y) * 100  # Smooth gradient

        # Forward DCT
        dct_block = forward_dct_block(smooth_block)

        # Calculate energy in different regions
        total_energy = np.sum(dct_block ** 2)

        # Energy in DC component (top-left)
        dc_energy = dct_block[0, 0] ** 2

        # Energy in low frequency region (top-left 4x4)
        low_freq_energy = np.sum(dct_block[:4, :4] ** 2)

        dc_ratio = dc_energy / total_energy * 100
        low_freq_ratio = low_freq_energy / total_energy * 100

        print(f"   DC component energy: {dc_ratio:.1f}%")
        print(f"   Low frequency (4x4) energy: {low_freq_ratio:.1f}%")

        # For smooth signals, most energy should be in low frequencies
        assert low_freq_ratio > 90, \
            f"Expected >90% energy in low frequencies, got {low_freq_ratio:.1f}%"

        print("   ✓ Energy compaction verified (>90% in low frequencies)")
        print("✅ DCT energy compaction test passed")
        return True

    except Exception as e:
        print(f"❌ DCT energy compaction test failed: {e}")
        return False


def test_block_split_merge():
    """Test block splitting and merging."""
    print("\n" + "=" * 60)
    print("Test 4: Block Split/Merge")
    print("=" * 60)

    try:
        test_cases = [
            ("Exact multiple (512x512)", (512, 512)),
            ("Non-multiple (510x510)", (510, 510)),
            ("Non-multiple (513x517)", (513, 517)),
            ("Small (15x17)", (15, 17)),
            ("Rectangular (256x512)", (256, 512)),
        ]

        for name, shape in test_cases:
            # Create random image
            img = np.random.randint(0, 65535, shape, dtype=np.uint16).astype(float)

            # Split into blocks
            blocks, pad_info = split_into_blocks(img, block_size=8)

            # Calculate expected number of blocks
            h, w = shape
            expected_blocks_h = (h + 7) // 8
            expected_blocks_w = (w + 7) // 8
            expected_total = expected_blocks_h * expected_blocks_w

            assert len(blocks) == expected_total, \
                f"Expected {expected_total} blocks, got {len(blocks)}"

            # Verify all blocks are 8x8
            for i, block in enumerate(blocks):
                assert block.shape == (8, 8), f"Block {i} has wrong shape: {block.shape}"

            # Merge back
            recovered = merge_blocks(blocks, shape, pad_info)

            # Verify exact recovery
            assert recovered.shape == shape, \
                f"Shape mismatch: expected {shape}, got {recovered.shape}"

            max_error = np.abs(img - recovered).max()
            assert np.allclose(img, recovered, atol=1e-10), \
                f"Split/merge not reversible! Max error: {max_error}"

            print(f"   ✓ {name}: {len(blocks)} blocks, recovered perfectly")

        print("✅ Block split/merge test passed")
        return True

    except Exception as e:
        print(f"❌ Block split/merge test failed: {e}")
        return False


def test_full_image_dct():
    """Test DCT/IDCT on full image with padding."""
    print("\n" + "=" * 60)
    print("Test 5: Full Image DCT/IDCT")
    print("=" * 60)

    try:
        # Test with various image sizes
        test_cases = [
            ("512x512", (512, 512)),
            ("510x510 (needs padding)", (510, 510)),
            ("367x835 (like DICOM)", (367, 835)),
        ]

        for name, shape in test_cases:
            # Create random 16-bit image
            img = np.random.randint(0, 65535, shape, dtype=np.uint16).astype(np.float64)

            # Split into blocks
            blocks, pad_info = split_into_blocks(img, block_size=8)

            # Apply DCT to all blocks
            dct_blocks = [forward_dct_block(block) for block in blocks]

            # Apply IDCT to all blocks
            idct_blocks = [inverse_dct_block(block) for block in dct_blocks]

            # Merge blocks
            recovered = merge_blocks(idct_blocks, shape, pad_info)

            # Verify
            max_error = np.abs(img - recovered).max()

            assert np.allclose(img, recovered, atol=1e-8), \
                f"Full image DCT not reversible! Max error: {max_error}"

            print(f"   ✓ {name}: max error = {max_error:.2e}")

        print("✅ Full image DCT/IDCT test passed")
        return True

    except Exception as e:
        print(f"❌ Full image DCT/IDCT test failed: {e}")
        return False


def test_dct_with_real_dicom():
    """Test DCT with real DICOM image."""
    print("\n" + "=" * 60)
    print("Test 6: DCT with Real DICOM Image")
    print("=" * 60)

    dicom_path = "data/2_skull_ct/DICOM/I0"

    if not os.path.exists(dicom_path):
        print(f"   ⚠️ DICOM file not found, skipping...")
        return True

    try:
        import pydicom
        from medcodec.transform import level_shift

        # Read DICOM
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

        img = img.astype(np.int16)
        print(f"   Image shape: {img.shape}")
        print(f"   Original range: [{img.min()}, {img.max()}]")

        # Level shift to unsigned
        img_shifted = level_shift(img, forward=True).astype(np.float64)

        # Split into blocks
        blocks, pad_info = split_into_blocks(img_shifted, block_size=8)
        print(f"   Number of blocks: {len(blocks)}")

        # DCT all blocks
        dct_blocks = [forward_dct_block(block) for block in blocks]

        # IDCT all blocks
        idct_blocks = [inverse_dct_block(block) for block in dct_blocks]

        # Merge
        recovered = merge_blocks(idct_blocks, img_shifted.shape, pad_info)

        # Level shift back
        recovered_int16 = level_shift(recovered, forward=False)

        # Verify
        max_error = np.abs(img - recovered_int16).max()
        print(f"   Max reconstruction error: {max_error:.2e}")

        assert np.allclose(img, recovered_int16, atol=1), \
            f"DICOM DCT not reversible! Max error: {max_error}"

        print("   ✓ DICOM image DCT/IDCT successful")
        print("✅ Real DICOM DCT test passed")
        return True

    except Exception as e:
        print(f"❌ Real DICOM DCT test failed: {e}")
        return False


def main():
    """Run all Checkpoint 2 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 2: DCT TRANSFORM VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Matrix orthogonality
    results.append(("DCT Matrix Orthogonality", test_dct_matrix_orthogonality()))

    # Test 2: Single block reversibility
    results.append(("Single Block Reversibility", test_single_block_reversibility()))

    # Test 3: Energy compaction
    results.append(("Energy Compaction", test_dct_energy_compaction()))

    # Test 4: Block split/merge
    results.append(("Block Split/Merge", test_block_split_merge()))

    # Test 5: Full image DCT
    results.append(("Full Image DCT/IDCT", test_full_image_dct()))

    # Test 6: Real DICOM
    results.append(("Real DICOM DCT", test_dct_with_real_dicom()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 2 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 2 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 2 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
