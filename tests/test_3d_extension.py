"""Test 3D Volume Codec Extension."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.codec import (
    MedicalImageEncoder, MedicalImageDecoder,
    VolumeEncoder, VolumeDecoder,
    calculate_3d_compression_stats
)
from medcodec.metrics import calculate_psnr, calculate_rmse


def create_synthetic_volume(num_slices=10, height=128, width=128, seed=42):
    """
    Create a synthetic CT-like volume with inter-slice correlation.

    Adjacent slices are similar (simulating real CT scans).
    """
    np.random.seed(seed)

    # Create base slice
    base = np.random.randint(-500, 500, (height, width), dtype=np.int16)

    # Add some structure (circular region)
    y, x = np.ogrid[:height, :width]
    center = (height // 2, width // 2)
    radius = min(height, width) // 4
    mask = (y - center[0])**2 + (x - center[1])**2 <= radius**2
    base[mask] += 500

    # Generate volume with inter-slice correlation
    volume = np.zeros((num_slices, height, width), dtype=np.int16)
    volume[0] = base

    for i in range(1, num_slices):
        # Add small random variation to previous slice
        noise = np.random.randint(-50, 50, (height, width), dtype=np.int16)
        volume[i] = np.clip(volume[i-1].astype(np.int32) + noise, -32768, 32767).astype(np.int16)

    return volume


def test_volume_roundtrip():
    """Test basic volume encode/decode roundtrip."""
    print("=" * 60)
    print("Test 1: Volume Encode/Decode Roundtrip")
    print("=" * 60)

    try:
        # Create synthetic volume
        volume = create_synthetic_volume(num_slices=5, height=64, width=64)
        print(f"   Volume shape: {volume.shape}")
        print(f"   Volume dtype: {volume.dtype}")
        print(f"   Value range: [{volume.min()}, {volume.max()}]")

        # Encode
        encoder = VolumeEncoder()
        compressed = encoder.encode(volume, quality=75)
        print(f"   Compressed size: {len(compressed):,} bytes")

        # Decode
        decoder = VolumeDecoder()
        recovered = decoder.decode(compressed)

        # Verify
        assert recovered.shape == volume.shape, "Shape mismatch"
        assert recovered.dtype == np.int16, "Dtype mismatch"

        # Calculate per-slice PSNR
        print("   Per-slice PSNR:")
        for i in range(volume.shape[0]):
            psnr = calculate_psnr(volume[i], recovered[i], bit_depth=16)
            print(f"     Slice {i}: {psnr:.2f} dB")

        print("   [PASS] Volume roundtrip test passed")
        return True

    except Exception as e:
        print(f"   [FAIL] Volume roundtrip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2d_vs_3d_compression():
    """Compare 2D (independent slices) vs 3D (inter-slice prediction) compression."""
    print("\n" + "=" * 60)
    print("Test 2: 2D vs 3D Compression Comparison")
    print("=" * 60)

    try:
        # Create volume with high inter-slice correlation
        volume = create_synthetic_volume(num_slices=10, height=128, width=128)
        print(f"   Volume: {volume.shape}, {volume.nbytes:,} bytes")

        quality = 75

        # 2D encoding: encode each slice independently
        encoder_2d = MedicalImageEncoder()
        compressed_2d_sizes = []

        for i in range(volume.shape[0]):
            compressed = encoder_2d.encode(volume[i], quality=quality)
            compressed_2d_sizes.append(len(compressed))

        total_2d = sum(compressed_2d_sizes)
        print(f"\n   2D Encoding (independent slices):")
        print(f"     Total size: {total_2d:,} bytes")
        print(f"     Average per slice: {total_2d // volume.shape[0]:,} bytes")

        # 3D encoding: exploit inter-slice prediction
        encoder_3d = VolumeEncoder()
        compressed_3d = encoder_3d.encode(volume, quality=quality)

        print(f"\n   3D Encoding (inter-slice prediction):")
        print(f"     Total size: {len(compressed_3d):,} bytes")

        # Get volume info
        decoder_3d = VolumeDecoder()
        info = decoder_3d.get_volume_info(compressed_3d)
        print(f"     I-slices: {info['i_slices']}, P-slices: {info['p_slices']}")

        # Calculate improvement
        stats = calculate_3d_compression_stats(volume, compressed_2d_sizes, len(compressed_3d))

        print(f"\n   Comparison:")
        print(f"     Original size:      {stats['original_bytes']:,} bytes")
        print(f"     2D compressed:      {stats['compressed_2d_bytes']:,} bytes (CR: {stats['compression_ratio_2d']:.2f}x)")
        print(f"     3D compressed:      {stats['compressed_3d_bytes']:,} bytes (CR: {stats['compression_ratio_3d']:.2f}x)")
        print(f"     3D improvement:     {stats['3d_improvement_percent']:.2f}%")
        print(f"     BPP (2D): {stats['bpp_2d']:.4f}, BPP (3D): {stats['bpp_3d']:.4f}")

        # Verify 3D is better
        assert stats['3d_improvement_percent'] > 0, "3D should be better than 2D"

        print("\n   [PASS] 2D vs 3D comparison test passed")
        return True, stats

    except Exception as e:
        print(f"   [FAIL] 2D vs 3D comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_gop_size_effect():
    """Test effect of GOP size on compression."""
    print("\n" + "=" * 60)
    print("Test 3: GOP Size Effect")
    print("=" * 60)

    try:
        volume = create_synthetic_volume(num_slices=20, height=64, width=64)
        print(f"   Volume: {volume.shape}")

        encoder = VolumeEncoder()
        decoder = VolumeDecoder()

        results = []
        for gop_size in [1, 5, 10, 20]:
            compressed = encoder.encode(volume, quality=75, gop_size=gop_size)
            info = decoder.get_volume_info(compressed)

            # Decode and check quality
            recovered = decoder.decode(compressed)
            psnr = calculate_psnr(volume, recovered, bit_depth=16)

            results.append({
                'gop_size': gop_size,
                'size': len(compressed),
                'i_slices': info['i_slices'],
                'p_slices': info['p_slices'],
                'psnr': psnr
            })

            print(f"   GOP={gop_size:2d}: {len(compressed):,} bytes, "
                  f"I={info['i_slices']:2d}, P={info['p_slices']:2d}, "
                  f"PSNR={psnr:.2f} dB")

        # GOP=1 should be largest (all I-slices)
        assert results[0]['size'] >= results[-1]['size'], \
            "GOP=1 should produce largest file"

        print("\n   [PASS] GOP size effect test passed")
        return True

    except Exception as e:
        print(f"   [FAIL] GOP size effect test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_levels():
    """Test 3D compression at different quality levels."""
    print("\n" + "=" * 60)
    print("Test 4: Quality Levels (3D)")
    print("=" * 60)

    try:
        volume = create_synthetic_volume(num_slices=5, height=64, width=64)
        print(f"   Volume: {volume.shape}")

        encoder = VolumeEncoder()
        decoder = VolumeDecoder()

        for quality in [25, 50, 85]:
            compressed = encoder.encode(volume, quality=quality)
            recovered = decoder.decode(compressed)

            psnr = calculate_psnr(volume, recovered, bit_depth=16)
            rmse = calculate_rmse(volume, recovered)
            cr = volume.nbytes / len(compressed)

            print(f"   Q={quality:2d}: Size={len(compressed):,} bytes, "
                  f"CR={cr:.2f}x, PSNR={psnr:.2f} dB, RMSE={rmse:.1f}")

        print("\n   [PASS] Quality levels test passed")
        return True

    except Exception as e:
        print(f"   [FAIL] Quality levels test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_dicom_volume():
    """Test with real DICOM volume if available."""
    print("\n" + "=" * 60)
    print("Test 5: Real DICOM Volume")
    print("=" * 60)

    dicom_dir = "data/2_skull_ct/DICOM"

    if not os.path.exists(dicom_dir):
        print("   [SKIP] DICOM directory not found")
        return True

    try:
        import pydicom

        # Load multiple slices
        dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.startswith('I')])

        if len(dicom_files) < 3:
            print("   [SKIP] Not enough DICOM slices")
            return True

        # Load first N slices
        num_slices = min(5, len(dicom_files))
        slices = []
        target_shape = None

        for fname in dicom_files[:num_slices]:
            ds = pydicom.dcmread(os.path.join(dicom_dir, fname))
            img = ds.pixel_array

            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            # Check shape consistency
            if target_shape is None:
                target_shape = img.shape
            elif img.shape != target_shape:
                print(f"   [SKIP] Inconsistent slice shapes: {target_shape} vs {img.shape}")
                return True

            slices.append(img.astype(np.int16))

        if len(slices) < 2:
            print("   [SKIP] Not enough valid slices")
            return True

        volume = np.stack(slices, axis=0)
        print(f"   Loaded {num_slices} DICOM slices: {volume.shape}")

        # Compare 2D vs 3D
        encoder_2d = MedicalImageEncoder()
        encoder_3d = VolumeEncoder()
        decoder_3d = VolumeDecoder()

        # 2D
        total_2d = sum(len(encoder_2d.encode(s, quality=75)) for s in volume)

        # 3D
        compressed_3d = encoder_3d.encode(volume, quality=75)
        recovered = decoder_3d.decode(compressed_3d)

        psnr = calculate_psnr(volume, recovered, bit_depth=16)

        improvement = (1 - len(compressed_3d) / total_2d) * 100

        print(f"   2D total: {total_2d:,} bytes")
        print(f"   3D total: {len(compressed_3d):,} bytes")
        print(f"   Improvement: {improvement:.2f}%")
        print(f"   PSNR: {psnr:.2f} dB")

        print("\n   [PASS] Real DICOM volume test passed")
        return True

    except Exception as e:
        print(f"   [FAIL] Real DICOM volume test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all 3D extension tests."""
    print("\n" + "=" * 60)
    print("3D VOLUME CODEC EXTENSION TESTS")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Basic roundtrip
    results.append(("Volume Roundtrip", test_volume_roundtrip()))

    # Test 2: 2D vs 3D comparison
    passed, stats = test_2d_vs_3d_compression()
    results.append(("2D vs 3D Comparison", passed))

    # Test 3: GOP size effect
    results.append(("GOP Size Effect", test_gop_size_effect()))

    # Test 4: Quality levels
    results.append(("Quality Levels", test_quality_levels()))

    # Test 5: Real DICOM
    results.append(("Real DICOM Volume", test_real_dicom_volume()))

    # Summary
    print("\n" + "=" * 60)
    print("3D EXTENSION TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All 3D extension tests passed!")
    else:
        print("Some tests failed")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
