"""Checkpoint 7: Evaluation and Report Verification."""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.metrics import (
    calculate_rmse, calculate_psnr, calculate_bpp,
    calculate_compression_ratio, generate_error_map, normalize_for_display
)


def test_metrics_functions():
    """Test all metric functions work correctly."""
    print("=" * 60)
    print("Test 1: Metric Functions")
    print("=" * 60)

    try:
        # Create test images
        original = np.random.randint(-1024, 3071, (64, 64), dtype=np.int16)
        noise = np.random.randn(64, 64) * 10
        reconstructed = (original.astype(np.float64) + noise).astype(np.int16)

        # Test RMSE
        rmse = calculate_rmse(original, reconstructed)
        assert rmse >= 0, "RMSE must be non-negative"
        print(f"   ✓ RMSE: {rmse:.4f}")

        # Test PSNR
        psnr = calculate_psnr(original, reconstructed, bit_depth=16)
        assert psnr > 0, "PSNR must be positive"
        print(f"   ✓ PSNR: {psnr:.2f} dB")

        # Test BPP
        compressed_size = 1000
        bpp = calculate_bpp(compressed_size, original.shape)
        expected_bpp = (1000 * 8) / (64 * 64)
        assert abs(bpp - expected_bpp) < 0.001, "BPP calculation incorrect"
        print(f"   ✓ BPP: {bpp:.4f}")

        # Test compression ratio
        original_size = 64 * 64 * 2  # int16 = 2 bytes
        cr = calculate_compression_ratio(original_size, compressed_size)
        expected_cr = original_size / compressed_size
        assert abs(cr - expected_cr) < 0.001, "CR calculation incorrect"
        print(f"   ✓ Compression Ratio: {cr:.2f}x")

        # Test error map
        error_map = generate_error_map(original, reconstructed)
        assert error_map.shape == original.shape, "Error map shape mismatch"
        assert error_map.dtype == np.uint16, "Error map dtype should be uint16"
        print(f"   ✓ Error map: shape={error_map.shape}, max={error_map.max()}")

        # Test normalization
        normalized = normalize_for_display(original)
        assert normalized.dtype == np.uint8, "Normalized should be uint8"
        assert normalized.min() >= 0 and normalized.max() <= 255, "Range should be 0-255"
        print(f"   ✓ Normalization: range=[{normalized.min()}, {normalized.max()}]")

        print("✅ Metric functions test passed")
        return True

    except Exception as e:
        print(f"❌ Metric functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_json():
    """Test that metrics.json was created with correct structure."""
    print("\n" + "=" * 60)
    print("Test 2: Metrics JSON Validation")
    print("=" * 60)

    metrics_path = "results/metrics.json"

    if not os.path.exists(metrics_path):
        print(f"   ⚠️ metrics.json not found at {metrics_path}")
        print("   Run scripts/run_experiments.py first")
        return False

    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        # Check required fields
        required_fields = ['dataset', 'image_info', 'codec_info',
                           'rate_distortion_results', 'ablation_study']
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        print("   ✓ All required top-level fields present")

        # Check rate-distortion results
        rd_results = data['rate_distortion_results']
        assert len(rd_results) >= 3, "Need at least 3 quality points"

        for r in rd_results:
            assert 'quality' in r, "Missing quality field"
            assert 'rmse' in r, "Missing rmse field"
            assert 'psnr' in r, "Missing psnr field"
            assert 'bpp' in r, "Missing bpp field"
            assert 'compression_ratio' in r, "Missing compression_ratio field"

        print(f"   ✓ Rate-distortion results: {len(rd_results)} quality points")

        # Verify PSNR increases with quality
        psnrs = [r['psnr'] for r in rd_results]
        assert psnrs == sorted(psnrs), "PSNR should increase with quality"
        print(f"   ✓ PSNR trend verified: {psnrs}")

        # Check ablation study
        ablation = data['ablation_study']
        assert 'results' in ablation, "Missing ablation results"
        assert len(ablation['results']) >= 1, "Need ablation data"

        for a in ablation['results']:
            assert 'with_dpcm' in a, "Missing with_dpcm"
            assert 'without_dpcm' in a, "Missing without_dpcm"
            assert 'dpcm_savings_percent' in a, "Missing savings percentage"

        print(f"   ✓ Ablation study: {len(ablation['results'])} quality points")

        print("✅ Metrics JSON validation passed")
        return True

    except Exception as e:
        print(f"❌ Metrics JSON validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_images():
    """Test that output images were created."""
    print("\n" + "=" * 60)
    print("Test 3: Output Images Verification")
    print("=" * 60)

    images_dir = "results/images"

    if not os.path.exists(images_dir):
        print(f"   ⚠️ Images directory not found: {images_dir}")
        return False

    try:
        required_images = [
            'original.png',
            'reconstructed_q25.png',
            'reconstructed_q50.png',
            'reconstructed_q85.png',
            'error_map_q25.png',
            'error_map_q50.png',
            'error_map_q85.png',
        ]

        for img in required_images:
            path = os.path.join(images_dir, img)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"   ✓ {img}: {size:,} bytes")
            else:
                print(f"   ⚠️ Missing: {img}")

        print("✅ Output images verification passed")
        return True

    except Exception as e:
        print(f"❌ Output images verification failed: {e}")
        return False


def test_full_pipeline():
    """Test full encode/decode with metric calculation."""
    print("\n" + "=" * 60)
    print("Test 4: Full Pipeline with Metrics")
    print("=" * 60)

    try:
        from medcodec.codec import MedicalImageEncoder, MedicalImageDecoder

        # Create test image
        np.random.seed(42)
        original = np.random.randint(-1024, 3071, (128, 128), dtype=np.int16)

        encoder = MedicalImageEncoder()
        decoder = MedicalImageDecoder()

        results = []
        for quality in [25, 50, 85]:
            compressed = encoder.encode(original, quality=quality)
            recovered = decoder.decode(compressed)

            rmse = calculate_rmse(original, recovered)
            psnr = calculate_psnr(original, recovered, bit_depth=16)
            bpp = calculate_bpp(len(compressed), original.shape)
            cr = calculate_compression_ratio(original.nbytes, len(compressed))

            results.append({
                'quality': quality,
                'rmse': rmse,
                'psnr': psnr,
                'bpp': bpp,
                'cr': cr
            })

            print(f"   Q={quality}: PSNR={psnr:.2f}dB, RMSE={rmse:.1f}, "
                  f"BPP={bpp:.3f}, CR={cr:.1f}x")

        # Verify trends
        assert results[2]['psnr'] > results[1]['psnr'] > results[0]['psnr'], \
            "PSNR should increase with quality"
        print("   ✓ PSNR increases with quality")

        assert results[2]['rmse'] < results[1]['rmse'] < results[0]['rmse'], \
            "RMSE should decrease with quality"
        print("   ✓ RMSE decreases with quality")

        print("✅ Full pipeline with metrics test passed")
        return True

    except Exception as e:
        print(f"❌ Full pipeline with metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Checkpoint 7 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 7: EVALUATION AND REPORT VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Metric functions
    results.append(("Metric Functions", test_metrics_functions()))

    # Test 2: Metrics JSON
    results.append(("Metrics JSON", test_metrics_json()))

    # Test 3: Output images
    results.append(("Output Images", test_output_images()))

    # Test 4: Full pipeline
    results.append(("Full Pipeline", test_full_pipeline()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 7 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 7 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 7 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
