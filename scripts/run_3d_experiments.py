"""Run 3D Volume Codec Experiments."""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.codec import (
    MedicalImageEncoder, MedicalImageDecoder,
    VolumeEncoder, VolumeDecoder,
    calculate_3d_compression_stats
)
from medcodec.metrics import calculate_psnr, calculate_rmse


def create_synthetic_ct_volume(num_slices=20, height=128, width=128, seed=42):
    """
    Create a synthetic CT-like volume with realistic inter-slice correlation.
    """
    np.random.seed(seed)

    # Create base slice with CT-like structures
    base = np.random.randint(-500, 500, (height, width), dtype=np.int16)

    # Add circular structure (simulating bone/organ)
    y, x = np.ogrid[:height, :width]
    center = (height // 2, width // 2)
    radius = min(height, width) // 4
    mask = (y - center[0])**2 + (x - center[1])**2 <= radius**2
    base[mask] += 500

    # Generate volume with inter-slice correlation
    volume = np.zeros((num_slices, height, width), dtype=np.int16)
    volume[0] = base

    for i in range(1, num_slices):
        # Small random variation (simulates adjacent CT slices)
        noise = np.random.randint(-30, 30, (height, width), dtype=np.int16)
        volume[i] = np.clip(volume[i-1].astype(np.int32) + noise, -32768, 32767).astype(np.int16)

    return volume


def run_2d_vs_3d_comparison(volume, quality=75):
    """Compare 2D (independent) vs 3D (inter-slice) compression."""
    encoder_2d = MedicalImageEncoder()
    encoder_3d = VolumeEncoder()
    decoder_3d = VolumeDecoder()

    # 2D encoding
    compressed_2d_sizes = []
    for i in range(volume.shape[0]):
        compressed = encoder_2d.encode(volume[i], quality=quality)
        compressed_2d_sizes.append(len(compressed))
    total_2d = sum(compressed_2d_sizes)

    # 3D encoding
    compressed_3d = encoder_3d.encode(volume, quality=quality)
    recovered_3d = decoder_3d.decode(compressed_3d)
    info_3d = decoder_3d.get_volume_info(compressed_3d)

    # Metrics
    psnr_3d = calculate_psnr(volume, recovered_3d, bit_depth=16)
    rmse_3d = calculate_rmse(volume, recovered_3d)

    stats = calculate_3d_compression_stats(volume, compressed_2d_sizes, len(compressed_3d))

    return {
        'quality': quality,
        '2d_size': total_2d,
        '3d_size': len(compressed_3d),
        'improvement_percent': stats['3d_improvement_percent'],
        'compression_ratio_2d': stats['compression_ratio_2d'],
        'compression_ratio_3d': stats['compression_ratio_3d'],
        'bpp_2d': stats['bpp_2d'],
        'bpp_3d': stats['bpp_3d'],
        'psnr_3d': psnr_3d,
        'rmse_3d': rmse_3d,
        'i_slices': info_3d['i_slices'],
        'p_slices': info_3d['p_slices'],
    }


def run_gop_size_experiment(volume, quality=75):
    """Test effect of GOP size on compression."""
    encoder = VolumeEncoder()
    decoder = VolumeDecoder()

    results = []
    for gop_size in [1, 5, 10, 20]:
        compressed = encoder.encode(volume, quality=quality, gop_size=gop_size, use_b_slices=False)
        recovered = decoder.decode(compressed)
        info = decoder.get_volume_info(compressed)

        psnr = calculate_psnr(volume, recovered, bit_depth=16)
        cr = volume.nbytes / len(compressed)

        results.append({
            'gop_size': gop_size,
            'size': len(compressed),
            'compression_ratio': cr,
            'psnr': psnr,
            'i_slices': info['i_slices'],
            'p_slices': info['p_slices'],
        })

    return results


def run_b_slice_experiment(volume):
    """Compare P-only vs I/P/B encoding at different quality levels."""
    encoder = VolumeEncoder()
    decoder = VolumeDecoder()

    results = []
    for quality in [50, 75, 85]:
        # P-only encoding
        comp_p = encoder.encode(volume, quality=quality, gop_size=6, use_b_slices=False)
        info_p = decoder.get_volume_info(comp_p)

        # I/P/B encoding
        comp_b = encoder.encode(volume, quality=quality, gop_size=6, use_b_slices=True)
        recovered_b = decoder.decode(comp_b)
        info_b = decoder.get_volume_info(comp_b)

        psnr = calculate_psnr(volume, recovered_b, bit_depth=16)
        improvement = (1 - len(comp_b) / len(comp_p)) * 100

        results.append({
            'quality': quality,
            'p_only_size': len(comp_p),
            'ipb_size': len(comp_b),
            'improvement_percent': improvement,
            'psnr': psnr,
            'i_slices': info_b['i_slices'],
            'p_slices': info_b['p_slices'],
            'b_slices': info_b.get('b_slices', 0),
        })

    return results


def main():
    """Run all 3D extension experiments."""
    print("=" * 70)
    print("3D VOLUME CODEC EXPERIMENTS (with B-slice)")
    print("=" * 70)
    print()

    # Create synthetic volume
    print("[1/5] Creating synthetic CT volume...")
    volume = create_synthetic_ct_volume(num_slices=20, height=128, width=128)
    print(f"      Volume shape: {volume.shape}")
    print(f"      Volume size: {volume.nbytes:,} bytes")
    print(f"      Value range: [{volume.min()}, {volume.max()}]")
    print()

    # Experiment 1: 2D vs 3D at different quality levels
    print("[2/5] Running 2D vs 3D comparison at different quality levels...")
    print("-" * 70)

    comparison_results = []
    for quality in [25, 50, 75, 85]:
        result = run_2d_vs_3d_comparison(volume, quality=quality)
        comparison_results.append(result)

        print(f"  Q={quality:2d}:")
        print(f"    2D: {result['2d_size']:,} bytes (CR: {result['compression_ratio_2d']:.2f}x)")
        print(f"    3D: {result['3d_size']:,} bytes (CR: {result['compression_ratio_3d']:.2f}x)")
        print(f"    Improvement: {result['improvement_percent']:.2f}%")
        print(f"    PSNR: {result['psnr_3d']:.2f} dB")
        print()

    # Experiment 2: GOP size effect
    print("[3/5] Running GOP size experiment (P-only)...")
    print("-" * 70)

    gop_results = run_gop_size_experiment(volume, quality=75)

    print(f"  {'GOP':>4} {'Size':>10} {'CR':>8} {'PSNR':>10} {'I-slices':>10} {'P-slices':>10}")
    print("  " + "-" * 60)
    for r in gop_results:
        print(f"  {r['gop_size']:>4} {r['size']:>10,} {r['compression_ratio']:>8.2f}x "
              f"{r['psnr']:>10.2f} {r['i_slices']:>10} {r['p_slices']:>10}")
    print()

    # Experiment 3: B-slice comparison
    print("[4/5] Running B-slice experiment (P-only vs I/P/B)...")
    print("-" * 70)

    b_slice_results = run_b_slice_experiment(volume)

    print(f"  {'Quality':>7} {'P-only':>12} {'I/P/B':>12} {'Improve':>10} {'PSNR':>10} {'I/P/B':>10}")
    print("  " + "-" * 65)
    for r in b_slice_results:
        print(f"  Q={r['quality']:2d}    {r['p_only_size']:>12,} {r['ipb_size']:>12,} "
              f"{r['improvement_percent']:>9.2f}% {r['psnr']:>10.2f} "
              f"{r['i_slices']}/{r['p_slices']}/{r['b_slices']}")
    print()

    # Save results
    print("[5/5] Saving results...")

    results = {
        'experiment': '3D Volume Codec Extension (with B-slice)',
        'volume_info': {
            'shape': list(volume.shape),
            'size_bytes': int(volume.nbytes),
            'dtype': str(volume.dtype),
        },
        '2d_vs_3d_comparison': comparison_results,
        'gop_size_experiment': gop_results,
        'b_slice_experiment': b_slice_results,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/3d_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("      Saved to: results/3d_metrics.json")
    print()

    # Summary
    print("=" * 70)
    print("3D EXTENSION SUMMARY")
    print("=" * 70)

    best_result = comparison_results[-1]  # Q=85
    print(f"  2D vs 3D (at Q=85):")
    print(f"    - 3D compression achieves {best_result['improvement_percent']:.1f}% better than 2D")
    print(f"    - Compression ratio: {best_result['compression_ratio_3d']:.2f}x (vs 2D: {best_result['compression_ratio_2d']:.2f}x)")
    print()
    print("  GOP Size Effect (P-only):")
    print(f"    - GOP=1 (all I-slices): {gop_results[0]['size']:,} bytes")
    print(f"    - GOP=20 (mostly P-slices): {gop_results[-1]['size']:,} bytes")
    print(f"    - Savings: {(1 - gop_results[-1]['size']/gop_results[0]['size'])*100:.1f}%")
    print()
    print("  B-slice Improvement (at Q=85):")
    best_b = b_slice_results[-1]
    print(f"    - P-only: {best_b['p_only_size']:,} bytes")
    print(f"    - I/P/B:  {best_b['ipb_size']:,} bytes")
    print(f"    - B-slice improvement: {best_b['improvement_percent']:.1f}%")
    print(f"    - Slice distribution: I={best_b['i_slices']}, P={best_b['p_slices']}, B={best_b['b_slices']}")
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
