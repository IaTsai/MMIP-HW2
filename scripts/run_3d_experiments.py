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
        compressed = encoder.encode(volume, quality=quality, gop_size=gop_size)
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


def main():
    """Run all 3D extension experiments."""
    print("=" * 70)
    print("3D VOLUME CODEC EXPERIMENTS")
    print("=" * 70)
    print()

    # Create synthetic volume
    print("[1/4] Creating synthetic CT volume...")
    volume = create_synthetic_ct_volume(num_slices=20, height=128, width=128)
    print(f"      Volume shape: {volume.shape}")
    print(f"      Volume size: {volume.nbytes:,} bytes")
    print(f"      Value range: [{volume.min()}, {volume.max()}]")
    print()

    # Experiment 1: 2D vs 3D at different quality levels
    print("[2/4] Running 2D vs 3D comparison at different quality levels...")
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
    print("[3/4] Running GOP size experiment...")
    print("-" * 70)

    gop_results = run_gop_size_experiment(volume, quality=75)

    print(f"  {'GOP':>4} {'Size':>10} {'CR':>8} {'PSNR':>10} {'I-slices':>10} {'P-slices':>10}")
    print("  " + "-" * 60)
    for r in gop_results:
        print(f"  {r['gop_size']:>4} {r['size']:>10,} {r['compression_ratio']:>8.2f}x "
              f"{r['psnr']:>10.2f} {r['i_slices']:>10} {r['p_slices']:>10}")
    print()

    # Save results
    print("[4/4] Saving results...")

    results = {
        'experiment': '3D Volume Codec Extension',
        'volume_info': {
            'shape': list(volume.shape),
            'size_bytes': int(volume.nbytes),
            'dtype': str(volume.dtype),
        },
        '2d_vs_3d_comparison': comparison_results,
        'gop_size_experiment': gop_results,
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
    print(f"  At Q=85:")
    print(f"    - 3D compression achieves {best_result['improvement_percent']:.1f}% better compression than 2D")
    print(f"    - Compression ratio: {best_result['compression_ratio_3d']:.2f}x (vs 2D: {best_result['compression_ratio_2d']:.2f}x)")
    print(f"    - PSNR: {best_result['psnr_3d']:.2f} dB")
    print()
    print("  GOP Size Effect:")
    print(f"    - GOP=1 (all I-slices): {gop_results[0]['size']:,} bytes")
    print(f"    - GOP=20 (mostly P-slices): {gop_results[-1]['size']:,} bytes")
    print(f"    - Savings: {(1 - gop_results[-1]['size']/gop_results[0]['size'])*100:.1f}%")
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
