#!/usr/bin/env python3
"""
Run experiments for Medical Image Codec evaluation.

Generates metrics.json and error map images for the report.
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pydicom

from medcodec.codec import MedicalImageEncoder, MedicalImageDecoder
from medcodec.metrics import (
    calculate_rmse,
    calculate_psnr,
    calculate_bpp,
    calculate_compression_ratio,
    generate_error_map,
    normalize_for_display,
)


def save_image_png(image: np.ndarray, path: str):
    """Save image as PNG using PIL."""
    try:
        from PIL import Image
        img_8bit = normalize_for_display(image)
        Image.fromarray(img_8bit).save(path)
        return True
    except ImportError:
        print("  Warning: PIL not available, skipping PNG output")
        return False


def run_experiment(image: np.ndarray, quality: int, encoder, decoder):
    """Run encode/decode experiment at a specific quality level."""
    # Encode
    compressed = encoder.encode(image, quality=quality)

    # Decode
    recovered = decoder.decode(compressed)

    # Calculate metrics
    rmse = calculate_rmse(image, recovered)
    psnr = calculate_psnr(image, recovered, bit_depth=16)
    bpp = calculate_bpp(len(compressed), image.shape)
    cr = calculate_compression_ratio(image.nbytes, len(compressed))

    # Generate error map
    error_map = generate_error_map(image, recovered)

    return {
        'quality': quality,
        'rmse': round(rmse, 4),
        'psnr': round(psnr, 2),
        'bpp': round(bpp, 4),
        'compression_ratio': round(cr, 2),
        'compressed_bytes': len(compressed),
        'original_bytes': image.nbytes,
        'max_error': int(error_map.max()),
        'mean_error': round(float(error_map.mean()), 2),
    }, recovered, error_map


def run_ablation_experiment(image: np.ndarray, quality: int, encoder, decoder):
    """Run ablation experiment comparing with/without DC DPCM."""
    # With DPCM (default)
    compressed_dpcm = encoder.encode(image, quality=quality, use_dpcm=True)
    recovered_dpcm = decoder.decode(compressed_dpcm)

    # Without DPCM
    compressed_no_dpcm = encoder.encode(image, quality=quality, use_dpcm=False)
    recovered_no_dpcm = decoder.decode(compressed_no_dpcm)

    return {
        'quality': quality,
        'with_dpcm': {
            'bpp': round(calculate_bpp(len(compressed_dpcm), image.shape), 4),
            'psnr': round(calculate_psnr(image, recovered_dpcm, 16), 2),
            'compressed_bytes': len(compressed_dpcm),
        },
        'without_dpcm': {
            'bpp': round(calculate_bpp(len(compressed_no_dpcm), image.shape), 4),
            'psnr': round(calculate_psnr(image, recovered_no_dpcm, 16), 2),
            'compressed_bytes': len(compressed_no_dpcm),
        },
        'dpcm_savings_percent': round(
            (1 - len(compressed_dpcm) / len(compressed_no_dpcm)) * 100, 2
        ) if len(compressed_no_dpcm) > 0 else 0,
    }


def main():
    """Run all experiments."""
    print("=" * 60)
    print("MEDICAL IMAGE CODEC - EXPERIMENT RUNNER")
    print("=" * 60)

    # Configuration
    dicom_path = "data/2_skull_ct/DICOM/I0"
    qualities = [25, 50, 85]
    results_dir = "results"
    images_dir = os.path.join(results_dir, "images")

    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)

    # Load DICOM
    if not os.path.exists(dicom_path):
        print(f"Error: DICOM file not found: {dicom_path}")
        sys.exit(1)

    print(f"\nLoading DICOM: {dicom_path}")
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array

    # Apply rescale if available
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        image = image * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    image = image.astype(np.int16)

    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min()}, {image.max()}]")
    print(f"  Size: {image.nbytes:,} bytes")

    # Save original image
    save_image_png(image, os.path.join(images_dir, "original.png"))
    print(f"  Saved: {images_dir}/original.png")

    # Initialize encoder/decoder
    encoder = MedicalImageEncoder()
    decoder = MedicalImageDecoder()

    # Run experiments at each quality level
    print("\n" + "=" * 60)
    print("RATE-DISTORTION EXPERIMENTS")
    print("=" * 60)

    all_results = []

    for quality in qualities:
        print(f"\n--- Quality = {quality} ---")

        result, recovered, error_map = run_experiment(
            image, quality, encoder, decoder
        )
        all_results.append(result)

        print(f"  RMSE:  {result['rmse']:.4f}")
        print(f"  PSNR:  {result['psnr']:.2f} dB")
        print(f"  BPP:   {result['bpp']:.4f}")
        print(f"  CR:    {result['compression_ratio']:.2f}x")
        print(f"  Size:  {result['compressed_bytes']:,} bytes")
        print(f"  Max Error: {result['max_error']}")

        # Save reconstructed and error map images
        save_image_png(recovered, os.path.join(images_dir, f"reconstructed_q{quality}.png"))
        save_image_png(error_map, os.path.join(images_dir, f"error_map_q{quality}.png"))
        print(f"  Saved: reconstructed_q{quality}.png, error_map_q{quality}.png")

    # Run ablation study
    print("\n" + "=" * 60)
    print("ABLATION STUDY: DC DPCM")
    print("=" * 60)

    ablation_results = []
    for quality in qualities:
        print(f"\n--- Quality = {quality} ---")

        ablation = run_ablation_experiment(image, quality, encoder, decoder)
        ablation_results.append(ablation)

        print(f"  With DPCM:    BPP={ablation['with_dpcm']['bpp']:.4f}, "
              f"Size={ablation['with_dpcm']['compressed_bytes']:,} bytes")
        print(f"  Without DPCM: BPP={ablation['without_dpcm']['bpp']:.4f}, "
              f"Size={ablation['without_dpcm']['compressed_bytes']:,} bytes")
        print(f"  DPCM Savings: {ablation['dpcm_savings_percent']:.2f}%")

    # Compile final results
    output = {
        "experiment_date": datetime.now().isoformat(),
        "dataset": {
            "name": "Medimodel Human Skull 2",
            "source_url": "https://medimodel.com/sample-dicom-files/",
            "file": dicom_path,
        },
        "image_info": {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min_value": int(image.min()),
            "max_value": int(image.max()),
            "original_bytes": int(image.nbytes),
        },
        "codec_info": {
            "block_size": 8,
            "transform": "DCT (matrix multiplication)",
            "quantization": "Uniform (16-bit adapted)",
            "entropy_coding": "Huffman with DC DPCM + AC RLE",
        },
        "rate_distortion_results": all_results,
        "ablation_study": {
            "description": "Comparison of compression with and without DC DPCM",
            "results": ablation_results,
        },
    }

    # Save results
    output_path = os.path.join(results_dir, "metrics.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print(f"Images saved to: {images_dir}/")

    # Summary table
    print("\n" + "-" * 60)
    print("SUMMARY TABLE")
    print("-" * 60)
    print(f"{'Quality':>8} {'RMSE':>10} {'PSNR (dB)':>12} {'BPP':>10} {'CR':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['quality']:>8} {r['rmse']:>10.4f} {r['psnr']:>12.2f} "
              f"{r['bpp']:>10.4f} {r['compression_ratio']:>9.2f}x")
    print("-" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
