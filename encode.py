#!/usr/bin/env python3
"""
Medical Image Encoder CLI

Usage:
    python encode.py --input <path> --output <path> --quality <q>

Example:
    python encode.py --input data/ct_scan.dcm --output output.mcd --quality 75
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from medcodec.io import read_medical_image
from medcodec.codec import MedicalImageEncoder
from medcodec.metrics import calculate_bpp, calculate_compression_ratio


def main():
    parser = argparse.ArgumentParser(
        description='Medical Image Encoder - Lossy compression for medical images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode DICOM file
  python encode.py --input data/ct.dcm --output output.mcd --quality 75

  # Encode with verbose output
  python encode.py --input data/image.npy --output output.mcd --quality 50 --verbose

  # Encode raw file (requires dimensions)
  python encode.py --input data/raw.bin --output output.mcd --quality 85 \\
      --width 512 --height 512 --bit-depth 16
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input image path (.dcm, .npy, or .raw)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output compressed file path (.mcd)')
    parser.add_argument('--quality', '-q', type=int, required=True,
                        help='Quality parameter (1-100). Higher = better quality, larger file')

    # Optional arguments
    parser.add_argument('--bit-depth', '-b', type=int, default=16,
                        help='Bit depth for raw files (default: 16)')
    parser.add_argument('--width', '-W', type=int,
                        help='Image width (required for raw files)')
    parser.add_argument('--height', '-H', type=int,
                        help='Image height (required for raw files)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Validate quality
    if not 1 <= args.quality <= 100:
        print(f"Error: Quality must be between 1 and 100, got {args.quality}",
              file=sys.stderr)
        sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check raw file requirements
    input_ext = os.path.splitext(args.input)[1].lower()
    if input_ext == '.raw':
        if args.width is None or args.height is None:
            print("Error: --width and --height are required for raw files",
                  file=sys.stderr)
            sys.exit(1)

    try:
        if args.verbose:
            print(f"Reading input: {args.input}")

        start_time = time.time()

        # Read input image
        if input_ext == '.raw':
            image = read_medical_image(args.input, width=args.width,
                                       height=args.height, bit_depth=args.bit_depth)
        elif input_ext in ['.dcm', '']:
            # Handle DICOM files (with or without extension)
            import pydicom
            ds = pydicom.dcmread(args.input)
            image = ds.pixel_array

            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                image = image * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            image = image.astype(np.int16)
            args.bit_depth = ds.BitsAllocated if hasattr(ds, 'BitsAllocated') else 16
        else:
            image = read_medical_image(args.input)

        if args.verbose:
            print(f"  Shape: {image.shape}")
            print(f"  Dtype: {image.dtype}")
            print(f"  Range: [{image.min()}, {image.max()}]")
            print(f"  Bit depth: {args.bit_depth}")

        # Encode
        if args.verbose:
            print(f"Encoding with quality={args.quality}...")

        encoder = MedicalImageEncoder()
        compressed = encoder.encode(image, quality=args.quality,
                                    bit_depth=args.bit_depth)

        # Write output
        with open(args.output, 'wb') as f:
            f.write(compressed)

        elapsed = time.time() - start_time

        # Calculate metrics
        original_size = image.nbytes
        compressed_size = len(compressed)
        bpp = calculate_bpp(compressed_size, image.shape)
        cr = calculate_compression_ratio(original_size, compressed_size)

        if args.verbose:
            print(f"\nResults:")
            print(f"  Original size:   {original_size:,} bytes")
            print(f"  Compressed size: {compressed_size:,} bytes")
            print(f"  Compression ratio: {cr:.2f}x")
            print(f"  Bits per pixel: {bpp:.3f}")
            print(f"  Encoding time: {elapsed:.2f}s")
            print(f"\nOutput written to: {args.output}")
        else:
            print(f"Encoded: {args.input} -> {args.output} "
                  f"({cr:.1f}x compression, {bpp:.3f} bpp)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
