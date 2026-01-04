#!/usr/bin/env python3
"""
Medical Image Decoder CLI

Usage:
    python decode.py --input <path> --output <path>

Example:
    python decode.py --input compressed.mcd --output recovered.npy
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from medcodec.io import write_medical_image
from medcodec.codec import MedicalImageDecoder


def main():
    parser = argparse.ArgumentParser(
        description='Medical Image Decoder - Decompress medical images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode to NumPy format
  python decode.py --input compressed.mcd --output recovered.npy

  # Decode with verbose output
  python decode.py --input compressed.mcd --output recovered.npy --verbose

  # Decode to raw format
  python decode.py --input compressed.mcd --output recovered.raw --format raw
        """
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                        help='Input compressed file path (.mcd)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output image path (.npy or .raw)')

    # Optional arguments
    parser.add_argument('--format', '-f', choices=['npy', 'raw'], default='npy',
                        help='Output format (default: npy)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.verbose:
            print(f"Reading compressed file: {args.input}")

        start_time = time.time()

        # Read compressed data
        with open(args.input, 'rb') as f:
            compressed = f.read()

        if args.verbose:
            print(f"  Compressed size: {len(compressed):,} bytes")

        # Decode
        if args.verbose:
            print("Decoding...")

        decoder = MedicalImageDecoder()
        image = decoder.decode(compressed)

        elapsed = time.time() - start_time

        if args.verbose:
            print(f"\nReconstructed image:")
            print(f"  Shape: {image.shape}")
            print(f"  Dtype: {image.dtype}")
            print(f"  Range: [{image.min()}, {image.max()}]")

        # Write output
        write_medical_image(image, args.output, format=args.format)

        if args.verbose:
            print(f"  Output size: {image.nbytes:,} bytes")
            print(f"  Decoding time: {elapsed:.2f}s")
            print(f"\nOutput written to: {args.output}")
        else:
            print(f"Decoded: {args.input} -> {args.output} "
                  f"({image.shape[0]}x{image.shape[1]})")

    except ValueError as e:
        print(f"Error: Invalid compressed file - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
