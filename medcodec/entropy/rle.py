"""Run-Length Encoding for AC coefficients."""

from typing import List, Tuple
import numpy as np

# Special markers
EOB = (0, 0)      # End of Block - remaining coefficients are all zeros
ZRL = (15, 0)     # Zero Run Length - 16 consecutive zeros


def rle_encode(ac_coeffs: List[int]) -> List[Tuple[int, int]]:
    """
    Apply Run-Length Encoding to AC coefficients.

    Each non-zero coefficient is encoded as (run, value) where:
    - run: number of preceding zeros (0-15)
    - value: the non-zero coefficient value

    Special cases:
    - EOB (0, 0): signals that all remaining coefficients are zero
    - ZRL (15, 0): represents 16 consecutive zeros (when run > 15)

    Args:
        ac_coeffs: List of 63 AC coefficients (after DC is removed)

    Returns:
        List of (run, value) tuples
    """
    result = []
    zero_count = 0

    for coeff in ac_coeffs:
        if coeff == 0:
            zero_count += 1
        else:
            # Handle runs longer than 15 with ZRL markers
            while zero_count >= 16:
                result.append(ZRL)
                zero_count -= 16

            result.append((zero_count, coeff))
            zero_count = 0

    # If we end with zeros, add EOB
    if zero_count > 0 or len(result) == 0:
        result.append(EOB)

    return result


def rle_decode(rle_pairs: List[Tuple[int, int]], length: int = 63) -> List[int]:
    """
    Decode Run-Length Encoded AC coefficients.

    Args:
        rle_pairs: List of (run, value) tuples
        length: Expected output length (default: 63 for AC coefficients)

    Returns:
        List of AC coefficients
    """
    result = []

    for run, value in rle_pairs:
        if (run, value) == EOB:
            # Fill remaining with zeros
            result.extend([0] * (length - len(result)))
            break
        elif (run, value) == ZRL:
            # 16 zeros
            result.extend([0] * 16)
        else:
            # run zeros followed by value
            result.extend([0] * run)
            result.append(value)

    # Ensure correct length
    if len(result) < length:
        result.extend([0] * (length - len(result)))
    elif len(result) > length:
        result = result[:length]

    return result
