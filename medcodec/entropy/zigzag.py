"""Zigzag scan for converting 2D blocks to 1D arrays."""

import numpy as np

# Pre-computed zigzag indices for 8x8 block
ZIGZAG_ORDER = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63
])

# Inverse mapping: position in zigzag sequence → (row, col)
INVERSE_ZIGZAG = np.argsort(ZIGZAG_ORDER)


def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """
    Apply zigzag scan to an 8x8 block.

    The zigzag pattern reads coefficients in order of increasing frequency,
    which tends to group zeros together at the end.

    Args:
        block: 8x8 input block

    Returns:
        1D array of 64 elements in zigzag order
    """
    assert block.shape == (8, 8), f"Expected 8x8 block, got {block.shape}"
    flat = block.flatten()
    return flat[ZIGZAG_ORDER]


def inverse_zigzag(array: np.ndarray) -> np.ndarray:
    """
    Convert zigzag-ordered 1D array back to 8x8 block.

    Args:
        array: 1D array of 64 elements in zigzag order

    Returns:
        8x8 block
    """
    assert len(array) == 64, f"Expected 64 elements, got {len(array)}"
    flat = np.zeros(64, dtype=array.dtype)
    flat[ZIGZAG_ORDER] = array
    return flat.reshape(8, 8)
