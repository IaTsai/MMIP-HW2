"""DPCM (Differential Pulse Code Modulation) for DC coefficients."""

import numpy as np
from typing import List


def dpcm_encode_dc(dc_values: np.ndarray) -> np.ndarray:
    """
    Apply DPCM encoding to DC coefficients.

    DPCM exploits the high correlation between adjacent blocks' DC values.
    Instead of encoding absolute values, we encode differences:
        diff[0] = DC[0]
        diff[i] = DC[i] - DC[i-1]  for i > 0

    Args:
        dc_values: Array of DC coefficients

    Returns:
        Array of differences (same length as input)
    """
    diff = np.zeros_like(dc_values)
    diff[0] = dc_values[0]

    if len(dc_values) > 1:
        diff[1:] = dc_values[1:] - dc_values[:-1]

    return diff


def dpcm_decode_dc(diff_values: np.ndarray) -> np.ndarray:
    """
    Decode DPCM-encoded DC coefficients.

    Args:
        diff_values: Array of differences

    Returns:
        Reconstructed DC values
    """
    return np.cumsum(diff_values)
