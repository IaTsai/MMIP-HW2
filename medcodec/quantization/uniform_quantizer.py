"""Uniform quantization for 16-bit medical images."""

import numpy as np


def get_quantization_step(quality: int, bit_depth: int = 16) -> int:
    """
    Map quality parameter (1-100) to quantization step.

    For 16-bit images, we need larger step values to achieve meaningful compression.
    The base step is scaled based on the dynamic range.

    Quality mapping:
        Q=1   → step ≈ 163850 (highest compression, lowest quality)
        Q=50  → step ≈ 3277   (medium)
        Q=100 → step = 1      (lowest compression, highest quality)

    Args:
        quality: Quality parameter (1-100)
        bit_depth: Bit depth of the image (default: 16)

    Returns:
        Quantization step value

    Raises:
        ValueError: If quality is not in range [1, 100]
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"Quality must be in range [1, 100], got {quality}")

    max_val = (1 << bit_depth) - 1  # 65535 for 16-bit
    base_step = max_val / 20        # ≈ 3277

    if quality < 50:
        # Quality 1 → factor 50.0, Quality 49 → factor ≈ 1.02
        factor = 50 / quality
    else:
        # Quality 50 → factor 1.0, Quality 100 → factor ≈ 0
        factor = (100 - quality) / 50

    step = int(base_step * factor)
    return max(1, step)


def quantize(coeffs: np.ndarray, step: int) -> np.ndarray:
    """
    Quantize DCT coefficients.

    For 16-bit medical images, DCT coefficients can exceed int16 range,
    especially the DC coefficient which can be very large (e.g., 270000+).
    We use int32 to safely store these values.

    Args:
        coeffs: DCT coefficient array
        step: Quantization step

    Returns:
        Quantized coefficients as int32
    """
    return np.round(coeffs / step).astype(np.int32)


def dequantize(quant_coeffs: np.ndarray, step: int) -> np.ndarray:
    """
    Dequantize coefficients.

    Args:
        quant_coeffs: Quantized coefficient array
        step: Quantization step

    Returns:
        Dequantized coefficients as float64
    """
    return quant_coeffs.astype(np.float64) * step
