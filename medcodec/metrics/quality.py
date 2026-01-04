"""Quality metrics for image codec evaluation."""

import numpy as np


def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        original: Original image
        reconstructed: Reconstructed image

    Returns:
        RMSE value
    """
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray,
                   bit_depth: int = 16) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX^2 / MSE)

    where MAX = 2^bit_depth - 1

    Args:
        original: Original image
        reconstructed: Reconstructed image
        bit_depth: Bit depth of the image (default: 16)

    Returns:
        PSNR in dB
    """
    max_val = (1 << bit_depth) - 1  # 65535 for 16-bit

    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)

    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr


def calculate_bpp(compressed_size: int, image_shape: tuple) -> float:
    """
    Calculate Bits Per Pixel (BPP).

    Args:
        compressed_size: Size of compressed data in bytes
        image_shape: Tuple of (height, width)

    Returns:
        BPP value
    """
    num_pixels = image_shape[0] * image_shape[1]
    return (compressed_size * 8) / num_pixels


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate compression ratio.

    Args:
        original_size: Size of original data in bytes
        compressed_size: Size of compressed data in bytes

    Returns:
        Compression ratio (original / compressed)
    """
    if compressed_size == 0:
        return float('inf')
    return original_size / compressed_size


def generate_error_map(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """
    Generate absolute error map between original and reconstructed images.

    Args:
        original: Original image
        reconstructed: Reconstructed image

    Returns:
        Absolute error map as uint16 array
    """
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    error_map = np.abs(diff)
    return error_map.astype(np.uint16)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize 16-bit image to 8-bit for display.

    Uses min-max normalization.

    Args:
        image: Input image (any dtype)

    Returns:
        8-bit normalized image
    """
    img_float = image.astype(np.float64)
    img_min = img_float.min()
    img_max = img_float.max()

    if img_max == img_min:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (img_float - img_min) / (img_max - img_min)
    return (normalized * 255).astype(np.uint8)
