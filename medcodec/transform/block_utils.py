"""Block processing utilities for image codec."""

import numpy as np
from typing import List, Tuple
from ..constants import BLOCK_SIZE


def split_into_blocks(image: np.ndarray, block_size: int = BLOCK_SIZE) -> Tuple[List[np.ndarray], dict]:
    """
    Split image into non-overlapping blocks with padding if necessary.

    Args:
        image: 2D input image
        block_size: Size of each block (default: 8)

    Returns:
        Tuple of (list of blocks, padding info dict)
        Blocks are returned in row-major order.
    """
    h, w = image.shape

    # Calculate padding needed
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    # Pad image if necessary (replicate edge values)
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        padded = image

    padded_h, padded_w = padded.shape
    n_blocks_h = padded_h // block_size
    n_blocks_w = padded_w // block_size

    # Extract blocks
    blocks = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            y_start = i * block_size
            x_start = j * block_size
            block = padded[y_start:y_start + block_size,
                          x_start:x_start + block_size].astype(np.float64)
            blocks.append(block)

    pad_info = {
        'original_h': h,
        'original_w': w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'n_blocks_h': n_blocks_h,
        'n_blocks_w': n_blocks_w
    }

    return blocks, pad_info


def merge_blocks(blocks: List[np.ndarray], original_shape: Tuple[int, int],
                 pad_info: dict) -> np.ndarray:
    """
    Merge blocks back into an image.

    Args:
        blocks: List of blocks in row-major order
        original_shape: Original (height, width) tuple
        pad_info: Padding info from split_into_blocks

    Returns:
        Reconstructed image with original dimensions
    """
    block_size = blocks[0].shape[0]
    n_blocks_h = pad_info['n_blocks_h']
    n_blocks_w = pad_info['n_blocks_w']

    # Create padded image
    padded_h = n_blocks_h * block_size
    padded_w = n_blocks_w * block_size
    padded = np.zeros((padded_h, padded_w), dtype=np.float64)

    # Place blocks
    for idx, block in enumerate(blocks):
        i = idx // n_blocks_w
        j = idx % n_blocks_w
        y_start = i * block_size
        x_start = j * block_size
        padded[y_start:y_start + block_size,
               x_start:x_start + block_size] = block

    # Crop to original size
    original_h, original_w = original_shape
    return padded[:original_h, :original_w]
