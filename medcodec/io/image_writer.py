"""Medical image writer supporting NumPy and raw formats."""

import numpy as np
from pathlib import Path


def write_medical_image(image: np.ndarray, path: str, format: str = None) -> None:
    """
    Write a medical image to file.

    Args:
        image: 2D numpy array
        path: Output file path
        format: Output format ('npy' or 'raw'). Auto-detected from extension if None.

    Raises:
        ValueError: If format is unsupported
    """
    path = Path(path)

    # Auto-detect format from extension
    if format is None:
        suffix = path.suffix.lower()
        if suffix == '.npy':
            format = 'npy'
        elif suffix == '.raw':
            format = 'raw'
        else:
            # Default to npy
            format = 'npy'
            path = path.with_suffix('.npy')

    # Ensure 2D
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got {image.ndim}D")

    if format == 'npy':
        _write_numpy(image, path)
    elif format == 'raw':
        _write_raw(image, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _write_numpy(image: np.ndarray, path: Path) -> None:
    """Write image as NumPy array."""
    np.save(str(path), image)


def _write_raw(image: np.ndarray, path: Path) -> None:
    """Write image as raw binary."""
    with open(path, 'wb') as f:
        f.write(image.tobytes())
