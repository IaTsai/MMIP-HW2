"""Medical image reader supporting DICOM, NumPy, and raw formats."""

import numpy as np
from pathlib import Path


def read_medical_image(path: str, width: int = None, height: int = None,
                       bit_depth: int = 16) -> np.ndarray:
    """
    Read a medical image from various formats.

    Args:
        path: Path to the image file (.dcm, .npy, or .raw)
        width: Image width (required for .raw files)
        height: Image height (required for .raw files)
        bit_depth: Bit depth for raw files (default: 16)

    Returns:
        2D numpy array with dtype int16

    Raises:
        ValueError: If format is unsupported or parameters are missing
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.dcm':
        return _read_dicom(path)
    elif suffix == '.npy':
        return _read_numpy(path)
    elif suffix == '.raw':
        if width is None or height is None:
            raise ValueError("Width and height are required for .raw files")
        return _read_raw(path, width, height, bit_depth)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _read_dicom(path: Path) -> np.ndarray:
    """Read a DICOM file and return pixel data as int16."""
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required to read DICOM files. Install with: pip install pydicom")

    ds = pydicom.dcmread(str(path))

    # Get pixel data
    pixel_array = ds.pixel_array

    # Apply rescale slope and intercept if present (for CT Hounsfield Units)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        pixel_array = pixel_array * slope + intercept

    # Convert to int16 to preserve signed values (CT HU can be negative)
    return pixel_array.astype(np.int16)


def _read_numpy(path: Path) -> np.ndarray:
    """Read a NumPy array file."""
    data = np.load(str(path))

    # Ensure 2D
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    # Convert to int16
    return data.astype(np.int16)


def _read_raw(path: Path, width: int, height: int, bit_depth: int) -> np.ndarray:
    """Read a raw binary file."""
    # Determine dtype based on bit depth
    if bit_depth <= 8:
        dtype = np.uint8
    elif bit_depth <= 16:
        dtype = np.int16
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    # Read raw data
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)

    # Reshape to 2D
    expected_size = width * height
    if len(data) != expected_size:
        raise ValueError(f"Data size mismatch. Expected {expected_size}, got {len(data)}")

    return data.reshape((height, width)).astype(np.int16)


def get_image_info(path: str) -> dict:
    """
    Get information about a medical image file.

    Returns:
        Dictionary with 'width', 'height', 'bit_depth', 'dtype'
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.dcm':
        import pydicom
        ds = pydicom.dcmread(str(path))
        return {
            'width': ds.Columns,
            'height': ds.Rows,
            'bit_depth': ds.BitsAllocated,
            'dtype': 'int16'
        }
    elif suffix == '.npy':
        data = np.load(str(path))
        return {
            'width': data.shape[1],
            'height': data.shape[0],
            'bit_depth': data.dtype.itemsize * 8,
            'dtype': str(data.dtype)
        }
    else:
        raise ValueError(f"Cannot get info for format: {suffix}")
