"""Level shift for converting between signed and unsigned integers."""

import numpy as np


def level_shift(data: np.ndarray, bit_depth: int = 16, forward: bool = True) -> np.ndarray:
    """
    Apply level shift to convert between signed and unsigned values.

    For 16-bit data:
    - Forward (encoding): signed int16 [-32768, 32767] → unsigned [0, 65535]
    - Inverse (decoding): unsigned [0, 65535] → signed int16 [-32768, 32767]

    This is essential for CT images where Hounsfield Units can be negative.

    Args:
        data: Input array
        bit_depth: Bit depth of the data (default: 16)
        forward: If True, shift from signed to unsigned.
                 If False, shift from unsigned to signed.

    Returns:
        Shifted array
    """
    offset = 2 ** (bit_depth - 1)  # 32768 for 16-bit

    if forward:
        # Signed → Unsigned
        # Use int32 to avoid overflow during addition
        return data.astype(np.int32) + offset
    else:
        # Unsigned → Signed
        # Use int32 for intermediate calculation to avoid overflow
        # Then clip to valid int16 range before casting
        result = data.astype(np.float64) - offset
        result = np.clip(result, -32768, 32767)
        return np.round(result).astype(np.int16)
