"""DCT (Discrete Cosine Transform) implementation using matrix multiplication."""

import numpy as np


def create_dct_matrix(N: int = 8) -> np.ndarray:
    """
    Generate the 1D DCT-II transform matrix of size N x N.

    The DCT matrix T has elements:
        T[i, j] = c[i] * cos((2j + 1) * i * pi / (2N))

    where:
        c[0] = 1/sqrt(N)
        c[k] = sqrt(2/N) for k > 0

    This matrix is orthogonal: T @ T.T = I

    Args:
        N: Size of the transform (default: 8)

    Returns:
        N x N DCT transform matrix
    """
    T = np.zeros((N, N))
    c0 = 1 / np.sqrt(N)
    ck = np.sqrt(2 / N)

    for i in range(N):
        for j in range(N):
            if i == 0:
                coeff = c0
            else:
                coeff = ck
            T[i, j] = coeff * np.cos((2 * j + 1) * i * np.pi / (2 * N))

    return T


# Pre-compute DCT matrices for 8x8 blocks (most common case)
DCT_MATRIX_8 = create_dct_matrix(8)
DCT_MATRIX_8_T = DCT_MATRIX_8.T


def forward_dct_block(block: np.ndarray) -> np.ndarray:
    """
    Perform 2D DCT on an 8x8 block using matrix multiplication.

    Formula: D = T @ B @ T'

    where:
        D = DCT coefficients
        T = DCT transform matrix
        B = Input block
        T' = Transpose of T

    Args:
        block: 8x8 input block

    Returns:
        8x8 DCT coefficient block
    """
    return DCT_MATRIX_8 @ block @ DCT_MATRIX_8_T


def inverse_dct_block(dct_block: np.ndarray) -> np.ndarray:
    """
    Perform 2D inverse DCT on an 8x8 block.

    Formula: B = T' @ D @ T

    Args:
        dct_block: 8x8 DCT coefficient block

    Returns:
        8x8 reconstructed block
    """
    return DCT_MATRIX_8_T @ dct_block @ DCT_MATRIX_8
