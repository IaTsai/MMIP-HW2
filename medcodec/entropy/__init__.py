"""Entropy coding modules for Medical Image Codec."""

from .zigzag import zigzag_scan, inverse_zigzag
from .dpcm import dpcm_encode_dc, dpcm_decode_dc
from .rle import rle_encode, rle_decode, EOB, ZRL
from .huffman import (
    HuffmanEncoder,
    HuffmanDecoder,
    get_category,
    encode_value,
    decode_value,
    build_huffman_tree,
    build_code_table,
    serialize_huffman_table,
    deserialize_huffman_table,
)

__all__ = [
    'zigzag_scan',
    'inverse_zigzag',
    'dpcm_encode_dc',
    'dpcm_decode_dc',
    'rle_encode',
    'rle_decode',
    'EOB',
    'ZRL',
    'HuffmanEncoder',
    'HuffmanDecoder',
    'get_category',
    'encode_value',
    'decode_value',
    'build_huffman_tree',
    'build_code_table',
    'serialize_huffman_table',
    'deserialize_huffman_table',
]
