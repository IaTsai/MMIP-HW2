"""Medical Image Decoder - Integrates all decoding stages."""

import io
import zlib
import numpy as np
from typing import Tuple, List

from ..constants import BLOCK_SIZE, HEADER_SIZE
from ..io.bitstream import BitstreamReader, unpack_header
from ..transform import level_shift, inverse_dct_block, merge_blocks
from ..quantization import get_quantization_step, dequantize
from ..entropy import (
    inverse_zigzag, dpcm_decode_dc, rle_decode,
    decode_value, HuffmanDecoder, build_huffman_tree,
    deserialize_huffman_table, EOB
)


class MedicalImageDecoder:
    """
    Decoder for medical images.

    Pipeline (reverse of encoder):
    1. Unpack header
    2. Verify CRC
    3. Decode Huffman tables
    4. Huffman decode
    5. DC: DPCM decode
    6. AC: RLE decode
    7. Inverse zigzag
    8. Dequantization
    9. Inverse DCT
    10. Merge blocks
    11. Level unshift
    """

    def decode(self, data: bytes) -> np.ndarray:
        """
        Decode compressed medical image.

        Args:
            data: Compressed data bytes

        Returns:
            Reconstructed 2D numpy array (int16)

        Raises:
            ValueError: If data is invalid or corrupted
        """
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Data too short: {len(data)} bytes, need at least {HEADER_SIZE}")

        # Step 1: Unpack header
        header = unpack_header(data[:HEADER_SIZE])

        h = header['height']
        w = header['width']
        bit_depth = header['bit_depth']
        quality = header['quality']
        pad_h = header['padding_h']
        pad_w = header['padding_w']
        data_len = header['data_len']
        use_dpcm = header.get('use_dpcm', True)  # Default to True for compatibility

        # Verify data length
        expected_len = HEADER_SIZE + data_len
        if len(data) < expected_len:
            raise ValueError(f"Data truncated: expected {expected_len} bytes, got {len(data)}")

        # Extract payload and CRC
        payload = data[HEADER_SIZE:HEADER_SIZE + data_len - 4]
        crc_received = int.from_bytes(data[HEADER_SIZE + data_len - 4:HEADER_SIZE + data_len], 'little')

        # Step 2: Verify CRC
        crc_computed = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_computed != crc_received:
            raise ValueError(f"CRC mismatch: expected {crc_received:08X}, got {crc_computed:08X}")

        # Step 3: Decode Huffman tables
        offset = 0

        dc_table_len = int.from_bytes(payload[offset:offset+2], 'little')
        offset += 2
        dc_code_table, _ = deserialize_huffman_table(payload[offset:offset+dc_table_len], is_ac=False)
        offset += dc_table_len

        ac_table_len = int.from_bytes(payload[offset:offset+2], 'little')
        offset += 2
        ac_code_table, _ = deserialize_huffman_table(payload[offset:offset+ac_table_len], is_ac=True)
        offset += ac_table_len

        # Build decoder trees from code tables
        dc_tree = self._build_tree_from_table(dc_code_table)
        ac_tree = self._build_tree_from_table(ac_code_table)

        decoder = HuffmanDecoder(dc_tree, ac_tree)

        # Calculate number of blocks
        padded_h = h + pad_h
        padded_w = w + pad_w
        n_blocks_h = padded_h // BLOCK_SIZE
        n_blocks_w = padded_w // BLOCK_SIZE
        num_blocks = n_blocks_h * n_blocks_w

        # Step 4 & 5 & 6: Decode coefficients
        bitstream = BitstreamReader(payload[offset:])

        # Decode DC values
        dc_diff = []
        for _ in range(num_blocks):
            cat = decoder.decode_dc(bitstream)
            if cat == 0:
                value = 0
            else:
                additional = bitstream.read_bits(cat)
                value = decode_value(cat, additional)
            dc_diff.append(value)

        # DC DPCM decode (or direct if no DPCM was used)
        dc_array = np.array(dc_diff, dtype=np.int32)
        if use_dpcm:
            dc_values = dpcm_decode_dc(dc_array)
        else:
            dc_values = dc_array  # Direct values, no cumsum

        # Decode AC values
        ac_blocks = []
        for _ in range(num_blocks):
            rle_pairs = []
            ac_count = 0

            while ac_count < 63:
                symbol = decoder.decode_ac(bitstream)
                run, cat = symbol

                if (run, cat) == EOB:
                    rle_pairs.append(EOB)
                    break
                elif run == 15 and cat == 0:
                    # ZRL
                    rle_pairs.append((15, 0))
                    ac_count += 16
                else:
                    if cat == 0:
                        value = 0
                    else:
                        additional = bitstream.read_bits(cat)
                        value = decode_value(cat, additional)
                    rle_pairs.append((run, value))
                    ac_count += run + 1

            # RLE decode
            ac_coeffs = rle_decode(rle_pairs, length=63)
            ac_blocks.append(ac_coeffs)

        # Step 7, 8, 9: Inverse zigzag, dequantize, IDCT
        step = get_quantization_step(quality, bit_depth)
        reconstructed_blocks = []

        for dc, ac in zip(dc_values, ac_blocks):
            # Combine DC and AC
            coeffs = [int(dc)] + ac
            coeffs = np.array(coeffs, dtype=np.int32)

            # Inverse zigzag
            quant_block = inverse_zigzag(coeffs)

            # Dequantize
            dct_block = dequantize(quant_block, step)

            # Inverse DCT
            block = inverse_dct_block(dct_block)

            reconstructed_blocks.append(block)

        # Step 10: Merge blocks
        pad_info = {
            'original_h': h,
            'original_w': w,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'n_blocks_h': n_blocks_h,
            'n_blocks_w': n_blocks_w
        }

        # The merge_blocks expects the original shape to crop to
        shifted = merge_blocks(reconstructed_blocks, (h, w), pad_info)

        # Step 11: Level unshift
        result = level_shift(shifted, bit_depth=bit_depth, forward=False)

        return result

    def _build_tree_from_table(self, code_table):
        """Build Huffman tree from code table for decoding."""
        from ..entropy.huffman import HuffmanNode

        if not code_table:
            return None

        root = HuffmanNode()

        for symbol, (code, length) in code_table.items():
            node = root

            for i in range(length - 1, -1, -1):
                bit = (code >> i) & 1

                if bit == 0:
                    if node.left is None:
                        node.left = HuffmanNode()
                    node = node.left
                else:
                    if node.right is None:
                        node.right = HuffmanNode()
                    node = node.right

            node.symbol = symbol

        return root
