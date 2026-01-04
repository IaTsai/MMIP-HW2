"""Medical Image Encoder - Integrates all encoding stages."""

import io
import zlib
import numpy as np
from typing import Tuple, List

from ..constants import BLOCK_SIZE, HEADER_SIZE
from ..io.bitstream import BitstreamWriter, pack_header
from ..transform import level_shift, forward_dct_block, split_into_blocks
from ..quantization import get_quantization_step, quantize
from ..entropy import (
    zigzag_scan, dpcm_encode_dc, rle_encode,
    get_category, encode_value, HuffmanEncoder,
    serialize_huffman_table, EOB
)


class MedicalImageEncoder:
    """
    Encoder for medical images.

    Pipeline:
    1. Level shift (signed → unsigned)
    2. Split into 8x8 blocks
    3. Forward DCT
    4. Quantization
    5. Zigzag scan
    6. DC: DPCM encoding
    7. AC: RLE encoding
    8. Huffman encoding
    9. Bitstream packing
    """

    def __init__(self):
        self.huffman_encoder = None

    def encode(self, image: np.ndarray, quality: int = 75,
               bit_depth: int = 16, use_dpcm: bool = True) -> bytes:
        """
        Encode a medical image.

        Args:
            image: 2D numpy array (int16)
            quality: Quality parameter (1-100)
            bit_depth: Bit depth of input image
            use_dpcm: Whether to use DC DPCM (for ablation study)

        Returns:
            Compressed data as bytes
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be 1-100, got {quality}")

        h, w = image.shape

        # Step 1: Level shift
        shifted = level_shift(image, bit_depth=bit_depth, forward=True)
        shifted = shifted.astype(np.float64)

        # Step 2: Split into blocks
        blocks, pad_info = split_into_blocks(shifted, BLOCK_SIZE)
        pad_h = pad_info['pad_h']
        pad_w = pad_info['pad_w']

        # Step 3 & 4: DCT and Quantize
        step = get_quantization_step(quality, bit_depth)
        quant_blocks = []

        for block in blocks:
            dct_block = forward_dct_block(block)
            quant_block = quantize(dct_block, step)
            quant_blocks.append(quant_block)

        # Step 5: Zigzag scan and separate DC/AC
        dc_values = []
        ac_lists = []

        for quant_block in quant_blocks:
            zigzag = zigzag_scan(quant_block)
            dc_values.append(int(zigzag[0]))  # DC coefficient
            ac_lists.append([int(x) for x in zigzag[1:]])  # AC coefficients

        # Step 6: DC DPCM (or direct encoding for ablation)
        dc_array = np.array(dc_values, dtype=np.int32)
        if use_dpcm:
            dc_diff = dpcm_encode_dc(dc_array)
        else:
            # No DPCM - encode DC values directly (for ablation study)
            dc_diff = dc_array

        # Step 7: AC RLE
        rle_blocks = [rle_encode(ac) for ac in ac_lists]

        # Collect statistics for Huffman training
        dc_categories = [get_category(d) for d in dc_diff]
        ac_symbols = []
        for rle in rle_blocks:
            for run, value in rle:
                cat = get_category(value)
                ac_symbols.append((run, cat))

        # Step 8: Train Huffman tables
        self.huffman_encoder = HuffmanEncoder()
        self.huffman_encoder.train(dc_categories, ac_symbols)

        # Step 9: Encode to bitstream
        payload = self._encode_payload(dc_diff, rle_blocks)

        # Add CRC32
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        # Pack header
        header = pack_header(
            height=h,
            width=w,
            bit_depth=bit_depth,
            quality=quality,
            pad_h=pad_h,
            pad_w=pad_w,
            data_len=len(payload) + 4,  # +4 for CRC
            use_dpcm=use_dpcm
        )

        # Combine: header + payload + CRC
        result = header + payload + crc.to_bytes(4, 'little')

        return result

    def _encode_payload(self, dc_diff: np.ndarray,
                        rle_blocks: List[List[Tuple[int, int]]]) -> bytes:
        """Encode the payload (Huffman tables + encoded data)."""
        buffer = io.BytesIO()
        writer = BitstreamWriter(buffer)

        # First, serialize Huffman tables
        dc_table_bytes = serialize_huffman_table(self.huffman_encoder.dc_code_table)
        ac_table_bytes = serialize_huffman_table(self.huffman_encoder.ac_code_table)

        # Write table lengths and data (byte-aligned)
        buffer.write(len(dc_table_bytes).to_bytes(2, 'little'))
        buffer.write(dc_table_bytes)
        buffer.write(len(ac_table_bytes).to_bytes(2, 'little'))
        buffer.write(ac_table_bytes)

        # Now write encoded data
        writer = BitstreamWriter(buffer)

        # Encode DC values
        for diff in dc_diff:
            diff = int(diff)
            cat, additional, num_bits = encode_value(diff)

            # Write Huffman code for category
            code, code_len = self.huffman_encoder.encode_dc(cat)
            writer.write_bits(code, code_len)

            # Write additional bits
            if num_bits > 0:
                writer.write_bits(additional, num_bits)

        # Encode AC values
        for rle in rle_blocks:
            for run, value in rle:
                value = int(value)
                cat = get_category(value)

                # Write Huffman code for (run, category)
                code, code_len = self.huffman_encoder.encode_ac(run, cat)
                writer.write_bits(code, code_len)

                # Write additional bits for value
                if cat > 0:
                    _, additional, num_bits = encode_value(value)
                    writer.write_bits(additional, num_bits)

        # Flush remaining bits
        writer.flush()

        return buffer.getvalue()
