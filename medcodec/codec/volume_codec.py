"""3D Volume Codec - Exploits inter-slice redundancy for better compression."""

import struct
import zlib
import numpy as np
from typing import List, Tuple, Optional

from .encoder import MedicalImageEncoder
from .decoder import MedicalImageDecoder

# Volume file magic number
VOLUME_MAGIC = b'MED3'
VOLUME_VERSION = 0x01

# Slice types
I_SLICE = 0  # Intra-coded (independent)
P_SLICE = 1  # Predictive (residual from previous)


class VolumeEncoder:
    """
    3D Volume Encoder with inter-slice prediction.

    Strategy:
    - First slice: I-slice (independent encoding)
    - Subsequent slices: P-slice (encode residual from previous reconstructed slice)

    This exploits the high correlation between adjacent CT/MR slices.
    """

    def __init__(self):
        self.encoder = MedicalImageEncoder()
        self.decoder = MedicalImageDecoder()

    def encode(self, volume: np.ndarray, quality: int = 75,
               bit_depth: int = 16, gop_size: int = 10) -> bytes:
        """
        Encode a 3D volume.

        Args:
            volume: 3D numpy array (slices, height, width) as int16
            quality: Quality parameter (1-100)
            bit_depth: Bit depth of input
            gop_size: Group of Pictures size (I-slice interval)
                      Every gop_size slices, insert an I-slice for random access

        Returns:
            Compressed volume as bytes
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D")

        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be 1-100, got {quality}")

        num_slices, height, width = volume.shape

        encoded_slices = []
        slice_types = []
        reconstructed_prev = None

        for i in range(num_slices):
            current_slice = volume[i]

            # Decide slice type: I-slice every gop_size or first slice
            if i % gop_size == 0:
                # I-slice: encode independently
                slice_type = I_SLICE
                to_encode = current_slice
            else:
                # P-slice: encode residual
                slice_type = P_SLICE
                # Compute residual (current - prediction)
                residual = current_slice.astype(np.int32) - reconstructed_prev.astype(np.int32)
                # Clip to int16 range
                residual = np.clip(residual, -32768, 32767).astype(np.int16)
                to_encode = residual

            # Encode the slice (or residual)
            encoded = self.encoder.encode(to_encode, quality=quality, bit_depth=bit_depth)
            encoded_slices.append(encoded)
            slice_types.append(slice_type)

            # Decode to get reconstructed slice for next prediction
            if slice_type == I_SLICE:
                reconstructed_prev = self.decoder.decode(encoded)
            else:
                # Decode residual and add back prediction
                decoded_residual = self.decoder.decode(encoded)
                reconstructed = reconstructed_prev.astype(np.int32) + decoded_residual.astype(np.int32)
                reconstructed = np.clip(reconstructed, -32768, 32767).astype(np.int16)
                reconstructed_prev = reconstructed

        # Build volume bitstream
        return self._pack_volume(
            num_slices=num_slices,
            height=height,
            width=width,
            bit_depth=bit_depth,
            quality=quality,
            gop_size=gop_size,
            slice_types=slice_types,
            encoded_slices=encoded_slices
        )

    def _pack_volume(self, num_slices: int, height: int, width: int,
                     bit_depth: int, quality: int, gop_size: int,
                     slice_types: List[int], encoded_slices: List[bytes]) -> bytes:
        """Pack encoded slices into volume bitstream."""

        # Volume header format:
        # Magic (4B) + Version (1B) + NumSlices (2B) + Height (2B) + Width (2B)
        # + BitDepth (1B) + Quality (1B) + GOPSize (1B) + Reserved (2B) = 16 bytes

        header = struct.pack(
            '<4sBHHHBBBH',
            VOLUME_MAGIC,
            VOLUME_VERSION,
            num_slices,
            height,
            width,
            bit_depth,
            quality,
            gop_size,
            0  # Reserved
        )

        # Slice directory: for each slice, store type (1B) + offset (4B) + size (4B)
        # Total: 9 bytes per slice
        slice_directory = b''
        current_offset = 16 + num_slices * 9  # After header and directory

        for i, (stype, encoded) in enumerate(zip(slice_types, encoded_slices)):
            slice_directory += struct.pack('<BII', stype, current_offset, len(encoded))
            current_offset += len(encoded)

        # Combine all parts
        payload = header + slice_directory + b''.join(encoded_slices)

        # Add CRC32 for entire volume
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        return payload + crc.to_bytes(4, 'little')


class VolumeDecoder:
    """
    3D Volume Decoder with inter-slice prediction.
    """

    def __init__(self):
        self.decoder = MedicalImageDecoder()

    def decode(self, data: bytes) -> np.ndarray:
        """
        Decode a compressed 3D volume.

        Args:
            data: Compressed volume bytes

        Returns:
            Reconstructed 3D numpy array (slices, height, width) as int16
        """
        # Verify minimum size
        if len(data) < 20:  # 16 header + 4 CRC minimum
            raise ValueError("Data too short for volume")

        # Verify CRC
        payload = data[:-4]
        crc_received = int.from_bytes(data[-4:], 'little')
        crc_computed = zlib.crc32(payload) & 0xFFFFFFFF

        if crc_computed != crc_received:
            raise ValueError(f"Volume CRC mismatch: expected {crc_received:08X}, got {crc_computed:08X}")

        # Parse header
        magic, version, num_slices, height, width, bit_depth, quality, gop_size, _ = \
            struct.unpack('<4sBHHHBBBH', data[:16])

        if magic != VOLUME_MAGIC:
            raise ValueError(f"Invalid volume magic: {magic}")

        if version != VOLUME_VERSION:
            raise ValueError(f"Unsupported volume version: {version}")

        # Parse slice directory
        slice_info = []
        dir_offset = 16

        for i in range(num_slices):
            stype, offset, size = struct.unpack('<BII', data[dir_offset:dir_offset+9])
            slice_info.append((stype, offset, size))
            dir_offset += 9

        # Decode slices
        reconstructed_slices = []
        reconstructed_prev = None

        for i, (stype, offset, size) in enumerate(slice_info):
            encoded_slice = data[offset:offset+size]

            if stype == I_SLICE:
                # I-slice: decode directly
                reconstructed = self.decoder.decode(encoded_slice)
            else:
                # P-slice: decode residual and add prediction
                decoded_residual = self.decoder.decode(encoded_slice)
                reconstructed = reconstructed_prev.astype(np.int32) + decoded_residual.astype(np.int32)
                reconstructed = np.clip(reconstructed, -32768, 32767).astype(np.int16)

            reconstructed_slices.append(reconstructed)
            reconstructed_prev = reconstructed

        return np.stack(reconstructed_slices, axis=0)

    def get_volume_info(self, data: bytes) -> dict:
        """Get volume metadata without full decoding."""
        magic, version, num_slices, height, width, bit_depth, quality, gop_size, _ = \
            struct.unpack('<4sBHHHBBBH', data[:16])

        # Count slice types
        i_count = 0
        p_count = 0
        dir_offset = 16

        for i in range(num_slices):
            stype, _, _ = struct.unpack('<BII', data[dir_offset:dir_offset+9])
            if stype == I_SLICE:
                i_count += 1
            else:
                p_count += 1
            dir_offset += 9

        return {
            'num_slices': num_slices,
            'height': height,
            'width': width,
            'bit_depth': bit_depth,
            'quality': quality,
            'gop_size': gop_size,
            'i_slices': i_count,
            'p_slices': p_count,
            'total_size': len(data)
        }


def calculate_3d_compression_stats(original_volume: np.ndarray,
                                    compressed_2d_sizes: List[int],
                                    compressed_3d_size: int) -> dict:
    """
    Calculate compression statistics comparing 2D vs 3D encoding.

    Args:
        original_volume: Original 3D volume
        compressed_2d_sizes: List of sizes when each slice encoded independently
        compressed_3d_size: Size of 3D encoded volume

    Returns:
        Dictionary with compression statistics
    """
    original_size = original_volume.nbytes
    total_2d_size = sum(compressed_2d_sizes)

    return {
        'original_bytes': original_size,
        'compressed_2d_bytes': total_2d_size,
        'compressed_3d_bytes': compressed_3d_size,
        'compression_ratio_2d': original_size / total_2d_size,
        'compression_ratio_3d': original_size / compressed_3d_size,
        '3d_improvement_percent': (1 - compressed_3d_size / total_2d_size) * 100,
        'bpp_2d': (total_2d_size * 8) / original_volume.size,
        'bpp_3d': (compressed_3d_size * 8) / original_volume.size,
    }
