"""3D Volume Codec - Exploits inter-slice redundancy for better compression."""

import struct
import zlib
import numpy as np
from typing import List, Tuple, Optional, Dict

from .encoder import MedicalImageEncoder
from .decoder import MedicalImageDecoder

# Volume file magic number
VOLUME_MAGIC = b'MED3'
VOLUME_VERSION = 0x02  # Version 2 with B-slice support

# Slice types
I_SLICE = 0  # Intra-coded (independent)
P_SLICE = 1  # Predictive (forward prediction from previous)
B_SLICE = 2  # Bidirectional (prediction from both previous and next)


class VolumeEncoder:
    """
    3D Volume Encoder with inter-slice prediction.

    Strategy:
    - I-slice: Independent encoding (reference frame)
    - P-slice: Forward prediction from previous reconstructed slice
    - B-slice: Bidirectional prediction from both previous and next reference

    GOP Structure (example with gop_size=4):
        Display order:  I  B  B  P  I  B  B  P  ...
        Slice index:    0  1  2  3  4  5  6  7  ...

    B-slices provide better compression by using bidirectional prediction,
    similar to video codecs like H.264/HEVC.
    """

    def __init__(self):
        self.encoder = MedicalImageEncoder()
        self.decoder = MedicalImageDecoder()

    def encode(self, volume: np.ndarray, quality: int = 75,
               bit_depth: int = 16, gop_size: int = 10,
               use_b_slices: bool = True) -> bytes:
        """
        Encode a 3D volume.

        Args:
            volume: 3D numpy array (slices, height, width) as int16
            quality: Quality parameter (1-100)
            bit_depth: Bit depth of input
            gop_size: Group of Pictures size (I-slice interval)
            use_b_slices: If True, use B-slices for better compression

        Returns:
            Compressed volume as bytes
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D")

        if not 1 <= quality <= 100:
            raise ValueError(f"Quality must be 1-100, got {quality}")

        num_slices, height, width = volume.shape

        if use_b_slices and gop_size >= 3:
            return self._encode_with_b_slices(
                volume, quality, bit_depth, gop_size
            )
        else:
            return self._encode_p_only(
                volume, quality, bit_depth, gop_size
            )

    def _encode_p_only(self, volume: np.ndarray, quality: int,
                       bit_depth: int, gop_size: int) -> bytes:
        """Original P-slice only encoding."""
        num_slices, height, width = volume.shape

        encoded_slices = []
        slice_types = []
        display_orders = []  # Same as encoding order for P-only
        reconstructed_prev = None

        for i in range(num_slices):
            current_slice = volume[i]

            if i % gop_size == 0:
                slice_type = I_SLICE
                to_encode = current_slice
            else:
                slice_type = P_SLICE
                residual = current_slice.astype(np.int32) - reconstructed_prev.astype(np.int32)
                residual = np.clip(residual, -32768, 32767).astype(np.int16)
                to_encode = residual

            encoded = self.encoder.encode(to_encode, quality=quality, bit_depth=bit_depth)
            encoded_slices.append(encoded)
            slice_types.append(slice_type)
            display_orders.append(i)

            if slice_type == I_SLICE:
                reconstructed_prev = self.decoder.decode(encoded)
            else:
                decoded_residual = self.decoder.decode(encoded)
                reconstructed = reconstructed_prev.astype(np.int32) + decoded_residual.astype(np.int32)
                reconstructed = np.clip(reconstructed, -32768, 32767).astype(np.int16)
                reconstructed_prev = reconstructed

        return self._pack_volume(
            num_slices=num_slices,
            height=height,
            width=width,
            bit_depth=bit_depth,
            quality=quality,
            gop_size=gop_size,
            slice_types=slice_types,
            display_orders=display_orders,
            encoded_slices=encoded_slices,
            use_b_slices=False
        )

    def _encode_with_b_slices(self, volume: np.ndarray, quality: int,
                               bit_depth: int, gop_size: int) -> bytes:
        """
        Encode with B-slices for better compression.

        GOP structure: I B B ... B P B B ... B P ... I
        - I at start of each GOP
        - P at regular intervals within GOP
        - B between reference frames (I or P)

        Encoding order: Reference frames first, then B-frames
        """
        num_slices, height, width = volume.shape

        # Determine slice types and encoding order
        slice_plan = self._plan_gop_structure(num_slices, gop_size)

        # Storage for encoded data
        encoded_data: Dict[int, bytes] = {}
        slice_types: Dict[int, int] = {}
        reconstructed: Dict[int, np.ndarray] = {}

        # First pass: encode reference frames (I and P slices) in order
        for display_idx, stype, refs in slice_plan:
            if stype == I_SLICE:
                # Encode I-slice
                encoded = self.encoder.encode(
                    volume[display_idx], quality=quality, bit_depth=bit_depth
                )
                encoded_data[display_idx] = encoded
                slice_types[display_idx] = I_SLICE
                reconstructed[display_idx] = self.decoder.decode(encoded)

            elif stype == P_SLICE:
                # Encode P-slice (forward prediction)
                ref_idx = refs[0]  # Previous reference
                if ref_idx not in reconstructed:
                    # Reference not yet available, encode as I-slice
                    encoded = self.encoder.encode(
                        volume[display_idx], quality=quality, bit_depth=bit_depth
                    )
                    encoded_data[display_idx] = encoded
                    slice_types[display_idx] = I_SLICE
                    reconstructed[display_idx] = self.decoder.decode(encoded)
                else:
                    residual = volume[display_idx].astype(np.int32) - reconstructed[ref_idx].astype(np.int32)
                    residual = np.clip(residual, -32768, 32767).astype(np.int16)
                    encoded = self.encoder.encode(residual, quality=quality, bit_depth=bit_depth)
                    encoded_data[display_idx] = encoded
                    slice_types[display_idx] = P_SLICE

                    # Reconstruct
                    decoded_residual = self.decoder.decode(encoded)
                    recon = reconstructed[ref_idx].astype(np.int32) + decoded_residual.astype(np.int32)
                    reconstructed[display_idx] = np.clip(recon, -32768, 32767).astype(np.int16)

        # Second pass: encode B-slices (need both references)
        for display_idx, stype, refs in slice_plan:
            if stype == B_SLICE:
                ref_prev, ref_next = refs[0], refs[1]

                if ref_prev in reconstructed and ref_next in reconstructed:
                    # Bidirectional prediction: average of two references
                    prediction = (
                        reconstructed[ref_prev].astype(np.int32) +
                        reconstructed[ref_next].astype(np.int32)
                    ) // 2
                    prediction = prediction.astype(np.int16)

                    residual = volume[display_idx].astype(np.int32) - prediction.astype(np.int32)
                    residual = np.clip(residual, -32768, 32767).astype(np.int16)

                    encoded = self.encoder.encode(residual, quality=quality, bit_depth=bit_depth)
                    encoded_data[display_idx] = encoded
                    slice_types[display_idx] = B_SLICE

                    # Reconstruct (for potential future use, though B-frames aren't used as refs)
                    decoded_residual = self.decoder.decode(encoded)
                    recon = prediction.astype(np.int32) + decoded_residual.astype(np.int32)
                    reconstructed[display_idx] = np.clip(recon, -32768, 32767).astype(np.int16)
                else:
                    # Fallback to P-slice if references not available
                    if ref_prev in reconstructed:
                        residual = volume[display_idx].astype(np.int32) - reconstructed[ref_prev].astype(np.int32)
                        residual = np.clip(residual, -32768, 32767).astype(np.int16)
                        encoded = self.encoder.encode(residual, quality=quality, bit_depth=bit_depth)
                        encoded_data[display_idx] = encoded
                        slice_types[display_idx] = P_SLICE

                        decoded_residual = self.decoder.decode(encoded)
                        recon = reconstructed[ref_prev].astype(np.int32) + decoded_residual.astype(np.int32)
                        reconstructed[display_idx] = np.clip(recon, -32768, 32767).astype(np.int16)
                    else:
                        # Last resort: I-slice
                        encoded = self.encoder.encode(
                            volume[display_idx], quality=quality, bit_depth=bit_depth
                        )
                        encoded_data[display_idx] = encoded
                        slice_types[display_idx] = I_SLICE
                        reconstructed[display_idx] = self.decoder.decode(encoded)

        # Build output in display order
        encoded_slices = [encoded_data[i] for i in range(num_slices)]
        slice_type_list = [slice_types[i] for i in range(num_slices)]
        display_orders = list(range(num_slices))

        return self._pack_volume(
            num_slices=num_slices,
            height=height,
            width=width,
            bit_depth=bit_depth,
            quality=quality,
            gop_size=gop_size,
            slice_types=slice_type_list,
            display_orders=display_orders,
            encoded_slices=encoded_slices,
            use_b_slices=True
        )

    def _plan_gop_structure(self, num_slices: int, gop_size: int) -> List[Tuple[int, int, List[int]]]:
        """
        Plan GOP structure with I, P, and B slices.

        Returns list of (display_index, slice_type, reference_indices)
        Ordered by encoding priority (references first, then B-slices)
        """
        plan = []

        # P-slice interval within GOP (place P-slices evenly)
        p_interval = max(2, gop_size // 3)

        for gop_start in range(0, num_slices, gop_size):
            gop_end = min(gop_start + gop_size, num_slices)
            gop_length = gop_end - gop_start

            # Determine reference frame positions within this GOP
            ref_positions = [0]  # I-slice at start
            pos = p_interval
            while pos < gop_length:
                ref_positions.append(pos)
                pos += p_interval

            # Add I-slice
            plan.append((gop_start, I_SLICE, []))

            # Add P-slices
            for i, rel_pos in enumerate(ref_positions[1:], 1):
                abs_pos = gop_start + rel_pos
                if abs_pos < gop_end:
                    prev_ref = gop_start + ref_positions[i - 1]
                    plan.append((abs_pos, P_SLICE, [prev_ref]))

            # Add B-slices (between references)
            for i in range(len(ref_positions)):
                start_ref = gop_start + ref_positions[i]
                if i + 1 < len(ref_positions):
                    end_ref = gop_start + ref_positions[i + 1]
                else:
                    end_ref = gop_end

                # B-slices between start_ref and end_ref
                for b_pos in range(start_ref + 1, min(end_ref, gop_end)):
                    if b_pos not in [gop_start + rp for rp in ref_positions]:
                        # Find nearest references
                        prev_ref = start_ref
                        next_ref = end_ref if end_ref < num_slices else start_ref
                        plan.append((b_pos, B_SLICE, [prev_ref, next_ref]))

        return plan

    def _pack_volume(self, num_slices: int, height: int, width: int,
                     bit_depth: int, quality: int, gop_size: int,
                     slice_types: List[int], display_orders: List[int],
                     encoded_slices: List[bytes], use_b_slices: bool) -> bytes:
        """Pack encoded slices into volume bitstream."""

        # Volume header format (17 bytes):
        # Magic (4B) + Version (1B) + NumSlices (2B) + Height (2B) + Width (2B)
        # + BitDepth (1B) + Quality (1B) + GOPSize (1B) + Flags (1B) + Reserved (2B)
        flags = 0x01 if use_b_slices else 0x00

        header = struct.pack(
            '<4sBHHHBBBBH',
            VOLUME_MAGIC,
            VOLUME_VERSION,
            num_slices,
            height,
            width,
            bit_depth,
            quality,
            gop_size,
            flags,
            0  # Reserved
        )

        # Slice directory: type (1B) + display_order (2B) + offset (4B) + size (4B) = 11 bytes
        slice_directory = b''
        current_offset = 17 + num_slices * 11  # After header and directory

        for i, (stype, disp_order, encoded) in enumerate(zip(slice_types, display_orders, encoded_slices)):
            slice_directory += struct.pack('<BH II', stype, disp_order, current_offset, len(encoded))
            current_offset += len(encoded)

        # Combine all parts
        payload = header + slice_directory + b''.join(encoded_slices)

        # Add CRC32
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        return payload + crc.to_bytes(4, 'little')


class VolumeDecoder:
    """
    3D Volume Decoder with inter-slice prediction (I/P/B slices).
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
        if len(data) < 21:  # 17 header + 4 CRC minimum
            raise ValueError("Data too short for volume")

        # Verify CRC
        payload = data[:-4]
        crc_received = int.from_bytes(data[-4:], 'little')
        crc_computed = zlib.crc32(payload) & 0xFFFFFFFF

        if crc_computed != crc_received:
            raise ValueError(f"Volume CRC mismatch: expected {crc_received:08X}, got {crc_computed:08X}")

        # Parse header
        header_data = struct.unpack('<4sBHHHBBBBH', data[:17])
        magic, version, num_slices, height, width, bit_depth, quality, gop_size, flags, _ = header_data

        if magic != VOLUME_MAGIC:
            raise ValueError(f"Invalid volume magic: {magic}")

        # Handle both v1 and v2 formats
        if version == 0x01:
            return self._decode_v1(data, num_slices, height, width)
        elif version == 0x02:
            use_b_slices = (flags & 0x01) != 0
            return self._decode_v2(data, num_slices, height, width, use_b_slices, gop_size)
        else:
            raise ValueError(f"Unsupported volume version: {version}")

    def _decode_v1(self, data: bytes, num_slices: int, height: int, width: int) -> np.ndarray:
        """Decode v1 format (P-slices only, 16-byte header, 9-byte directory entries)."""
        slice_info = []
        dir_offset = 16

        for i in range(num_slices):
            stype, offset, size = struct.unpack('<BII', data[dir_offset:dir_offset+9])
            slice_info.append((i, stype, offset, size))
            dir_offset += 9

        reconstructed_slices = [None] * num_slices
        reconstructed_prev = None

        for display_idx, stype, offset, size in slice_info:
            encoded_slice = data[offset:offset+size]

            if stype == I_SLICE:
                reconstructed = self.decoder.decode(encoded_slice)
            else:
                decoded_residual = self.decoder.decode(encoded_slice)
                reconstructed = reconstructed_prev.astype(np.int32) + decoded_residual.astype(np.int32)
                reconstructed = np.clip(reconstructed, -32768, 32767).astype(np.int16)

            reconstructed_slices[display_idx] = reconstructed
            reconstructed_prev = reconstructed

        return np.stack(reconstructed_slices, axis=0)

    def _decode_v2(self, data: bytes, num_slices: int, height: int, width: int,
                   use_b_slices: bool, gop_size: int) -> np.ndarray:
        """Decode v2 format with B-slice support."""
        # Parse slice directory (11 bytes per entry)
        slice_info = []
        dir_offset = 17

        for i in range(num_slices):
            entry = struct.unpack('<BH II', data[dir_offset:dir_offset+11])
            stype, display_order, offset, size = entry
            slice_info.append((display_order, stype, offset, size))
            dir_offset += 11

        # Sort by display order
        slice_info.sort(key=lambda x: x[0])

        reconstructed: Dict[int, np.ndarray] = {}

        # Determine reference structure for decoding
        p_interval = max(2, gop_size // 3)

        # First pass: decode I and P slices
        for display_idx, stype, offset, size in slice_info:
            if stype == I_SLICE:
                encoded_slice = data[offset:offset+size]
                reconstructed[display_idx] = self.decoder.decode(encoded_slice)
            elif stype == P_SLICE:
                # Find previous reference
                prev_ref = self._find_prev_reference(display_idx, reconstructed, gop_size)
                if prev_ref is not None and prev_ref in reconstructed:
                    encoded_slice = data[offset:offset+size]
                    decoded_residual = self.decoder.decode(encoded_slice)
                    recon = reconstructed[prev_ref].astype(np.int32) + decoded_residual.astype(np.int32)
                    reconstructed[display_idx] = np.clip(recon, -32768, 32767).astype(np.int16)

        # Second pass: decode B slices
        for display_idx, stype, offset, size in slice_info:
            if stype == B_SLICE:
                prev_ref, next_ref = self._find_b_references(display_idx, reconstructed, gop_size)

                encoded_slice = data[offset:offset+size]
                decoded_residual = self.decoder.decode(encoded_slice)

                if prev_ref in reconstructed and next_ref in reconstructed:
                    # Bidirectional prediction
                    prediction = (
                        reconstructed[prev_ref].astype(np.int32) +
                        reconstructed[next_ref].astype(np.int32)
                    ) // 2
                elif prev_ref in reconstructed:
                    prediction = reconstructed[prev_ref].astype(np.int32)
                elif next_ref in reconstructed:
                    prediction = reconstructed[next_ref].astype(np.int32)
                else:
                    # Fallback: treat as I-slice residual is the actual data
                    reconstructed[display_idx] = decoded_residual
                    continue

                recon = prediction + decoded_residual.astype(np.int32)
                reconstructed[display_idx] = np.clip(recon, -32768, 32767).astype(np.int16)

        # Build output array in display order
        result = [reconstructed[i] for i in range(num_slices)]
        return np.stack(result, axis=0)

    def _find_prev_reference(self, idx: int, reconstructed: Dict[int, np.ndarray],
                              gop_size: int) -> Optional[int]:
        """Find previous reference frame for P-slice."""
        for i in range(idx - 1, -1, -1):
            if i in reconstructed:
                return i
        return None

    def _find_b_references(self, idx: int, reconstructed: Dict[int, np.ndarray],
                            gop_size: int) -> Tuple[int, int]:
        """Find previous and next reference frames for B-slice."""
        prev_ref = None
        next_ref = None

        # Find previous reference
        for i in range(idx - 1, -1, -1):
            if i in reconstructed:
                prev_ref = i
                break

        # Find next reference
        for i in range(idx + 1, idx + gop_size + 1):
            if i in reconstructed:
                next_ref = i
                break

        # Fallback
        if prev_ref is None:
            prev_ref = 0
        if next_ref is None:
            next_ref = prev_ref

        return prev_ref, next_ref

    def get_volume_info(self, data: bytes) -> dict:
        """Get volume metadata without full decoding."""
        # Try v2 header first
        try:
            header_data = struct.unpack('<4sBHHHBBBBH', data[:17])
            magic, version, num_slices, height, width, bit_depth, quality, gop_size, flags, _ = header_data
        except:
            # Fallback to v1 header
            header_data = struct.unpack('<4sBHHHBBBH', data[:16])
            magic, version, num_slices, height, width, bit_depth, quality, gop_size, _ = header_data
            flags = 0

        # Count slice types
        i_count = 0
        p_count = 0
        b_count = 0

        if version == 0x01:
            dir_offset = 16
            entry_size = 9
        else:
            dir_offset = 17
            entry_size = 11

        for i in range(num_slices):
            stype = data[dir_offset]
            if stype == I_SLICE:
                i_count += 1
            elif stype == P_SLICE:
                p_count += 1
            else:
                b_count += 1
            dir_offset += entry_size

        return {
            'version': version,
            'num_slices': num_slices,
            'height': height,
            'width': width,
            'bit_depth': bit_depth,
            'quality': quality,
            'gop_size': gop_size,
            'use_b_slices': (flags & 0x01) != 0 if version >= 2 else False,
            'i_slices': i_count,
            'p_slices': p_count,
            'b_slices': b_count,
            'total_size': len(data)
        }


def calculate_3d_compression_stats(original_volume: np.ndarray,
                                    compressed_2d_sizes: List[int],
                                    compressed_3d_size: int) -> dict:
    """
    Calculate compression statistics comparing 2D vs 3D encoding.
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
