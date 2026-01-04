"""Bitstream reader and writer for binary data."""

import struct
from ..constants import MAGIC, VERSION, HEADER_FORMAT, HEADER_SIZE


class BitstreamWriter:
    """Bit-level writer for binary data."""

    def __init__(self, f):
        """
        Initialize bitstream writer.

        Args:
            f: File object opened in binary write mode
        """
        self.f = f
        self.accumulator = 0
        self.bit_count = 0

    def write_bit(self, bit: int) -> None:
        """Write a single bit."""
        self.accumulator = (self.accumulator << 1) | (bit & 1)
        self.bit_count += 1
        if self.bit_count == 8:
            self._flush_byte()

    def write_bits(self, value: int, num_bits: int) -> None:
        """Write multiple bits from value (MSB first)."""
        for i in range(num_bits - 1, -1, -1):
            bit = (value >> i) & 1
            self.write_bit(bit)

    def write_byte(self, byte_val: int) -> None:
        """Write a single byte."""
        self.write_bits(byte_val, 8)

    def write_uint16(self, value: int) -> None:
        """Write a 16-bit unsigned integer."""
        self.write_bits(value, 16)

    def write_uint32(self, value: int) -> None:
        """Write a 32-bit unsigned integer."""
        self.write_bits(value, 32)

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        for byte in data:
            self.write_byte(byte)

    def _flush_byte(self) -> None:
        """Flush accumulated bits as a byte."""
        self.f.write(bytes([self.accumulator]))
        self.accumulator = 0
        self.bit_count = 0

    def flush(self) -> None:
        """Pad remaining bits with 0s to reach byte boundary."""
        if self.bit_count > 0:
            padding = 8 - self.bit_count
            self.accumulator = (self.accumulator << padding)
            self.f.write(bytes([self.accumulator]))
            self.accumulator = 0
            self.bit_count = 0


class BitstreamReader:
    """Bit-level reader for binary data."""

    def __init__(self, data_bytes: bytes):
        """
        Initialize bitstream reader.

        Args:
            data_bytes: Binary data to read from
        """
        self.data = data_bytes
        self.byte_ptr = 0
        self.bit_ptr = 0  # 0 to 7, current bit index (MSB=0)

    def read_bit(self) -> int:
        """Read a single bit."""
        if self.byte_ptr >= len(self.data):
            raise EOFError("End of bitstream")

        byte = self.data[self.byte_ptr]
        bit = (byte >> (7 - self.bit_ptr)) & 1

        self.bit_ptr += 1
        if self.bit_ptr == 8:
            self.bit_ptr = 0
            self.byte_ptr += 1

        return bit

    def read_bits(self, num_bits: int) -> int:
        """Read multiple bits and return as integer."""
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value

    def read_byte(self) -> int:
        """Read a single byte."""
        return self.read_bits(8)

    def read_uint16(self) -> int:
        """Read a 16-bit unsigned integer."""
        return self.read_bits(16)

    def read_uint32(self) -> int:
        """Read a 32-bit unsigned integer."""
        return self.read_bits(32)

    def bytes_remaining(self) -> int:
        """Return number of full bytes remaining."""
        remaining = len(self.data) - self.byte_ptr
        if self.bit_ptr > 0:
            remaining -= 1
        return max(0, remaining)


def pack_header(height: int, width: int, bit_depth: int, quality: int,
                pad_h: int, pad_w: int, data_len: int,
                use_dpcm: bool = True) -> bytes:
    """
    Pack metadata into a 20-byte binary header.

    Args:
        height: Image height
        width: Image width
        bit_depth: Original bit depth (12/14/16)
        quality: Quality parameter (1-100)
        pad_h: Vertical padding applied
        pad_w: Horizontal padding applied
        data_len: Length of payload in bytes
        use_dpcm: Whether DC DPCM was used (stored in reserved byte)

    Returns:
        20-byte header as bytes
    """
    # Use reserved byte bit 0 for DPCM flag
    flags = 0x01 if use_dpcm else 0x00

    return struct.pack(
        HEADER_FORMAT,
        MAGIC,          # Magic number
        VERSION,        # Version
        height,
        width,
        bit_depth,
        quality,
        pad_h,
        pad_w,
        data_len,
        flags           # Reserved/Flags byte
    )


def unpack_header(header_bytes: bytes) -> dict:
    """
    Unpack the 20-byte binary header.

    Args:
        header_bytes: 20-byte header data

    Returns:
        Dictionary with header fields

    Raises:
        ValueError: If header is invalid
    """
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Header size mismatch. Expected {HEADER_SIZE}, got {len(header_bytes)}")

    magic, ver, h, w, bd, q, ph, pw, dlen, flags = struct.unpack(HEADER_FORMAT, header_bytes)

    if magic != MAGIC:
        raise ValueError(f"Invalid file signature: {magic}. Expected {MAGIC}")

    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")

    return {
        'height': h,
        'width': w,
        'bit_depth': bd,
        'quality': q,
        'padding_h': ph,
        'padding_w': pw,
        'data_len': dlen,
        'use_dpcm': bool(flags & 0x01)  # Extract DPCM flag from bit 0
    }
