"""Constants for Medical Image Codec."""

import struct

# Magic number: 'MEDC' (Medical Codec)
MAGIC = b'MEDC'
VERSION = 0x01

# Block size for DCT
BLOCK_SIZE = 8

# Header format (Little-endian, 20 bytes total)
# 4s: Magic (4B), B: Version (1B), H: Height (2B), H: Width (2B)
# B: Bit depth (1B), B: Quality (1B), H: Pad_H (2B), H: Pad_W (2B)
# I: Data Length (4B), B: Reserved (1B)
HEADER_FORMAT = '<4sBHHBBHHIB'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 20 bytes

# Entropy coding markers
MARKER_EOB = (0, 0)   # End of Block
MARKER_ZRL = (15, 0)  # Zero Run Length (16 consecutive zeros)

# Default bit depth for medical images
DEFAULT_BIT_DEPTH = 16
