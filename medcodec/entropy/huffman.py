"""Huffman coding implementation for entropy encoding."""

import heapq
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np


class HuffmanNode:
    """Node in Huffman tree."""

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

    def is_leaf(self):
        return self.left is None and self.right is None


def build_huffman_tree(freq_table: Dict) -> HuffmanNode:
    """
    Build Huffman tree from frequency table.

    Args:
        freq_table: Dictionary of symbol -> frequency

    Returns:
        Root of Huffman tree
    """
    if not freq_table:
        return None

    # Create leaf nodes
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
    heapq.heapify(heap)

    # Build tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)

    return heap[0] if heap else None


def build_code_table(tree: HuffmanNode) -> Dict:
    """
    Build code table from Huffman tree.

    Args:
        tree: Root of Huffman tree

    Returns:
        Dictionary of symbol -> (code_bits, code_length)
    """
    if tree is None:
        return {}

    code_table = {}

    def traverse(node, code, length):
        if node.is_leaf():
            code_table[node.symbol] = (code, length)
            return

        if node.left:
            traverse(node.left, (code << 1), length + 1)
        if node.right:
            traverse(node.right, (code << 1) | 1, length + 1)

    # Handle single-symbol case
    if tree.is_leaf():
        code_table[tree.symbol] = (0, 1)
    else:
        traverse(tree, 0, 0)

    return code_table


def get_category(value) -> int:
    """
    Get category (number of bits needed) for a value.

    Category 0: value = 0
    Category n: 2^(n-1) <= |value| < 2^n

    Args:
        value: Integer value (can be numpy int or Python int)

    Returns:
        Category (0-16)
    """
    if value == 0:
        return 0
    # Convert to Python int to use bit_length()
    return int(abs(value)).bit_length()


def encode_value(value: int) -> Tuple[int, int, int]:
    """
    Encode a value using category + additional bits (JPEG style).

    For positive values: additional bits = value
    For negative values: additional bits = value + (2^category - 1)

    Args:
        value: Integer value

    Returns:
        (category, additional_bits, num_additional_bits)
    """
    if value == 0:
        return (0, 0, 0)

    cat = get_category(value)

    if value > 0:
        additional = value
    else:
        # One's complement for negative values
        additional = value + (1 << cat) - 1

    return (cat, additional, cat)


def decode_value(category: int, additional: int) -> int:
    """
    Decode a value from category + additional bits.

    Args:
        category: Category (number of bits)
        additional: Additional bits value

    Returns:
        Decoded integer value
    """
    if category == 0:
        return 0

    # Check if value is negative (MSB is 0)
    threshold = 1 << (category - 1)
    if additional < threshold:
        # Negative value
        return additional - (1 << category) + 1
    else:
        # Positive value
        return additional


class HuffmanEncoder:
    """Huffman encoder for medical image codec."""

    def __init__(self):
        self.dc_code_table = None
        self.ac_code_table = None
        self.dc_tree = None
        self.ac_tree = None

    def train(self, dc_categories: List[int], ac_symbols: List[Tuple[int, int]]):
        """
        Train Huffman tables from data.

        Args:
            dc_categories: List of DC coefficient categories
            ac_symbols: List of AC (run, category) tuples
        """
        # Build DC table
        dc_freq = Counter(dc_categories)
        self.dc_tree = build_huffman_tree(dc_freq)
        self.dc_code_table = build_code_table(self.dc_tree)

        # Build AC table
        ac_freq = Counter(ac_symbols)
        self.ac_tree = build_huffman_tree(ac_freq)
        self.ac_code_table = build_code_table(self.ac_tree)

    def encode_dc(self, category: int) -> Tuple[int, int]:
        """
        Encode DC category.

        Returns:
            (code_bits, code_length)
        """
        if category in self.dc_code_table:
            return self.dc_code_table[category]
        # Fallback for unknown category
        return (category, 8)

    def encode_ac(self, run: int, category: int) -> Tuple[int, int]:
        """
        Encode AC symbol (run, category).

        Returns:
            (code_bits, code_length)
        """
        symbol = (run, category)
        if symbol in self.ac_code_table:
            return self.ac_code_table[symbol]
        # Fallback
        return ((run << 4) | category, 8)


class HuffmanDecoder:
    """Huffman decoder for medical image codec."""

    def __init__(self, dc_tree: HuffmanNode, ac_tree: HuffmanNode):
        self.dc_tree = dc_tree
        self.ac_tree = ac_tree

    def decode_dc(self, bitstream) -> int:
        """
        Decode one DC category from bitstream.

        Args:
            bitstream: BitstreamReader

        Returns:
            Category value
        """
        return self._decode_symbol(bitstream, self.dc_tree)

    def decode_ac(self, bitstream) -> Tuple[int, int]:
        """
        Decode one AC symbol from bitstream.

        Args:
            bitstream: BitstreamReader

        Returns:
            (run, category) tuple
        """
        return self._decode_symbol(bitstream, self.ac_tree)

    def _decode_symbol(self, bitstream, tree):
        """Decode one symbol using tree traversal."""
        node = tree

        if node is None:
            raise ValueError("Empty Huffman tree")

        while not node.is_leaf():
            bit = bitstream.read_bit()
            if bit == 0:
                node = node.left
            else:
                node = node.right

            if node is None:
                raise ValueError("Invalid Huffman code")

        return node.symbol


def serialize_huffman_table(code_table: Dict) -> bytes:
    """
    Serialize Huffman table for storage in bitstream.

    Format:
    - 2 bytes: number of symbols
    - For each symbol:
        - 4 bytes: symbol (as int32 for flexibility)
        - 1 byte: code length
        - ceil(code_length/8) bytes: code bits

    Args:
        code_table: Dictionary of symbol -> (code_bits, code_length)

    Returns:
        Serialized bytes
    """
    import struct

    result = bytearray()

    # Number of symbols
    num_symbols = len(code_table)
    result.extend(struct.pack('<H', num_symbols))

    for symbol, (code_bits, code_length) in code_table.items():
        # Symbol - handle tuples for AC symbols
        if isinstance(symbol, tuple):
            # Pack as (run << 8) | category
            packed = (symbol[0] << 8) | symbol[1]
            result.extend(struct.pack('<i', packed))
        else:
            result.extend(struct.pack('<i', symbol))

        # Code length
        result.append(code_length)

        # Code bits (padded to bytes)
        num_bytes = (code_length + 7) // 8
        for i in range(num_bytes):
            shift = max(0, code_length - 8 * (i + 1))
            byte_val = (code_bits >> shift) & 0xFF
            result.append(byte_val)

    return bytes(result)


def deserialize_huffman_table(data: bytes, is_ac: bool = False) -> Tuple[Dict, int]:
    """
    Deserialize Huffman table from bytes.

    Args:
        data: Serialized bytes
        is_ac: If True, interpret symbols as (run, category) tuples

    Returns:
        (code_table, bytes_consumed)
    """
    import struct

    offset = 0

    # Number of symbols
    num_symbols = struct.unpack_from('<H', data, offset)[0]
    offset += 2

    code_table = {}

    for _ in range(num_symbols):
        # Symbol
        packed = struct.unpack_from('<i', data, offset)[0]
        offset += 4

        if is_ac:
            symbol = ((packed >> 8) & 0xFF, packed & 0xFF)
        else:
            symbol = packed

        # Code length
        code_length = data[offset]
        offset += 1

        # Code bits
        num_bytes = (code_length + 7) // 8
        code_bits = 0
        for i in range(num_bytes):
            shift = max(0, code_length - 8 * (i + 1))
            code_bits |= data[offset + i] << shift
        offset += num_bytes

        code_table[symbol] = (code_bits, code_length)

    return code_table, offset
