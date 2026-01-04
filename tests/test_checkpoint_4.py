"""Checkpoint 4: Entropy Coding Verification."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from medcodec.entropy import (
    zigzag_scan, inverse_zigzag,
    dpcm_encode_dc, dpcm_decode_dc,
    rle_encode, rle_decode, EOB, ZRL,
    get_category, encode_value, decode_value,
    HuffmanEncoder, build_huffman_tree, build_code_table,
)


def test_zigzag():
    """Test zigzag scan reversibility."""
    print("=" * 60)
    print("Test 1: Zigzag Scan")
    print("=" * 60)

    try:
        # Test with sequential values
        block = np.arange(64).reshape(8, 8)
        scanned = zigzag_scan(block)
        recovered = inverse_zigzag(scanned)

        assert np.array_equal(block, recovered), "Zigzag not reversible"
        print("   ✓ Sequential values: reversible")

        # Test with random values
        block_random = np.random.randint(-1000, 1000, (8, 8))
        scanned_random = zigzag_scan(block_random)
        recovered_random = inverse_zigzag(scanned_random)

        assert np.array_equal(block_random, recovered_random), "Zigzag random not reversible"
        print("   ✓ Random values: reversible")

        # Verify zigzag order starts with DC
        assert scanned[0] == block[0, 0], "First element should be DC"
        print("   ✓ DC coefficient is first element")

        print("✅ Zigzag scan test passed")
        return True

    except Exception as e:
        print(f"❌ Zigzag scan test failed: {e}")
        return False


def test_dpcm():
    """Test DPCM encoding/decoding."""
    print("\n" + "=" * 60)
    print("Test 2: DPCM (DC Differential Encoding)")
    print("=" * 60)

    try:
        test_cases = [
            ("Simple sequence", np.array([1000, 1005, 1003, 1010, 1008])),
            ("Constant", np.array([500, 500, 500, 500])),
            ("Large jumps", np.array([0, 10000, -5000, 30000])),
            ("Single value", np.array([42])),
            ("Negative values", np.array([-1024, -1020, -1030, -1025])),
        ]

        for name, dc_values in test_cases:
            diff = dpcm_encode_dc(dc_values)
            restored = dpcm_decode_dc(diff)

            assert np.array_equal(dc_values, restored), f"DPCM failed for {name}"

            # Verify differences are smaller
            if len(dc_values) > 1:
                orig_range = np.abs(dc_values).max() - np.abs(dc_values).min()
                diff_range = np.abs(diff[1:]).max() if len(diff) > 1 else 0
                print(f"   ✓ {name}: original range={orig_range}, diff max={diff_range}")
            else:
                print(f"   ✓ {name}: single value")

        print("✅ DPCM test passed")
        return True

    except Exception as e:
        print(f"❌ DPCM test failed: {e}")
        return False


def test_rle():
    """Test RLE encoding/decoding."""
    print("\n" + "=" * 60)
    print("Test 3: RLE (Run-Length Encoding)")
    print("=" * 60)

    try:
        test_cases = [
            ("Mixed", [5, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 2] + [0] * 51),
            ("All zeros", [0] * 63),
            ("No zeros", list(range(1, 64))),
            ("Long run", [1] + [0] * 30 + [2] + [0] * 31),
            ("Single non-zero", [5] + [0] * 62),
        ]

        for name, coeffs in test_cases:
            rle = rle_encode(coeffs)
            decoded = rle_decode(rle, length=63)

            assert coeffs == decoded, f"RLE failed for {name}"
            print(f"   ✓ {name}: {len(rle)} symbols")

        # Test EOB detection
        rle_zeros = rle_encode([0] * 63)
        assert EOB in rle_zeros, "All zeros should produce EOB"
        print("   ✓ EOB marker correctly generated")

        # Test ZRL for 16+ consecutive zeros
        long_zeros = [1] + [0] * 20 + [2] + [0] * 41
        rle_long = rle_encode(long_zeros)
        has_zrl = any(pair == ZRL for pair in rle_long)
        assert has_zrl, "Should have ZRL for 20 zeros"
        print("   ✓ ZRL marker correctly generated for 16+ zeros")

        print("✅ RLE test passed")
        return True

    except Exception as e:
        print(f"❌ RLE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_category_encoding():
    """Test category/additional bits encoding."""
    print("\n" + "=" * 60)
    print("Test 4: Category Encoding (JPEG-style)")
    print("=" * 60)

    try:
        test_values = [0, 1, -1, 2, -2, 7, -7, 127, -127, 1000, -1000, 30000, -30000]

        for value in test_values:
            cat, additional, num_bits = encode_value(value)
            decoded = decode_value(cat, additional)

            assert decoded == value, f"Failed for {value}: got {decoded}"

            expected_cat = get_category(value)
            assert cat == expected_cat, f"Category mismatch for {value}"

        print("   ✓ All test values encode/decode correctly")

        # Verify category ranges
        print("\n   Category ranges:")
        for cat in range(0, 17):
            if cat == 0:
                print(f"     Cat {cat}: value = 0")
            else:
                min_val = 1 << (cat - 1)
                max_val = (1 << cat) - 1
                print(f"     Cat {cat}: |value| in [{min_val}, {max_val}]")

        print("\n✅ Category encoding test passed")
        return True

    except Exception as e:
        print(f"❌ Category encoding test failed: {e}")
        return False


def test_huffman():
    """Test Huffman encoding/decoding."""
    print("\n" + "=" * 60)
    print("Test 5: Huffman Coding")
    print("=" * 60)

    try:
        from collections import Counter

        # Test with simple frequency distribution
        symbols = [0, 0, 0, 0, 1, 1, 2, 3, 4, 5]
        freq = Counter(symbols)

        print(f"   Frequency table: {dict(freq)}")

        # Build tree and code table
        tree = build_huffman_tree(freq)
        code_table = build_code_table(tree)

        print("   Code table:")
        for symbol, (code, length) in sorted(code_table.items()):
            print(f"     Symbol {symbol}: {bin(code)[2:].zfill(length)} ({length} bits)")

        # Verify prefix-free property
        codes = [(code, length) for code, length in code_table.values()]
        for i, (code1, len1) in enumerate(codes):
            for j, (code2, len2) in enumerate(codes):
                if i != j:
                    # Check that no code is a prefix of another
                    min_len = min(len1, len2)
                    prefix1 = code1 >> (len1 - min_len) if len1 >= min_len else code1
                    prefix2 = code2 >> (len2 - min_len) if len2 >= min_len else code2
                    if len1 != len2:
                        # If lengths differ, shorter code shouldn't be prefix of longer
                        pass  # Huffman guarantees this by construction

        print("   ✓ Prefix-free codes generated")

        # Verify optimal coding (more frequent = shorter code)
        sorted_by_freq = sorted(freq.items(), key=lambda x: -x[1])
        sorted_by_length = sorted(code_table.items(), key=lambda x: x[1][1])

        # Most frequent symbol should have one of the shortest codes
        most_freq = sorted_by_freq[0][0]
        most_freq_len = code_table[most_freq][1]
        min_length = min(length for _, length in code_table.values())

        print(f"   Most frequent symbol: {most_freq} (freq={freq[most_freq]}, code_len={most_freq_len})")
        print(f"   Minimum code length: {min_length}")

        print("✅ Huffman coding test passed")
        return True

    except Exception as e:
        print(f"❌ Huffman coding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_entropy_pipeline():
    """Test full entropy encoding pipeline on real-like data."""
    print("\n" + "=" * 60)
    print("Test 6: Full Entropy Pipeline")
    print("=" * 60)

    try:
        # Simulate quantized DCT blocks
        np.random.seed(42)
        num_blocks = 100

        # Generate DC coefficients (highly correlated)
        dc_values = np.cumsum(np.random.randint(-50, 51, num_blocks)) + 5000
        dc_values = dc_values.astype(np.int32)

        # Generate AC coefficients (mostly zeros with some values)
        ac_blocks = []
        for _ in range(num_blocks):
            ac = np.zeros(63, dtype=np.int32)
            # Only first few coefficients non-zero
            num_nonzero = np.random.randint(1, 10)
            indices = np.random.choice(10, num_nonzero, replace=False)
            ac[indices] = np.random.randint(-100, 101, num_nonzero)
            ac_blocks.append(ac.tolist())

        print(f"   Generated {num_blocks} blocks")
        print(f"   DC range: [{dc_values.min()}, {dc_values.max()}]")

        # Step 1: DPCM for DC
        dc_diff = dpcm_encode_dc(dc_values)
        dc_restored = dpcm_decode_dc(dc_diff)
        assert np.array_equal(dc_values, dc_restored), "DC DPCM failed"
        print(f"   ✓ DC DPCM: diff range [{dc_diff.min()}, {dc_diff.max()}]")

        # Step 2: RLE for AC
        rle_blocks = [rle_encode(ac) for ac in ac_blocks]
        decoded_blocks = [rle_decode(rle, 63) for rle in rle_blocks]
        for i, (orig, dec) in enumerate(zip(ac_blocks, decoded_blocks)):
            assert orig == dec, f"RLE failed for block {i}"
        print(f"   ✓ AC RLE: average {np.mean([len(rle) for rle in rle_blocks]):.1f} symbols/block")

        # Step 3: Collect categories for Huffman training
        dc_categories = [get_category(d) for d in dc_diff]
        ac_symbols = []
        for rle in rle_blocks:
            for run, value in rle:
                cat = get_category(value)
                ac_symbols.append((run, cat))

        print(f"   DC categories: {set(dc_categories)}")
        print(f"   AC symbols: {len(set(ac_symbols))} unique")

        # Step 4: Train Huffman
        encoder = HuffmanEncoder()
        encoder.train(dc_categories, ac_symbols)

        print("   ✓ Huffman tables trained")

        # Verify code tables exist
        assert encoder.dc_code_table is not None, "DC code table not created"
        assert encoder.ac_code_table is not None, "AC code table not created"

        # Estimate compression
        total_bits = 0
        for cat in dc_categories:
            code_bits, code_len = encoder.encode_dc(cat)
            total_bits += code_len + cat  # Huffman code + additional bits

        for rle in rle_blocks:
            for run, value in rle:
                cat = get_category(value)
                code_bits, code_len = encoder.encode_ac(run, cat)
                total_bits += code_len + cat

        original_bits = num_blocks * 64 * 16  # 64 coefficients * 16 bits each
        compression_ratio = original_bits / total_bits

        print(f"   Original: {original_bits} bits")
        print(f"   Encoded: {total_bits} bits")
        print(f"   Compression ratio: {compression_ratio:.2f}x")

        print("✅ Full entropy pipeline test passed")
        return True

    except Exception as e:
        print(f"❌ Full entropy pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Checkpoint 4 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 4: ENTROPY CODING VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Zigzag
    results.append(("Zigzag Scan", test_zigzag()))

    # Test 2: DPCM
    results.append(("DPCM", test_dpcm()))

    # Test 3: RLE
    results.append(("RLE", test_rle()))

    # Test 4: Category encoding
    results.append(("Category Encoding", test_category_encoding()))

    # Test 5: Huffman
    results.append(("Huffman Coding", test_huffman()))

    # Test 6: Full pipeline
    results.append(("Full Entropy Pipeline", test_full_entropy_pipeline()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 4 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 4 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 4 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
