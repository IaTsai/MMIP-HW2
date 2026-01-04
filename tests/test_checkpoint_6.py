"""Checkpoint 6: CLI Interface Verification."""

import sys
import os
import subprocess
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_command(cmd, check=True):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def test_encode_decode_roundtrip():
    """Test basic encode/decode CLI roundtrip."""
    print("=" * 60)
    print("Test 1: Encode/Decode CLI Roundtrip")
    print("=" * 60)

    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            input_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.mcd', delete=False) as f:
            compressed_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            output_path = f.name

        # Create test image
        img = np.random.randint(-1024, 3071, (64, 64), dtype=np.int16)
        np.save(input_path, img)

        # Encode
        cmd = f"python encode.py --input {input_path} --output {compressed_path} --quality 75 --verbose"
        rc, stdout, stderr = run_command(cmd)

        if rc != 0:
            print(f"   Encode failed: {stderr}")
            return False

        print(f"   Encode output:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"     {line}")

        # Check compressed file exists
        assert os.path.exists(compressed_path), "Compressed file not created"
        compressed_size = os.path.getsize(compressed_path)
        print(f"   Compressed file size: {compressed_size} bytes")

        # Decode
        cmd = f"python decode.py --input {compressed_path} --output {output_path} --verbose"
        rc, stdout, stderr = run_command(cmd)

        if rc != 0:
            print(f"   Decode failed: {stderr}")
            return False

        print(f"\n   Decode output:")
        for line in stdout.split('\n'):
            if line.strip():
                print(f"     {line}")

        # Load and compare
        recovered = np.load(output_path)
        assert recovered.shape == img.shape, "Shape mismatch"
        assert recovered.dtype == np.int16, "Dtype mismatch"

        max_error = np.abs(img - recovered).max()
        print(f"\n   Max reconstruction error: {max_error}")

        # Cleanup
        os.unlink(input_path)
        os.unlink(compressed_path)
        os.unlink(output_path)

        print("✅ Encode/decode CLI roundtrip test passed")
        return True

    except Exception as e:
        print(f"❌ Encode/decode CLI roundtrip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dicom_encoding():
    """Test encoding DICOM file."""
    print("\n" + "=" * 60)
    print("Test 2: DICOM Encoding")
    print("=" * 60)

    dicom_path = "data/2_skull_ct/DICOM/I0"

    if not os.path.exists(dicom_path):
        print("   ⚠️ DICOM file not found, skipping...")
        return True

    try:
        with tempfile.NamedTemporaryFile(suffix='.mcd', delete=False) as f:
            compressed_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            output_path = f.name

        # Encode DICOM
        cmd = f"python encode.py --input {dicom_path} --output {compressed_path} --quality 75"
        rc, stdout, stderr = run_command(cmd)

        if rc != 0:
            print(f"   Encode failed: {stderr}")
            return False

        print(f"   {stdout.strip()}")

        # Decode
        cmd = f"python decode.py --input {compressed_path} --output {output_path}"
        rc, stdout, stderr = run_command(cmd)

        if rc != 0:
            print(f"   Decode failed: {stderr}")
            return False

        print(f"   {stdout.strip()}")

        # Cleanup
        os.unlink(compressed_path)
        os.unlink(output_path)

        print("✅ DICOM encoding test passed")
        return True

    except Exception as e:
        print(f"❌ DICOM encoding test failed: {e}")
        return False


def test_error_handling_cli():
    """Test CLI error handling."""
    print("\n" + "=" * 60)
    print("Test 3: CLI Error Handling")
    print("=" * 60)

    try:
        # Test 1: Non-existent input file
        cmd = "python encode.py --input nonexistent.dcm --output test.mcd --quality 75"
        rc, stdout, stderr = run_command(cmd, check=False)
        assert rc != 0, "Should fail for non-existent file"
        print("   ✓ Non-existent file: exit code != 0")

        # Test 2: Invalid quality
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_input = f.name
        np.save(temp_input, np.zeros((8, 8), dtype=np.int16))

        cmd = f"python encode.py --input {temp_input} --output test.mcd --quality 0"
        rc, stdout, stderr = run_command(cmd, check=False)
        assert rc != 0, "Should fail for quality=0"
        print("   ✓ Invalid quality (0): exit code != 0")

        cmd = f"python encode.py --input {temp_input} --output test.mcd --quality 101"
        rc, stdout, stderr = run_command(cmd, check=False)
        assert rc != 0, "Should fail for quality=101"
        print("   ✓ Invalid quality (101): exit code != 0")

        os.unlink(temp_input)

        # Test 3: Invalid compressed file
        with tempfile.NamedTemporaryFile(suffix='.mcd', delete=False) as f:
            f.write(b"INVALID_DATA")
            invalid_path = f.name

        cmd = f"python decode.py --input {invalid_path} --output test.npy"
        rc, stdout, stderr = run_command(cmd, check=False)
        assert rc != 0, "Should fail for invalid compressed file"
        print("   ✓ Invalid compressed file: exit code != 0")

        os.unlink(invalid_path)

        print("✅ CLI error handling test passed")
        return True

    except Exception as e:
        print(f"❌ CLI error handling test failed: {e}")
        return False


def test_help_messages():
    """Test help messages."""
    print("\n" + "=" * 60)
    print("Test 4: Help Messages")
    print("=" * 60)

    try:
        # Test encode --help
        rc, stdout, stderr = run_command("python encode.py --help")
        assert rc == 0, "encode.py --help should succeed"
        assert "--input" in stdout, "Help should mention --input"
        assert "--output" in stdout, "Help should mention --output"
        assert "--quality" in stdout, "Help should mention --quality"
        print("   ✓ encode.py --help works")

        # Test decode --help
        rc, stdout, stderr = run_command("python decode.py --help")
        assert rc == 0, "decode.py --help should succeed"
        assert "--input" in stdout, "Help should mention --input"
        assert "--output" in stdout, "Help should mention --output"
        print("   ✓ decode.py --help works")

        print("✅ Help messages test passed")
        return True

    except Exception as e:
        print(f"❌ Help messages test failed: {e}")
        return False


def main():
    """Run all Checkpoint 6 tests."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 6: CLI INTERFACE VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Basic roundtrip
    results.append(("Encode/Decode Roundtrip", test_encode_decode_roundtrip()))

    # Test 2: DICOM encoding
    results.append(("DICOM Encoding", test_dicom_encoding()))

    # Test 3: Error handling
    results.append(("CLI Error Handling", test_error_handling_cli()))

    # Test 4: Help messages
    results.append(("Help Messages", test_help_messages()))

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT 6 SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CHECKPOINT 6 PASSED - All tests successful!")
    else:
        print("⚠️  CHECKPOINT 6 FAILED - Some tests did not pass")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
