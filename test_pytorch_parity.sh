#!/bin/bash
# Test script for PyTorch parity framework
# This creates mock snapshots and runs the parity test

set -e

echo "=== PyTorch Parity Test Script ==="

# Create mock snapshot directory
SNAPSHOT_DIR="test_pytorch_snapshots"
mkdir -p "$SNAPSHOT_DIR"

echo "Creating mock PyTorch snapshots in $SNAPSHOT_DIR..."

# Create a simple Python script to generate mock snapshots
cat > /tmp/generate_mock_snapshots.py << 'PYTHON_EOF'
import numpy as np
import sys
import os

snapshot_dir = sys.argv[1] if len(sys.argv) > 1 else "test_pytorch_snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

# Mock configuration (matches a small Qwen model)
seq_len = 5
d_model = 512
vocab_size = 151936

print(f"Generating mock snapshots:")
print(f"  Sequence length: {seq_len}")
print(f"  Hidden dimension: {d_model}")
print(f"  Vocabulary size: {vocab_size}")

# Generate mock snapshots with realistic shapes
snapshots = {
    "EMBEDDING_-1": (seq_len, d_model),
    "ATTENTION_OUTPUT_0": (seq_len, d_model),
    "FFN_DOWN_0": (seq_len, d_model),
    "FINAL_NORM_-1": (seq_len, d_model),
    "LM_HEAD_-1": (seq_len, vocab_size),
}

for name, shape in snapshots.items():
    # Generate random data with reasonable values (mean ~0, std ~1)
    data = np.random.randn(*shape).astype(np.float32) * 0.1
    filepath = os.path.join(snapshot_dir, f"{name}.npy")
    np.save(filepath, data)
    print(f"  Created: {filepath} with shape {shape}")

print(f"\nMock snapshots created in {snapshot_dir}/")
print("Note: These are random values for testing the infrastructure.")
print("For real validation, use: python python/reference/run_reference.py")
PYTHON_EOF

# Generate mock snapshots
python3 /tmp/generate_mock_snapshots.py "$SNAPSHOT_DIR"

# Set environment variables
export PYTORCH_SNAPSHOT_DIR="$SNAPSHOT_DIR"
export PYTORCH_SNAPSHOT_TOKENS="1,2,3,4,5"
export LLAMINAR_PARITY_CAPTURE=1

echo ""
echo "Environment variables set:"
echo "  PYTORCH_SNAPSHOT_DIR=$PYTORCH_SNAPSHOT_DIR"
echo "  PYTORCH_SNAPSHOT_TOKENS=$PYTORCH_SNAPSHOT_TOKENS"
echo "  LLAMINAR_PARITY_CAPTURE=$LLAMINAR_PARITY_CAPTURE"

echo ""
echo "=== Running PyTorch Parity Test ==="
echo ""

# Run the test (will skip if no model found, which is expected)
./build/test_parity_framework --gtest_filter="*PyTorchReference*" || {
    EXIT_CODE=$?
    echo ""
    echo "=== Test Result ==="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Test PASSED"
    else
        echo "Test exited with code: $EXIT_CODE"
        echo ""
        echo "Expected behavior:"
        echo "  - If no model found: Test will SKIP"
        echo "  - If model found but snapshots don't match: Test will show differences"
        echo "  - For real validation: Generate snapshots with run_reference.py"
    fi
}

echo ""
echo "=== Cleanup ==="
echo "To remove mock snapshots: rm -rf $SNAPSHOT_DIR"
