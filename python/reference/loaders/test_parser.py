"""
Test script for GGUF parser.

Quick validation that the parser can read a GGUF file.

Usage:
    python -m python.reference.loaders.test_parser models/qwen2.5-0.5b-instruct-q4_0.gguf
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.reference.loaders.gguf_parser import GGUFParser


def test_parser(gguf_path: str):
    """Test GGUF parser on a file"""
    print(f"\n{'='*80}")
    print(f"Testing GGUF Parser on: {gguf_path}")
    print(f"{'='*80}\n")
    
    # Parse file
    with GGUFParser(gguf_path) as parser:
        parser.parse()
        
        print(f"\n{'='*80}")
        print("HEADER INFORMATION")
        print(f"{'='*80}")
        print(f"Version: {parser.version}")
        print(f"Tensor count: {parser.tensor_count}")
        print(f"Metadata KV count: {parser.metadata_kv_count}")
        print(f"Data offset: {parser.data_offset}")
        
        print(f"\n{'='*80}")
        print("MODEL CONFIGURATION")
        print(f"{'='*80}")
        config = parser.get_config_dict()
        for key, value in sorted(config.items()):
            print(f"{key:30s} = {value}")
        
        print(f"\n{'='*80}")
        print("METADATA SAMPLE (first 20 keys)")
        print(f"{'='*80}")
        for i, (key, value) in enumerate(sorted(parser.metadata.items())):
            if i >= 20:
                print(f"... and {len(parser.metadata) - 20} more")
                break
            # Truncate long values
            if isinstance(value, (list, str)) and len(str(value)) > 60:
                value_str = str(value)[:60] + "..."
            else:
                value_str = str(value)
            print(f"{key:50s} = {value_str}")
        
        print(f"\n{'='*80}")
        print("TENSOR INFORMATION (first 20 tensors)")
        print(f"{'='*80}")
        for i, tensor in enumerate(parser.tensors):
            if i >= 20:
                print(f"... and {len(parser.tensors) - 20} more")
                break
            print(f"{tensor.name:50s} {str(tensor.shape):20s} {tensor.type.name:10s} "
                  f"offset={tensor.offset}")
        
        print(f"\n{'='*80}")
        print("TENSOR DATA SAMPLE (first tensor)")
        print(f"{'='*80}")
        if parser.tensors:
            tensor = parser.tensors[0]
            print(f"Reading data for: {tensor.name}")
            data = parser.read_tensor_data(tensor)
            print(f"Data size: {len(data)} bytes")
            print(f"First 64 bytes (hex): {data[:64].hex()}")
        
        print(f"\n{'='*80}")
        print("PARSER TEST COMPLETE ✓")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m python.reference.loaders.test_parser <gguf_file>")
        print("\nExample:")
        print("  python -m python.reference.loaders.test_parser models/qwen2.5-0.5b-instruct-q4_0.gguf")
        sys.exit(1)
    
    test_parser(sys.argv[1])
