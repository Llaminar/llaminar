#!/usr/bin/env python3
"""
Helper script to extract .npz archives to individual .npy files.

The C++ test framework can easily load individual .npy files but parsing
ZIP archives in C++ adds complexity. This script bridges the gap.

Usage:
    python tests/npz_to_npy.py snapshots.npz output_dir/
"""

import sys
import argparse
from pathlib import Path
import numpy as np


def extract_npz(npz_path: Path, output_dir: Path, verbose: bool = False):
    """
    Extract .npz archive to individual .npy files.
    
    Args:
        npz_path: Path to .npz file
        output_dir: Directory to write .npy files
        verbose: Print progress
    """
    if not npz_path.exists():
        print(f"Error: {npz_path} not found", file=sys.stderr)
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading {npz_path}...")
    
    try:
        data = np.load(npz_path)
        
        if verbose:
            print(f"Found {len(data.files)} arrays")
        
        for key in data.files:
            array = data[key]
            output_path = output_dir / f"{key}.npy"
            
            if verbose:
                print(f"  {key:30} shape={str(array.shape):20} → {output_path.name}")
            
            np.save(output_path, array)
        
        if verbose:
            print(f"\n✓ Extracted {len(data.files)} arrays to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error extracting {npz_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract .npz archive to individual .npy files for C++ loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to same directory as .npz
  python tests/npz_to_npy.py snapshots.npz
  
  # Extract to specific directory
  python tests/npz_to_npy.py snapshots.npz extracted_snapshots/
  
  # Verbose output
  python tests/npz_to_npy.py snapshots.npz -v
"""
    )
    
    parser.add_argument(
        "npz_file",
        type=Path,
        help="Path to .npz file to extract"
    )
    
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory (default: same as .npz file)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.npz_file.parent / args.npz_file.stem
    
    success = extract_npz(args.npz_file, args.output_dir, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
