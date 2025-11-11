#!/usr/bin/env python3
"""
@file generate_integer_gemm_instantiations.py
@brief Generate explicit instantiations for Integer GEMM kernel variants

Generates template instantiations for IntegerGemmKernel covering the full
configuration space of ISA, tile sizes, and tuning parameters.

This enables:
1. Parallel compilation across all CPU cores (64+ files)
2. Tile sweep benchmarking to gather training data
3. ML model training for runtime kernel selection

Template parameters:
- ISA: simd::AVX512VNNITag (INT8 support required)
- MR: Micro-kernel M dimension (1, 2, 4, 8, 16, 32)
- NR: Micro-kernel N dimension (fixed at 32 for Q8_0 alignment)
- UNROLL_K: K-loop unroll factor (1, 2, 4, 8, 16)
- PREFETCH_DIST: Prefetch distance (0, 1, 2, 3, 5)
- MC: M cache block size (128, 256, 512, 1024)
- KC: K cache block size (256, 512, 1024, 2048)
- NC: N cache block size (64, 128, 256, 512)

@author David Sanftenberg
@date November 11, 2025
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# Configuration Space
# ============================================================================

# ISA support (INT8 VNNI required)
ISA_TYPES = ["simd::AVX512VNNITag"]

# Micro-kernel tile sizes
MR_VALUES = [1, 2, 4, 8, 16, 32]  # M tile sizes
NR_VALUE = 32  # Fixed at 32 for Q8_0 block alignment

# K-loop unroll factors
UNROLL_K_VALUES = [1, 2, 4, 8, 16]

# Prefetch distances
PREFETCH_DIST_VALUES = [0, 1, 2, 3, 5]

# Cache blocking parameters
MC_VALUES = [128, 256, 512, 1024]  # M cache block
KC_VALUES = [256, 512, 1024, 2048]  # K cache block (must be multiple of 32)
NC_VALUES = [64, 128, 256, 512]    # N cache block

# Output configuration
NUM_FILES = 64  # Parallel compilation shards
OUTPUT_DIR = Path(__file__).parent.parent / "int8" / "generated"

# ============================================================================
# Combination Generation
# ============================================================================

def generate_all_combinations() -> List[Tuple]:
    """
    Generate all valid (ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC) combinations.
    
    Filters out invalid configurations:
    - Register pressure: MR * NR must fit in register file
    - KC alignment: KC must be multiple of 32 (Q8_0 block size)
    - Cache hierarchy: MC*KC and KC*NC should fit in L2/L3
    """
    combinations = []
    
    for isa in ISA_TYPES:
        for mr in MR_VALUES:
            nr = NR_VALUE  # Fixed at 32
            
            # Register pressure check (AVX512VNNI: 32 ZMM registers)
            # Need MR * (NR/32) ZMM registers for accumulation
            # (NR=32 fits in 1 ZMM for INT32 accumulators)
            zmm_needed = mr  # Each row needs 1 ZMM for 32×INT32
            if zmm_needed > 24:  # Leave 8 ZMM for A/B panels + misc
                continue
            
            for unroll_k in UNROLL_K_VALUES:
                for prefetch_dist in PREFETCH_DIST_VALUES:
                    for mc in MC_VALUES:
                        for kc in KC_VALUES:
                            # KC must be multiple of 32 (Q8_0 block size)
                            if kc % 32 != 0:
                                continue
                            
                            for nc in NC_VALUES:
                                # Cache hierarchy sanity checks
                                # MC*KC panel should fit in L2 (typical 256KB-1MB per core)
                                # Assume Q8_0: 1 byte per element + 2 bytes scale per 32 elements
                                bytes_per_q8_element = 1 + 2/32  # ~1.0625 bytes
                                mc_kc_bytes = mc * kc * bytes_per_q8_element
                                
                                # KC*NC panel check
                                kc_nc_bytes = kc * nc * bytes_per_q8_element
                                
                                # Skip if panels too large for L3 (typ 32MB)
                                if mc_kc_bytes > 4 * 1024 * 1024:  # 4MB limit
                                    continue
                                if kc_nc_bytes > 4 * 1024 * 1024:
                                    continue
                                
                                combinations.append((isa, mr, nr, unroll_k, prefetch_dist, mc, kc, nc))
    
    return combinations

def distribute_combinations(combinations: List, num_files: int) -> List[List]:
    """Distribute combinations across files for balanced compilation."""
    num_per_file = len(combinations) // num_files
    remainder = len(combinations) % num_files
    
    distributed = []
    idx = 0
    
    for i in range(num_files):
        count = num_per_file + (1 if i < remainder else 0)
        distributed.append(combinations[idx:idx+count])
        idx += count
    
    return distributed

# ============================================================================
# Code Generation
# ============================================================================

def generate_file_header(file_index: int, total_files: int) -> str:
    """Generate file header with includes and namespace."""
    return f"""/**
 * @file IntegerGemmInstantiations_{file_index:02d}.cpp
 * @brief Explicit template instantiations for IntegerGemmKernel (shard {file_index}/{total_files})
 *
 * AUTO-GENERATED by generate_integer_gemm_instantiations.py
 * DO NOT EDIT MANUALLY
 *
 * This file contains explicit instantiations for a subset of the integer GEMM
 * configuration space. Instantiations are distributed across {total_files} files
 * to enable parallel compilation.
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#include "../../IntegerGemmKernelTemplate.h"

namespace llaminar2 {{
namespace kernels {{
namespace gemm {{

// Explicit instantiations (shard {file_index}/{total_files})

"""

def generate_instantiation(isa: str, mr: int, nr: int, unroll_k: int, 
                          prefetch_dist: int, mc: int, kc: int, nc: int) -> str:
    """
    Generate explicit template instantiation for one kernel configuration.
    
    Example output:
    ```cpp
    template class IntegerGemmKernel<simd::AVX512VNNITag, 8, 32, 4, 2, 256, 512, 128>;
    ```
    """
    return (f"template class IntegerGemmKernel<{isa}, {mr}, {nr}, "
            f"{unroll_k}, {prefetch_dist}, {mc}, {kc}, {nc}>;\n")

def generate_file_footer() -> str:
    """Generate file footer with namespace closures."""
    return """
} // namespace gemm
} // namespace kernels
} // namespace llaminar2
"""

def generate_cmake_fragment(num_files: int, output_dir: Path) -> str:
    """Generate CMakeLists.txt fragment for including generated files."""
    # Paths relative to src/v2/ directory
    sources = [f"    kernels/cpu/gemm/int8/generated/IntegerGemmInstantiations_{i:02d}.cpp" 
               for i in range(num_files)]
    return "\n".join(sources)

# ============================================================================
# Main Generation Logic
# ============================================================================

def main():
    print("🔧 Generating Integer GEMM template instantiations...")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Shards: {NUM_FILES}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    combinations = generate_all_combinations()
    print(f"   Total combinations: {len(combinations)}")
    
    if len(combinations) == 0:
        print("   ❌ ERROR: No valid combinations generated!")
        sys.exit(1)
    
    # Statistics
    isa_counts = {}
    mr_counts = {}
    for combo in combinations:
        isa, mr = combo[0], combo[1]
        isa_counts[isa] = isa_counts.get(isa, 0) + 1
        mr_counts[mr] = mr_counts.get(mr, 0) + 1
    
    print(f"\n   Configuration breakdown:")
    print(f"   ISA: {dict(isa_counts)}")
    print(f"   MR tile sizes: {dict(sorted(mr_counts.items()))}")
    
    # Distribute across files
    distributed = distribute_combinations(combinations, NUM_FILES)
    
    # Filter out empty shards
    distributed = [shard for shard in distributed if len(shard) > 0]
    actual_num_files = len(distributed)
    
    print(f"\n   Generating {actual_num_files} non-empty shards...")
    
    # Generate each file
    for file_idx, file_combinations in enumerate(distributed):
        output_path = OUTPUT_DIR / f"IntegerGemmInstantiations_{file_idx:02d}.cpp"
        
        with open(output_path, 'w') as f:
            f.write(generate_file_header(file_idx, actual_num_files))
            
            for combo in file_combinations:
                isa, mr, nr, unroll_k, prefetch_dist, mc, kc, nc = combo
                f.write(generate_instantiation(isa, mr, nr, unroll_k, 
                                              prefetch_dist, mc, kc, nc))
            
            f.write(generate_file_footer())
        
        print(f"   ✅ Generated {output_path.name} ({len(file_combinations)} instantiations)")
    
    # Generate CMakeLists.txt fragment
    cmake_fragment_path = OUTPUT_DIR / "sources.cmake"
    with open(cmake_fragment_path, 'w') as f:
        f.write("# AUTO-GENERATED by generate_integer_gemm_instantiations.py\n")
        f.write("# Include this in src/v2/CMakeLists.txt\n\n")
        f.write("set(INTEGER_GEMM_INSTANTIATION_SOURCES\n")
        f.write(generate_cmake_fragment(actual_num_files, OUTPUT_DIR))
        f.write("\n)\n")
    
    print(f"\n✅ Generated {actual_num_files} files with {len(combinations)} total instantiations")
    print(f"   CMake fragment: {cmake_fragment_path}")
    print(f"\n📋 Next steps:")
    print(f"   1. Include sources.cmake in src/v2/CMakeLists.txt")
    print(f"   2. Add INTEGER_GEMM_INSTANTIATION_SOURCES to llaminar2_core target")
    print(f"   3. Build with -j56 for parallel compilation")
    print(f"   4. Run tile sweep benchmarks to gather training data")
    print(f"   5. Train ML model for runtime kernel selection")

if __name__ == "__main__":
    main()
