#!/usr/bin/env python3
"""
@file generate_cuda_gemm_variants.py
@brief Auto-generate CUDA GEMM kernel variant instantiations

This script generates multiple .cu files, each containing a handful of explicit
template instantiations of iq4nl_gemm_kernel_variant<...>. This allows:
  1. Parallel compilation (nvcc compiles 32-64 files simultaneously)
  2. Incremental builds (changing one variant doesn't rebuild all)
  3. Faster linking (smaller object files)

Pattern adapted from CPU GEMM microkernel generator.

@author David Sanftenberg
"""

import os
import sys
import itertools
from typing import List, Tuple, Dict

# =============================================================================
# Configuration Space
# =============================================================================

# Tile dimensions - MAXIMIZED for RTX 3090 hardware capabilities
# RTX 3090 (sm_86): 100 KB shared memory per SM, 48 KB default per block (100 KB with opt-in)
# Max threads per block: 1024
# Strategy: Cover full range from tiny (8×8) for small ops to huge (256×256) for max throughput
TILE_M_VALUES = [8, 16, 32, 64, 128, 256]
TILE_N_VALUES = [8, 16, 32, 64, 128, 256]

# TILE_K: Full range for different arithmetic intensity and occupancy trade-offs
# Small (8, 16): High occupancy, low shared memory
# Medium (24, 32): Balanced
# Large (48, 64, 96, 128): High arithmetic intensity, may require opt-in shared memory
TILE_K_VALUES = [8, 16, 24, 32, 48, 64, 96, 128]

# Thread block configuration - expanded to support larger tiles
# Max 1024 threads per block = 32×32
BLOCK_THREADS_X_VALUES = [4, 8, 16, 32]
BLOCK_THREADS_Y_VALUES = [4, 8, 16, 32]

# Work items per thread (register pressure)
# Larger values (16) enable very large tiles with fewer threads
WORK_ITEMS_X_VALUES = [1, 2, 4, 8, 16]
WORK_ITEMS_Y_VALUES = [1, 2, 4, 8, 16]

# Prefetching stages (0 = no prefetch, 1-2 = double/triple buffering)
PREFETCH_STAGES_VALUES = [0, 1, 2]

# Shared memory optimizations
USE_TRANSPOSE_VALUES = [False, True]

# Vectorized loads (1 = scalar, 2/4 = vector)
VECTORIZE_VALUES = [1, 2, 4]

# Atom configuration (MMA instruction shapes)
# ATOM_TYPE: 0 = SM80_16x8x16 (K=16), 1 = SM80_16x8x8 (K=8)
# Different K dimensions may perform better on different shapes
ATOM_TYPE_VALUES = [0, 1]

# ATOM_LAYOUT: How MMA atoms tile within a thread block tile
# [M, N, K] repetition of the MMA atom (16×8×K)
# Higher values increase tensor core utilization on large tiles
# but may harm small tiles due to register pressure
ATOM_LAYOUT_M_VALUES = [1, 2, 4]  # 1×16=16, 2×16=32, 4×16=64 rows
ATOM_LAYOUT_N_VALUES = [1, 2, 4]  # 1×8=8, 2×8=16, 4×8=32 cols
ATOM_LAYOUT_K_VALUES = [1]        # Always 1 for SM80 (K handled by atom type)

# Output configuration
# PARALLELISM: More files = better nvcc parallelization
# With 56 cores and 37,380 configs, target ~100-150 configs per file
# This gives us ~250-370 files, ensuring all cores stay busy
NUM_FILES = 250  # Split across 250 .inc files for maximum parallelism
OUTPUT_DIR = "generated"
OUTPUT_CMAKE_FILE = os.path.join(OUTPUT_DIR, "sources.cmake")

# =============================================================================
# CUDA Device Capabilities
# =============================================================================

# Shared memory limits by compute capability
SM_70_SHARED_MEM_LIMIT = 48 * 1024    # V100 (48KB default)
SM_80_SHARED_MEM_LIMIT = 164 * 1024   # A100 (164KB with opt-in)
SM_86_SHARED_MEM_LIMIT = 100 * 1024   # RTX 3090 (100KB per SM, measured)

# Use full hardware capability - configs > 48KB will require cudaFuncSetAttribute
# We'll generate ALL valid configs and handle opt-in at runtime
SHARED_MEM_LIMIT = SM_86_SHARED_MEM_LIMIT
SHARED_MEM_DEFAULT_LIMIT = 48 * 1024  # Configs <= 48KB work without opt-in

# Register file size (64K 32-bit registers per SM on sm_70)
MAX_REGISTERS_PER_THREAD = 255

# =============================================================================
# Configuration Validation
# =============================================================================

def estimate_shared_memory(tile_m: int, tile_n: int, tile_k: int,
                           prefetch_stages: int, use_transpose: bool) -> int:
    """
    Estimate shared memory usage for a kernel variant.
    
    Shared memory layout:
      - s_A[TILE_M][TILE_K_PADDED] - A tile (FP32) with +1 padding
      - s_B[TILE_N][TILE_K_PADDED] - B tile (FP32, or transposed) with +1 padding
      - Prefetch buffers multiply by (1 + prefetch_stages)
    
    NOTE: We add +1 padding to TILE_K to avoid shared memory bank conflicts
    when TILE_K is a power of 2 (e.g., 32).
    
    Returns: Shared memory bytes
    """
    bytes_per_float = 4
    
    # Add +1 padding to avoid bank conflicts (always applied in kernel)
    tile_k_padded = tile_k + 1
    
    # Base allocation with padding
    s_A_bytes = tile_m * tile_k_padded * bytes_per_float
    s_B_bytes = tile_n * tile_k_padded * bytes_per_float
    
    # Double/triple buffering for prefetch
    buffering_factor = 1 + prefetch_stages
    
    total_bytes = (s_A_bytes + s_B_bytes) * buffering_factor
    
    return total_bytes

def estimate_registers(tile_m: int, tile_n: int, work_m: int,
                       work_n: int, vectorize: int) -> int:
    """
    Estimate register usage per thread.
    
    Registers used for:
      - Accumulator array: work_m * work_n * 1 (FP32)
      - Vectorized loads: vectorize elements
      - Indexing, loop counters: ~10-20 registers
    
    Returns: Estimated registers per thread
    """
    # Accumulator registers (C tile fragment)
    accum_regs = work_m * work_n
    
    # Vectorized load buffers
    vector_regs = vectorize * 2  # A and B vectors
    
    # Indexing and temps
    overhead_regs = 20
    
    total_regs = accum_regs + vector_regs + overhead_regs
    
    return total_regs

def is_valid_config(tile_m: int, tile_n: int, tile_k: int,
                    threads_m: int, threads_n: int,
                    work_m: int, work_n: int,
                    prefetch_stages: int, use_transpose: bool,
                    vectorize: int,
                    atom_type: int = 0, atom_layout_m: int = 1,
                    atom_layout_n: int = 1, atom_layout_k: int = 1) -> bool:
    """
    Validate configuration against hardware constraints and kernel requirements.
    
    Constraints:
      1. Shared memory <= 48KB (sm_70)
      2. Registers per thread <= 255
      3. Block threads <= 1024
      4. Tile must equal (block_threads * work_items) - CRITICAL for kernel
      5. Vectorize must divide tile_k for alignment
      6. Atom layout must be compatible with tile dimensions
    
    Note: threads_m/work_m are M dimension (rows)
          threads_n/work_n are N dimension (cols)
    """
    # CRITICAL: These must match kernel static_assert requirements
    # tile_m == threads_m * work_m
    # tile_n == threads_n * work_n
    if tile_m != threads_m * work_m:
        return False
    if tile_n != threads_n * work_n:
        return False
    
    # Atom compatibility checks
    # Atom type 0: SM80_16x8x16 (M=16, N=8, K=16)
    # Atom type 1: SM80_16x8x8  (M=16, N=8, K=8)
    atom_m_dim = 16
    atom_n_dim = 8
    atom_k_dim = 16 if atom_type == 0 else 8
    
    # Tile must be divisible by atom layout dimensions
    if tile_m % (atom_m_dim * atom_layout_m) != 0:
        return False
    if tile_n % (atom_n_dim * atom_layout_n) != 0:
        return False
    if tile_k % (atom_k_dim * atom_layout_k) != 0:
        return False
    
    # Check shared memory
    shared_mem = estimate_shared_memory(tile_m, tile_n, tile_k,
                                        prefetch_stages, use_transpose)
    if shared_mem > SHARED_MEM_LIMIT:
        return False
    
    # Check registers
    regs = estimate_registers(tile_m, tile_n, work_m, work_n, vectorize)
    if regs > MAX_REGISTERS_PER_THREAD:
        return False
    
    # Check block size
    block_threads = threads_m * threads_n
    if block_threads > 1024 or block_threads == 0:
        return False
    
    # Check vectorization alignment
    if tile_k % vectorize != 0:
        return False
    
    return True

# =============================================================================
# Configuration Generation
# =============================================================================

def generate_all_configs() -> List[Tuple]:
    """
    Generate all valid kernel configurations.
    
    Returns: List of tuples (tile_m, tile_n, tile_k, threads_m, threads_n,
                              work_m, work_n, prefetch_stages, use_transpose, vectorize,
                              atom_type, atom_layout_m, atom_layout_n, atom_layout_k)
    Note: threads_m/work_m correspond to M dimension (rows, Y)
          threads_n/work_n correspond to N dimension (cols, X)
    """
    valid_configs = []
    
    # Nested loops over parameter space
    for atom_type in ATOM_TYPE_VALUES:
        for atom_m in ATOM_LAYOUT_M_VALUES:
            for atom_n in ATOM_LAYOUT_N_VALUES:
                for atom_k in ATOM_LAYOUT_K_VALUES:
                    for tile_m in TILE_M_VALUES:
                        for tile_n in TILE_N_VALUES:
                            for tile_k in TILE_K_VALUES:
                                for block_threads_x in BLOCK_THREADS_X_VALUES:
                                    for block_threads_y in BLOCK_THREADS_Y_VALUES:
                                        for work_items_x in WORK_ITEMS_X_VALUES:
                                            for work_items_y in WORK_ITEMS_Y_VALUES:
                                                for prefetch_stages in PREFETCH_STAGES_VALUES:
                                                    for use_transpose in USE_TRANSPOSE_VALUES:
                                                        for vectorize in VECTORIZE_VALUES:
                                                            # CRITICAL: Map X/Y to M/N correctly
                                                            # M dimension (rows) uses Y values
                                                            # N dimension (cols) uses X values
                                                            config = (tile_m, tile_n, tile_k,
                                                                     block_threads_y, block_threads_x,  # threads_m, threads_n
                                                                     work_items_y, work_items_x,        # work_m, work_n
                                                                     prefetch_stages, use_transpose,
                                                                     vectorize,
                                                                     atom_type, atom_m, atom_n, atom_k)
                                                            
                                                            if is_valid_config(*config):
                                                                valid_configs.append(config)
    
    return valid_configs

def select_top_configs(all_configs: List[Tuple], max_count: int = 200) -> List[Tuple]:
    """
    Select top configurations using analytical performance model.
    
    Scoring criteria:
      1. Arithmetic intensity (FLOP/byte)
      2. Occupancy potential (blocks per SM)
      3. Balance between tile dimensions
      4. Prefetching effectiveness
      5. Atom layout efficiency (tensor core utilization)
    
    Returns: Top N configurations
    """
    def score_config(config):
        (tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n, 
         prefetch, transpose, vec, atom_type, atom_m, atom_n, atom_k) = config
        
        # Arithmetic intensity: more FLOPs per memory access
        flops = tile_m * tile_n * tile_k * 2  # GEMM FLOPs
        bytes_loaded = (tile_m * tile_k + tile_n * tile_k) * 4  # A + B tiles
        arithmetic_intensity = flops / bytes_loaded
        
        # Occupancy: more threads per block is better
        threads_per_block = threads_m * threads_n
        occupancy_score = threads_per_block / 1024.0  # Normalize to max
        
        # Work per thread: larger work items reduce overhead
        work_per_thread = work_m * work_n
        
        # Prefetching: double-buffering helps latency hiding
        prefetch_score = 1.0 + (0.2 * prefetch)
        
        # Vectorization: wider loads are faster
        vectorize_score = vec / 4.0  # Normalize to max
        
        # Balance: prefer square-ish tiles
        aspect_ratio = min(tile_m, tile_n) / max(tile_m, tile_n)
        balance_score = aspect_ratio
        
        # Atom layout: larger layouts may improve tensor core utilization
        # but also increase register pressure
        atom_layout_score = (atom_m * atom_n) / 16.0  # Max is 4×4=16
        
        # Combined score
        score = (arithmetic_intensity * 10.0 +
                 occupancy_score * 50.0 +
                 work_per_thread * 5.0 +
                 prefetch_score * 20.0 +
                 vectorize_score * 10.0 +
                 balance_score * 15.0 +
                 atom_layout_score * 5.0)
        
        return score
    
    # Score all configs
    scored_configs = [(score_config(cfg), cfg) for cfg in all_configs]
    
    # Sort by score descending
    scored_configs.sort(reverse=True, key=lambda x: x[0])
    
    # Take top N
    top_configs = [cfg for (score, cfg) in scored_configs[:max_count]]
    
    return top_configs

# =============================================================================
# Code Generation
# =============================================================================

def generate_file_header(file_index: int) -> str:
    """Generate file header - just comments for .inc files."""
    return f"""/**
 * @file CudaGemmVariants_{file_index:02d}.inc
 * @brief CUDA GEMM kernel variant dispatches (shard {file_index}/{NUM_FILES})
 * 
 * AUTO-GENERATED by generate_cuda_gemm_variants.py
 * DO NOT EDIT MANUALLY
 * 
 * This file is included inside launchIQ4NLGemmVariant() in CudaGemmVariants.cu
 * 
 * @author David Sanftenberg
 */

"""

def generate_config_instantiation(config: Tuple) -> str:
    """Generate LAUNCH_VARIANT macro call for a configuration."""
    tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n, prefetch, transpose, vec = config
    
    transpose_str = "true" if transpose else "false"
    
    return f"""// Tile: {tile_m}×{tile_n}×{tile_k}, Threads: {threads_m}×{threads_n}, Work: {work_m}×{work_n}, Prefetch: {prefetch}, Transpose: {transpose_str}, Vec: {vec}
LAUNCH_VARIANT({tile_m}, {tile_n}, {tile_k}, {threads_m}, {threads_n}, {work_m}, {work_n}, {prefetch}, {transpose_str}, {vec});
"""

def generate_file_footer() -> str:
    """Generate file footer - nothing needed for .inc files."""
    return ""

def write_variant_file(file_index: int, configs: List[Tuple], output_dir: str) -> str:
    """
    Write a single .cu file with kernel instantiations + registry.
    Uses __attribute__((constructor)) pattern like CPU microkernel.
    
    Each file:
    1. Explicitly instantiates ~150 kernel templates
    2. Creates launcher wrapper functions
    3. Registers launchers with CudaGemmKernelRegistry
    
    Returns: Filename
    """
    filename = f"CudaGemmVariants_{file_index:02d}.cu"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        # Header
        f.write(f"""/**
 * @file {filename}
 * @brief Auto-generated CUDA GEMM kernel variant instantiations (shard {file_index}/{NUM_FILES})
 * 
 * AUTO-GENERATED by generate_cuda_gemm_variants.py
 * DO NOT EDIT MANUALLY
 */

#include "../CudaGemmKernelRegistry.h"
#include "../IQ4_NL_BlockDecoder.h"
#include <cuda_runtime.h>

// Include kernel template definition (needed for implicit instantiation)
// Note: This is safe because each .cu file compiles separately
#include "../CudaGemmVariantsBaseline.cu"

// Force-link symbol (called by CudaGemmKernelInit.cu)
extern "C" void forceLink_CudaGemmVariants_{file_index:02d}() {{}}

// Launcher wrappers + registration
namespace llaminar2 {{
namespace cuda {{

""")
        
        # For each config: instantiate + create launcher + register
        for config in configs:
            (tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n, 
             prefetch, transpose, vec, atom_type, atom_m, atom_n, atom_k) = config
            transpose_str = "true" if transpose else "false"
            
            # Unique name for this variant
            variant_name = f"variant_{tile_m}_{tile_n}_{tile_k}_{threads_m}_{threads_n}_{work_m}_{work_n}_{prefetch}_{1 if transpose else 0}_{vec}_{atom_type}_{atom_m}_{atom_n}_{atom_k}"
            
            # Launcher wrapper (no explicit instantiation - template will be implicitly instantiated when called)
            f.write(f"""// Config: {tile_m}x{tile_n}x{tile_k}, threads={threads_m}x{threads_n}, work={work_m}x{work_n}, prefetch={prefetch}, transpose={transpose}, vec={vec}, atom={atom_type}({atom_m}x{atom_n}x{atom_k})
// Launcher wrapper
cudaError_t launch_{variant_name}(
    const float *A, const IQ4_NLBlock *B_blocks, float *C,
    int m, int n, int k, dim3 gridDim, dim3 blockDim, cudaStream_t stream)
{{
    const int num_k_blocks = k / 32;
    IQ4_NL_Decoder<IQ4_NLBlock> decoder(B_blocks, n, num_k_blocks);
    quantized_gemm_kernel_variant<IQ4_NL_Decoder<IQ4_NLBlock>, {tile_m}, {tile_n}, {tile_k}, 
        {threads_m}, {threads_n}, {work_m}, {work_n}, {prefetch}, {transpose_str}, {vec},
        {atom_type}, {atom_m}, {atom_n}, {atom_k}>
        <<<gridDim, blockDim, 0, stream>>>(A, C, m, n, k, decoder);
    return cudaGetLastError();
}}

// Auto-register with __attribute__((constructor))
namespace {{
    __attribute__((constructor)) void register_{variant_name}() {{
        CudaGemmKernelRegistry::instance().register_kernel(
            {tile_m}, {tile_n}, {tile_k}, {threads_m}, {threads_n}, {work_m}, {work_n},
            {prefetch}, {"true" if transpose else "false"}, {vec},
            {atom_type}, {atom_m}, {atom_n}, {atom_k}, &launch_{variant_name});
    }}
}}

""")
        
        # Close namespaces AFTER all variants
        f.write("""}  // namespace cuda
}  // namespace llaminar2
""")
    
    return filename

def write_cmake_file(filenames: List[str], output_dir: str):
    """Generate CMakeLists.txt fragment to add all .cu files to build."""
    cmake_path = os.path.join(output_dir, "CMakeVariantSources.txt")
    
    with open(cmake_path, 'w') as f:
        f.write("# AUTO-GENERATED by generate_cuda_gemm_variants.py\n")
        f.write("# Include this file in parent CMakeLists.txt to add all variant sources\n")
        f.write(f"# Total files: {len(filenames)} (enables {len(filenames)}-way parallel compilation)\n\n")
        f.write("set(CUDA_GEMM_VARIANT_SOURCES\n")
        for filename in filenames:
            f.write(f"    kernels/cuda/generated/{filename}\n")
        f.write(")\n")

# =============================================================================
# Main
# =============================================================================

def generate_config_struct(config: Tuple) -> str:
    """Generate C++ struct initialization for a configuration."""
    (tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n, 
     prefetch, transpose, vec, atom_type, atom_m, atom_n, atom_k) = config
    
    transpose_str = "true" if transpose else "false"
    vec_str = "true" if vec > 1 else "false"
    
    return f"""        {{.tile_m={tile_m}, .tile_n={tile_n}, .tile_k={tile_k},
          .threads_m={threads_m}, .threads_n={threads_n},
          .work_per_thread_m={work_m}, .work_per_thread_n={work_n},
          .prefetch_stages={prefetch}, .transpose_smem={transpose_str}, .vectorize_load={vec_str},
          .atom_type={atom_type}, .atom_layout_m={atom_m}, .atom_layout_n={atom_n}, .atom_layout_k={atom_k}}}"""

def write_config_header(configs: List[Tuple], output_dir: str):
    """Write C++ header with generated config list for auto-tuner."""
    header_path = os.path.join(output_dir, "GeneratedConfigs.h")
    
    with open(header_path, 'w') as f:
        f.write("""/**
 * @file GeneratedConfigs.h
 * @brief Auto-generated list of instantiated CUDA GEMM kernel configurations
 * 
 * This file is AUTO-GENERATED by generate_cuda_gemm_variants.py
 * DO NOT EDIT MANUALLY - your changes will be overwritten!
 */

#pragma once

#include "../CudaGemmConfig.h"
#include <vector>

namespace llaminar {
namespace v2 {

/**
 * @brief Get list of all instantiated kernel configurations
 * 
 * This list matches the LAUNCH_VARIANT calls in CudaGemmVariants_*.inc files.
 * The auto-tuner should ONLY use configurations from this list.
 */
inline std::vector<CudaGemmConfig> getGeneratedConfigs() {
    return {
""")
        
        # Write each config
        for i, config in enumerate(configs):
            f.write(generate_config_struct(config))
            if i < len(configs) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        
        f.write("""    };
}

} // namespace v2
} // namespace llaminar
""")
    
    print(f"  Generated {header_path}")


def main():
    """Main entry point."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating CUDA GEMM kernel variants...")
    
    # Generate all valid configurations
    all_configs = generate_all_configs()
    print(f"  Total valid configs: {len(all_configs)}")
    
    # Use ALL configurations to ensure auto-tuner candidates have compiled kernels
    # Previously limited to 200, but auto-tuner can select any of the 648 valid configs
    # This ensures launchIQ4NLGemmVariant always finds a matching LAUNCH_VARIANT
    top_configs = all_configs  # Use all 648 configs instead of selecting top 200
    print(f"  Generating all configs: {len(top_configs)}")
    
    # Distribute configs across files
    configs_per_file = (len(top_configs) + NUM_FILES - 1) // NUM_FILES
    print(f"  Configs per file: ~{configs_per_file}")
    
    # Generate files
    filenames = []
    for file_index in range(NUM_FILES):
        start_idx = file_index * configs_per_file
        end_idx = min(start_idx + configs_per_file, len(top_configs))
        
        if start_idx >= len(top_configs):
            break  # No more configs to distribute
        
        file_configs = top_configs[start_idx:end_idx]
        filename = write_variant_file(file_index, file_configs, OUTPUT_DIR)
        filenames.append(filename)
    
    print(f"  Generated {len(filenames)} .inc files")
    
    # Generate CMake file (informational only)
    write_cmake_file(filenames, OUTPUT_DIR)
    print(f"  Generated {OUTPUT_CMAKE_FILE} (informational)")
    
    # Generate C++ header with config list for auto-tuner
    write_config_header(top_configs, OUTPUT_DIR)
    
    print("Done!")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Total variant dispatches: {len(top_configs)}")
    print(f"  Files: {len(filenames)}")
    print(f"\n  To use: #include these files inside launchIQ4NLGemmVariant() in CudaGemmVariants.cu")

if __name__ == "__main__":
    main()
