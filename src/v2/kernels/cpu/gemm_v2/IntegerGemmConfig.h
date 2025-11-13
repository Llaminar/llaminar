/**
 * @file IntegerGemmConfig.h
 * @brief Runtime-configurable parameters for Integer GEMM kernel
 * 
 * Provides environment-variable based configuration for:
 * - K-block processing width (32/64/128-byte)
 * - Tile sizes (TILE_M, TILE_N)
 * - Prefetching behavior
 * - Cache blocking parameters
 * 
 * All settings can be tuned without recompilation via environment variables.
 * Falls back to compile-time defaults if not specified.
 * 
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#pragma once

#include <cstdlib>
#include <cstdint>
#include <string>
#include <iostream>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {
            /**
             * @brief Integer GEMM runtime configuration
             * 
             * All parameters are read once on first access (lazy static initialization).
             * Thread-safe via C++11 magic statics.
             */
            struct IntegerGemmConfig
            {
                // ============================================================
                // K-BLOCK PROCESSING
                // ============================================================
                
                /**
                 * @brief Number of Q8_0 blocks to process per K-loop iteration
                 * 
                 * - 1 block = 32 bytes (baseline, uses 8 DPBUSD lanes)
                 * - 2 blocks = 64 bytes (optimal for single AVX512 register, 16 lanes)
                 * - 4 blocks = 128 bytes (requires 2 AVX512 registers, potential for unrolling)
                 * 
                 * Environment: LLAMINAR_INT_GEMM_K_BLOCKS_PER_ITER
                 * Default: 2 (64-byte processing)
                 * Valid: 1, 2, 4
                 */
                int k_blocks_per_iter;
                
                /**
                 * @brief Prefetch distance in K-loop iterations
                 * 
                 * How many iterations ahead to prefetch A/B panels.
                 * - 0: No prefetching
                 * - 1: Prefetch next panel (minimal latency hiding)
                 * - 2: Prefetch 2 panels ahead (recommended for AVX512)
                 * - 3+: Aggressive prefetching (may pollute cache)
                 * 
                 * Environment: LLAMINAR_INT_GEMM_PREFETCH_DIST
                 * Default: 0 (disabled until validated)
                 * Valid: 0-8
                 */
                int k_prefetch_distance;
                
                // ============================================================
                // TILE SIZES
                // ============================================================
                
                /**
                 * @brief Micro-kernel M dimension (rows)
                 * 
                 * Number of output rows processed per micro-kernel invocation.
                 * Larger values reduce outer loop overhead but increase register pressure.
                 * 
                 * Environment: LLAMINAR_INT_GEMM_TILE_M
                 * Default: Template parameter MR (typically 16)
                 * Valid: 1-32 (must divide evenly into M dimension)
                 */
                int tile_m;
                
                /**
                 * @brief Micro-kernel N dimension (columns)
                 * 
                 * Number of output columns processed per micro-kernel invocation.
                 * MUST be 32 for Q8_0 block alignment (not configurable at runtime).
                 * 
                 * Environment: N/A (fixed by Q8_0 format)
                 * Default: 32
                 * Valid: 32 only
                 */
                static constexpr int tile_n = 32;
                
                // ============================================================
                // CACHE BLOCKING
                // ============================================================
                
                /**
                 * @brief M-dimension cache block size
                 * 
                 * Controls L1/L2 cache blocking for large matrices.
                 * - 0: Disabled (process entire M dimension)
                 * - >0: Block M into chunks of this size
                 * 
                 * Environment: LLAMINAR_INT_GEMM_MC
                 * Default: 0 (disabled)
                 * Valid: 0, 64, 128, 256, 512
                 */
                int mc;
                
                /**
                 * @brief K-dimension cache block size
                 * 
                 * Controls how many K blocks to process before moving to next M/N tile.
                 * Affects A-panel reuse.
                 * 
                 * Environment: LLAMINAR_INT_GEMM_KC
                 * Default: 0 (disabled, process all K at once)
                 * Valid: 0, 128, 256, 512, 1024
                 */
                int kc;
                
                /**
                 * @brief N-dimension cache block size
                 * 
                 * Controls B-panel cache blocking.
                 * 
                 * Environment: LLAMINAR_INT_GEMM_NC
                 * Default: 0 (disabled)
                 * Valid: 0, 64, 128, 256, 512
                 */
                int nc;
                
                // ============================================================
                // DIAGNOSTICS
                // ============================================================
                
                /**
                 * @brief Print configuration on first use
                 * 
                 * Environment: LLAMINAR_INT_GEMM_VERBOSE
                 * Default: 0 (silent)
                 */
                bool verbose;
                
                /**
                 * @brief Get singleton instance (lazy initialization)
                 */
                static const IntegerGemmConfig& instance()
                {
                    static IntegerGemmConfig config = load_from_environment();
                    return config;
                }
                
            private:
                /**
                 * @brief Load configuration from environment variables
                 */
                static IntegerGemmConfig load_from_environment()
                {
                    IntegerGemmConfig cfg;
                    
                    // K-block processing width
                    cfg.k_blocks_per_iter = get_env_int("LLAMINAR_INT_GEMM_K_BLOCKS_PER_ITER", 2);
                    if (cfg.k_blocks_per_iter != 1 && cfg.k_blocks_per_iter != 2 && cfg.k_blocks_per_iter != 4) {
                        std::cerr << "[IntegerGemmConfig] Invalid K_BLOCKS_PER_ITER=" << cfg.k_blocks_per_iter 
                                  << ", forcing to 2 (valid: 1,2,4)\n";
                        cfg.k_blocks_per_iter = 2;
                    }
                    
                    // Prefetching
                    cfg.k_prefetch_distance = get_env_int("LLAMINAR_INT_GEMM_PREFETCH_DIST", 0);
                    if (cfg.k_prefetch_distance < 0 || cfg.k_prefetch_distance > 8) {
                        std::cerr << "[IntegerGemmConfig] Invalid PREFETCH_DIST=" << cfg.k_prefetch_distance 
                                  << ", clamping to [0,8]\n";
                        cfg.k_prefetch_distance = std::max(0, std::min(8, cfg.k_prefetch_distance));
                    }
                    
                    // Tile sizes (runtime can override template defaults)
                    cfg.tile_m = get_env_int("LLAMINAR_INT_GEMM_TILE_M", 0);  // 0 = use template default
                    if (cfg.tile_m < 0 || cfg.tile_m > 32) {
                        std::cerr << "[IntegerGemmConfig] Invalid TILE_M=" << cfg.tile_m 
                                  << ", using template default\n";
                        cfg.tile_m = 0;
                    }
                    
                    // Cache blocking
                    cfg.mc = get_env_int("LLAMINAR_INT_GEMM_MC", 0);
                    cfg.kc = get_env_int("LLAMINAR_INT_GEMM_KC", 0);
                    cfg.nc = get_env_int("LLAMINAR_INT_GEMM_NC", 0);
                    
                    // Diagnostics
                    cfg.verbose = get_env_int("LLAMINAR_INT_GEMM_VERBOSE", 0) != 0;
                    
                    if (cfg.verbose) {
                        cfg.print();
                    }
                    
                    return cfg;
                }
                
                /**
                 * @brief Read integer from environment variable
                 */
                static int get_env_int(const char* name, int default_value)
                {
                    const char* val = std::getenv(name);
                    if (!val) return default_value;
                    
                    try {
                        return std::stoi(val);
                    } catch (...) {
                        std::cerr << "[IntegerGemmConfig] Failed to parse " << name 
                                  << "=" << val << ", using default=" << default_value << "\n";
                        return default_value;
                    }
                }
                
                /**
                 * @brief Print current configuration
                 */
                void print() const
                {
                    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
                    std::cout << "║ INTEGER GEMM CONFIGURATION                                    ║\n";
                    std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
                    std::cout << "║ K-Block Processing                                            ║\n";
                    std::cout << "║   Blocks per iteration:  " << k_blocks_per_iter 
                              << " (" << (k_blocks_per_iter * 32) << " bytes)";
                    for (int i = 0; i < 23 - std::to_string(k_blocks_per_iter * 32).length(); ++i) std::cout << " ";
                    std::cout << "║\n";
                    std::cout << "║   Prefetch distance:     " << k_prefetch_distance;
                    for (int i = 0; i < 39 - std::to_string(k_prefetch_distance).length(); ++i) std::cout << " ";
                    std::cout << "║\n";
                    std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
                    std::cout << "║ Tile Sizes                                                    ║\n";
                    std::cout << "║   TILE_M:                " << (tile_m ? std::to_string(tile_m) : "template default");
                    for (int i = 0; i < 39 - (tile_m ? std::to_string(tile_m).length() : 16); ++i) std::cout << " ";
                    std::cout << "║\n";
                    std::cout << "║   TILE_N:                32 (fixed)                           ║\n";
                    std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
                    std::cout << "║ Cache Blocking                                                ║\n";
                    std::cout << "║   MC (M-block):          " << (mc ? std::to_string(mc) : "disabled");
                    for (int i = 0; i < 39 - (mc ? std::to_string(mc).length() : 8); ++i) std::cout << " ";
                    std::cout << "║\n";
                    std::cout << "║   KC (K-block):          " << (kc ? std::to_string(kc) : "disabled");
                    for (int i = 0; i < 39 - (kc ? std::to_string(kc).length() : 8); ++i) std::cout << " ";
                    std::cout << "║\n";
                    std::cout << "║   NC (N-block):          " << (nc ? std::to_string(nc) : "disabled");
                    for (int i = 0; i < 39 - (nc ? std::to_string(nc).length() : 8); ++i) std::cout << " ";
                    std::cout << "║\n";
                    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
                }
            };

        } // namespace gemm
    }     // namespace kernels
} // namespace llaminar2
