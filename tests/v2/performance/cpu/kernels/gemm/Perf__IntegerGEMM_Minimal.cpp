/**
 * @file Perf__IntegerGEMM_Minimal.cpp
 * @brief Minimal test harness for INT8 GEMM performance profiling
 *
 * Purpose:
 * - Isolate and reproduce the catastrophic performance bug (1.1% efficiency)
 * - Allow rapid iteration on fixes without full test suite overhead
 * - Support profiling with perf/vtune/gdb
 * - Test all configuration parameters (MR, NR, UNROLL_K, PREFETCH_DIST, etc.)
 *
 * Baseline: FP32 GEMM achieves 335-1208 GFLOPS (proven working)
 * Bug: INT8 GEMM achieves 0.04-2.0 GFLOPS (1.1% efficiency)
 * Root cause: Double buffering in IntegerGemmMicroKernel wrapper
 *
 * @author David Sanftenberg
 * @date November 11, 2025
 */

#include "kernels/cpu/gemm_v2/IntegerGemmKernelTemplate.h"
#include "kernels/cpu/gemm/GemmWeightCache.h"
#include "kernels/cpu/SimdTraits.h"
#include "tensors/Tensors.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>

using namespace llaminar2;
using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;

// Simple Q8_0 block provider for testing (direct pointer access)
struct SimpleBlockProvider : public Q8_0BlockProvider {
    const Q8_0Block* blocks;
    size_t k_blocks_;
    size_t num_rows_;
    
    SimpleBlockProvider(const Q8_0Block* b, size_t kb, size_t num_rows)
        : blocks(b), k_blocks_(kb), num_rows_(num_rows) {}
    
    const Q8_0Block* get_q8_block(size_t row_idx, size_t k_block_offset) override {
        return &blocks[row_idx * k_blocks_ + k_block_offset];
    }
    
    void warmup_cache(size_t, size_t, size_t, size_t) override {}
    size_t k_blocks() const override { return k_blocks_; }
    size_t num_rows() const override { return num_rows_; }
    bool is_zero_copy() const override { return true; }
};

// Configuration knobs (modifiable via recompilation or command-line args)
struct BenchmarkConfig {
    int m = 128;                  // Batched prefill tokens (stress scenario)
    int n = 2048;                 // Wider projection head for prefill
    int k = 2048;                 // Matching inner dimension for prefill
    int MR = 16;                  // Microkernel M dimension
    int NR = 32;                  // Microkernel N dimension (fixed for Q8_0)
    int UNROLL_K = 4;             // K-loop unroll factor
    int PREFETCH_DIST = 2;        // Prefetch distance
    int MC = 256;                 // M-dimension cache block
    int KC = 512;                 // K-dimension cache block
    int NC = 128;                 // N-dimension cache block
    int warmup_iters = 3;         // Warmup iterations
    int timed_iters = 10;         // Timed iterations
    bool verbose = false;         // Print detailed output
    bool check_correctness = true; // Run scalar reference correctness validation
    double correctness_tol = 0.0;  // Allowed absolute diff tolerance (0 = exact)
    bool dump_fp = false;          // Dump raw fp_accumulators_ for first tile
    bool force_simd = false;       // Enable SIMD accumulation path in microkernel
};

// Helper: Generate random Q8_0 blocks
void generate_random_q8_blocks(Q8_0Block* blocks, size_t num_blocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    
    for (size_t i = 0; i < num_blocks; ++i) {
        blocks[i].d = fp32_to_fp16(scale_dist(rng));
        for (int j = 0; j < 32; ++j) {
            blocks[i].qs[j] = static_cast<int8_t>(dist(rng));
        }
    }
}

// Instantiate kernel with specific configuration
template<int MR, int NR, int UNROLL_K, int PREFETCH_DIST, int MC, int KC, int NC>
double run_benchmark_templated(const BenchmarkConfig& cfg) {
    using KernelType = IntegerGemmKernel<AVX512VNNITag, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;
    KernelType::MicroKernel_t::setUseSimd(cfg.force_simd);
    
    // Calculate block counts
    const size_t m_blocks = (cfg.m + 31) / 32;
    const size_t k_blocks = (cfg.k + 31) / 32;
    const size_t n_blocks = (cfg.n + 31) / 32;
    
    // Allocate aligned buffers
    // A: m rows × k_blocks blocks (activation format)
    // B: n rows × k_blocks blocks (weight format - ROWS not blocks!)
    // C: m rows × n_blocks blocks (output format)
    Q8_0Block* A = static_cast<Q8_0Block*>(aligned_alloc(64, cfg.m * k_blocks * sizeof(Q8_0Block)));
    Q8_0Block* B = static_cast<Q8_0Block*>(aligned_alloc(64, cfg.n * k_blocks * sizeof(Q8_0Block)));
    Q8_0Block* C = static_cast<Q8_0Block*>(aligned_alloc(64, cfg.m * n_blocks * sizeof(Q8_0Block)));
    
    if (!A || !B || !C) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return -1.0;
    }
    
    // Initialize with random data
    std::mt19937 rng(42);
    generate_random_q8_blocks(A, cfg.m * k_blocks, rng);
    generate_random_q8_blocks(B, cfg.n * k_blocks, rng);
    std::memset(C, 0, cfg.m * n_blocks * sizeof(Q8_0Block));
    
    // Create weight provider (zero-copy Q8_0 access)
    // Provider expects (blocks, k_blocks, num_rows) where num_rows = cfg.n (NOT n_blocks!)
    SimpleBlockProvider B_provider(B, k_blocks, cfg.n);
    
    // Optional correctness check BEFORE timing (single multiply compared to scalar reference)
    if (cfg.check_correctness) {
        // Create a scalar reference compute for first output block C[0,*]
        // We'll compute full C_ref for first tile (m x n) via naive int32 accumulation.
        // Build fully-dequantized FP32 panels and perform naive GEMM reference.
        std::vector<float> ref(cfg.m * cfg.n, 0.0f);
        std::vector<float> A_fp(cfg.m * cfg.k, 0.0f);
        std::vector<float> B_fp(cfg.n * cfg.k, 0.0f);
        auto dequantA = [&](int row, size_t kb) {
            const Q8_0Block &blk = A[row * k_blocks + kb];
            float scale = fp16_to_fp32(blk.d);
            for (int t = 0; t < 32 && (kb * 32 + t) < cfg.k; ++t)
                A_fp[row * cfg.k + kb * 32 + t] = static_cast<float>(blk.qs[t]) * scale;
        };
        auto dequantB = [&](int col, size_t kb) {
            const Q8_0Block &blk = B[col * k_blocks + kb];
            float scale = fp16_to_fp32(blk.d);
            for (int t = 0; t < 32 && (kb * 32 + t) < cfg.k; ++t)
                B_fp[col * cfg.k + kb * 32 + t] = static_cast<float>(blk.qs[t]) * scale;
        };
        for (int row = 0; row < cfg.m; ++row)
            for (size_t kb = 0; kb < k_blocks; ++kb)
                dequantA(row, kb);
        for (int col = 0; col < cfg.n; ++col)
            for (size_t kb = 0; kb < k_blocks; ++kb)
                dequantB(col, kb);
        for (int row = 0; row < cfg.m; ++row) {
            for (int col = 0; col < cfg.n; ++col) {
                double acc = 0.0;
                const float *a_row = &A_fp[row * cfg.k];
                const float *b_col = &B_fp[col * cfg.k];
                for (int kk = 0; kk < cfg.k; ++kk)
                    acc += static_cast<double>(a_row[kk]) * static_cast<double>(b_col[kk]);
                ref[row * cfg.n + col] = static_cast<float>(acc);
            }
        }
        // Run one kernel multiply to produce C
        KernelType::multiply(A, B_provider, C, cfg.m, cfg.n, cfg.k);
        // Quantized comparison: quantize reference rows in 32-wide blocks identically to kernel
        size_t total_blocks = static_cast<size_t>(cfg.m) * n_blocks;
        size_t scale_mismatches = 0, code_mismatches = 0;
        float max_scale_diff = 0.0f;
        int8_t max_code_diff = 0;
        for (int row = 0; row < cfg.m; ++row) {
            for (size_t nb = 0; nb < n_blocks; ++nb) {
                int base_col = static_cast<int>(nb * 32);
                // Build ref block slice (≤32 elements if tail)
                float ref_slice[32];
                int valid = 0;
                for (int j = 0; j < 32 && (base_col + j) < cfg.n; ++j) {
                    ref_slice[j] = ref[row * cfg.n + base_col + j];
                    valid++;
                }
                // Find max abs
                float amax = 0.0f;
                for (int j = 0; j < valid; ++j) amax = std::max(amax, std::fabs(ref_slice[j]));
                float ref_scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                float inv_ref_scale = 1.0f / ref_scale;
                // Quantize reference
                int8_t ref_q[32];
                for (int j = 0; j < valid; ++j) {
                    int32_t q = static_cast<int32_t>(std::round(ref_slice[j] * inv_ref_scale));
                    q = std::max(-127, std::min(127, q));
                    ref_q[j] = static_cast<int8_t>(q);
                }
                // Fetch kernel block
                const Q8_0Block &blk = C[row * n_blocks + nb];
                float ker_scale = fp16_to_fp32(blk.d);
                float ref_scale_fp16 = fp16_to_fp32(fp32_to_fp16(ref_scale));
                float scale_diff = std::fabs(ker_scale - ref_scale_fp16);
                if (scale_diff > max_scale_diff) max_scale_diff = scale_diff;
                if (scale_diff > cfg.correctness_tol) scale_mismatches++;
                for (int j = 0; j < valid; ++j) {
                    int8_t ker_q = blk.qs[j];
                    int8_t diff = static_cast<int8_t>(std::abs(static_cast<int>(ker_q) - static_cast<int>(ref_q[j])));
                    if (diff != 0) {
                        code_mismatches++;
                        if (diff > max_code_diff) max_code_diff = diff;
                    }
                }
            }
        }
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ QUANTIZED CORRECTNESS (ROW-BLOCK)                          ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Scale mismatches: " << std::setw(10) << scale_mismatches << " / " << total_blocks << std::setw(15) << " ║\n";
        std::cout << "║ Code  mismatches: " << std::setw(10) << code_mismatches << " / " << (cfg.m * cfg.n) << std::setw(12) << " ║\n";
        std::cout << "║ Max scale diff : " << std::setw(10) << std::fixed << std::setprecision(6) << max_scale_diff << std::setw(16) << " ║\n";
        std::cout << "║ Max code diff  : " << std::setw(10) << static_cast<int>(max_code_diff) << std::setw(22) << " ║\n";
        bool pass = (scale_mismatches == 0 && code_mismatches == 0);
        if (pass) {
            std::cout << "║ ✓ PASS (exact quantized match)                             ║\n";
        } else {
            std::cout << "║ ✗ FAIL (quantized mismatch)                                ║\n";
        }
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        // Reset C before warmup timing
        std::memset(C, 0, cfg.m * n_blocks * sizeof(Q8_0Block));
        // Reset C before warmup timing (avoid residuals influencing prefetch/cache effects)
        std::memset(C, 0, cfg.m * n_blocks * sizeof(Q8_0Block));
    }

    // Warmup iterations (excluded from correctness check timing)
    for (int iter = 0; iter < cfg.warmup_iters; ++iter) {
        KernelType::multiply(A, B_provider, C, cfg.m, cfg.n, cfg.k);
    }
    
    // Timed iterations
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < cfg.timed_iters; ++iter) {
        KernelType::multiply(A, B_provider, C, cfg.m, cfg.n, cfg.k);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms = elapsed_ms / cfg.timed_iters;
    
    // Calculate GFLOPS (INT8 GEMM: 2*m*n*k operations per multiply)
    double ops = 2.0 * cfg.m * cfg.n * cfg.k;
    double gflops = ops / (avg_ms * 1e6);
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    
    return gflops;
}

// Runtime dispatcher for different MR values
double run_benchmark(const BenchmarkConfig& cfg) {
    // Dispatch based on MR (NR fixed at 32)
    // Only instantiate configurations we actually use
    
    if (cfg.MR == 1 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2) {
        return run_benchmark_templated<1, 32, 4, 2, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 2 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2) {
        return run_benchmark_templated<2, 32, 4, 2, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 4 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2) {
        return run_benchmark_templated<4, 32, 4, 2, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 8 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2) {
        return run_benchmark_templated<8, 32, 4, 2, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 16 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2) {
        return run_benchmark_templated<16, 32, 4, 2, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 16 && cfg.UNROLL_K == 8 && cfg.PREFETCH_DIST == 2) {
        return run_benchmark_templated<16, 32, 8, 2, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 16 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 0) {
        return run_benchmark_templated<16, 32, 4, 0, 256, 512, 128>(cfg);
    }
    else if (cfg.MR == 16 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 3) {
        return run_benchmark_templated<16, 32, 4, 3, 256, 512, 128>(cfg);
    }
    else {
        std::cerr << "Unsupported configuration: MR=" << cfg.MR 
                  << " UNROLL_K=" << cfg.UNROLL_K 
                  << " PREFETCH_DIST=" << cfg.PREFETCH_DIST << std::endl;
        std::cerr << "Add template instantiation for this config" << std::endl;
        return -1.0;
    }
}

// Dump raw fp accumulators for first tile using manual microkernel drive (ignores OpenMP parallelization)
template<int MR, int NR, int UNROLL_K, int PREFETCH_DIST, int MC, int KC, int NC>
void dump_first_tile_fp(const BenchmarkConfig& cfg,
                        const Q8_0Block* A,
                        Q8_0BlockProvider &B_provider,
                        int m, int n, int k) {
    using KernelType = IntegerGemmKernel<AVX512VNNITag, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC>;
    using MicroKernel_t = typename KernelType::MicroKernel_t;
    MicroKernel_t::setUseSimd(cfg.force_simd);
    const size_t k_blocks_a = (k + 31) / 32;
    const int tile_m = std::min(MR, m);
    const int tile_n = std::min(NR, n);
    alignas(64) int8_t A_panel[MR * 32];
    alignas(64) int8_t B_panel[NR * 32];
    alignas(64) double a_scales[MR];
    alignas(64) double b_scales[NR];
    MicroKernel_t ukernel;
    ukernel.zero();
    // Reference fused FP32 accumulator per (i,j)
    std::vector<float> ref_fp(MR * NR, 0.0f);
    for (size_t kb = 0; kb < k_blocks_a; ++kb) {
        // Pack panels using helpers from kernel template
        // Replicate loadAndPackA logic
        for (int i = 0; i < tile_m; ++i) {
            const Q8_0Block *block = &A[i * k_blocks_a + kb];
            std::memcpy(A_panel + i * 32, block->qs, 32);
            a_scales[i] = static_cast<double>(fp16_to_fp32(block->d));
        }
        for (int i = tile_m; i < MR; ++i) {
            std::memset(A_panel + i * 32, 0, 32);
            a_scales[i] = 1.0;
        }
        for (int j = 0; j < tile_n; ++j) {
            const Q8_0Block *block = B_provider.get_q8_block(j, kb);
            std::memcpy(B_panel + j * 32, block->qs, 32);
            b_scales[j] = static_cast<double>(fp16_to_fp32(block->d));
        }
        for (int j = tile_n; j < NR; ++j) {
            std::memset(B_panel + j * 32, 0, 32);
            b_scales[j] = 1.0;
        }
        ukernel.accumulate(A_panel, B_panel, 32, a_scales, b_scales);
        // Accumulate reference using the same quantized panels & scales (32 elements)
        for (int i = 0; i < tile_m; ++i) {
            const int8_t* a_row = A_panel + i * 32;
            float as = static_cast<float>(a_scales[i]);
            for (int j = 0; j < tile_n; ++j) {
                const int8_t* b_col = B_panel + j * 32;
                float bs = static_cast<float>(b_scales[j]);
                int32_t dot = 0;
                for (int t = 0; t < 32; ++t) dot += static_cast<int32_t>(a_row[t]) * static_cast<int32_t>(b_col[t]);
                ref_fp[i * NR + j] += static_cast<float>(dot) * as * bs;
            }
        }
    }
    const float* fp_raw = ukernel.raw_fp_accumulators();
    std::cout << "\nRAW FP ACCUMULATORS (first tile, MR=" << MR << ", NR=" << NR << ")\n";
    for (int i = 0; i < tile_m; ++i) {
        std::cout << "Row " << i << ":";
        for (int j = 0; j < tile_n; ++j) {
            std::cout << ' ' << std::fixed << std::setprecision(6) << fp_raw[i * NR + j];
        }
        std::cout << '\n';
    }
    // Compute diff statistics
    double max_abs = 0.0, sum_abs = 0.0;
    size_t count = 0;
    for (int i = 0; i < tile_m; ++i) {
        for (int j = 0; j < tile_n; ++j) {
            float diff = fp_raw[i * NR + j] - ref_fp[i * NR + j];
            float ad = std::fabs(diff);
            if (ad > max_abs) max_abs = ad;
            sum_abs += ad;
            count++;
        }
    }
    double mean_abs = (count ? sum_abs / count : 0.0);
    std::cout << "FP REF DIFF: max_abs=" << std::setprecision(6) << max_abs
              << " mean_abs=" << mean_abs << " (" << count << " elements)\n";
}

void print_config(const BenchmarkConfig& cfg) {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ INT8 GEMM PERFORMANCE PROFILING HARNESS                    ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Matrix dimensions: " << std::setw(5) << cfg.m << " × " 
              << std::setw(5) << cfg.n << " × " << std::setw(5) << cfg.k 
              << std::setw(14) << " ║" << std::endl;
    std::cout << "║ Microkernel config: MR=" << std::setw(2) << cfg.MR 
              << " NR=" << std::setw(2) << cfg.NR << std::setw(24) << " ║" << std::endl;
    std::cout << "║ Tuning params: UNROLL_K=" << std::setw(2) << cfg.UNROLL_K 
              << " PREFETCH=" << std::setw(2) << cfg.PREFETCH_DIST 
              << std::setw(16) << " ║" << std::endl;
    std::cout << "║ Cache blocks: MC=" << std::setw(4) << cfg.MC 
              << " KC=" << std::setw(4) << cfg.KC 
              << " NC=" << std::setw(4) << cfg.NC << std::setw(10) << " ║" << std::endl;
    std::cout << "║ Iterations: warmup=" << cfg.warmup_iters 
              << " timed=" << cfg.timed_iters << std::setw(24) << " ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
}

void print_results(const BenchmarkConfig& cfg, double gflops) {
    // Theoretical peak for INT8 VNNI on Ice Lake
    // Base: 2.8 GHz, 2 ports (FMA units), 64 INT8 ops per dpbusd
    const double theoretical_peak = 2.8 * 2.0 * 64.0; // 358.4 GFLOPS
    const double efficiency = (gflops / theoretical_peak) * 100.0;
    
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ PERFORMANCE RESULTS                                        ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Throughput:     " << std::setw(10) << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS" << std::setw(21) << " ║" << std::endl;
    std::cout << "║ Theoretical:    " << std::setw(10) << theoretical_peak 
              << " GFLOPS" << std::setw(21) << " ║" << std::endl;
    std::cout << "║ Efficiency:     " << std::setw(10) << efficiency 
              << " %" << std::setw(25) << " ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════╣" << std::endl;
    
    if (efficiency < 10.0) {
        std::cout << "║ ⚠ WARNING: CATASTROPHIC PERFORMANCE (<10% efficiency)     ║" << std::endl;
        std::cout << "║   Known bug: Double buffering in IntegerGemmMicroKernel   ║" << std::endl;
    } else if (efficiency < 50.0) {
        std::cout << "║ ⚠ WARNING: Poor efficiency (<50%)                         ║" << std::endl;
    } else {
        std::cout << "║ ✓ Good efficiency (>50%)                                   ║" << std::endl;
    }
    
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
}

int main(int argc, char** argv) {
    BenchmarkConfig cfg;
    
    // Simple command-line parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--m" && i + 1 < argc) {
            cfg.m = std::atoi(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            cfg.n = std::atoi(argv[++i]);
        } else if (arg == "--k" && i + 1 < argc) {
            cfg.k = std::atoi(argv[++i]);
        } else if (arg == "--mr" && i + 1 < argc) {
            cfg.MR = std::atoi(argv[++i]);
        } else if (arg == "--unroll" && i + 1 < argc) {
            cfg.UNROLL_K = std::atoi(argv[++i]);
        } else if (arg == "--prefetch" && i + 1 < argc) {
            cfg.PREFETCH_DIST = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iters = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.timed_iters = std::atoi(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            cfg.verbose = true;
        } else if (arg == "--nocorrect") {
            cfg.check_correctness = false;
        } else if (arg == "--tol" && i + 1 < argc) {
            cfg.correctness_tol = std::atof(argv[++i]);
        } else if (arg == "--dump-fp") {
            cfg.dump_fp = true;
        } else if (arg == "--simd") {
            cfg.force_simd = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --m <size>         M dimension (default: 128)" << std::endl;
            std::cout << "  --n <size>         N dimension (default: 2048)" << std::endl;
            std::cout << "  --k <size>         K dimension (default: 2048)" << std::endl;
            std::cout << "  --mr <tiles>       Microkernel M tiles (default: 16)" << std::endl;
            std::cout << "  --unroll <factor>  K-loop unroll (default: 4)" << std::endl;
            std::cout << "  --prefetch <dist>  Prefetch distance (default: 2)" << std::endl;
            std::cout << "  --warmup <iters>   Warmup iterations (default: 3)" << std::endl;
            std::cout << "  --iters <count>    Timed iterations (default: 10)" << std::endl;
            std::cout << "  --verbose, -v      Verbose output" << std::endl;
            std::cout << "  --nocorrect        Disable correctness validation" << std::endl;
            std::cout << "  --tol <value>      Absolute tolerance for correctness (default 0)" << std::endl;
            std::cout << "  --dump-fp          Dump raw fp accumulators for first tile (debug)" << std::endl;
            std::cout << "  --simd             Force SIMD accumulation path (experimental)" << std::endl;
            std::cout << "  --help, -h         Show this help" << std::endl;
            std::cout << std::endl;
            std::cout << "Examples:" << std::endl;
            std::cout << "  # Test bug with MR=16 (slow)" << std::endl;
            std::cout << "  " << argv[0] << " --mr 16" << std::endl;
            std::cout << std::endl;
            std::cout << "  # Test with MR=1 (less overhead)" << std::endl;
            std::cout << "  " << argv[0] << " --mr 1" << std::endl;
            std::cout << std::endl;
            std::cout << "  # Profile large matrix" << std::endl;
            std::cout << "  perf record -g " << argv[0] << " --m 512 --n 4096 --k 4096 --iters 100" << std::endl;
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
    }
    
    print_config(cfg);
    
    double gflops = run_benchmark(cfg);

    // Optional FP accumulator dump (first tile only) after performance run
    if (cfg.dump_fp) {
        // Reconstruct buffers (small extra allocation acceptable for debug)
        const size_t k_blocks = (cfg.k + 31) / 32;
        Q8_0Block* A = static_cast<Q8_0Block*>(aligned_alloc(64, cfg.m * k_blocks * sizeof(Q8_0Block)));
        Q8_0Block* B = static_cast<Q8_0Block*>(aligned_alloc(64, cfg.n * k_blocks * sizeof(Q8_0Block)));
        if (!A || !B) {
            std::cerr << "Allocation failed for dump_fp buffers" << std::endl;
        } else {
            std::mt19937 rng(1337);
            generate_random_q8_blocks(A, cfg.m * k_blocks, rng);
            generate_random_q8_blocks(B, cfg.n * k_blocks, rng);
            SimpleBlockProvider B_provider(B, k_blocks, cfg.n);
            // Dispatch matching benchmark kernel configuration
            if (cfg.MR == 1 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2)
                dump_first_tile_fp<1,32,4,2,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 2 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2)
                dump_first_tile_fp<2,32,4,2,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 4 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2)
                dump_first_tile_fp<4,32,4,2,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 8 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2)
                dump_first_tile_fp<8,32,4,2,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 16 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 2)
                dump_first_tile_fp<16,32,4,2,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 16 && cfg.UNROLL_K == 8 && cfg.PREFETCH_DIST == 2)
                dump_first_tile_fp<16,32,8,2,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 16 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 0)
                dump_first_tile_fp<16,32,4,0,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else if (cfg.MR == 16 && cfg.UNROLL_K == 4 && cfg.PREFETCH_DIST == 3)
                dump_first_tile_fp<16,32,4,3,256,512,128>(cfg, A, B_provider, cfg.m, cfg.n, cfg.k);
            else
                std::cerr << "Unsupported configuration for dump_fp path" << std::endl;
        }
        free(A); free(B);
    }
    
    if (gflops < 0) {
        std::cerr << "Benchmark failed" << std::endl;
        return 1;
    }
    
    print_results(cfg, gflops);
    
    return 0;
}
