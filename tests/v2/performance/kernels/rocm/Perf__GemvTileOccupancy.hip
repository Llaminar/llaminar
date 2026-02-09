/**
 * @file Perf__GemvTileOccupancy.hip
 * @brief GEMV tile/occupancy tuning sweep for grid_kpar kernel on MI50 (gfx906)
 *
 * Sweeps TN (tile width), CPT (columns per thread), and KB (K-blocks) for
 * the grid_kpar INT8 VNNI GEMV kernel to find optimal occupancy parameters.
 *
 * Key finding from assembly analysis: current TN=128/CPT=4 gives 32 threads/block,
 * which is only half a wavefront on GCN (64-wide). This benchmark evaluates
 * whether full-wavefront or multi-wavefront configurations are faster.
 *
 * Target shapes (Qwen2.5-7B decode M=1):
 *   QKV fused:  4608 × 3584
 *   Wo:         3584 × 3584
 *   GateUp:    37888 × 3584  (currently uses wide_vec4)
 *   FFN Down:   3584 × 18944
 *   LM Head:  152064 × 3584  (currently uses wide_vec4)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <string>

// ============================================================================
// Grid-level K-parallel VNNI kernel (copy from production, templated)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_grid_kpar_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per = (K + kblocks - 1) / kblocks;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base + c) * 4);

                acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

// ============================================================================
// Benchmark infrastructure
// ============================================================================

struct BenchConfig {
    int TN;
    int CPT;
    int KB;
    int threads_per_block;
    const char* label;
};

struct BenchResult {
    double min_us;
    double mean_us;
    double gbps;
    int grid_x;
    int grid_y;
    int threads;
    int wavefronts;  // total wavefronts launched
};

// Pack VNNI weights on host: [K][N] row-major → [K/4][N][4]
static void packVnniWeights(const std::vector<int8_t>& row_major,
                            int N, int K,
                            std::vector<int8_t>& vnni_out)
{
    vnni_out.resize(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; k += 4) {
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < 4; ++i) {
                vnni_out[static_cast<size_t>((k / 4) * N + n) * 4 + i] =
                    row_major[static_cast<size_t>(k + i) * N + n];
            }
        }
    }
}

template <int TN, int CPT>
static BenchResult benchmarkConfig(
    const int8_t* d_A_int8,
    const int8_t* d_B_vnni,
    int32_t* d_C_int32,
    int N, int K, int KB,
    int warmup_iters, int bench_iters)
{
    BenchResult result{};

    constexpr int threads = TN / CPT;
    const int grid_n = (N + TN - 1) / TN;

    result.grid_x = grid_n;
    result.grid_y = KB;
    result.threads = threads;
    result.wavefronts = grid_n * KB;  // each block = 1 wavefront for threads <= 64

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), nullptr);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_grid_kpar_t<TN, CPT>),
            dim3(grid_n, KB), dim3(threads), 0, nullptr,
            d_A_int8, d_B_vnni, d_C_int32, N, K, KB);
    }
    hipDeviceSynchronize();

    // Timed runs
    std::vector<double> times;
    times.reserve(bench_iters);

    for (int i = 0; i < bench_iters; ++i) {
        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), nullptr);
        hipDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_grid_kpar_t<TN, CPT>),
            dim3(grid_n, KB), dim3(threads), 0, nullptr,
            d_A_int8, d_B_vnni, d_C_int32, N, K, KB);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times.push_back(us);
    }

    std::sort(times.begin(), times.end());
    result.min_us = times[0];

    double sum = 0;
    for (auto t : times) sum += t;
    result.mean_us = sum / times.size();

    // Bandwidth: weight bytes read = K * N (INT8)
    double weight_bytes = static_cast<double>(K) * N;
    result.gbps = (weight_bytes / (result.min_us * 1e-6)) / 1e9;

    return result;
}

// ============================================================================
// Shape definitions
// ============================================================================

struct Shape {
    const char* name;
    int N;
    int K;
};

static const Shape kQwen7BShapes[] = {
    {"QKV fused",  4608,  3584},
    {"Wo",         3584,  3584},
    {"GateUp",    37888,  3584},
    {"FFN Down",   3584, 18944},
    {"LM Head",  152064,  3584},
};
static constexpr int kNumShapes = sizeof(kQwen7BShapes) / sizeof(kQwen7BShapes[0]);

// ============================================================================
// Main sweep
// ============================================================================

// Dispatch helper to call the right template instantiation
template <int TN, int CPT>
static BenchResult runBench(const int8_t* A, const int8_t* B, int32_t* C,
                            int N, int K, int KB)
{
    return benchmarkConfig<TN, CPT>(A, B, C, N, K, KB, 5, 30);
}

int main()
{
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No HIP device found\n");
        return 1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    hipSetDevice(0);
    fprintf(stderr, "\nDevice: %s  (%d CUs)\n", prop.name, prop.multiProcessorCount);
    fprintf(stderr, "Wavefront size: %d\n\n", prop.warpSize);

    // Maximum shape sizes — allocate once
    const int maxN = 152064;
    const int maxK = 18944;

    // Host data
    std::mt19937 rng(42);
    std::vector<int8_t> h_A(maxK);
    std::vector<int8_t> h_B_row(static_cast<size_t>(maxK) * maxN);

    for (auto& v : h_A) v = static_cast<int8_t>(rng() % 255 - 127);
    for (auto& v : h_B_row) v = static_cast<int8_t>(rng() % 255 - 127);

    // Device allocations
    int8_t* d_A = nullptr;
    int8_t* d_B_vnni = nullptr;
    int32_t* d_C = nullptr;

    hipMalloc(&d_A, maxK * sizeof(int8_t));
    hipMalloc(&d_B_vnni, static_cast<size_t>(maxK) * maxN * sizeof(int8_t));
    hipMalloc(&d_C, maxN * sizeof(int32_t));

    hipMemcpy(d_A, h_A.data(), maxK * sizeof(int8_t), hipMemcpyHostToDevice);

    // KB values to sweep
    const int kb_values[] = {8, 16, 24, 32, 48, 56, 64, 96, 128};
    const int num_kb = sizeof(kb_values) / sizeof(kb_values[0]);

    // ========================================================================
    // EXPERIMENT 1: TN/CPT sweep at default KB
    // ========================================================================
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  EXPERIMENT 1: TN × CPT Sweep (KB = production default per shape)                              ║\n");
    fprintf(stderr, "╠═══════════════════╦═════╦═════╦═══════╦═══════╦═══════════╦═══════════╦══════════╦══════════════╣\n");
    fprintf(stderr, "║ Shape             ║  TN ║ CPT ║ Thds  ║  KB   ║  Min(μs)  ║ Mean(μs)  ║  GB/s    ║ Wavefronts   ║\n");
    fprintf(stderr, "╠═══════════════════╬═════╬═════╬═══════╬═══════╬═══════════╬═══════════╬══════════╬══════════════╣\n");

    for (int si = 0; si < kNumShapes; ++si) {
        const auto& shape = kQwen7BShapes[si];
        const int N = shape.N;
        const int K = shape.K;

        // Pack VNNI for this shape
        int B_size = K * N;
        std::vector<int8_t> h_B_sub(h_B_row.begin(), h_B_row.begin() + B_size);
        std::vector<int8_t> h_B_vnni;
        packVnniWeights(h_B_sub, N, K, h_B_vnni);
        hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size(), hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        // Production KB formula
        const int default_kb = std::max(2, std::min(64, (K + 63) / 64));

        // Test configurations: (TN, CPT, threads_per_block)
        // Current production: TN=128, CPT=4 → 32 threads (HALF wavefront!)
        struct Config { int TN; int CPT; const char* tag; };
        Config configs[] = {
            {64,  1, "64/1"},   // 64 threads = 1 wavefront
            {64,  2, "64/2"},   // 32 threads = 0.5 wavefront
            {128, 1, "128/1"},  // 128 threads = 2 wavefronts
            {128, 2, "128/2"},  // 64 threads = 1 wavefront ★
            {128, 4, "128/4"},  // 32 threads = 0.5 wavefront (CURRENT)
            {256, 1, "256/1"},  // 256 threads = 4 wavefronts
            {256, 2, "256/2"},  // 128 threads = 2 wavefronts
            {256, 4, "256/4"},  // 64 threads = 1 wavefront ★
            {256, 8, "256/8"},  // 32 threads = 0.5 wavefront
            {512, 4, "512/4"},  // 128 threads = 2 wavefronts
            {512, 8, "512/8"},  // 64 threads = 1 wavefront ★
        };

        // N must be divisible by TN for each config
        for (const auto& cfg : configs) {
            if (N % cfg.TN != 0 && (N + cfg.TN - 1) / cfg.TN * cfg.TN > N + cfg.TN) continue;
            // CPT must divide TN, and N must be aligned to CPT
            if (cfg.TN % cfg.CPT != 0) continue;
            if (N % cfg.CPT != 0) continue;

            BenchResult r;
            int kb = default_kb;

            // Dispatch to template specialization
            if (cfg.TN == 64 && cfg.CPT == 1)
                r = runBench<64, 1>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 64 && cfg.CPT == 2)
                r = runBench<64, 2>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 128 && cfg.CPT == 1)
                r = runBench<128, 1>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 128 && cfg.CPT == 2)
                r = runBench<128, 2>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 128 && cfg.CPT == 4)
                r = runBench<128, 4>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 256 && cfg.CPT == 1)
                r = runBench<256, 1>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 256 && cfg.CPT == 2)
                r = runBench<256, 2>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 256 && cfg.CPT == 4)
                r = runBench<256, 4>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 256 && cfg.CPT == 8)
                r = runBench<256, 8>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 512 && cfg.CPT == 4)
                r = runBench<512, 4>(d_A, d_B_vnni, d_C, N, K, kb);
            else if (cfg.TN == 512 && cfg.CPT == 8)
                r = runBench<512, 8>(d_A, d_B_vnni, d_C, N, K, kb);
            else
                continue;

            const char* marker = (cfg.TN == 128 && cfg.CPT == 4) ? " ◄PROD" : "";

            fprintf(stderr, "║ %-17s ║ %3d ║ %3d ║ %5d ║ %5d ║ %9.1f ║ %9.1f ║ %8.1f ║ %10d%s ║\n",
                    shape.name, cfg.TN, cfg.CPT, cfg.TN / cfg.CPT, kb,
                    r.min_us, r.mean_us, r.gbps, r.wavefronts, marker);
        }

        fprintf(stderr, "╠═══════════════════╬═════╬═════╬═══════╬═══════╬═══════════╬═══════════╬══════════╬══════════════╣\n");
    }

    fprintf(stderr, "╚═══════════════════╩═════╩═════╩═══════╩═══════╩═══════════╩═══════════╩══════════╩══════════════╝\n\n");

    // ========================================================================
    // EXPERIMENT 2: KB sweep for best TN/CPT configs
    // ========================================================================
    // Only sweep grid_kpar shapes (QKV, Wo, FFN Down — the 3 that use grid_kpar)
    const Shape kpar_shapes[] = {
        {"QKV fused",  4608,  3584},
        {"Wo",         3584,  3584},
        {"FFN Down",   3584, 18944},
    };

    fprintf(stderr, "╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  EXPERIMENT 2: KB Sweep for top TN/CPT configs (grid_kpar shapes only)                     ║\n");
    fprintf(stderr, "╠═══════════════════╦════════════╦═══════╦═══════════╦═══════════╦══════════╦════════════════╣\n");
    fprintf(stderr, "║ Shape             ║ TN/CPT     ║  KB   ║  Min(μs)  ║ Mean(μs)  ║  GB/s    ║ Wavefronts     ║\n");
    fprintf(stderr, "╠═══════════════════╬════════════╬═══════╬═══════════╬═══════════╬══════════╬════════════════╣\n");

    // Test the most promising configs from Experiment 1
    struct TopConfig { int TN; int CPT; const char* tag; };
    TopConfig top_configs[] = {
        {128, 2, "128/2"},   // 64 threads = 1 wavefront
        {128, 4, "128/4"},   // 32 threads = 0.5 wavefront (current prod)
        {256, 4, "256/4"},   // 64 threads = 1 wavefront
    };

    for (int si = 0; si < 3; ++si) {
        const auto& shape = kpar_shapes[si];
        const int N = shape.N;
        const int K = shape.K;

        // Pack VNNI
        int B_size = K * N;
        std::vector<int8_t> h_B_sub(h_B_row.begin(), h_B_row.begin() + B_size);
        std::vector<int8_t> h_B_vnni;
        packVnniWeights(h_B_sub, N, K, h_B_vnni);
        hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size(), hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        for (const auto& cfg : top_configs) {
            if (N % cfg.CPT != 0) continue;

            for (int ki = 0; ki < num_kb; ++ki) {
                int kb = kb_values[ki];
                // Skip KB values that don't make sense for K
                int k_per = (K + kb - 1) / kb;
                if (k_per < 4) continue;  // less than 1 VNNI group per block = useless

                BenchResult r;
                if (cfg.TN == 128 && cfg.CPT == 2)
                    r = runBench<128, 2>(d_A, d_B_vnni, d_C, N, K, kb);
                else if (cfg.TN == 128 && cfg.CPT == 4)
                    r = runBench<128, 4>(d_A, d_B_vnni, d_C, N, K, kb);
                else if (cfg.TN == 256 && cfg.CPT == 4)
                    r = runBench<256, 4>(d_A, d_B_vnni, d_C, N, K, kb);
                else
                    continue;

                const int prod_kb = std::max(2, std::min(64, (K + 63) / 64));
                const char* marker = (cfg.TN == 128 && cfg.CPT == 4 && kb == prod_kb) ? " ◄PROD" : "";

                fprintf(stderr, "║ %-17s ║ %-10s ║ %5d ║ %9.1f ║ %9.1f ║ %8.1f ║ %12d%s ║\n",
                        shape.name, cfg.tag, kb,
                        r.min_us, r.mean_us, r.gbps, r.grid_x * kb, marker);
            }

            fprintf(stderr, "╠═══════════════════╬════════════╬═══════╬═══════════╬═══════════╬══════════╬════════════════╣\n");
        }
    }

    fprintf(stderr, "╚═══════════════════╩════════════╩═══════╩═══════════╩═══════════╩══════════╩════════════════╝\n\n");

    // ========================================================================
    // EXPERIMENT 3: Wide kernel tuning for GateUp/LM Head
    // ========================================================================
    // wide_vec4 handles GateUp (37888×3584) and LM Head (152064×3584)
    // Compare with grid_kpar alternatives that might be better
    fprintf(stderr, "╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  EXPERIMENT 3: Wide shapes — grid_kpar vs wide_vec4 dispatch                                ║\n");
    fprintf(stderr, "╠═══════════════════╦════════════╦═══════╦═══════════╦═══════════╦══════════╦════════════════╣\n");
    fprintf(stderr, "║ Shape             ║ Kernel     ║  KB   ║  Min(μs)  ║ Mean(μs)  ║  GB/s    ║ Notes          ║\n");
    fprintf(stderr, "╠═══════════════════╬════════════╬═══════╬═══════════╬═══════════╬══════════╬════════════════╣\n");

    const Shape wide_shapes[] = {
        {"GateUp",   37888, 3584},
        {"LM Head", 152064, 3584},
    };

    for (int si = 0; si < 2; ++si) {
        const auto& shape = wide_shapes[si];
        const int N = shape.N;
        const int K = shape.K;

        // Pack VNNI
        int B_size = K * N;
        std::vector<int8_t> h_B_sub(h_B_row.begin(), h_B_row.begin() + B_size);
        std::vector<int8_t> h_B_vnni;
        packVnniWeights(h_B_sub, N, K, h_B_vnni);
        hipMemcpy(d_B_vnni, h_B_vnni.data(), h_B_vnni.size(), hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        // Test grid_kpar<256,4> with various KB
        int kb_sweep[] = {16, 32, 48, 56, 64, 96};
        for (int kb : kb_sweep) {
            auto r = runBench<256, 4>(d_A, d_B_vnni, d_C, N, K, kb);
            fprintf(stderr, "║ %-17s ║ kpar 256/4 ║ %5d ║ %9.1f ║ %9.1f ║ %8.1f ║                ║\n",
                    shape.name, kb, r.min_us, r.mean_us, r.gbps);
        }
        // Also test grid_kpar<128,2>
        int kb_sweep2[] = {32, 56, 64};
        for (int kb : kb_sweep2) {
            auto r = runBench<128, 2>(d_A, d_B_vnni, d_C, N, K, kb);
            fprintf(stderr, "║ %-17s ║ kpar 128/2 ║ %5d ║ %9.1f ║ %9.1f ║ %8.1f ║                ║\n",
                    shape.name, kb, r.min_us, r.mean_us, r.gbps);
        }

        fprintf(stderr, "╠═══════════════════╬════════════╬═══════╬═══════════╬═══════════╬══════════╬════════════════╣\n");
    }

    fprintf(stderr, "╚═══════════════════╩════════════╩═══════╩═══════════╩═══════════╩══════════╩════════════════╝\n\n");

    // Cleanup
    hipFree(d_A);
    hipFree(d_B_vnni);
    hipFree(d_C);

    fprintf(stderr, "Done.\n");
    return 0;
}
