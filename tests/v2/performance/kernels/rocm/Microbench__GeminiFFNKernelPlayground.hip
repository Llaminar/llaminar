/**
 * @file Microbench__GeminiFFNKernelPlayground.hip
 * @brief Fast standalone ROCm micro-harness for Qwen2.5 FFN Up/Gate kernel tuning.
 *
 * Purpose:
 * - Reproduce the exact hot-path shape family used in production FFN tuning:
 *   - Qwen2.5-0.5B FFN Up/Gate:  M=128, N=4864,  K=896
 *   - Qwen2.5-3B   FFN Up/Gate:  M=128, N=11008, K=2048
 * - Keep a stable production-like baseline kernel side-by-side with a
 *   "candidate" kernel section for rapid edits by an external coding agent.
 * - Measure candidate speedup and correctness (exact INT32 parity) per shape.
 *
 * Build target (added in tests/v2/CMakeLists.txt):
 *   v2_perf_rocm_gemini_ffn_kernel_playground
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define HIP_CHECK(cmd)                                                          \
    do                                                                          \
    {                                                                           \
        hipError_t err__ = (cmd);                                               \
        if (err__ != hipSuccess)                                                \
        {                                                                       \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                         hipGetErrorString(err__), __FILE__, __LINE__);         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace
{

struct ShapeCase
{
    const char *name;
    int M;
    int N;
    int K;
};

constexpr std::array<ShapeCase, 4> kShapeCases = {
    ShapeCase{"Qwen2.5-0.5B_FFN_Up", 128, 4864, 896},
    ShapeCase{"Qwen2.5-0.5B_FFN_Gate", 128, 4864, 896},
    ShapeCase{"Qwen2.5-3B_FFN_Up", 128, 11008, 2048},
    ShapeCase{"Qwen2.5-3B_FFN_Gate", 128, 11008, 2048},
};

struct RunConfig
{
    int warmup = 2;
    int iters = 8;
    int split_k_slices = 4;
    int seed = 1337;
    bool check = true;
};

struct TimingResult
{
    float avg_ms = 0.0f;
    double tops = 0.0;
};

inline int clamp_positive(int x, int fallback)
{
    return x > 0 ? x : fallback;
}

RunConfig parseArgs(int argc, char **argv)
{
    RunConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--warmup" && (i + 1) < argc)
        {
            cfg.warmup = clamp_positive(std::atoi(argv[++i]), cfg.warmup);
        }
        else if (arg == "--iters" && (i + 1) < argc)
        {
            cfg.iters = clamp_positive(std::atoi(argv[++i]), cfg.iters);
        }
        else if (arg == "--split-k" && (i + 1) < argc)
        {
            cfg.split_k_slices = clamp_positive(std::atoi(argv[++i]), cfg.split_k_slices);
        }
        else if (arg == "--seed" && (i + 1) < argc)
        {
            cfg.seed = std::atoi(argv[++i]);
        }
        else if (arg == "--no-check")
        {
            cfg.check = false;
        }
        else if (arg == "--help")
        {
            std::printf("Usage: %s [--warmup N] [--iters N] [--split-k N] [--seed N] [--no-check]\n", argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

void fillRandomInt8(std::vector<int8_t> &data, int seed, int lo = -8, int hi = 8)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    for (auto &v : data)
    {
        v = static_cast<int8_t>(dist(rng));
    }
}

// Layout: B_vnni[(kg * N + n) * 4 + lane], lane in [0..3]
void makeRandomBVNNI(std::vector<int8_t> &b_vnni, int N, int K, int seed)
{
    if ((K % 4) != 0)
    {
        std::fprintf(stderr, "K must be multiple of 4 for VNNI pack (K=%d)\n", K);
        std::exit(1);
    }
    fillRandomInt8(b_vnni, seed);
    (void)N;
}

template <bool IsCandidate>
__global__ void gridkpar_32x8_cpt4_kernel(
    const int8_t *__restrict__ d_A_int8,
    const int8_t *__restrict__ d_B_int8_vnni,
    int32_t *__restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    constexpr int CPT = 4;

    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int total_k_groups = K / 4;
    const int k_begin = split_idx * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return;
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    const bool full_tile = (n_base + CPT - 1) < N;

    const int8_t *a_ptr0 = d_A_int8 + static_cast<int64_t>(m) * K + static_cast<int64_t>(k_begin) * 4;
    const int8_t *b_ptr0 = d_B_int8_vnni + (static_cast<int64_t>(k_begin) * N + n_base) * 4;

    int32_t acc[4] = {0, 0, 0, 0};

    // ---------------------------------------------------------------------
    // GEMINI TUNING ZONE START
    //
    // Edit the candidate branch only. Keep baseline untouched to preserve
    // apples-to-apples comparisons.
    // ---------------------------------------------------------------------
    if constexpr (IsCandidate)
    {
        if (full_tile)
        {
            const int8_t *a_ptr = a_ptr0;
            const int32_t *b_ptr = reinterpret_cast<const int32_t *>(b_ptr0);
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t *>(a_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_ptr[0], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_ptr[1], acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b_ptr[2], acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b_ptr[3], acc[3], false);
                a_ptr += 4;
                b_ptr += b_row_stride_i32;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t *a_ptr = a_ptr0;
            const int32_t *b0 = reinterpret_cast<const int32_t *>(b_ptr0);
            const int32_t *b1 = b0 + 1;
            const int32_t *b2 = b0 + 2;
            const int32_t *b3 = b0 + 3;
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t *>(a_ptr);
                if (valid > 0)
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                if (valid > 1)
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                if (valid > 2)
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, *b2, acc[2], false);
                if (valid > 3)
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, *b3, acc[3], false);
                a_ptr += 4;
                b0 += b_row_stride_i32;
                b1 += b_row_stride_i32;
                b2 += b_row_stride_i32;
                b3 += b_row_stride_i32;
            }
        }
    }
    else
    {
        // Baseline (production-like) reference path
        if (full_tile)
        {
            const int8_t *a_ptr = a_ptr0;
            const int32_t *b_ptr = reinterpret_cast<const int32_t *>(b_ptr0);
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t *>(a_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_ptr[0], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_ptr[1], acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b_ptr[2], acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b_ptr[3], acc[3], false);
                a_ptr += 4;
                b_ptr += b_row_stride_i32;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t *a_ptr = a_ptr0;
            const int32_t *b0 = reinterpret_cast<const int32_t *>(b_ptr0);
            const int32_t *b1 = b0 + 1;
            const int32_t *b2 = b0 + 2;
            const int32_t *b3 = b0 + 3;
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t *>(a_ptr);
                if (valid > 0)
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                if (valid > 1)
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                if (valid > 2)
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, *b2, acc[2], false);
                if (valid > 3)
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, *b3, acc[3], false);
                a_ptr += 4;
                b0 += b_row_stride_i32;
                b1 += b_row_stride_i32;
                b2 += b_row_stride_i32;
                b3 += b_row_stride_i32;
            }
        }
    }
    // ---------------------------------------------------------------------
    // GEMINI TUNING ZONE END
    // ---------------------------------------------------------------------

    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
        }
    }
}

template <bool IsCandidate>
TimingResult runKernelCase(
    const ShapeCase &shape,
    const RunConfig &cfg,
    const int8_t *d_A,
    const int8_t *d_B_vnni,
    int32_t *d_C)
{
    const int total_k_groups = shape.K / 4;
    const int slices = std::max(1, std::min(cfg.split_k_slices, total_k_groups));
    const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

    constexpr int BX = 32;
    constexpr int BY = 8;
    constexpr int CPT = 4;
    const dim3 block(BX, BY, 1);
    const dim3 grid(
        static_cast<unsigned>((shape.N + BX * CPT - 1) / (BX * CPT)),
        static_cast<unsigned>((shape.M + BY - 1) / BY),
        static_cast<unsigned>(slices));

    hipEvent_t start_evt = nullptr;
    hipEvent_t stop_evt = nullptr;
    HIP_CHECK(hipEventCreate(&start_evt));
    HIP_CHECK(hipEventCreate(&stop_evt));

    const size_t c_bytes = static_cast<size_t>(shape.M) * shape.N * sizeof(int32_t);

    for (int i = 0; i < cfg.warmup; ++i)
    {
        HIP_CHECK(hipMemset(d_C, 0, c_bytes));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gridkpar_32x8_cpt4_kernel<IsCandidate>),
            grid,
            block,
            0,
            nullptr,
            d_A,
            d_B_vnni,
            d_C,
            shape.M,
            shape.N,
            shape.K,
            k_groups_per_slice);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    float elapsed_sum_ms = 0.0f;
    for (int i = 0; i < cfg.iters; ++i)
    {
        HIP_CHECK(hipMemset(d_C, 0, c_bytes));
        HIP_CHECK(hipEventRecord(start_evt));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gridkpar_32x8_cpt4_kernel<IsCandidate>),
            grid,
            block,
            0,
            nullptr,
            d_A,
            d_B_vnni,
            d_C,
            shape.M,
            shape.N,
            shape.K,
            k_groups_per_slice);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop_evt));
        HIP_CHECK(hipEventSynchronize(stop_evt));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_evt, stop_evt));
        elapsed_sum_ms += ms;
    }

    HIP_CHECK(hipEventDestroy(start_evt));
    HIP_CHECK(hipEventDestroy(stop_evt));

    TimingResult out;
    out.avg_ms = elapsed_sum_ms / static_cast<float>(cfg.iters);
    const double ops = 2.0 * static_cast<double>(shape.M) * static_cast<double>(shape.N) * static_cast<double>(shape.K);
    out.tops = (ops / (out.avg_ms / 1000.0)) / 1.0e12;
    return out;
}

int64_t maxAbsDiff(const std::vector<int32_t> &a, const std::vector<int32_t> &b)
{
    int64_t max_diff = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i)
    {
        const int64_t d = std::llabs(static_cast<long long>(a[i]) - static_cast<long long>(b[i]));
        max_diff = std::max(max_diff, d);
    }
    return max_diff;
}

void computeCpuReferenceOpenMP(
    const ShapeCase &shape,
    const std::vector<int8_t> &h_A,
    const std::vector<int8_t> &h_B_vnni,
    std::vector<int32_t> &h_C_ref)
{
    const int M = shape.M;
    const int N = shape.N;
    const int K = shape.K;
    const int total_k_groups = K / 4;

    h_C_ref.assign(static_cast<size_t>(M) * N, 0);

    #pragma omp parallel for schedule(static) collapse(2)
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            int32_t acc = 0;
            for (int kg = 0; kg < total_k_groups; ++kg)
            {
                const int64_t a_off = static_cast<int64_t>(m) * K + static_cast<int64_t>(kg) * 4;
                const int64_t b_off = (static_cast<int64_t>(kg) * N + n) * 4;

                acc += static_cast<int32_t>(h_A[a_off + 0]) * static_cast<int32_t>(h_B_vnni[b_off + 0]);
                acc += static_cast<int32_t>(h_A[a_off + 1]) * static_cast<int32_t>(h_B_vnni[b_off + 1]);
                acc += static_cast<int32_t>(h_A[a_off + 2]) * static_cast<int32_t>(h_B_vnni[b_off + 2]);
                acc += static_cast<int32_t>(h_A[a_off + 3]) * static_cast<int32_t>(h_B_vnni[b_off + 3]);
            }
            h_C_ref[static_cast<int64_t>(m) * N + n] = acc;
        }
    }
}

int getOpenMpThreadCount()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void runOneShape(const ShapeCase &shape, const RunConfig &cfg)
{
    std::printf("\n[Shape] %s (M=%d N=%d K=%d)\n", shape.name, shape.M, shape.N, shape.K);

    const size_t a_elems = static_cast<size_t>(shape.M) * shape.K;
    const size_t b_elems = static_cast<size_t>(shape.K) * shape.N; // VNNI bytes == K*N
    const size_t c_elems = static_cast<size_t>(shape.M) * shape.N;

    std::vector<int8_t> h_A(a_elems);
    std::vector<int8_t> h_B_vnni(b_elems);
    fillRandomInt8(h_A, cfg.seed + shape.K + shape.N + 17);
    makeRandomBVNNI(h_B_vnni, shape.N, shape.K, cfg.seed + shape.K + shape.N + 31);

    int8_t *d_A = nullptr;
    int8_t *d_B = nullptr;
    int32_t *d_C_baseline = nullptr;
    int32_t *d_C_candidate = nullptr;

    HIP_CHECK(hipMalloc(&d_A, a_elems * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_B, b_elems * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_C_baseline, c_elems * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_C_candidate, c_elems * sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), a_elems * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B_vnni.data(), b_elems * sizeof(int8_t), hipMemcpyHostToDevice));

    const TimingResult baseline = runKernelCase<false>(shape, cfg, d_A, d_B, d_C_baseline);
    const TimingResult candidate = runKernelCase<true>(shape, cfg, d_A, d_B, d_C_candidate);

    int64_t max_abs_diff_base_vs_ref = -1;
    int64_t max_abs_diff_cand_vs_ref = -1;
    int64_t max_abs_diff_base_vs_cand = -1;
    if (cfg.check)
    {
        std::vector<int32_t> h_base(c_elems);
        std::vector<int32_t> h_cand(c_elems);
        std::vector<int32_t> h_ref;

        HIP_CHECK(hipMemcpy(h_base.data(), d_C_baseline, c_elems * sizeof(int32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_cand.data(), d_C_candidate, c_elems * sizeof(int32_t), hipMemcpyDeviceToHost));
        computeCpuReferenceOpenMP(shape, h_A, h_B_vnni, h_ref);

        max_abs_diff_base_vs_ref = maxAbsDiff(h_base, h_ref);
        max_abs_diff_cand_vs_ref = maxAbsDiff(h_cand, h_ref);
        max_abs_diff_base_vs_cand = maxAbsDiff(h_base, h_cand);
    }

    const float speedup = baseline.avg_ms / std::max(candidate.avg_ms, 1e-9f);

    std::printf("  baseline:  %8.4f ms   %6.3f TOPS\n", baseline.avg_ms, baseline.tops);
    std::printf("  candidate: %8.4f ms   %6.3f TOPS\n", candidate.avg_ms, candidate.tops);
    std::printf("  speedup (baseline/candidate): %.4fx\n", speedup);
    if (cfg.check)
    {
        std::printf("  max_abs_diff(INT32) baseline_vs_cpu_ref:  %lld\n", static_cast<long long>(max_abs_diff_base_vs_ref));
        std::printf("  max_abs_diff(INT32) candidate_vs_cpu_ref: %lld\n", static_cast<long long>(max_abs_diff_cand_vs_ref));
        std::printf("  max_abs_diff(INT32) baseline_vs_candidate:%lld\n", static_cast<long long>(max_abs_diff_base_vs_cand));
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C_baseline));
    HIP_CHECK(hipFree(d_C_candidate));
}

} // namespace

int main(int argc, char **argv)
{
    const RunConfig cfg = parseArgs(argc, argv);

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count <= 0)
    {
        std::fprintf(stderr, "No ROCm device found.\n");
        return 1;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::printf("============================================================\n");
    std::printf("Gemini FFN Kernel Playground (ROCm HIP)\n");
    std::printf("Device: %s\n", prop.name);
    std::printf("Config: warmup=%d iters=%d split_k=%d check=%s omp_threads=%d\n",
                cfg.warmup,
                cfg.iters,
                cfg.split_k_slices,
                cfg.check ? "on" : "off",
                getOpenMpThreadCount());
    std::printf("============================================================\n");

    for (const auto &shape : kShapeCases)
    {
        runOneShape(shape, cfg);
    }

    std::printf("\nDone. Edit candidate branch in gridkpar_32x8_cpt4_kernel<true> and rerun.\n");
    return 0;
}
