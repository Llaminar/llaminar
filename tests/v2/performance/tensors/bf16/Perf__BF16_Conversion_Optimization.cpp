/**
 * @file Perf__BF16_Conversion_Optimization.cpp
 * @brief Targeted BF16 conversion performance optimization benchmark
 *
 * Purpose: Measure baseline BF16 conversion bandwidth and test optimizations side-by-side.
 *
 * Test Strategy:
 * 1. Baseline: Current SIMDHelpers.h implementation
 * 2. Opt1: Loop unrolling (4× unroll)
 * 3. Opt2: Loop unrolling + software prefetching
 * 4. Opt3: Loop unrolling + prefetching + aligned loads
 * 5. Opt4: Full optimization (unroll + prefetch + aligned + streaming stores)
 *
 * Metrics: GB/s bandwidth, speedup vs baseline, CPU cycles/element
 */

#include "utils/MPISessionOwner.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "tensors/SIMDHelpers.h"

using namespace llaminar2;

namespace
{

    // ============================================================================
    // Baseline Implementation (Current)
    // ============================================================================

    inline void convert_bf16_to_fp32_baseline(const uint16_t *src, float *dst, size_t count)
    {
        simd::convert_bf16_to_fp32(src, dst, count);
    }

    inline void convert_fp32_to_bf16_baseline(const float *src, uint16_t *dst, size_t count)
    {
        simd::convert_fp32_to_bf16(src, dst, count);
    }

    // ============================================================================
    // Optimization 1: Loop Unrolling (4× unroll)
    // ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
    inline void convert_bf16_to_fp32_opt1_unroll4(const uint16_t *src, float *dst, size_t count)
    {
        size_t i = 0;

        // Main loop: Process 128 elements (4×32) per iteration
        for (; i + 128 <= count; i += 128)
        {
            // Unroll 1
            __m512i bf16_vec0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo0 = _mm512_extracti64x4_epi64(bf16_vec0, 0);
            __m256i bf16_hi0 = _mm512_extracti64x4_epi64(bf16_vec0, 1);
            __m512i fp32_lo0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo0), 16);
            __m512i fp32_hi0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi0), 16);
            _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(fp32_lo0));
            _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi0));

            // Unroll 2
            __m512i bf16_vec1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i + 32));
            __m256i bf16_lo1 = _mm512_extracti64x4_epi64(bf16_vec1, 0);
            __m256i bf16_hi1 = _mm512_extracti64x4_epi64(bf16_vec1, 1);
            __m512i fp32_lo1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo1), 16);
            __m512i fp32_hi1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi1), 16);
            _mm512_storeu_ps(dst + i + 32, _mm512_castsi512_ps(fp32_lo1));
            _mm512_storeu_ps(dst + i + 48, _mm512_castsi512_ps(fp32_hi1));

            // Unroll 3
            __m512i bf16_vec2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i + 64));
            __m256i bf16_lo2 = _mm512_extracti64x4_epi64(bf16_vec2, 0);
            __m256i bf16_hi2 = _mm512_extracti64x4_epi64(bf16_vec2, 1);
            __m512i fp32_lo2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo2), 16);
            __m512i fp32_hi2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi2), 16);
            _mm512_storeu_ps(dst + i + 64, _mm512_castsi512_ps(fp32_lo2));
            _mm512_storeu_ps(dst + i + 80, _mm512_castsi512_ps(fp32_hi2));

            // Unroll 4
            __m512i bf16_vec3 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i + 96));
            __m256i bf16_lo3 = _mm512_extracti64x4_epi64(bf16_vec3, 0);
            __m256i bf16_hi3 = _mm512_extracti64x4_epi64(bf16_vec3, 1);
            __m512i fp32_lo3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo3), 16);
            __m512i fp32_hi3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi3), 16);
            _mm512_storeu_ps(dst + i + 96, _mm512_castsi512_ps(fp32_lo3));
            _mm512_storeu_ps(dst + i + 112, _mm512_castsi512_ps(fp32_hi3));
        }

        // Tail: Process remaining elements with baseline
        for (; i < count; i += 32)
        {
            __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
            __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);
            __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
            __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);
            _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(fp32_lo));
            _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi));
        }
    }
#else
    inline void convert_bf16_to_fp32_opt1_unroll4(const uint16_t *src, float *dst, size_t count)
    {
        convert_bf16_to_fp32_baseline(src, dst, count);
    }
#endif

    // ============================================================================
    // Optimization 2: Unroll + Prefetching
    // ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
    inline void convert_bf16_to_fp32_opt2_prefetch(const uint16_t *src, float *dst, size_t count)
    {
        constexpr size_t PREFETCH_DISTANCE = 512; // Prefetch 512 bytes ahead (8 cache lines)
        size_t i = 0;

        // Main loop: Process 128 elements (4×32) per iteration
        for (; i + 128 <= count; i += 128)
        {
            // Prefetch next iteration's data
            _mm_prefetch(reinterpret_cast<const char *>(src + i + PREFETCH_DISTANCE), _MM_HINT_T0);

            // Unroll 1
            __m512i bf16_vec0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo0 = _mm512_extracti64x4_epi64(bf16_vec0, 0);
            __m256i bf16_hi0 = _mm512_extracti64x4_epi64(bf16_vec0, 1);
            __m512i fp32_lo0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo0), 16);
            __m512i fp32_hi0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi0), 16);
            _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(fp32_lo0));
            _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi0));

            // Unroll 2
            __m512i bf16_vec1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i + 32));
            __m256i bf16_lo1 = _mm512_extracti64x4_epi64(bf16_vec1, 0);
            __m256i bf16_hi1 = _mm512_extracti64x4_epi64(bf16_vec1, 1);
            __m512i fp32_lo1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo1), 16);
            __m512i fp32_hi1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi1), 16);
            _mm512_storeu_ps(dst + i + 32, _mm512_castsi512_ps(fp32_lo1));
            _mm512_storeu_ps(dst + i + 48, _mm512_castsi512_ps(fp32_hi1));

            // Unroll 3
            __m512i bf16_vec2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i + 64));
            __m256i bf16_lo2 = _mm512_extracti64x4_epi64(bf16_vec2, 0);
            __m256i bf16_hi2 = _mm512_extracti64x4_epi64(bf16_vec2, 1);
            __m512i fp32_lo2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo2), 16);
            __m512i fp32_hi2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi2), 16);
            _mm512_storeu_ps(dst + i + 64, _mm512_castsi512_ps(fp32_lo2));
            _mm512_storeu_ps(dst + i + 80, _mm512_castsi512_ps(fp32_hi2));

            // Unroll 4
            __m512i bf16_vec3 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i + 96));
            __m256i bf16_lo3 = _mm512_extracti64x4_epi64(bf16_vec3, 0);
            __m256i bf16_hi3 = _mm512_extracti64x4_epi64(bf16_vec3, 1);
            __m512i fp32_lo3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo3), 16);
            __m512i fp32_hi3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi3), 16);
            _mm512_storeu_ps(dst + i + 96, _mm512_castsi512_ps(fp32_lo3));
            _mm512_storeu_ps(dst + i + 112, _mm512_castsi512_ps(fp32_hi3));
        }

        // Tail
        for (; i < count; i += 32)
        {
            __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
            __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);
            __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
            __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);
            _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(fp32_lo));
            _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi));
        }
    }
#else
    inline void convert_bf16_to_fp32_opt2_prefetch(const uint16_t *src, float *dst, size_t count)
    {
        convert_bf16_to_fp32_baseline(src, dst, count);
    }
#endif

    // ============================================================================
    // Optimization 3: Unroll + Prefetch + Aligned Loads
    // ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
    inline void convert_bf16_to_fp32_opt3_aligned(const uint16_t *__restrict__ src, float *__restrict__ dst, size_t count)
    {
        // Assumes src and dst are 64-byte aligned
        constexpr size_t PREFETCH_DISTANCE = 512;
        size_t i = 0;

        for (; i + 128 <= count; i += 128)
        {
            _mm_prefetch(reinterpret_cast<const char *>(src + i + PREFETCH_DISTANCE), _MM_HINT_T0);

            // Unroll 1 (aligned load)
            __m512i bf16_vec0 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo0 = _mm512_extracti64x4_epi64(bf16_vec0, 0);
            __m256i bf16_hi0 = _mm512_extracti64x4_epi64(bf16_vec0, 1);
            __m512i fp32_lo0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo0), 16);
            __m512i fp32_hi0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi0), 16);
            _mm512_store_ps(dst + i, _mm512_castsi512_ps(fp32_lo0));
            _mm512_store_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi0));

            // Unroll 2
            __m512i bf16_vec1 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i + 32));
            __m256i bf16_lo1 = _mm512_extracti64x4_epi64(bf16_vec1, 0);
            __m256i bf16_hi1 = _mm512_extracti64x4_epi64(bf16_vec1, 1);
            __m512i fp32_lo1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo1), 16);
            __m512i fp32_hi1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi1), 16);
            _mm512_store_ps(dst + i + 32, _mm512_castsi512_ps(fp32_lo1));
            _mm512_store_ps(dst + i + 48, _mm512_castsi512_ps(fp32_hi1));

            // Unroll 3
            __m512i bf16_vec2 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i + 64));
            __m256i bf16_lo2 = _mm512_extracti64x4_epi64(bf16_vec2, 0);
            __m256i bf16_hi2 = _mm512_extracti64x4_epi64(bf16_vec2, 1);
            __m512i fp32_lo2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo2), 16);
            __m512i fp32_hi2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi2), 16);
            _mm512_store_ps(dst + i + 64, _mm512_castsi512_ps(fp32_lo2));
            _mm512_store_ps(dst + i + 80, _mm512_castsi512_ps(fp32_hi2));

            // Unroll 4
            __m512i bf16_vec3 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i + 96));
            __m256i bf16_lo3 = _mm512_extracti64x4_epi64(bf16_vec3, 0);
            __m256i bf16_hi3 = _mm512_extracti64x4_epi64(bf16_vec3, 1);
            __m512i fp32_lo3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo3), 16);
            __m512i fp32_hi3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi3), 16);
            _mm512_store_ps(dst + i + 96, _mm512_castsi512_ps(fp32_lo3));
            _mm512_store_ps(dst + i + 112, _mm512_castsi512_ps(fp32_hi3));
        }

        // Tail
        for (; i < count; i += 32)
        {
            __m512i bf16_vec = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
            __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);
            __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
            __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);
            _mm512_store_ps(dst + i, _mm512_castsi512_ps(fp32_lo));
            _mm512_store_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi));
        }
    }
#else
    inline void convert_bf16_to_fp32_opt3_aligned(const uint16_t *src, float *dst, size_t count)
    {
        convert_bf16_to_fp32_baseline(src, dst, count);
    }
#endif

    // ============================================================================
    // Optimization 4: Full Optimization (Streaming Stores)
    // ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
    inline void convert_bf16_to_fp32_opt4_streaming(const uint16_t *__restrict__ src, float *__restrict__ dst, size_t count)
    {
        constexpr size_t PREFETCH_DISTANCE = 512;
        size_t i = 0;

        for (; i + 128 <= count; i += 128)
        {
            _mm_prefetch(reinterpret_cast<const char *>(src + i + PREFETCH_DISTANCE), _MM_HINT_T0);

            // Unroll 1 (streaming stores bypass cache)
            __m512i bf16_vec0 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo0 = _mm512_extracti64x4_epi64(bf16_vec0, 0);
            __m256i bf16_hi0 = _mm512_extracti64x4_epi64(bf16_vec0, 1);
            __m512i fp32_lo0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo0), 16);
            __m512i fp32_hi0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi0), 16);
            _mm512_stream_ps(dst + i, _mm512_castsi512_ps(fp32_lo0));
            _mm512_stream_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi0));

            // Unroll 2
            __m512i bf16_vec1 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i + 32));
            __m256i bf16_lo1 = _mm512_extracti64x4_epi64(bf16_vec1, 0);
            __m256i bf16_hi1 = _mm512_extracti64x4_epi64(bf16_vec1, 1);
            __m512i fp32_lo1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo1), 16);
            __m512i fp32_hi1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi1), 16);
            _mm512_stream_ps(dst + i + 32, _mm512_castsi512_ps(fp32_lo1));
            _mm512_stream_ps(dst + i + 48, _mm512_castsi512_ps(fp32_hi1));

            // Unroll 3
            __m512i bf16_vec2 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i + 64));
            __m256i bf16_lo2 = _mm512_extracti64x4_epi64(bf16_vec2, 0);
            __m256i bf16_hi2 = _mm512_extracti64x4_epi64(bf16_vec2, 1);
            __m512i fp32_lo2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo2), 16);
            __m512i fp32_hi2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi2), 16);
            _mm512_stream_ps(dst + i + 64, _mm512_castsi512_ps(fp32_lo2));
            _mm512_stream_ps(dst + i + 80, _mm512_castsi512_ps(fp32_hi2));

            // Unroll 4
            __m512i bf16_vec3 = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i + 96));
            __m256i bf16_lo3 = _mm512_extracti64x4_epi64(bf16_vec3, 0);
            __m256i bf16_hi3 = _mm512_extracti64x4_epi64(bf16_vec3, 1);
            __m512i fp32_lo3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo3), 16);
            __m512i fp32_hi3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi3), 16);
            _mm512_stream_ps(dst + i + 96, _mm512_castsi512_ps(fp32_lo3));
            _mm512_stream_ps(dst + i + 112, _mm512_castsi512_ps(fp32_hi3));
        }

        // Tail
        for (; i < count; i += 32)
        {
            __m512i bf16_vec = _mm512_load_si512(reinterpret_cast<const __m512i *>(src + i));
            __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0);
            __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1);
            __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
            __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);
            _mm512_stream_ps(dst + i, _mm512_castsi512_ps(fp32_lo));
            _mm512_stream_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi));
        }

        // Ensure all streaming stores complete before function returns
        _mm_sfence();
    }
#else
    inline void convert_bf16_to_fp32_opt4_streaming(const uint16_t *src, float *dst, size_t count)
    {
        convert_bf16_to_fp32_baseline(src, dst, count);
    }
#endif

    // ============================================================================
    // Benchmark Infrastructure
    // ============================================================================

    struct ConversionBenchmarkResult
    {
        std::string name;
        double bandwidth_gb_s;
        double time_ms;
        double speedup_vs_baseline;
        double stddev_gb_s;
    };

    /**
     * @brief Benchmark a single conversion function
     */
    template <typename ConvFunc>
    ConversionBenchmarkResult benchmarkConversion(
        const std::string &name,
        ConvFunc func,
        const uint16_t *src,
        float *dst,
        size_t count,
        int warmup_iters,
        int bench_iters,
        int num_trials,
        double baseline_bandwidth = 0.0)
    {
        std::vector<double> bandwidths;
        bandwidths.reserve(num_trials);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        for (int trial = 0; trial < num_trials; ++trial)
        {
            // Warmup
            for (int i = 0; i < warmup_iters; ++i)
            {
                func(src, dst, count);
            }

            // Benchmark
            MPI_Barrier(MPI_COMM_WORLD);
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < bench_iters; ++i)
            {
                func(src, dst, count);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto t1 = std::chrono::high_resolution_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double time_per_iter_ms = elapsed_ms / bench_iters;

            // Calculate bandwidth (read BF16 + write FP32)
            double bytes_transferred = count * (sizeof(uint16_t) + sizeof(float));
            double bandwidth_gb_s = (bytes_transferred / 1e9) / (time_per_iter_ms / 1000.0);

            bandwidths.push_back(bandwidth_gb_s);
        }

        // Calculate statistics
        double mean_bw = std::accumulate(bandwidths.begin(), bandwidths.end(), 0.0) / bandwidths.size();
        double variance = 0.0;
        for (auto bw : bandwidths)
        {
            variance += (bw - mean_bw) * (bw - mean_bw);
        }
        double stddev_bw = std::sqrt(variance / bandwidths.size());

        // Calculate mean time
        double bytes_transferred = count * (sizeof(uint16_t) + sizeof(float));
        double mean_time_ms = (bytes_transferred / 1e9) / mean_bw * 1000.0;

        ConversionBenchmarkResult result;
        result.name = name;
        result.bandwidth_gb_s = mean_bw;
        result.time_ms = mean_time_ms;
        result.stddev_gb_s = stddev_bw;
        result.speedup_vs_baseline = (baseline_bandwidth > 0.0) ? (mean_bw / baseline_bandwidth) : 1.0;

        return result;
    }

    // ============================================================================
    // Test Cases
    // ============================================================================

    /**
     * @brief Benchmark all optimization variants side-by-side
     */
    TEST(BF16ConversionOpt, CompareAllVariants)
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Test configuration
        const size_t array_size = 128 * 896 * 896; // ~100M elements (realistic matrix size)
        const int warmup = 5;
        const int iterations = 50;
        const int trials = 3;

        // Allocate arrays (unaligned for baseline/opt1/opt2, aligned for opt3/opt4)
        std::vector<uint16_t> src_unaligned(array_size, 0xBF80); // BF16: 1.0
        std::vector<float> dst_unaligned(array_size, 0.0f);

        // Aligned arrays for opt3/opt4
        uint16_t *src_aligned = static_cast<uint16_t *>(aligned_alloc(64, array_size * sizeof(uint16_t)));
        float *dst_aligned = static_cast<float *>(aligned_alloc(64, array_size * sizeof(float)));
        std::memcpy(src_aligned, src_unaligned.data(), array_size * sizeof(uint16_t));

        if (rank == 0)
        {
            std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║    BF16→FP32 CONVERSION OPTIMIZATION BENCHMARK               ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Array size:  " << std::setw(10) << array_size << " elements (~"
                      << std::fixed << std::setprecision(2) << (array_size * sizeof(uint16_t) / 1e9) << " GB)        ║\n";
            std::cout << "║ Iterations:  " << std::setw(10) << iterations << " per trial                       ║\n";
            std::cout << "║ Trials:      " << std::setw(10) << trials << "                                  ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        }

        // Run benchmarks
        std::vector<ConversionBenchmarkResult> results;

        // Baseline
        auto baseline = benchmarkConversion(
            "Baseline (Current SIMDHelpers)",
            convert_bf16_to_fp32_baseline,
            src_unaligned.data(), dst_unaligned.data(), array_size,
            warmup, iterations, trials);
        results.push_back(baseline);

        // Opt1: Loop unrolling
        results.push_back(benchmarkConversion(
            "Opt1: Loop Unrolling (4×)",
            convert_bf16_to_fp32_opt1_unroll4,
            src_unaligned.data(), dst_unaligned.data(), array_size,
            warmup, iterations, trials, baseline.bandwidth_gb_s));

        // Opt2: Unroll + prefetch
        results.push_back(benchmarkConversion(
            "Opt2: Unroll + Prefetching",
            convert_bf16_to_fp32_opt2_prefetch,
            src_unaligned.data(), dst_unaligned.data(), array_size,
            warmup, iterations, trials, baseline.bandwidth_gb_s));

        // Opt3: Unroll + prefetch + aligned
        results.push_back(benchmarkConversion(
            "Opt3: Unroll + Prefetch + Aligned",
            convert_bf16_to_fp32_opt3_aligned,
            src_aligned, dst_aligned, array_size,
            warmup, iterations, trials, baseline.bandwidth_gb_s));

        // Opt4: Full optimization (streaming)
        results.push_back(benchmarkConversion(
            "Opt4: Full (+ Streaming Stores)",
            convert_bf16_to_fp32_opt4_streaming,
            src_aligned, dst_aligned, array_size,
            warmup, iterations, trials, baseline.bandwidth_gb_s));

        // Print results
        if (rank == 0)
        {
            std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                             BENCHMARK RESULTS                                          ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Variant                          │ Bandwidth  │   Time   │  Speedup │    Stddev        ║\n";
            std::cout << "╠══════════════════════════════════╪════════════╪══════════╪══════════╪══════════════════╣\n";

            for (const auto &result : results)
            {
                std::cout << "║ " << std::left << std::setw(32) << result.name << " │ "
                          << std::right << std::setw(7) << std::fixed << std::setprecision(2) << result.bandwidth_gb_s << " GB/s │ "
                          << std::setw(6) << std::fixed << std::setprecision(2) << result.time_ms << " ms │ "
                          << std::setw(6) << std::fixed << std::setprecision(2) << result.speedup_vs_baseline << "× │ "
                          << std::setw(7) << std::fixed << std::setprecision(2) << result.stddev_gb_s << " GB/s   ║\n";
            }

            std::cout << "╚══════════════════════════════════╧════════════╧══════════╧══════════╧══════════════════╝\n\n";

            // Print improvement summary
            std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                   OPTIMIZATION SUMMARY                         ║\n";
            std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Best variant:        " << std::left << std::setw(37) << results.back().name << " ║\n";
            std::cout << "║ Best bandwidth:      " << std::setw(6) << std::fixed << std::setprecision(2)
                      << results.back().bandwidth_gb_s << " GB/s                            ║\n";
            std::cout << "║ Overall speedup:     " << std::setw(6) << std::fixed << std::setprecision(2)
                      << results.back().speedup_vs_baseline << "× vs baseline                     ║\n";
            std::cout << "║ Time improvement:    " << std::setw(6) << std::fixed << std::setprecision(2)
                      << (baseline.time_ms - results.back().time_ms) << " ms saved ("
                      << std::setw(3) << std::fixed << std::setprecision(0)
                      << ((baseline.time_ms - results.back().time_ms) / baseline.time_ms * 100) << "%)              ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        }

        // Cleanup
        free(src_aligned);
        free(dst_aligned);

        EXPECT_GT(results.back().speedup_vs_baseline, 1.0);
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    if (auto terminal = llaminar2::MPISessionOwner::enterProcess(argc, argv))
        return *terminal;

    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
