/**
 * @file CPUBenchmark.cpp
 * @brief CPU benchmark implementation for memory bandwidth and compute throughput
 *
 * Design Goals:
 * - Complete in < 100ms for quick preset, < 500ms for thorough
 * - Use realistic patterns that match inference workloads
 * - NUMA-aware for accurate multi-socket measurements
 * - Minimal false sharing in parallel sections
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CPUBenchmark.h"
#include "../../utils/CPUFeatures.h"
#include "utils/Logger.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h> // mmap for NUMA-aware allocation
#include <omp.h>

#if defined(__AVX512F__) || defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace
    {
        /// High-resolution timing
        double getTimeMs()
        {
            using clock = std::chrono::high_resolution_clock;
            auto now = clock::now();
            auto duration = now.time_since_epoch();
            return std::chrono::duration<double, std::milli>(duration).count();
        }

        /// Prevent compiler from optimizing away benchmark results
        template <typename T>
        void doNotOptimize(T &&value)
        {
#if defined(__GNUC__) || defined(__clang__)
            asm volatile("" : : "g"(value) : "memory");
#else
            volatile auto unused = value;
            (void)unused;
#endif
        }

        /// Memory barrier
        void memoryFence()
        {
#if defined(__GNUC__) || defined(__clang__)
            asm volatile("" ::: "memory");
#endif
        }

    } // anonymous namespace

    CPUBenchmark::CPUBenchmark(const BenchmarkConfig &config)
        : config_(config)
    {
    }

    double CPUBenchmark::estimatedDurationMs() const
    {
        // Estimate based on config
        double iterations = config_.warmup_iterations + config_.iterations;
        // ~10ms per iteration for memory tests, ~5ms for compute
        return iterations * 15.0 * 5; // 5 tests
    }

    int CPUBenchmark::getNumThreads() const
    {
        return omp_get_max_threads();
    }

    float *CPUBenchmark::allocateNUMABuffer(size_t num_elements)
    {
        size_t bytes = num_elements * sizeof(float);
        // Align to page boundary for optimal NUMA first-touch
        void *ptr = aligned_alloc(4096, bytes);
        if (!ptr)
        {
            LOG_ERROR("Failed to allocate " << (bytes / 1024 / 1024) << " MB for benchmark");
            return nullptr;
        }

// First-touch initialization for NUMA-local allocation
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_elements; ++i)
        {
            static_cast<float *>(ptr)[i] = 0.0f;
        }

        return static_cast<float *>(ptr);
    }

    void CPUBenchmark::freeNUMABuffer(float *buffer, size_t /*num_elements*/)
    {
        free(buffer);
    }

    double CPUBenchmark::benchmarkMemoryRead(float *buffer, size_t num_elements)
    {
        double total_time_ms = 0.0;
        double bytes_read = num_elements * sizeof(float);

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum) schedule(static)
            for (size_t i = 0; i < num_elements; ++i)
            {
                sum += buffer[i];
            }
            doNotOptimize(sum);
        }

        // Timed runs - time all iterations together to minimize timer overhead
        float sum = 0.0f;
        memoryFence();
        double start = getTimeMs();

        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            float local_sum = 0.0f;
#pragma omp parallel for reduction(+ : local_sum) schedule(static)
            for (size_t i = 0; i < num_elements; ++i)
            {
                local_sum += buffer[i];
            }
            sum += local_sum;
        }

        memoryFence();
        double end = getTimeMs();
        total_time_ms = end - start;
        doNotOptimize(sum);

        double total_bytes = bytes_read * config_.iterations;
        double time_sec = total_time_ms / 1000.0;
        return (total_bytes / time_sec) / 1e9; // GB/s
    }

    double CPUBenchmark::benchmarkMemoryWrite(float *buffer, size_t num_elements)
    {
        double total_time_ms = 0.0;
        double bytes_written = num_elements * sizeof(float);

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            const float value = static_cast<float>(w);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < num_elements; ++i)
            {
                buffer[i] = value;
            }
        }
        memoryFence();

        // Timed runs - time all iterations together to minimize timer overhead
        memoryFence();
        double start = getTimeMs();

        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            const float value = static_cast<float>(iter);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < num_elements; ++i)
            {
                buffer[i] = value;
            }
        }

        memoryFence();
        double end = getTimeMs();
        total_time_ms = end - start;

        double total_bytes = bytes_written * config_.iterations;
        double time_sec = total_time_ms / 1000.0;
        return (total_bytes / time_sec) / 1e9; // GB/s
    }

    double CPUBenchmark::benchmarkMemoryCopy(float *src, float *dst, size_t num_elements)
    {
        double total_time_ms = 0.0;
        // Copy reads from src and writes to dst
        double bytes_transferred = 2 * num_elements * sizeof(float);

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < num_elements; ++i)
            {
                dst[i] = src[i];
            }
        }
        memoryFence();

        // Timed runs - time all iterations together to minimize timer overhead
        memoryFence();
        double start = getTimeMs();

        for (int iter = 0; iter < config_.iterations; ++iter)
        {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < num_elements; ++i)
            {
                dst[i] = src[i];
            }
        }

        memoryFence();
        double end = getTimeMs();
        total_time_ms = end - start;

        double total_bytes = bytes_transferred * config_.iterations;
        double time_sec = total_time_ms / 1000.0;
        return (total_bytes / time_sec) / 1e9; // GB/s
    }

    double CPUBenchmark::benchmarkComputeFP32(size_t num_ops)
    {
        // Each FMA is 2 FLOPS, we want num_ops total FLOPS
        size_t num_fma = num_ops / 2;
        size_t ops_per_iter = num_fma * 2; // FMA = 2 ops

        double total_time_ms = 0.0;
        double total_ops = ops_per_iter * config_.iterations;

        int num_threads = getNumThreads();
        size_t fma_per_thread = num_fma / static_cast<size_t>(num_threads);

        const bool use_avx512 = (activeISALevel() >= ISALevel::AVX512);

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
#pragma omp parallel
            {
                float a = 1.0001f;
                float b = 0.9999f;
                float c = 1.0f;
                if (use_avx512)
                {
#if defined(__AVX512F__)
                    __m512 va = _mm512_set1_ps(a);
                    __m512 vb = _mm512_set1_ps(b);
                    __m512 vc = _mm512_set1_ps(c);
                    for (size_t i = 0; i < fma_per_thread / 16; ++i)
                    {
                        vc = _mm512_fmadd_ps(va, vb, vc);
                        va = _mm512_fmadd_ps(vb, vc, va);
                        vb = _mm512_fmadd_ps(vc, va, vb);
                        vc = _mm512_fmadd_ps(va, vb, vc);
                    }
                    float result[16];
                    _mm512_storeu_ps(result, vc);
                    doNotOptimize(result[0]);
#endif
                }
                else
                {
                    for (size_t i = 0; i < fma_per_thread; ++i)
                    {
                        c = a * b + c;
                        a = b * c + a;
                        b = c * a + b;
                        c = a * b + c;
                    }
                    doNotOptimize(c);
                }
            }
        }

        // Timed runs - time all iterations together to minimize timer overhead
        memoryFence();
        double start = getTimeMs();

        for (int iter = 0; iter < config_.iterations; ++iter)
        {
#pragma omp parallel
            {
                float a = 1.0001f;
                float b = 0.9999f;
                float c = 1.0f;
                if (use_avx512)
                {
#if defined(__AVX512F__)
                    __m512 va = _mm512_set1_ps(a);
                    __m512 vb = _mm512_set1_ps(b);
                    __m512 vc = _mm512_set1_ps(c);
                    // Each iteration: 4 FMAs * 16 elements = 128 FLOPS
                    for (size_t i = 0; i < fma_per_thread / 16; ++i)
                    {
                        vc = _mm512_fmadd_ps(va, vb, vc);
                        va = _mm512_fmadd_ps(vb, vc, va);
                        vb = _mm512_fmadd_ps(vc, va, vb);
                        vc = _mm512_fmadd_ps(va, vb, vc);
                    }
                    float result[16];
                    _mm512_storeu_ps(result, vc);
                    doNotOptimize(result[0]);
#endif
                }
                else
                {
                    for (size_t i = 0; i < fma_per_thread; ++i)
                    {
                        c = a * b + c;
                        a = b * c + a;
                        b = c * a + b;
                        c = a * b + c;
                    }
                    doNotOptimize(c);
                }
            }
        }

        memoryFence();
        double end = getTimeMs();
        total_time_ms = end - start;

        double time_sec = total_time_ms / 1000.0;
        return (total_ops / time_sec) / 1e9; // GFLOPS
    }

    double CPUBenchmark::benchmarkComputeINT8(size_t num_ops)
    {
        // Simulate INT8 dot products (used in quantized inference)
        // Each element multiply-add is 2 ops
        size_t num_elements = num_ops / 2;
        double total_ops = num_ops;

        double total_time_ms = 0.0;
        int num_threads = getNumThreads();
        size_t elements_per_thread = num_elements / static_cast<size_t>(num_threads);

        const bool use_vnni = (activeISALevel() >= ISALevel::AVX512);

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
#pragma omp parallel
            {
                int32_t acc = 0;
                int8_t a = 1;
                int8_t b = 2;
                if (use_vnni)
                {
#if defined(__AVX512VNNI__)
                    // Use VNNI for true INT8 throughput measurement
                    __m512i va = _mm512_set1_epi8(a);
                    __m512i vb = _mm512_set1_epi8(b);
                    __m512i vacc = _mm512_setzero_si512();
                    for (size_t i = 0; i < elements_per_thread / 64; ++i)
                    {
                        vacc = _mm512_dpbusd_epi32(vacc, va, vb);
                    }
                    int32_t result[16];
                    _mm512_storeu_si512(result, vacc);
                    doNotOptimize(result[0]);
#endif
                }
                else
                {
                    // Scalar fallback
                    for (size_t i = 0; i < elements_per_thread; ++i)
                    {
                        acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
                        a = static_cast<int8_t>(acc & 0x7F);
                        b = static_cast<int8_t>((acc >> 7) & 0x7F);
                    }
                    doNotOptimize(acc);
                }
            }
        }

        // Timed runs - time all iterations together to minimize timer overhead
        memoryFence();
        double start = getTimeMs();

        for (int iter = 0; iter < config_.iterations; ++iter)
        {
#pragma omp parallel
            {
                int32_t acc = 0;
                int8_t a = 1;
                int8_t b = 2;
                if (use_vnni)
                {
#if defined(__AVX512VNNI__)
                    __m512i va = _mm512_set1_epi8(a);
                    __m512i vb = _mm512_set1_epi8(b);
                    __m512i vacc = _mm512_setzero_si512();
                    for (size_t i = 0; i < elements_per_thread / 64; ++i)
                    {
                        vacc = _mm512_dpbusd_epi32(vacc, va, vb);
                    }
                    int32_t result[16];
                    _mm512_storeu_si512(result, vacc);
                    doNotOptimize(result[0]);
#endif
                }
                else
                {
                    for (size_t i = 0; i < elements_per_thread; ++i)
                    {
                        acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
                        a = static_cast<int8_t>(acc & 0x7F);
                        b = static_cast<int8_t>((acc >> 7) & 0x7F);
                    }
                    doNotOptimize(acc);
                }
            }
        }

        memoryFence();
        double end = getTimeMs();
        total_time_ms = end - start;

        double total_ops_all_iters = total_ops * config_.iterations;
        double time_sec = total_time_ms / 1000.0;
        return (total_ops_all_iters / time_sec) / 1e9; // GOPS
    }

    DeviceBenchmarkResult CPUBenchmark::run()
    {
        DeviceBenchmarkResult result;
        result.device = DeviceId::cpu();

        double start_time = getTimeMs();

        LOG_DEBUG("Starting CPU benchmark with " << getNumThreads() << " threads");
        LOG_DEBUG("Memory test size: " << (config_.memory_test_bytes / 1024 / 1024) << " MB");

#ifndef NDEBUG
        LOG_WARN("CPU benchmark running in Debug mode (-O0) - results will be much lower than actual hardware capability");
#endif

        // Allocate buffers
        size_t num_elements = config_.memory_test_bytes / sizeof(float);
        float *buffer1 = allocateNUMABuffer(num_elements);
        float *buffer2 = allocateNUMABuffer(num_elements);

        if (!buffer1 || !buffer2)
        {
            LOG_ERROR("Failed to allocate benchmark buffers");
            result.valid = false;
            return result;
        }

        // Fill with non-zero values
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < num_elements; ++i)
        {
            buffer1[i] = static_cast<float>(i % 1000) * 0.001f;
            buffer2[i] = static_cast<float>((i + 500) % 1000) * 0.001f;
        }

        // Run benchmarks
        LOG_DEBUG("Running memory read benchmark...");
        result.memory_read_gbps = benchmarkMemoryRead(buffer1, num_elements);

        LOG_DEBUG("Running memory write benchmark...");
        result.memory_write_gbps = benchmarkMemoryWrite(buffer2, num_elements);

        LOG_DEBUG("Running memory copy benchmark...");
        result.memory_copy_gbps = benchmarkMemoryCopy(buffer1, buffer2, num_elements);

        // Compute benchmarks - scale ops count based on desired test time
        size_t compute_ops = config_.memory_test_bytes; // 1 op per byte of mem test

        LOG_DEBUG("Running FP32 compute benchmark...");
        result.compute_fp32_gflops = benchmarkComputeFP32(compute_ops);

        LOG_DEBUG("Running INT8 compute benchmark...");
        result.compute_int8_gops = benchmarkComputeINT8(compute_ops);

        // Cleanup
        freeNUMABuffer(buffer1, num_elements);
        freeNUMABuffer(buffer2, num_elements);

        result.benchmark_duration_ms = getTimeMs() - start_time;
        result.valid = true;

        LOG_INFO("CPU Benchmark completed in " << result.benchmark_duration_ms << " ms");
        LOG_INFO("  Memory Read:  " << result.memory_read_gbps << " GB/s");
        LOG_INFO("  Memory Write: " << result.memory_write_gbps << " GB/s");
        LOG_INFO("  Memory Copy:  " << result.memory_copy_gbps << " GB/s");
        LOG_INFO("  FP32 Compute: " << result.compute_fp32_gflops << " GFLOPS");
        LOG_INFO("  INT8 Compute: " << result.compute_int8_gops << " GOPS");

        return result;
    }

} // namespace llaminar2
