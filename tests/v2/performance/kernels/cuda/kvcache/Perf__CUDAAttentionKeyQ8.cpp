/**
 * @file Perf__CUDAAttentionKeyQ8.cpp
 * @brief CUDA economy and isolated-profiler harness for attention-key Q8.
 *
 * The production sweep replays one-node retained graphs over decode, grouped
 * verifier, and prefill geometries. All allocations and host transfers happen
 * before timing. Two separately filterable tests expose exactly one eager
 * production dispatch so Nsight Compute can attribute registers, occupancy,
 * memory traffic, and spills to one kernel rather than to fixture work.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include "kernels/cuda/kvcache/CUDAAttentionKeyQ8Kernels.h"
#include "tensors/BlockStructures.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        constexpr int kWarmupReplays = 20;
        constexpr int kTimedReplays = 1000;

        /** @brief Whether this process can create a CUDA context. */
        bool hasCUDADevice()
        {
            int count = 0;
            return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
        }

        /** @brief One released-model cache geometry used by the economy sweep. */
        struct Geometry
        {
            const char *label; ///< Stable evidence-row label.
            int block_count;   ///< Number of token/head vectors in one launch.
            int head_dim;      ///< Physical attention-head width.
        };

        /** @brief Timed operation selected without a boolean mode flag. */
        enum class Operation : uint8_t
        {
            Quantize,
            Dequantize,
        };

        /** @brief Measured retained-graph result. */
        struct Measurement
        {
            double latency_us = 0.0;   ///< Average graph replay latency.
            double effective_gb_s = 0.0; ///< Logical input plus output traffic.
        };

        /** @brief Return the physical encoded block size for a supported width. */
        size_t encodedBlockBytes(int head_dim)
        {
            switch (head_dim)
            {
            case 64:
                return sizeof(AttentionKeyQ8Block<64>);
            case 128:
                return sizeof(AttentionKeyQ8Block<128>);
            case 256:
                return sizeof(AttentionKeyQ8Block<256>);
            default:
                return 0;
            }
        }

        /**
         * @brief Own persistent CUDA buffers, stream, events, and retained graphs.
         *
         * A harness instance has exactly one immutable geometry. This mirrors the
         * production graph-cache contract: pointers and launch geometry cannot be
         * rebound after capture.
         */
        class CUDAAttentionKeyQ8Harness final
        {
        public:
            /** @brief Allocate, initialize, and capture both one-node graphs. */
            explicit CUDAAttentionKeyQ8Harness(const Geometry &geometry)
                : geometry_(geometry)
            {
                initialize();
            }

            /** @brief Release all benchmark-owned CUDA resources. */
            ~CUDAAttentionKeyQ8Harness()
            {
                if (quantize_exec_)
                    (void)cudaGraphExecDestroy(quantize_exec_);
                if (dequantize_exec_)
                    (void)cudaGraphExecDestroy(dequantize_exec_);
                if (quantize_graph_)
                    (void)cudaGraphDestroy(quantize_graph_);
                if (dequantize_graph_)
                    (void)cudaGraphDestroy(dequantize_graph_);
                if (start_)
                    (void)cudaEventDestroy(start_);
                if (stop_)
                    (void)cudaEventDestroy(stop_);
                if (input_)
                    (void)cudaFree(input_);
                if (encoded_)
                    (void)cudaFree(encoded_);
                if (decoded_)
                    (void)cudaFree(decoded_);
                if (stream_)
                    (void)cudaStreamDestroy(stream_);
            }

            CUDAAttentionKeyQ8Harness(const CUDAAttentionKeyQ8Harness &) = delete;
            CUDAAttentionKeyQ8Harness &operator=(const CUDAAttentionKeyQ8Harness &) = delete;

            /** @brief Whether setup and graph capture completed successfully. */
            [[nodiscard]] bool ready() const { return ready_; }

            /** @brief First setup failure, suitable for a test assertion. */
            [[nodiscard]] const std::string &error() const { return error_; }

            /**
             * @brief Measure repeated retained-graph replay with CUDA events.
             *
             * @param operation Exact one-node graph to replay.
             * @param timed_replays Number of launches included in the sample.
             */
            [[nodiscard]] Measurement measure(Operation operation, int timed_replays)
            {
                Measurement result;
                cudaGraphExec_t executable = operation == Operation::Quantize
                                                 ? quantize_exec_
                                                 : dequantize_exec_;
                if (!ready_ || !executable || timed_replays <= 0)
                    return result;

                // Warmups establish clocks and page mappings outside the sample.
                for (int replay = 0; replay < kWarmupReplays; ++replay)
                {
                    if (cudaGraphLaunch(executable, stream_) != cudaSuccess)
                        return result;
                }

                if (cudaEventRecord(start_, stream_) != cudaSuccess)
                    return result;
                for (int replay = 0; replay < timed_replays; ++replay)
                {
                    if (cudaGraphLaunch(executable, stream_) != cudaSuccess)
                        return result;
                }
                if (cudaEventRecord(stop_, stream_) != cudaSuccess ||
                    cudaEventSynchronize(stop_) != cudaSuccess)
                {
                    return result;
                }

                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) != cudaSuccess)
                    return result;

                result.latency_us =
                    static_cast<double>(elapsed_ms) * 1000.0 / timed_replays;
                const size_t fp32_bytes = static_cast<size_t>(geometry_.block_count) *
                                          geometry_.head_dim * sizeof(float);
                const size_t encoded_bytes = static_cast<size_t>(geometry_.block_count) *
                                             encodedBlockBytes(geometry_.head_dim);
                const double logical_bytes = static_cast<double>(fp32_bytes + encoded_bytes);
                result.effective_gb_s =
                    logical_bytes / (result.latency_us * 1.0e-6) / 1.0e9;
                return result;
            }

            /**
             * @brief Submit one eager dispatch for an isolated profiler launch.
             *
             * Fixture initialization is synchronized before this call. The final
             * synchronization is result collection, not part of the hot operation.
             */
            [[nodiscard]] bool launchExactlyOnce(Operation operation)
            {
                if (!ready_)
                    return false;
                const bool launched = operation == Operation::Quantize
                                          ? cudaAttentionKeyQ8Quantize(
                                                input_, encoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_)
                                          : cudaAttentionKeyQ8Dequantize(
                                                encoded_, decoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_);
                return launched && cudaStreamSynchronize(stream_) == cudaSuccess;
            }

        private:
            /** @brief Store the first setup failure and make the harness unusable. */
            void fail(const char *message)
            {
                if (error_.empty())
                    error_ = message;
                ready_ = false;
            }

            /** @brief Allocate persistent state, seed it, and capture immutable graphs. */
            void initialize()
            {
                if (!hasCUDADevice())
                {
                    fail("No CUDA device");
                    return;
                }
                if (encodedBlockBytes(geometry_.head_dim) == 0 ||
                    geometry_.block_count <= 0)
                {
                    fail("Unsupported geometry");
                    return;
                }

                const size_t element_count = static_cast<size_t>(geometry_.block_count) *
                                             geometry_.head_dim;
                const size_t encoded_bytes = static_cast<size_t>(geometry_.block_count) *
                                             encodedBlockBytes(geometry_.head_dim);
                std::vector<float> host_input(element_count);
                for (size_t index = 0; index < element_count; ++index)
                {
                    // Add periodic Qwen-scale key outliers so the threshold path
                    // represents the numerical workload that exposed the defect.
                    host_input[index] = index % geometry_.head_dim == 0
                                            ? 128.0f
                                            : static_cast<float>(static_cast<int>(index % 31) - 15) *
                                                  0.25f;
                }

                if (cudaSetDevice(0) != cudaSuccess ||
                    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
                    cudaEventCreate(&start_) != cudaSuccess ||
                    cudaEventCreate(&stop_) != cudaSuccess ||
                    cudaMalloc(&input_, element_count * sizeof(float)) != cudaSuccess ||
                    cudaMalloc(&encoded_, encoded_bytes) != cudaSuccess ||
                    cudaMalloc(&decoded_, element_count * sizeof(float)) != cudaSuccess ||
                    cudaMemcpyAsync(input_, host_input.data(), element_count * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                    cudaMemsetAsync(encoded_, 0, encoded_bytes, stream_) != cudaSuccess ||
                    cudaMemsetAsync(decoded_, 0, element_count * sizeof(float), stream_) !=
                        cudaSuccess ||
                    cudaStreamSynchronize(stream_) != cudaSuccess)
                {
                    fail("CUDA persistent-state setup failed");
                    return;
                }

                if (!capture(Operation::Quantize, quantize_graph_, quantize_exec_) ||
                    !capture(Operation::Dequantize, dequantize_graph_, dequantize_exec_))
                {
                    fail("CUDA graph capture failed");
                    return;
                }
                ready_ = true;
            }

            /** @brief Capture and instantiate one immutable operation graph. */
            bool capture(Operation operation, cudaGraph_t &graph, cudaGraphExec_t &executable)
            {
                if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) !=
                    cudaSuccess)
                {
                    return false;
                }
                const bool launched = operation == Operation::Quantize
                                          ? cudaAttentionKeyQ8Quantize(
                                                input_, encoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_)
                                          : cudaAttentionKeyQ8Dequantize(
                                                encoded_, decoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_);
                if (!launched || cudaStreamEndCapture(stream_, &graph) != cudaSuccess)
                    return false;

                size_t node_count = 0;
                return cudaGraphGetNodes(graph, nullptr, &node_count) == cudaSuccess &&
                       node_count == 1 &&
                       cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) ==
                           cudaSuccess;
            }

            Geometry geometry_;
            cudaStream_t stream_ = nullptr;
            cudaEvent_t start_ = nullptr;
            cudaEvent_t stop_ = nullptr;
            cudaGraph_t quantize_graph_ = nullptr;
            cudaGraph_t dequantize_graph_ = nullptr;
            cudaGraphExec_t quantize_exec_ = nullptr;
            cudaGraphExec_t dequantize_exec_ = nullptr;
            float *input_ = nullptr;
            void *encoded_ = nullptr;
            float *decoded_ = nullptr;
            bool ready_ = false;
            std::string error_;
        };

        constexpr std::array<Geometry, 6> kProductionGeometries{{
            {"qwen2_decode", 2, 64},
            {"qwen2_mtp_depth3", 8, 64},
            {"qwen3_decode", 8, 128},
            {"qwen35_decode", 2, 256},
            {"qwen2_prefill_512", 1024, 64},
            {"qwen35_prefill_512", 1024, 256},
        }};

        /** @brief Run one profiler-only dispatch for one exact specialization. */
        void runExactProfilerLaunch(Operation operation, int head_dim)
        {
            CUDAAttentionKeyQ8Harness harness({"profiler_prefill", 1024, head_dim});
            ASSERT_TRUE(harness.ready()) << harness.error();
            ASSERT_TRUE(harness.launchExactlyOnce(operation));
        }
    } // namespace

    TEST(AttentionKeyQ8CUDAEconomy, ProductionGeometryRetainedGraphSweep)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";

        std::cout << "backend,geometry,operation,blocks,head_dim,latency_us,effective_gb_s\n";
        for (const Geometry &geometry : kProductionGeometries)
        {
            CUDAAttentionKeyQ8Harness harness(geometry);
            ASSERT_TRUE(harness.ready()) << geometry.label << ": " << harness.error();
            for (const Operation operation : {Operation::Quantize, Operation::Dequantize})
            {
                const Measurement measurement = harness.measure(operation, kTimedReplays);
                ASSERT_GT(measurement.latency_us, 0.0) << geometry.label;
                ASSERT_GT(measurement.effective_gb_s, 0.0) << geometry.label;
                if (geometry.block_count <= 8)
                {
                    // The observed RTX 3090 range is 2.2-2.3 us. A 10 us
                    // ceiling catches launch serialization or an accidental
                    // multi-kernel implementation while retaining broad CI
                    // headroom for clocks and shared-device load.
                    EXPECT_LT(measurement.latency_us, 10.0) << geometry.label;
                }
                else
                {
                    // The slowest production prefill row currently exceeds
                    // 100 GB/s. This deliberately conservative floor rejects
                    // scalarized/serialized regressions without overfitting one GPU.
                    EXPECT_GT(measurement.effective_gb_s, 50.0) << geometry.label;
                }
                std::cout << "cuda," << geometry.label << ','
                          << (operation == Operation::Quantize ? "quantize" : "dequantize")
                          << ',' << geometry.block_count << ',' << geometry.head_dim << ','
                          << std::fixed << std::setprecision(4) << measurement.latency_us << ','
                          << measurement.effective_gb_s << '\n';
            }
        }
    }

    TEST(AttentionKeyQ8CUDAEconomy, ExactQuantizeProfilerLaunch)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";
        runExactProfilerLaunch(Operation::Quantize, 64);
    }

    TEST(AttentionKeyQ8CUDAEconomy, ExactQuantizeD128ProfilerLaunch)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";
        runExactProfilerLaunch(Operation::Quantize, 128);
    }

    TEST(AttentionKeyQ8CUDAEconomy, ExactQuantizeD256ProfilerLaunch)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";
        runExactProfilerLaunch(Operation::Quantize, 256);
    }

    TEST(AttentionKeyQ8CUDAEconomy, ExactDequantizeProfilerLaunch)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";
        runExactProfilerLaunch(Operation::Dequantize, 64);
    }

    TEST(AttentionKeyQ8CUDAEconomy, ExactDequantizeD128ProfilerLaunch)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";
        runExactProfilerLaunch(Operation::Dequantize, 128);
    }

    TEST(AttentionKeyQ8CUDAEconomy, ExactDequantizeD256ProfilerLaunch)
    {
        if (!hasCUDADevice())
            GTEST_SKIP() << "No CUDA device";
        runExactProfilerLaunch(Operation::Dequantize, 256);
    }
} // namespace llaminar2

#else

TEST(AttentionKeyQ8CUDAEconomy, CUDAUnavailable)
{
    GTEST_SKIP() << "CUDA support is disabled";
}

#endif
