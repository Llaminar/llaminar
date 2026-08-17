/**
 * @file Perf__ROCmAttentionKeyQ8.cpp
 * @brief ROCm economy and isolated-profiler harness for attention-key Q8.
 *
 * The timed sweep uses immutable one-node HIP graphs and persistent buffers at
 * production decode, grouped-verifier, and prefill geometries. Separately
 * filterable one-dispatch cases give rocprof and ISA inspection an uncontaminated
 * attachment point for occupancy, VGPR, LDS, scratch, and throughput evidence.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM

#include "kernels/rocm/kvcache/ROCmAttentionKeyQ8Kernels.h"
#include "tensors/BlockStructures.h"

#include <hip/hip_runtime.h>

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

        /** @brief Whether this process can create a HIP context. */
        bool hasROCmDevice()
        {
            int count = 0;
            return hipGetDeviceCount(&count) == hipSuccess && count > 0;
        }

        /** @brief One released-model cache geometry used by the economy sweep. */
        struct Geometry
        {
            const char *label; ///< Stable evidence-row label.
            int block_count;   ///< Number of token/head vectors in one launch.
            int head_dim;      ///< Physical attention-head width.
        };

        /** @brief Timed operation selected without boolean mode ambiguity. */
        enum class Operation : uint8_t
        {
            Quantize,
            Dequantize,
        };

        /** @brief Measured retained-graph result. */
        struct Measurement
        {
            double latency_us = 0.0;     ///< Average graph replay latency.
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
         * @brief Own persistent HIP buffers, stream, events, and retained graphs.
         *
         * Pointer identity and geometry remain immutable after capture, matching
         * the production graph-cache ownership contract.
         */
        class ROCmAttentionKeyQ8Harness final
        {
        public:
            /** @brief Allocate, initialize, and capture both one-node graphs. */
            explicit ROCmAttentionKeyQ8Harness(const Geometry &geometry)
                : geometry_(geometry)
            {
                initialize();
            }

            /** @brief Release all benchmark-owned HIP resources. */
            ~ROCmAttentionKeyQ8Harness()
            {
                if (quantize_exec_)
                    (void)hipGraphExecDestroy(quantize_exec_);
                if (dequantize_exec_)
                    (void)hipGraphExecDestroy(dequantize_exec_);
                if (quantize_graph_)
                    (void)hipGraphDestroy(quantize_graph_);
                if (dequantize_graph_)
                    (void)hipGraphDestroy(dequantize_graph_);
                if (start_)
                    (void)hipEventDestroy(start_);
                if (stop_)
                    (void)hipEventDestroy(stop_);
                if (input_)
                    (void)hipFree(input_);
                if (encoded_)
                    (void)hipFree(encoded_);
                if (decoded_)
                    (void)hipFree(decoded_);
                if (stream_)
                    (void)hipStreamDestroy(stream_);
            }

            ROCmAttentionKeyQ8Harness(const ROCmAttentionKeyQ8Harness &) = delete;
            ROCmAttentionKeyQ8Harness &operator=(const ROCmAttentionKeyQ8Harness &) = delete;

            /** @brief Whether setup and graph capture completed successfully. */
            [[nodiscard]] bool ready() const { return ready_; }

            /** @brief First setup failure, suitable for a test assertion. */
            [[nodiscard]] const std::string &error() const { return error_; }

            /** @brief Measure repeated retained-graph replay with HIP events. */
            [[nodiscard]] Measurement measure(Operation operation, int timed_replays)
            {
                Measurement result;
                hipGraphExec_t executable = operation == Operation::Quantize
                                                ? quantize_exec_
                                                : dequantize_exec_;
                if (!ready_ || !executable || timed_replays <= 0)
                    return result;

                for (int replay = 0; replay < kWarmupReplays; ++replay)
                {
                    if (hipGraphLaunch(executable, stream_) != hipSuccess)
                        return result;
                }

                if (hipEventRecord(start_, stream_) != hipSuccess)
                    return result;
                for (int replay = 0; replay < timed_replays; ++replay)
                {
                    if (hipGraphLaunch(executable, stream_) != hipSuccess)
                        return result;
                }
                if (hipEventRecord(stop_, stream_) != hipSuccess ||
                    hipEventSynchronize(stop_) != hipSuccess)
                {
                    return result;
                }

                float elapsed_ms = 0.0f;
                if (hipEventElapsedTime(&elapsed_ms, start_, stop_) != hipSuccess)
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

            /** @brief Submit one eager production dispatch for isolated profiling. */
            [[nodiscard]] bool launchExactlyOnce(Operation operation)
            {
                if (!ready_)
                    return false;
                const bool launched = operation == Operation::Quantize
                                          ? rocmAttentionKeyQ8Quantize(
                                                input_, encoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_)
                                          : rocmAttentionKeyQ8Dequantize(
                                                encoded_, decoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_);
                return launched && hipStreamSynchronize(stream_) == hipSuccess;
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
                if (!hasROCmDevice())
                {
                    fail("No ROCm device");
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
                    host_input[index] = index % geometry_.head_dim == 0
                                            ? 128.0f
                                            : static_cast<float>(static_cast<int>(index % 31) - 15) *
                                                  0.25f;
                }

                if (hipSetDevice(0) != hipSuccess ||
                    hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess ||
                    hipEventCreate(&start_) != hipSuccess ||
                    hipEventCreate(&stop_) != hipSuccess ||
                    hipMalloc(&input_, element_count * sizeof(float)) != hipSuccess ||
                    hipMalloc(&encoded_, encoded_bytes) != hipSuccess ||
                    hipMalloc(&decoded_, element_count * sizeof(float)) != hipSuccess ||
                    hipMemcpyAsync(input_, host_input.data(), element_count * sizeof(float),
                                   hipMemcpyHostToDevice, stream_) != hipSuccess ||
                    hipMemsetAsync(encoded_, 0, encoded_bytes, stream_) != hipSuccess ||
                    hipMemsetAsync(decoded_, 0, element_count * sizeof(float), stream_) !=
                        hipSuccess ||
                    hipStreamSynchronize(stream_) != hipSuccess)
                {
                    fail("HIP persistent-state setup failed");
                    return;
                }

                if (!capture(Operation::Quantize, quantize_graph_, quantize_exec_) ||
                    !capture(Operation::Dequantize, dequantize_graph_, dequantize_exec_))
                {
                    fail("HIP graph capture failed");
                    return;
                }
                ready_ = true;
            }

            /** @brief Capture and instantiate one immutable operation graph. */
            bool capture(Operation operation, hipGraph_t &graph, hipGraphExec_t &executable)
            {
                if (hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal) != hipSuccess)
                    return false;

                const bool launched = operation == Operation::Quantize
                                          ? rocmAttentionKeyQ8Quantize(
                                                input_, encoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_)
                                          : rocmAttentionKeyQ8Dequantize(
                                                encoded_, decoded_, geometry_.block_count,
                                                geometry_.head_dim, stream_);
                if (!launched || hipStreamEndCapture(stream_, &graph) != hipSuccess)
                    return false;

                size_t node_count = 0;
                return hipGraphGetNodes(graph, nullptr, &node_count) == hipSuccess &&
                       node_count == 1 &&
                       hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0) == hipSuccess;
            }

            Geometry geometry_;
            hipStream_t stream_ = nullptr;
            hipEvent_t start_ = nullptr;
            hipEvent_t stop_ = nullptr;
            hipGraph_t quantize_graph_ = nullptr;
            hipGraph_t dequantize_graph_ = nullptr;
            hipGraphExec_t quantize_exec_ = nullptr;
            hipGraphExec_t dequantize_exec_ = nullptr;
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
            ROCmAttentionKeyQ8Harness harness({"profiler_prefill", 1024, head_dim});
            ASSERT_TRUE(harness.ready()) << harness.error();
            ASSERT_TRUE(harness.launchExactlyOnce(operation));
        }
    } // namespace

    TEST(AttentionKeyQ8ROCmEconomy, ProductionGeometryRetainedGraphSweep)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";

        std::cout << "backend,geometry,operation,blocks,head_dim,latency_us,effective_gb_s\n";
        for (const Geometry &geometry : kProductionGeometries)
        {
            ROCmAttentionKeyQ8Harness harness(geometry);
            ASSERT_TRUE(harness.ready()) << geometry.label << ": " << harness.error();
            for (const Operation operation : {Operation::Quantize, Operation::Dequantize})
            {
                const Measurement measurement = harness.measure(operation, kTimedReplays);
                ASSERT_GT(measurement.latency_us, 0.0) << geometry.label;
                ASSERT_GT(measurement.effective_gb_s, 0.0) << geometry.label;
                if (geometry.block_count <= 8)
                {
                    // gfx906 currently measures 9.8-12 us. The ceiling keeps
                    // enough headroom for shared-device clocks while detecting
                    // blocking, extra dispatches, or host involvement.
                    EXPECT_LT(measurement.latency_us, 30.0) << geometry.label;
                }
                else
                {
                    // The slowest observed prefill row is above 22 GB/s. A
                    // 10 GB/s floor is intentionally portable but still rules
                    // out scalar replay and transfer-contaminated timing.
                    EXPECT_GT(measurement.effective_gb_s, 10.0) << geometry.label;
                }
                std::cout << "rocm," << geometry.label << ','
                          << (operation == Operation::Quantize ? "quantize" : "dequantize")
                          << ',' << geometry.block_count << ',' << geometry.head_dim << ','
                          << std::fixed << std::setprecision(4) << measurement.latency_us << ','
                          << measurement.effective_gb_s << '\n';
            }
        }
    }

    TEST(AttentionKeyQ8ROCmEconomy, ExactQuantizeProfilerLaunch)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";
        runExactProfilerLaunch(Operation::Quantize, 64);
    }

    TEST(AttentionKeyQ8ROCmEconomy, ExactQuantizeD128ProfilerLaunch)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";
        runExactProfilerLaunch(Operation::Quantize, 128);
    }

    TEST(AttentionKeyQ8ROCmEconomy, ExactQuantizeD256ProfilerLaunch)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";
        runExactProfilerLaunch(Operation::Quantize, 256);
    }

    TEST(AttentionKeyQ8ROCmEconomy, ExactDequantizeProfilerLaunch)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";
        runExactProfilerLaunch(Operation::Dequantize, 64);
    }

    TEST(AttentionKeyQ8ROCmEconomy, ExactDequantizeD128ProfilerLaunch)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";
        runExactProfilerLaunch(Operation::Dequantize, 128);
    }

    TEST(AttentionKeyQ8ROCmEconomy, ExactDequantizeD256ProfilerLaunch)
    {
        if (!hasROCmDevice())
            GTEST_SKIP() << "No ROCm device";
        runExactProfilerLaunch(Operation::Dequantize, 256);
    }
} // namespace llaminar2

#else

TEST(AttentionKeyQ8ROCmEconomy, ROCmUnavailable)
{
    GTEST_SKIP() << "ROCm support is disabled";
}

#endif
