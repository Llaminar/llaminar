/**
 * @file Perf__CUDAFlashDecodeStableGraph.cpp
 * @brief Benchmark one graph-stable CUDA flash-decode executable across live KV lengths.
 *
 * Production decode captures one physical launch envelope from the immutable
 * KV-cache capacity. The device-resident committed-token counter then selects
 * the active split prefix on every replay. This benchmark measures that exact
 * contract: the graph contains the device-count-to-attention-parameters kernel,
 * the split phase, and the reduction. No allocation, transfer, host decision,
 * or synchronization occurs between timed graph launches.
 *
 * The full sweep crosses every short/long split boundary and extends through a
 * 16K context. Set `LLAMINAR_ATTN_PROFILE_KV_LEN` to one listed length when
 * invoking ncu; profiler mode performs one timed replay so hardware-counter
 * collection remains focused on one production launch.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include "tensors/FP16Utils.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr int kWarmupReplays = 10;
    constexpr int kTimedReplays = 200;
    constexpr int kMaxKVLength = 16384;
    constexpr std::array<int, 10> kKVLengths{
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    struct DecodeGeometry
    {
        const char *label;
        int n_heads;
        int n_kv_heads;
        int head_dim;
    };

    struct DecodeMeasurement
    {
        int kv_len;
        double latency_us;
        double effective_gb_s;
    };

    /**
     * @brief Return the requested one-length profiler run, or zero for the full sweep.
     */
    int profileKVLength()
    {
        const char *raw = std::getenv("LLAMINAR_ATTN_PROFILE_KV_LEN");
        if (!raw || !*raw)
            return 0;

        const int value = std::atoi(raw);
        return value > 0 ? value : 0;
    }

    /**
     * @brief Build deterministic FP16 cache contents without GPU setup work.
     */
    std::vector<uint16_t> makeFP16Data(size_t count, uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
        std::vector<uint16_t> result(count);
        std::generate(result.begin(), result.end(), [&]
        {
            return fp32_to_fp16(distribution(rng));
        });
        return result;
    }

    /**
     * @brief Own and benchmark one immutable CUDA decode graph.
     *
     * The host `kv_len` argument passed while recording is deliberately the
     * cache capacity. Replay-time attention length comes exclusively from
     * `cached_tokens_` through the captured parameter-publication kernel.
     */
    class CapturedCUDAFlashDecode final
    {
    public:
        explicit CapturedCUDAFlashDecode(const DecodeGeometry &geometry)
            : geometry_(geometry),
              device_(DeviceId::cuda(0)),
              q_({size_t{1}, static_cast<size_t>(geometry.n_heads * geometry.head_dim)}),
              k_({static_cast<size_t>(kMaxKVLength),
                  static_cast<size_t>(geometry.n_kv_heads * geometry.head_dim)},
                 makeFP16Data(
                     static_cast<size_t>(kMaxKVLength) *
                         geometry.n_kv_heads * geometry.head_dim,
                     0xC0FFEEu)),
              v_({static_cast<size_t>(kMaxKVLength),
                  static_cast<size_t>(geometry.n_kv_heads * geometry.head_dim)},
                 makeFP16Data(
                     static_cast<size_t>(kMaxKVLength) *
                         geometry.n_kv_heads * geometry.head_dim,
                     0xBADC0DEu)),
              output_({size_t{1}, static_cast<size_t>(geometry.n_heads * geometry.head_dim)}),
              cached_tokens_({size_t{1}}, std::vector<int32_t>{kKVLengths.front()}),
              mpi_(0, 1, MPI_COMM_SELF),
              kernel_(0)
        {
            initialize();
        }

        ~CapturedCUDAFlashDecode()
        {
            if (graph_exec_)
                (void)cudaGraphExecDestroy(graph_exec_);
            if (graph_)
                (void)cudaGraphDestroy(graph_);
            if (start_)
                (void)cudaEventDestroy(start_);
            if (stop_)
                (void)cudaEventDestroy(stop_);
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }

        CapturedCUDAFlashDecode(const CapturedCUDAFlashDecode &) = delete;
        CapturedCUDAFlashDecode &operator=(const CapturedCUDAFlashDecode &) = delete;

        bool ready() const { return ready_; }
        const std::string &error() const { return error_; }

        /**
         * @brief Measure repeated graph replay for one live device-owned KV length.
         */
        DecodeMeasurement measure(int kv_len)
        {
            DecodeMeasurement result{kv_len, 0.0, 0.0};
            if (!ready_ || kv_len <= 0 || kv_len > kMaxKVLength)
                return result;

            /*
             * This H2D copy is benchmark control-plane setup, not part of the
             * measured graph. Production updates the same counter from the
             * captured KV append. Same-stream ordering makes the subsequent
             * warmups consume the new value without a host/device fence.
             */
            if (cudaMemcpyAsync(
                    cached_tokens_.gpu_data_ptr(),
                    &kv_len,
                    sizeof(kv_len),
                    cudaMemcpyHostToDevice,
                    stream_) != cudaSuccess)
            {
                return result;
            }

            for (int replay = 0; replay < kWarmupReplays; ++replay)
            {
                if (cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess)
                    return result;
            }

            const int timed_replays = profileKVLength() > 0 ? 1 : kTimedReplays;
            if (cudaEventRecord(start_, stream_) != cudaSuccess)
                return result;
            for (int replay = 0; replay < timed_replays; ++replay)
            {
                if (cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess)
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
            const double bytes =
                static_cast<double>(geometry_.n_heads * geometry_.head_dim) *
                    sizeof(float) * 2.0 +
                static_cast<double>(kv_len) * geometry_.n_kv_heads *
                    geometry_.head_dim * sizeof(uint16_t) * 2.0;
            result.effective_gb_s =
                bytes / (result.latency_us * 1.0e-6) / 1.0e9;
            return result;
        }

        /**
         * @brief Copy only the final output after all timed work and validate it.
         */
        bool finalOutputIsFinite()
        {
            std::vector<float> host_output(
                static_cast<size_t>(geometry_.n_heads * geometry_.head_dim));
            if (cudaMemcpyAsync(
                    host_output.data(),
                    output_.gpu_data_ptr(),
                    host_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_) != cudaSuccess ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                return false;
            }
            return std::all_of(host_output.begin(), host_output.end(), [](float value)
            {
                return std::isfinite(value);
            });
        }

    private:
        void fail(const std::string &message)
        {
            error_ = message;
            ready_ = false;
        }

        void initialize()
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
            {
                fail("No CUDA device");
                return;
            }
            if (cudaSetDevice(0) != cudaSuccess ||
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreate(&start_) != cudaSuccess ||
                cudaEventCreate(&stop_) != cudaSuccess)
            {
                fail("CUDA stream/event setup failed");
                return;
            }

            auto &transfer = TransferEngine::instance();
            if (!transfer.uploadFull(&q_, device_, stream_).success ||
                !transfer.uploadFull(&k_, device_, stream_).success ||
                !transfer.uploadFull(&v_, device_, stream_).success ||
                !transfer.uploadFull(&output_, device_, stream_).success ||
                !transfer.uploadFull(&cached_tokens_, device_, stream_).success)
            {
                fail("Persistent tensor upload failed");
                return;
            }

            kernel_.setGPUStream(stream_);
            const auto requirements =
                kernel_.getWorkspaceRequirements(
                    1, geometry_.n_heads, geometry_.head_dim);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                device_, requirements.total_bytes_with_alignment() + 4096);
            if (!workspace_->allocate(requirements))
            {
                fail("Attention workspace allocation failed");
                return;
            }
            kernel_.bindWorkspace(workspace_.get());

            /*
             * Prime backend attributes and workspace pointers before capture.
             * This setup launch is intentionally outside the measured graph.
             */
            if (!kernel_.prepareDynamicAttnParamsFromDeviceSequenceState(
                    static_cast<const int *>(cached_tokens_.gpu_data_ptr()),
                    1,
                    1,
                    stream_,
                    kMaxKVLength) ||
                !kernel_.compute_tensor(
                    &q_, &k_, &v_, &output_,
                    1, 1, kMaxKVLength,
                    geometry_.n_heads,
                    geometry_.n_kv_heads,
                    geometry_.head_dim,
                    true, -1,
                    nullptr, nullptr,
                    &mpi_, 0,
                    0,
                    geometry_.n_heads,
                    geometry_.n_kv_heads,
                    geometry_.n_heads / geometry_.n_kv_heads) ||
                cudaStreamSynchronize(stream_) != cudaSuccess)
            {
                fail("CUDA attention warmup failed");
                return;
            }

            {
                GraphCaptureGuard guard;
                if (cudaStreamBeginCapture(
                        stream_, cudaStreamCaptureModeGlobal) != cudaSuccess)
                {
                    fail("CUDA graph begin-capture failed");
                    return;
                }
                const bool params_recorded =
                    kernel_.prepareDynamicAttnParamsFromDeviceSequenceState(
                        static_cast<const int *>(cached_tokens_.gpu_data_ptr()),
                        1, 1, stream_, kMaxKVLength);
                const bool attention_recorded =
                    params_recorded && kernel_.compute_tensor(
                        &q_, &k_, &v_, &output_,
                        1, 1, kMaxKVLength,
                        geometry_.n_heads,
                        geometry_.n_kv_heads,
                        geometry_.head_dim,
                        true, -1,
                        nullptr, nullptr,
                        &mpi_, 0,
                        0,
                        geometry_.n_heads,
                        geometry_.n_kv_heads,
                        geometry_.n_heads / geometry_.n_kv_heads);
                const cudaError_t end_status =
                    cudaStreamEndCapture(stream_, &graph_);
                if (!params_recorded || !attention_recorded ||
                    end_status != cudaSuccess)
                {
                    fail("CUDA attention graph capture failed");
                    return;
                }
            }
            if (!graph_ ||
                cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0) !=
                    cudaSuccess)
            {
                fail("CUDA attention graph instantiation failed");
                return;
            }

            size_t node_count = 0;
            if (cudaGraphGetNodes(graph_, nullptr, &node_count) != cudaSuccess ||
                node_count < 3)
            {
                fail("Captured CUDA graph does not contain publication, phase, and reduction");
                return;
            }
            ready_ = true;
        }

        DecodeGeometry geometry_;
        DeviceId device_;
        FP32Tensor q_;
        FP16Tensor k_;
        FP16Tensor v_;
        FP32Tensor output_;
        INT32Tensor cached_tokens_;
        MPIContext mpi_;
        llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        cudaStream_t stream_ = nullptr;
        cudaEvent_t start_ = nullptr;
        cudaEvent_t stop_ = nullptr;
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t graph_exec_ = nullptr;
        bool ready_ = false;
        std::string error_;
    };

    class Perf__CUDAFlashDecodeStableGraph : public ::testing::Test
    {
    protected:
        void run(const DecodeGeometry &geometry)
        {
            CapturedCUDAFlashDecode benchmark(geometry);
            if (!benchmark.ready())
            {
                if (benchmark.error() == "No CUDA device")
                    GTEST_SKIP() << benchmark.error();
                FAIL() << benchmark.error();
            }

            const int requested_kv_len = profileKVLength();
            if (requested_kv_len > 0)
            {
                ASSERT_NE(
                    std::find(
                        kKVLengths.begin(), kKVLengths.end(), requested_kv_len),
                    kKVLengths.end())
                    << "LLAMINAR_ATTN_PROFILE_KV_LEN must select a listed length";
            }

            std::cout << "\nCUDA stable captured flash decode: " << geometry.label
                      << " (" << geometry.n_heads << "q/"
                      << geometry.n_kv_heads << "kv, d=" << geometry.head_dim
                      << ", capacity=" << kMaxKVLength << ")\n"
                      << "kv_len,latency_us,effective_gb_s\n";

            for (const int kv_len : kKVLengths)
            {
                if (requested_kv_len > 0 && requested_kv_len != kv_len)
                    continue;
                const DecodeMeasurement measurement = benchmark.measure(kv_len);
                ASSERT_GT(measurement.latency_us, 0.0) << "kv_len=" << kv_len;
                ASSERT_TRUE(std::isfinite(measurement.effective_gb_s));
                std::cout << measurement.kv_len << ","
                          << std::fixed << std::setprecision(3)
                          << measurement.latency_us << ","
                          << std::setprecision(2)
                          << measurement.effective_gb_s << "\n";
            }
            EXPECT_TRUE(benchmark.finalOutputIsFinite());
        }
    };

    TEST_F(Perf__CUDAFlashDecodeStableGraph, Qwen36SingleDevice)
    {
        run(DecodeGeometry{"Qwen3.6 SingleDevice", 16, 4, 256});
    }

    TEST_F(Perf__CUDAFlashDecodeStableGraph, Qwen36LocalTP2)
    {
        run(DecodeGeometry{"Qwen3.6 LocalTP2", 8, 2, 256});
    }
} // namespace

#else

TEST(Perf__CUDAFlashDecodeStableGraph, NoCUDA)
{
    GTEST_SKIP() << "CUDA backend is not enabled";
}

#endif
