/**
 * @file Perf__ROCmFlashDecodeStableGraph.cpp
 * @brief Benchmark one graph-stable ROCm flash-decode executable across live KV lengths.
 *
 * The captured graph records a cache-capacity-sized physical split grid and a
 * 256-thread physical block. Device-resident sequence metadata selects both
 * the active split prefix and the active wavefront prefix for each replay.
 * This permits a long-sequence sequence-parallel mode shift without changing
 * graph topology or consulting the host.
 *
 * Timed replay contains only device parameter publication, attention phase,
 * and reduction kernels. Persistent tensors, workspace, graph, stream, and
 * events are created before timing. Set `LLAMINAR_ATTN_PROFILE_KV_LEN` to one
 * listed length to reduce rocprof collection to one measured replay.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "tensors/FP16Utils.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

    int profileKVLength()
    {
        const char *raw = std::getenv("LLAMINAR_ATTN_PROFILE_KV_LEN");
        if (!raw || !*raw)
            return 0;
        const int value = std::atoi(raw);
        return value > 0 ? value : 0;
    }

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
     * @brief Own and benchmark one immutable HIP decode graph.
     */
    class CapturedROCmFlashDecode final
    {
    public:
        explicit CapturedROCmFlashDecode(const DecodeGeometry &geometry)
            : geometry_(geometry),
              device_(DeviceId::rocm(0)),
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

        ~CapturedROCmFlashDecode()
        {
            if (graph_exec_)
                (void)hipGraphExecDestroy(graph_exec_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
            if (start_)
                (void)hipEventDestroy(start_);
            if (stop_)
                (void)hipEventDestroy(stop_);
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        CapturedROCmFlashDecode(const CapturedROCmFlashDecode &) = delete;
        CapturedROCmFlashDecode &operator=(const CapturedROCmFlashDecode &) = delete;

        bool ready() const { return ready_; }
        const std::string &error() const { return error_; }

        DecodeMeasurement measure(int kv_len)
        {
            DecodeMeasurement result{kv_len, 0.0, 0.0};
            if (!ready_ || kv_len <= 0 || kv_len > kMaxKVLength)
                return result;

            /*
             * The benchmark controller changes the counter before timing.
             * Production's captured cache append writes this value on device;
             * no transfer belongs to the measured graph replay.
             */
            if (hipMemcpyAsync(
                    cached_tokens_.gpu_data_ptr(),
                    &kv_len,
                    sizeof(kv_len),
                    hipMemcpyHostToDevice,
                    stream_) != hipSuccess)
            {
                return result;
            }

            for (int replay = 0; replay < kWarmupReplays; ++replay)
            {
                if (hipGraphLaunch(graph_exec_, stream_) != hipSuccess)
                    return result;
            }

            const int timed_replays = profileKVLength() > 0 ? 1 : kTimedReplays;
            if (hipEventRecord(start_, stream_) != hipSuccess)
                return result;
            for (int replay = 0; replay < timed_replays; ++replay)
            {
                if (hipGraphLaunch(graph_exec_, stream_) != hipSuccess)
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
            const double bytes =
                static_cast<double>(geometry_.n_heads * geometry_.head_dim) *
                    sizeof(float) * 2.0 +
                static_cast<double>(kv_len) * geometry_.n_kv_heads *
                    geometry_.head_dim * sizeof(uint16_t) * 2.0;
            result.effective_gb_s =
                bytes / (result.latency_us * 1.0e-6) / 1.0e9;
            return result;
        }

        bool finalOutputIsFinite()
        {
            std::vector<float> host_output(
                static_cast<size_t>(geometry_.n_heads * geometry_.head_dim));
            if (hipMemcpyAsync(
                    host_output.data(),
                    output_.gpu_data_ptr(),
                    host_output.size() * sizeof(float),
                    hipMemcpyDeviceToHost,
                    stream_) != hipSuccess ||
                hipStreamSynchronize(stream_) != hipSuccess)
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
            if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0)
            {
                fail("No ROCm device");
                return;
            }
            if (hipSetDevice(0) != hipSuccess ||
                hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) != hipSuccess ||
                hipEventCreate(&start_) != hipSuccess ||
                hipEventCreate(&stop_) != hipSuccess)
            {
                fail("HIP stream/event setup failed");
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
                hipStreamSynchronize(stream_) != hipSuccess)
            {
                fail("ROCm attention warmup failed");
                return;
            }

            {
                GraphCaptureGuard guard;
                if (hipStreamBeginCapture(
                        stream_, hipStreamCaptureModeGlobal) != hipSuccess)
                {
                    fail("HIP graph begin-capture failed");
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
                const hipError_t end_status =
                    hipStreamEndCapture(stream_, &graph_);
                if (!params_recorded || !attention_recorded ||
                    end_status != hipSuccess)
                {
                    fail("ROCm attention graph capture failed");
                    return;
                }
            }
            if (!graph_ ||
                hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0) !=
                    hipSuccess)
            {
                fail("ROCm attention graph instantiation failed");
                return;
            }

            size_t node_count = 0;
            if (hipGraphGetNodes(graph_, nullptr, &node_count) != hipSuccess ||
                node_count < 3)
            {
                fail("Captured ROCm graph does not contain publication, phase, and reduction");
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
        llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        hipStream_t stream_ = nullptr;
        hipEvent_t start_ = nullptr;
        hipEvent_t stop_ = nullptr;
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t graph_exec_ = nullptr;
        bool ready_ = false;
        std::string error_;
    };

    class Perf__ROCmFlashDecodeStableGraph : public ::testing::Test
    {
    protected:
        void run(const DecodeGeometry &geometry)
        {
            CapturedROCmFlashDecode benchmark(geometry);
            if (!benchmark.ready())
            {
                if (benchmark.error() == "No ROCm device")
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

            std::cout << "\nROCm stable captured flash decode: " << geometry.label
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

    TEST_F(Perf__ROCmFlashDecodeStableGraph, Qwen36SingleDevice)
    {
        run(DecodeGeometry{"Qwen3.6 SingleDevice", 16, 4, 256});
    }

    TEST_F(Perf__ROCmFlashDecodeStableGraph, Qwen36LocalTP2)
    {
        run(DecodeGeometry{"Qwen3.6 LocalTP2", 8, 2, 256});
    }
} // namespace

#else

TEST(Perf__ROCmFlashDecodeStableGraph, NoROCm)
{
    GTEST_SKIP() << "ROCm backend is not enabled";
}

#endif
