#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"

#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoERuntimeTable.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
extern "C" bool cudaMoE_group_tokens_small_float(
    const float *routing_indices,
    const float *routing_weights,
    int *expert_counts,
    int *expert_offsets,
    int *grouped_token_indices,
    float *grouped_weights,
    int *active_expert_ids,
    int total_slots,
    int num_experts,
    int top_k,
    int max_active_experts,
    int device_idx,
    void *stream);

#endif

/**
 * @file Test__CUDAMoEKernel.cpp
 * @brief Focused CUDA MoE kernel parity tests against the CPU implementation.
 *
 * These tests exercise the public `IMoEKernel` surface used by compute stages:
 * tensor-aware routing, gather/scatter, fallback SwiGLU, zeroing, and the
 * device-side grouping path that drives per-expert CUDA prefill dispatch. The
 * fixtures use tiny deterministic tensors and skip cleanly when CUDA hardware
 * is unavailable.
 */

namespace
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    std::shared_ptr<llaminar2::FP32Tensor> makeTensor(const std::vector<size_t> &shape,
                                                      const std::vector<float> &values)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        std::copy(values.begin(), values.end(), data);
        return tensor;
    }

    std::shared_ptr<llaminar2::FP32Tensor> makeZeros(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_shared<llaminar2::FP32Tensor>(shape);
        std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), 0.0f);
        return tensor;
    }

    void expectNearArray(const float *actual, const float *expected, size_t count, float tolerance = 1.0e-5f)
    {
        for (size_t i = 0; i < count; ++i)
            ASSERT_NEAR(actual[i], expected[i], tolerance) << "at element " << i;
    }

    bool hasCudaDevice()
    {
#ifdef HAVE_CUDA
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
        return false;
#endif
    }

#ifdef HAVE_CUDA
    /// @brief RAII owner for tiny CUDA buffers used by descriptor-upload tests.
    class CudaAllocation
    {
    public:
        explicit CudaAllocation(size_t bytes)
        {
            EXPECT_EQ(cudaMalloc(&ptr_, bytes), cudaSuccess);
        }

        ~CudaAllocation()
        {
            if (ptr_)
                cudaFree(ptr_);
        }

        CudaAllocation(const CudaAllocation &) = delete;
        CudaAllocation &operator=(const CudaAllocation &) = delete;

        CudaAllocation(CudaAllocation &&other) noexcept
            : ptr_(other.ptr_)
        {
            other.ptr_ = nullptr;
        }

        CudaAllocation &operator=(CudaAllocation &&other) noexcept
        {
            if (this != &other)
            {
                if (ptr_)
                    cudaFree(ptr_);
                ptr_ = other.ptr_;
                other.ptr_ = nullptr;
            }
            return *this;
        }

        void *get() const { return ptr_; }

    private:
        void *ptr_ = nullptr;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };

    /// @brief Build a minimal native-VNNI descriptor backed by CUDA device pointers.
    llaminar2::DeviceNativeVNNIMatrixDesc makeCudaNativeDesc(
        const CudaAllocation &payload,
        const CudaAllocation &scales,
        int rows,
        int cols)
    {
        llaminar2::DeviceNativeVNNIMatrixDesc desc{};
        desc.payload = static_cast<const uint8_t *>(payload.get());
        desc.scales = scales.get();
        desc.n = rows;
        desc.k = cols;
        desc.blocks_per_row = static_cast<uint32_t>(cols / 32);
        desc.codebook_id = 0;
        return desc;
    }
#endif

    class Test__CUDAMoEKernel : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifndef HAVE_CUDA
            GTEST_SKIP() << "CUDA support not compiled";
#else
            if (!hasCudaDevice())
                GTEST_SKIP() << "No CUDA device available";
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
            cuda_kernel_ = KernelFactory::getOrCreateMoEKernel(llaminar2::DeviceId::cuda(0));
            cpu_kernel_ = KernelFactory::getOrCreateMoEKernel(llaminar2::DeviceId::cpu());
            ASSERT_NE(cuda_kernel_, nullptr);
            ASSERT_NE(cpu_kernel_, nullptr);
            cuda_kernel_->setGPUStream(stream_);
#endif
        }

        void TearDown() override
        {
#ifdef HAVE_CUDA
            if (stream_)
                cudaStreamDestroy(stream_);
#endif
        }

#ifdef HAVE_CUDA
        cudaStream_t stream_ = nullptr;
#endif
        llaminar2::IMoEKernel *cuda_kernel_ = nullptr;
        llaminar2::IMoEKernel *cpu_kernel_ = nullptr;
    };
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRuntimeTableMatchesCPU)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 3;
    constexpr int top_k = 2;
    constexpr int d_model = 4;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int i = 0; i < num_experts; ++i)
        update.experts[i].logical_expert_id = i;

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    // Deterministic hidden/gate tensors (1 token, d_model=4)
    auto hidden = makeTensor({seq_len, d_model}, {0.2f, -0.1f, 0.3f, 0.5f});
    auto gate = makeTensor({num_experts, d_model}, {0.1f, 0.2f, 0.3f, 0.4f,
                                                    -0.2f, 0.0f, 0.1f, 0.2f,
                                                    0.5f, -0.3f, 0.2f, -0.1f});
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    // CUDA decode uses the mirrored runtime table; CPU routeWithTensors is the
    // host reference for the same one-token top-k routing result.
    auto *cuda_layer = cuda_table.deviceLayerState(0);
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, cuda_indices.get(), cuda_weights.get(), true, true));
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // Compare legacy outputs
    expectNearArray(cuda_indices->data(), cpu_indices->data(), seq_len * top_k, 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), seq_len * top_k, 1e-5f);

    llaminar2::DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = num_layers;
    hist_config.num_experts = num_experts;
    hist_config.top_k = top_k;
    hist_config.window_size = 8;
    hist_config.sockets = {llaminar2::DeviceId(llaminar2::DeviceType::CPU, 0)};
    hist_config.expert_to_socket.assign(num_experts, 0);
    llaminar2::DecodeExpertHistogram runtime_histogram(hist_config);

    int expected_counts[num_experts] = {};
    const float *cpu_index_data = cpu_indices->data();
    for (int k = 0; k < top_k; ++k)
        ++expected_counts[static_cast<int>(cpu_index_data[k])];

    runtime_histogram.recordTokenBoundary(0);
    ASSERT_TRUE(cuda_table.syncDecodeHistogramToHost(runtime_histogram, stream_, true));
    EXPECT_EQ(runtime_histogram.windowTokenCount(), 1u);
    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(runtime_histogram.activationCount(0, expert), static_cast<uint64_t>(expected_counts[expert]));

    ASSERT_TRUE(cuda_table.syncDecodeHistogramToHost(runtime_histogram, stream_, true));
    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(runtime_histogram.activationCount(0, expert), static_cast<uint64_t>(expected_counts[expert]));
#endif
}

TEST_F(Test__CUDAMoEKernel, DecodeRouteSelectRuntimeOnlyDoesNotRequireLegacyOutputs)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoELayerRuntime;
    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 4;
    constexpr int top_k = 2;
    constexpr int d_model = 4;
    constexpr int seq_len = 1;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 0);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
        update.experts[expert].logical_expert_id = expert;

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    auto hidden = makeTensor({seq_len, d_model}, {0.4f, -0.2f, 0.1f, 0.7f});
    auto gate = makeTensor({num_experts, d_model}, {0.2f, -0.1f, 0.5f, 0.3f,
                                                    -0.3f, 0.4f, 0.2f, 0.1f,
                                                    0.6f, 0.1f, -0.4f, 0.2f,
                                                    0.0f, -0.5f, 0.3f, 0.8f});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    auto *cuda_layer = cuda_table.deviceLayerState(0);
    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        cuda_layer, hidden.get(), gate.get(), d_model, num_experts, top_k,
        true, nullptr, nullptr, /*write_legacy_outputs=*/false, /*update_runtime_histogram=*/true));

    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));

    int32_t runtime_ids[top_k] = {};
    float runtime_weights[top_k] = {};
    const char *runtime_base = reinterpret_cast<const char *>(cuda_layer);
    const auto *ids_device = reinterpret_cast<const int32_t *>(
        runtime_base + offsetof(DeviceMoELayerRuntime, topk_expert_ids));
    const auto *weights_device = reinterpret_cast<const float *>(
        runtime_base + offsetof(DeviceMoELayerRuntime, topk_weights));

    ASSERT_EQ(cudaMemcpyAsync(runtime_ids, ids_device, sizeof(runtime_ids),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(runtime_weights, weights_device, sizeof(runtime_weights),
                              cudaMemcpyDeviceToHost, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *cpu_index_data = cpu_indices->data();
    const float *cpu_weight_data = cpu_weights->data();
    int expected_counts[num_experts] = {};
    for (int k = 0; k < top_k; ++k)
    {
        EXPECT_EQ(runtime_ids[k], static_cast<int32_t>(cpu_index_data[k]));
        EXPECT_NEAR(runtime_weights[k], cpu_weight_data[k], 1.0e-5f);
        ++expected_counts[static_cast<int>(cpu_index_data[k])];
    }

    llaminar2::DecodeExpertHistogramConfig hist_config;
    hist_config.num_layers = num_layers;
    hist_config.num_experts = num_experts;
    hist_config.top_k = top_k;
    hist_config.window_size = 8;
    hist_config.sockets = {llaminar2::DeviceId(llaminar2::DeviceType::CPU, 0)};
    hist_config.expert_to_socket.assign(num_experts, 0);
    llaminar2::DecodeExpertHistogram runtime_histogram(hist_config);

    runtime_histogram.recordTokenBoundary(0);
    ASSERT_TRUE(cuda_table.syncDecodeHistogramToHost(runtime_histogram, stream_, true));
    EXPECT_EQ(runtime_histogram.windowTokenCount(), 1u);
    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(runtime_histogram.activationCount(0, expert), static_cast<uint64_t>(expected_counts[expert]));
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsMatchesCPU)
{
    constexpr int seq_len = 3;
    constexpr int d_model = 4;
    constexpr int num_experts = 5;
    constexpr int top_k = 2;

    auto hidden = makeTensor({seq_len, d_model}, {
                                                     0.25f,
                                                     -0.50f,
                                                     0.75f,
                                                     1.00f,
                                                     -1.00f,
                                                     0.50f,
                                                     0.25f,
                                                     -0.75f,
                                                     0.60f,
                                                     0.10f,
                                                     -0.30f,
                                                     0.90f,
                                                 });
    auto gate = makeTensor({num_experts, d_model}, {
                                                       0.10f,
                                                       0.20f,
                                                       -0.30f,
                                                       0.40f,
                                                       -0.40f,
                                                       0.10f,
                                                       0.30f,
                                                       0.20f,
                                                       0.50f,
                                                       -0.20f,
                                                       0.10f,
                                                       -0.10f,
                                                       -0.30f,
                                                       -0.60f,
                                                       0.40f,
                                                       0.70f,
                                                       0.20f,
                                                       0.80f,
                                                       -0.50f,
                                                       0.30f,
                                                   });
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 1.0e-5f);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
    ASSERT_EQ(cuda_host_result.router_logits.size(), cpu_host_result.router_logits.size());
    expectNearArray(cuda_host_result.router_logits.data(), cpu_host_result.router_logits.data(),
                    cuda_host_result.router_logits.size(), 1.0e-5f);
    for (int token = 0; token < seq_len; ++token)
    {
        float prob_sum = 0.0f;
        for (int expert = 0; expert < num_experts; ++expert)
            prob_sum += cuda_host_result.router_logits[static_cast<size_t>(token) * num_experts + expert];
        EXPECT_NEAR(prob_sum, 1.0f, 1.0e-5f);
    }
#endif
}

TEST_F(Test__CUDAMoEKernel, RouteWithTensorsSingleTokenQwenScalePopulatesSnapshotOutputs)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 1;
    constexpr int d_model = 2048;
    constexpr int num_experts = 256;
    constexpr int top_k = 8;

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    std::vector<float> gate_values(static_cast<size_t>(num_experts) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.07f * std::sin(0.013f * static_cast<float>(i + 1)) +
                           0.03f * std::cos(0.029f * static_cast<float>(i + 3));
    for (size_t i = 0; i < gate_values.size(); ++i)
        gate_values[i] = 0.05f * std::sin(0.017f * static_cast<float>(i + 5)) -
                         0.02f * std::cos(0.031f * static_cast<float>(i + 7));

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto gate = makeTensor({num_experts, d_model}, gate_values);
    auto cuda_indices = makeZeros({seq_len, top_k});
    auto cuda_weights = makeZeros({seq_len, top_k});
    auto cpu_indices = makeZeros({seq_len, top_k});
    auto cpu_weights = makeZeros({seq_len, top_k});

    llaminar2::MoERoutingResult cuda_host_result;
    llaminar2::MoERoutingResult cpu_host_result;
    ASSERT_TRUE(cuda_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                               num_experts, top_k, true,
                                               cuda_indices.get(), cuda_weights.get(), cuda_host_result));
    ASSERT_TRUE(cpu_kernel_->routeWithTensors(hidden.get(), gate.get(), seq_len, d_model,
                                              num_experts, top_k, true,
                                              cpu_indices.get(), cpu_weights.get(), cpu_host_result));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    expectNearArray(cuda_indices->data(), cpu_indices->data(), cuda_indices->numel(), 0.0f);
    expectNearArray(cuda_weights->data(), cpu_weights->data(), cuda_weights->numel(), 1.0e-4f);

#ifdef ENABLE_PIPELINE_SNAPSHOTS
    ASSERT_EQ(cuda_host_result.router_logits.size(), static_cast<size_t>(num_experts));
    const int nonzero_logits = std::count_if(
        cuda_host_result.router_logits.begin(),
        cuda_host_result.router_logits.end(),
        [](float v)
        { return v != 0.0f; });
    EXPECT_GT(nonzero_logits, 0);
    expectNearArray(cuda_host_result.router_logits.data(), cpu_host_result.router_logits.data(),
                    cuda_host_result.router_logits.size(), 1.0e-4f);
#endif
#endif
}

TEST_F(Test__CUDAMoEKernel, GatherAndScatterMatchCPU)
{
    constexpr int rows = 4;
    constexpr int d_model = 5;
    const std::vector<int> token_indices = {3, 1, 0};
    const std::vector<float> weights = {0.25f, 0.50f, 0.75f};

    auto hidden = makeTensor({rows, d_model}, {
                                                  0.0f,
                                                  0.1f,
                                                  0.2f,
                                                  0.3f,
                                                  0.4f,
                                                  1.0f,
                                                  1.1f,
                                                  1.2f,
                                                  1.3f,
                                                  1.4f,
                                                  2.0f,
                                                  2.1f,
                                                  2.2f,
                                                  2.3f,
                                                  2.4f,
                                                  3.0f,
                                                  3.1f,
                                                  3.2f,
                                                  3.3f,
                                                  3.4f,
                                              });
    auto cuda_batch = makeZeros({token_indices.size(), d_model});
    auto cpu_batch = makeZeros({token_indices.size(), d_model});

    cuda_kernel_->gatherTokenBatchFromTensors(hidden.get(), cuda_batch.get(),
                                              token_indices.data(), static_cast<int>(token_indices.size()), d_model);
    cpu_kernel_->gatherTokenBatchFromTensors(hidden.get(), cpu_batch.get(),
                                             token_indices.data(), static_cast<int>(token_indices.size()), d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(cuda_batch->data(), cpu_batch->data(), cuda_batch->numel());

    auto cuda_output = makeZeros({rows, d_model});
    auto cpu_output = makeZeros({rows, d_model});
    cuda_kernel_->scatterAddWeightedFromTensors(cuda_output.get(), cuda_batch.get(),
                                                token_indices.data(), weights.data(),
                                                static_cast<int>(token_indices.size()), d_model);
    cpu_kernel_->scatterAddWeightedFromTensors(cpu_output.get(), cpu_batch.get(),
                                               token_indices.data(), weights.data(),
                                               static_cast<int>(token_indices.size()), d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(cuda_output->data(), cpu_output->data(), cuda_output->numel());
}

TEST_F(Test__CUDAMoEKernel, SwiGLUAndZeroMatchCPU)
{
    constexpr int count = 12;
    auto gate_cuda = makeTensor({count}, {-1.0f, -0.75f, -0.25f, 0.0f, 0.25f, 0.5f,
                                          0.75f, 1.0f, 1.25f, 1.5f, -1.5f, 2.0f});
    auto gate_cpu = makeTensor({count}, {-1.0f, -0.75f, -0.25f, 0.0f, 0.25f, 0.5f,
                                         0.75f, 1.0f, 1.25f, 1.5f, -1.5f, 2.0f});
    auto up = makeTensor({count}, {0.5f, -0.25f, 0.75f, 1.0f, -1.25f, 0.33f,
                                   1.2f, -0.8f, 0.9f, 0.4f, -0.6f, 0.1f});

    cuda_kernel_->swiGLUFromTensors(gate_cuda.get(), up.get(), count);
    cpu_kernel_->swiGLUFromTensors(gate_cpu.get(), up.get(), count);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(gate_cuda->data(), gate_cpu->data(), count, 1.0e-5f);

    cuda_kernel_->zeroBuffer(gate_cuda.get(), gate_cuda->size_bytes());
    cpu_kernel_->zeroBuffer(gate_cpu.get(), gate_cpu->size_bytes());
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(gate_cuda->data(), gate_cpu->data(), count, 0.0f);
}

TEST_F(Test__CUDAMoEKernel, PrepareExpertGroupsMatchesCPUGroups)
{
    constexpr int seq_len = 4;
    constexpr int top_k = 2;
    constexpr int num_experts = 4;
    constexpr int d_model = 3;

    auto routing_indices = makeTensor({seq_len, top_k}, {
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                            1.0f,
                                                            3.0f,
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                        });
    auto routing_weights = makeTensor({seq_len, top_k}, {
                                                            0.7f,
                                                            0.3f,
                                                            0.6f,
                                                            0.4f,
                                                            0.8f,
                                                            0.2f,
                                                            0.55f,
                                                            0.45f,
                                                        });
    auto hidden = makeTensor({seq_len, d_model}, {
                                                     1.0f,
                                                     1.1f,
                                                     1.2f,
                                                     2.0f,
                                                     2.1f,
                                                     2.2f,
                                                     3.0f,
                                                     3.1f,
                                                     3.2f,
                                                     4.0f,
                                                     4.1f,
                                                     4.2f,
                                                 });

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroups(routing_indices.get(), routing_weights.get(),
                                                  seq_len, num_experts, top_k));
    ASSERT_TRUE(cpu_kernel_->prepareExpertGroups(routing_indices.get(), routing_weights.get(),
                                                 seq_len, num_experts, top_k));

    for (int expert = 0; expert < num_experts; ++expert)
        EXPECT_EQ(cuda_kernel_->getExpertTokenCount(expert), cpu_kernel_->getExpertTokenCount(expert));

    auto cuda_expert_one = makeZeros({static_cast<size_t>(cuda_kernel_->getExpertTokenCount(1)), d_model});
    auto cpu_expert_one = makeZeros({static_cast<size_t>(cpu_kernel_->getExpertTokenCount(1)), d_model});
    cuda_kernel_->gatherExpertBatch(hidden.get(), cuda_expert_one.get(), 1, d_model);
    cpu_kernel_->gatherExpertBatch(hidden.get(), cpu_expert_one.get(), 1, d_model);
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
    expectNearArray(cuda_expert_one->data(), cpu_expert_one->data(), cuda_expert_one->numel());
}

TEST_F(Test__CUDAMoEKernel, PrepareExpertGroupsAsyncAcceptsDeviceRoutingTensors)
{
    constexpr int seq_len = 4;
    constexpr int top_k = 2;
    constexpr int num_experts = 4;

    auto routing_indices = makeTensor({seq_len, top_k}, {
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                            1.0f,
                                                            3.0f,
                                                            1.0f,
                                                            2.0f,
                                                            0.0f,
                                                        });
    auto routing_weights = makeTensor({seq_len, top_k}, {
                                                            0.7f,
                                                            0.3f,
                                                            0.6f,
                                                            0.4f,
                                                            0.8f,
                                                            0.2f,
                                                            0.55f,
                                                            0.45f,
                                                        });

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(routing_indices.get(), routing_weights.get(),
                                                       seq_len, num_experts, top_k));
#ifdef HAVE_CUDA
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, SmallFloatGroupingEmitsCompactActiveExperts)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 4;
    constexpr int top_k = 4;
    constexpr int num_experts = 16;
    constexpr int total_slots = seq_len * top_k;
    constexpr int max_active_experts = total_slots;

    const std::vector<float> routing_indices = {
        7.0f, 3.0f, 7.0f, 1.0f,
        5.0f, 1.0f, 9.0f, 3.0f,
        7.0f, 5.0f, 11.0f, 1.0f,
        3.0f, 9.0f, 11.0f, 5.0f,
    };
    std::vector<float> routing_weights(static_cast<size_t>(total_slots));
    for (int i = 0; i < total_slots; ++i)
        routing_weights[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i + 1);

    float *d_indices = nullptr;
    float *d_weights = nullptr;
    int *d_counts = nullptr;
    int *d_offsets = nullptr;
    int *d_grouped_tokens = nullptr;
    float *d_grouped_weights = nullptr;
    int *d_active = nullptr;
    ASSERT_EQ(cudaMalloc(&d_indices, routing_indices.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_weights, routing_weights.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_counts, num_experts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_offsets, num_experts * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_grouped_tokens, total_slots * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_grouped_weights, total_slots * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_active, max_active_experts * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpyAsync(d_indices, routing_indices.data(),
                              routing_indices.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_weights, routing_weights.data(),
                              routing_weights.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              cudaSuccess);

    ASSERT_TRUE(cudaMoE_group_tokens_small_float(
        d_indices, d_weights, d_counts, d_offsets, d_grouped_tokens, d_grouped_weights,
        d_active, total_slots, num_experts, top_k, max_active_experts, 0, stream_));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    std::vector<int> counts(num_experts);
    std::vector<int> offsets(num_experts);
    std::vector<int> grouped_tokens(total_slots);
    std::vector<float> grouped_weights(total_slots);
    std::vector<int> active(max_active_experts);
    ASSERT_EQ(cudaMemcpy(counts.data(), d_counts, counts.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets.data(), d_offsets, offsets.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_tokens.data(), d_grouped_tokens, grouped_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grouped_weights.data(), d_grouped_weights, grouped_weights.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(active.data(), d_active, active.size() * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    std::vector<int> expected_counts(num_experts, 0);
    for (float expert : routing_indices)
        ++expected_counts[static_cast<int>(expert)];

    std::vector<int> expected_offsets(num_experts, 0);
    int running = 0;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        expected_offsets[expert] = running;
        running += expected_counts[expert];
    }
    EXPECT_EQ(counts, expected_counts);
    EXPECT_EQ(offsets, expected_offsets);

    std::vector<int> expected_active;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        if (expected_counts[expert] > 0)
            expected_active.push_back(expert);
    }
    ASSERT_LE(expected_active.size(), active.size());
    for (size_t i = 0; i < expected_active.size(); ++i)
        EXPECT_EQ(active[i], expected_active[i]) << "active slot " << i;
    for (size_t i = expected_active.size(); i < active.size(); ++i)
        EXPECT_EQ(active[i], -1) << "inactive slot " << i;

    for (int expert = 0; expert < num_experts; ++expert)
    {
        int local = 0;
        for (int slot = 0; slot < total_slots; ++slot)
        {
            if (static_cast<int>(routing_indices[static_cast<size_t>(slot)]) != expert)
                continue;
            const int dest = offsets[expert] + local++;
            EXPECT_EQ(grouped_tokens[dest], slot / top_k);
            EXPECT_FLOAT_EQ(grouped_weights[dest], routing_weights[static_cast<size_t>(slot)]);
        }
    }

    EXPECT_FALSE(cudaMoE_group_tokens_small_float(
        d_indices, d_weights, d_counts, d_offsets, d_grouped_tokens, d_grouped_weights,
        d_active, total_slots, num_experts, top_k, max_active_experts, 0, nullptr));

    cudaFree(d_indices);
    cudaFree(d_weights);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_grouped_tokens);
    cudaFree(d_grouped_weights);
    cudaFree(d_active);
#endif
}

TEST_F(Test__CUDAMoEKernel, UploadGroupedDescriptorTablesAcceptValidCudaDescriptors)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int d_model = 128;
    constexpr int intermediate = 128;
    constexpr int num_experts = 2;
    constexpr size_t descriptor_bytes = 4096;

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);

    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    EXPECT_GE(down_table, 0);

    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    EXPECT_GE(gateup_table, 0);

    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillUsesCompactActiveExpertGrid)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    llaminar2::PerfStatsCollector::reset();

    constexpr int seq_len = 4;
    constexpr int top_k = 4;
    constexpr int num_experts = 16;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.01f * static_cast<float>(static_cast<int>(i % 13) - 6);
    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto output = makeZeros({seq_len, d_model});

    auto routing_indices = makeTensor({seq_len, top_k}, {
                                                            7.0f,
                                                            3.0f,
                                                            7.0f,
                                                            1.0f,
                                                            5.0f,
                                                            1.0f,
                                                            9.0f,
                                                            3.0f,
                                                            7.0f,
                                                            5.0f,
                                                            11.0f,
                                                            1.0f,
                                                            3.0f,
                                                            9.0f,
                                                            11.0f,
                                                            5.0f,
                                                        });
    auto routing_weights = makeTensor({seq_len, top_k}, {
                                                            0.40f,
                                                            0.30f,
                                                            0.20f,
                                                            0.10f,
                                                            0.45f,
                                                            0.25f,
                                                            0.20f,
                                                            0.10f,
                                                            0.35f,
                                                            0.30f,
                                                            0.20f,
                                                            0.15f,
                                                            0.50f,
                                                            0.20f,
                                                            0.20f,
                                                            0.10f,
                                                        });

    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (size_t i = 0; i < output->numel(); ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    const auto grouping_records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_small_prefill_grouping_calls"});
    ASSERT_FALSE(grouping_records.empty());

    const auto grid_records =
        llaminar2::PerfStatsCollector::snapshot({"kernel.cuda_moe_grouped_prefill_active_expert_grid_calls"});
    ASSERT_FALSE(grid_records.empty());
    EXPECT_EQ(grid_records.front().tags.at("active_expert_slots"), std::to_string(seq_len * top_k));
    EXPECT_EQ(grid_records.front().tags.at("num_experts"), std::to_string(num_experts));

    llaminar2::PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAMoEKernel, FixedTopologyRuntimeGroupedPrefillLargeAllExpertPathCapturesAndReplays)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int seq_len = 33;
    constexpr int top_k = 2;
    constexpr int num_experts = 128;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(static_cast<size_t>(seq_len) * d_model);
    for (size_t i = 0; i < hidden_values.size(); ++i)
        hidden_values[i] = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);

    std::vector<float> routing_indices(static_cast<size_t>(seq_len) * top_k);
    std::vector<float> routing_weights(static_cast<size_t>(seq_len) * top_k);
    for (int slot = 0; slot < seq_len * top_k; ++slot)
    {
        routing_indices[static_cast<size_t>(slot)] =
            static_cast<float>((slot * 17 + 5) % num_experts);
        routing_weights[static_cast<size_t>(slot)] =
            (slot % top_k == 0) ? 0.625f : 0.375f;
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto output = makeZeros({seq_len, d_model});
    auto routing_tensor = makeTensor({seq_len, top_k}, routing_indices);
    auto weights_tensor = makeTensor({seq_len, top_k}, routing_weights);

    // Warmup allocates grouping/prefill scratch and pins tensor residency before capture.
    ASSERT_TRUE(cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k));
    ASSERT_TRUE(cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_grouping = cuda_kernel_->prepareExpertGroupsAsync(
        routing_tensor.get(), weights_tensor.get(), seq_len, num_experts, top_k);
    const bool captured_prefill = cuda_kernel_->executeGroupedPrefillPipeline(
        hidden.get(), output.get(), gateup_table, down_table,
        seq_len, d_model, intermediate, num_experts, top_k);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_grouping);
    EXPECT_TRUE(captured_prefill);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    ASSERT_NE(graph, nullptr);

    cudaGraphExec_t executable = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(executable, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (size_t i = 0; i < output->numel(); ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, RuntimeGroupedDecodeDescriptorPathCapturesAfterWarmup)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    using llaminar2::DeviceMoERuntimeTable;
    constexpr int num_layers = 1;
    constexpr int num_experts = 2;
    constexpr int top_k = 2;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr int seq_len = 1;
    constexpr size_t descriptor_bytes = 4096;

    DeviceMoERuntimeTable::Config cuda_config;
    cuda_config.device_id = llaminar2::DeviceId::cuda(0);
    cuda_config.num_layers = num_layers;
    cuda_config.num_experts = num_experts;
    cuda_config.top_k = top_k;
    cuda_config.mirror_to_device = true;
    DeviceMoERuntimeTable cuda_table(cuda_config);

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemsetAsync(payloads.back().get(), 0, descriptor_bytes, stream_), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpyAsync(scales.back().get(), host_scales.data(),
                                  host_scales.size() * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.reserve(num_experts);
    gate_descs.reserve(num_experts);
    up_descs.reserve(num_experts);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        down_descs.push_back(add_desc(d_model, intermediate));
        gate_descs.push_back(add_desc(intermediate, d_model));
        up_descs.push_back(add_desc(intermediate, d_model));
    }

    llaminar2::MoEPlacementUpdate update;
    update.epoch = 1;
    update.expert_count = num_experts;
    update.experts.resize(num_experts);
    update.local_compute_mask.assign(num_experts, 1);
    update.replica_role.resize(num_experts, 0);
    for (int expert = 0; expert < num_experts; ++expert)
    {
        update.experts[expert].logical_expert_id = expert;
        update.experts[expert].gate = gate_descs[expert];
        update.experts[expert].up = up_descs[expert];
        update.experts[expert].down = down_descs[expert];
    }

    ASSERT_TRUE(cuda_table.prepareInactiveBank(0, update));
    ASSERT_TRUE(cuda_table.flipActiveBank(0, update.epoch, stream_));

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(d_model);
    std::vector<float> router_values(static_cast<size_t>(num_experts) * d_model);
    for (int i = 0; i < d_model; ++i)
    {
        hidden_values[i] = 0.01f * static_cast<float>((i % 7) - 3);
        router_values[i] = 0.02f * static_cast<float>((i % 5) - 2);
        router_values[static_cast<size_t>(d_model) + i] = 0.03f * static_cast<float>((i % 3) - 1);
    }

    auto hidden = makeTensor({seq_len, d_model}, hidden_values);
    auto router = makeTensor({num_experts, d_model}, router_values);
    auto gate0 = makeZeros({intermediate});
    auto gate1 = makeZeros({intermediate});
    auto up0 = makeZeros({intermediate});
    auto up1 = makeZeros({intermediate});
    auto output = makeZeros({d_model});

    std::array<llaminar2::ITensor *, top_k> gate_outputs = {gate0.get(), gate1.get()};
    std::array<llaminar2::ITensor *, top_k> up_outputs = {up0.get(), up1.get()};
    auto *runtime_layer = cuda_table.deviceLayerState(0);

    ASSERT_TRUE(cuda_kernel_->decodeRouteSelect(
        runtime_layer, hidden.get(), router.get(), d_model, num_experts, top_k,
        true, nullptr, nullptr, /*write_legacy_outputs=*/false, /*update_runtime_histogram=*/false));
    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromRuntime(
        gate_outputs.data(), up_outputs.data(), runtime_layer, down_table, top_k,
        output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (int i = 0; i < d_model; ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_gateup = cuda_kernel_->groupedExpertGateUpDecodeFromRuntime(
        runtime_layer, hidden.get(), gateup_table, top_k,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate);
    const bool captured_down = cuda_kernel_->groupedExpertDownDecodeFromRuntime(
        gate_outputs.data(), up_outputs.data(), runtime_layer, down_table, top_k,
        output.get(), d_model, intermediate);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_gateup);
    EXPECT_TRUE(captured_down);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    if (graph)
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}

TEST_F(Test__CUDAMoEKernel, StaticGroupedDecodeDescriptorPathCapturesAfterWarmup)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    constexpr int num_experts = 1;
    constexpr int num_active = 1;
    constexpr int d_model = 32;
    constexpr int intermediate = 32;
    constexpr size_t descriptor_bytes = 4096;
    const int expert_ids[num_active] = {0};
    const float expert_weights[num_active] = {1.0f};

    std::vector<CudaAllocation> payloads;
    std::vector<CudaAllocation> scales;
    payloads.reserve(static_cast<size_t>(num_experts * 3));
    scales.reserve(static_cast<size_t>(num_experts * 3));

    auto add_desc = [&](int rows, int cols)
    {
        payloads.emplace_back(descriptor_bytes);
        scales.emplace_back(descriptor_bytes);
        EXPECT_EQ(cudaMemset(payloads.back().get(), 0, descriptor_bytes), cudaSuccess);

        const int scale_count = rows * (cols / 32);
        std::vector<uint16_t> host_scales(static_cast<size_t>(scale_count), 0x3c00u);
        EXPECT_EQ(cudaMemcpy(scales.back().get(), host_scales.data(),
                             host_scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  cudaSuccess);
        return makeCudaNativeDesc(payloads.back(), scales.back(), rows, cols);
    };

    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> down_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> gate_descs;
    std::vector<llaminar2::DeviceNativeVNNIMatrixDesc> up_descs;
    down_descs.push_back(add_desc(d_model, intermediate));
    gate_descs.push_back(add_desc(intermediate, d_model));
    up_descs.push_back(add_desc(intermediate, d_model));

    const int down_table = cuda_kernel_->uploadGroupedExpertDownDescriptorTable(
        down_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(down_table, 0);
    const int gateup_table = cuda_kernel_->uploadGroupedExpertGateUpDescriptorTables(
        gate_descs.data(), up_descs.data(), num_experts, d_model, intermediate);
    ASSERT_GE(gateup_table, 0);

    std::vector<float> hidden_values(d_model);
    for (int i = 0; i < d_model; ++i)
        hidden_values[i] = 0.01f * static_cast<float>((i % 7) - 3);

    auto hidden = makeTensor({1, d_model}, hidden_values);
    auto gate = makeZeros({intermediate});
    auto up = makeZeros({intermediate});
    auto output = makeZeros({d_model});

    ASSERT_TRUE(hidden->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));
    ASSERT_TRUE(gate->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));
    ASSERT_TRUE(up->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));
    ASSERT_TRUE(output->ensureOnDevice(llaminar2::DeviceId::cuda(0), stream_));

    std::array<llaminar2::ITensor *, num_active> gate_outputs = {gate.get()};
    std::array<llaminar2::ITensor *, num_active> up_outputs = {up.get()};

    ASSERT_TRUE(cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate));
    ASSERT_TRUE(cuda_kernel_->groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate));
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const float *output_data = output->data();
    for (int i = 0; i < d_model; ++i)
        ASSERT_TRUE(std::isfinite(output_data[i])) << "output element " << i;

    ASSERT_EQ(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), cudaSuccess);
    const bool captured_gateup = cuda_kernel_->groupedExpertGateUpDecodeFromTable(
        hidden.get(), expert_ids, gateup_table, num_active,
        gate_outputs.data(), up_outputs.data(), d_model, intermediate);
    const bool captured_down = cuda_kernel_->groupedExpertDownDecodeFromTable(
        gate_outputs.data(), up_outputs.data(), expert_ids, expert_weights,
        down_table, num_active, output.get(), d_model, intermediate);
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph);
    EXPECT_TRUE(captured_gateup);
    EXPECT_TRUE(captured_down);
    ASSERT_EQ(capture_status, cudaSuccess) << cudaGetErrorString(capture_status);
    if (graph)
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
#endif
}
