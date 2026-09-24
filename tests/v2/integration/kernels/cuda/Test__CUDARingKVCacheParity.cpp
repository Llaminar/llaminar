/**
 * @file Test__CUDARingKVCacheParity.cpp
 * @brief Parity tests for CUDA Ring Buffer KV Cache
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests:
 * 1. Basic append and retrieval
 * 2. Ring buffer wrap-around behavior
 * 3. O(1) eviction correctness
 * 4. Sliding window pattern
 * 5. Batched gather
 * 6. Multi-precision (FP32, FP16, BF16)
 * 7. Captured unequal-length continuation and seed-authenticated diagnostic
 *    partitions, with external producer events joined before capture.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "transfer/TransferEngine.h"
#include "kernels/cuda/kvcache/CUDARingKVCache.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    // Check CUDA availability
    bool hasCUDA()
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }

    /**
     * @brief Owns a CUDA stream for integration tests that exercise stream-aware KV APIs.
     */
    class ScopedCudaStream
    {
    public:
        ScopedCudaStream()
        {
            EXPECT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
        }

        ~ScopedCudaStream()
        {
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        void *opaque() const { return static_cast<void *>(stream_); }
        cudaStream_t stream() const { return stream_; }

        void synchronize() const
        {
            ASSERT_NE(stream_, nullptr);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        }

    private:
        cudaStream_t stream_ = nullptr;
    };

    // Generate random FP32 data
    std::vector<float> generateRandomFP32(size_t count, unsigned seed = 42)
    {
        std::vector<float> data(count);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : data)
        {
            val = dist(rng);
        }
        return data;
    }

    // Compute max absolute error
    float computeMaxError(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    std::unique_ptr<DeviceWorkspaceManager> bindRequiredWorkspace(
        IWorkspaceConsumer *consumer,
        int m,
        int n = 0,
        int k = 0)
    {
        auto reqs = consumer->getWorkspaceRequirements(m, n, k);
        const size_t budget = reqs.total_bytes_with_alignment() + 4096;
        auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cuda(0), budget);
        EXPECT_TRUE(workspace->allocate(reqs));
        consumer->bindWorkspace(workspace.get());
        EXPECT_TRUE(consumer->hasWorkspace());
        return workspace;
    }

    /**
     * @brief Prove one captured resident gather follows a growing device count.
     *
     * Production prefill captures a chunk-shaped graph only once. The cache
     * count can be larger on every later replay, so the gather's physical
     * request stride and launch topology must come from the resident cache
     * capacity rather than the prefix visible while capture occurred. This
     * helper captures at eight rows, appends twelve more rows, and then requires
     * the unchanged graph to publish all twenty native rows byte for byte.
     *
     * @tparam Precision Native CUDA cache format under test.
     * @param format_name Human-readable format included in assertion output.
     */
    template <ActivationPrecision Precision>
    void runCapturedResidentGatherGrowthByteExact(const char *format_name)
    {
        using DataT = typename llaminar2::detail::CUDAKVCacheType<Precision>::Type;

        constexpr int layer = 0;
        constexpr int batch_size = 1;
        constexpr int max_seq_len = 32;
        constexpr int capture_rows = 8;
        constexpr int continuation_rows = 12;
        constexpr int live_rows = capture_rows + continuation_rows;
        constexpr int n_kv_heads = 2;
        constexpr int head_dim = 32;
        constexpr int logical_kv_dim = n_kv_heads * head_dim;
        constexpr int storage_dim =
            Precision == ActivationPrecision::Q8_1
                ? logical_kv_dim / static_cast<int>(Q8_1Block::BLOCK_SIZE)
                : logical_kv_dim;

        SCOPED_TRACE(format_name);
        ASSERT_GT(storage_dim, 0);

        CUDARingKVCache<Precision> cache(
            /*n_layers=*/1,
            batch_size,
            max_seq_len,
            n_kv_heads,
            head_dim,
            /*device_id=*/0);
        const WorkspaceRequirements requirements =
            cache.getWorkspaceRequirements(
                capture_rows,
                batch_size,
                head_dim);
        DeviceWorkspaceManager workspace(
            DeviceId::cuda(0),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements));
        cache.bindWorkspace(&workspace);
        ASSERT_TRUE(cache.hasWorkspace());

        std::vector<DataT> source_k(
            static_cast<size_t>(live_rows) * storage_dim);
        std::vector<DataT> source_v(
            static_cast<size_t>(live_rows) * storage_dim);
        auto fillNativeBytes = [](std::vector<DataT> &values, uint8_t salt)
        {
            auto *bytes = reinterpret_cast<uint8_t *>(values.data());
            const size_t byte_count = values.size() * sizeof(DataT);
            for (size_t index = 0; index < byte_count; ++index)
            {
                bytes[index] = static_cast<uint8_t>(
                    1u + ((static_cast<unsigned int>(salt) +
                           static_cast<unsigned int>(index * 29u)) %
                          251u));
            }
        };
        fillNativeBytes(source_k, 17);
        fillNativeBytes(source_v, 113);

        ScopedCudaStream stream;
        DataT *device_k = nullptr;
        DataT *device_v = nullptr;
        const size_t source_bytes = source_k.size() * sizeof(DataT);
        ASSERT_EQ(cudaMalloc(&device_k, source_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&device_v, source_bytes), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_k, source_k.data(), source_bytes,
                cudaMemcpyHostToDevice, stream.stream()),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_v, source_v.data(), source_bytes,
                cudaMemcpyHostToDevice, stream.stream()),
            cudaSuccess);
        ASSERT_TRUE(cache.append(
            layer, /*seq_idx=*/0, device_k, device_v,
            capture_rows, stream.stream()));

        /*
         * Establish cache-owned tensor wrappers before global capture. The
         * subsequent call is the operation whose launch geometry is recorded.
         */
        ITensor *gathered_k = nullptr;
        ITensor *gathered_v = nullptr;
        ASSERT_TRUE(cache.get_kv_batched_device_view(
            layer, /*first_seq_idx=*/0, batch_size,
            &gathered_k, &gathered_v, stream.opaque()));
        stream.synchronize();

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            cudaStreamBeginCapture(
                stream.stream(), cudaStreamCaptureModeGlobal),
            cudaSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = cache.get_kv_batched_device_view(
                layer, /*first_seq_idx=*/0, batch_size,
                &gathered_k, &gathered_v, stream.opaque());
        }
        ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            cudaSuccess);

        ASSERT_TRUE(cache.append(
            layer,
            /*seq_idx=*/0,
            device_k + static_cast<size_t>(capture_rows) * storage_dim,
            device_v + static_cast<size_t>(capture_rows) * storage_dim,
            continuation_rows,
            stream.stream()));
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();

        ASSERT_NE(gathered_k, nullptr);
        ASSERT_NE(gathered_v, nullptr);
        const size_t live_elements =
            static_cast<size_t>(live_rows) * storage_dim;
        std::vector<DataT> actual_k(live_elements);
        std::vector<DataT> actual_v(live_elements);
        const size_t live_bytes = live_elements * sizeof(DataT);
        ASSERT_EQ(
            cudaMemcpyAsync(
                actual_k.data(), gathered_k->gpu_data_ptr(), live_bytes,
                cudaMemcpyDeviceToHost, stream.stream()),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                actual_v.data(), gathered_v->gpu_data_ptr(), live_bytes,
                cudaMemcpyDeviceToHost, stream.stream()),
            cudaSuccess);
        stream.synchronize();

        EXPECT_EQ(
            std::memcmp(actual_k.data(), source_k.data(), live_bytes), 0)
            << "Captured resident K gather stopped at its capture-time horizon for "
            << format_name;
        EXPECT_EQ(
            std::memcmp(actual_v.data(), source_v.data(), live_bytes), 0)
            << "Captured resident V gather stopped at its capture-time horizon for "
            << format_name;

        EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        EXPECT_EQ(cudaFree(device_k), cudaSuccess);
        EXPECT_EQ(cudaFree(device_v), cudaSuccess);
        cache.unbindWorkspace();
    }

} // namespace

/**
 * @brief Every standard CUDA cache format grows beyond capture byte-exactly.
 */
TEST(Test__CUDARingKVCache, CapturedResidentGatherGrowthAllFormatsIsByteExact)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    runCapturedResidentGatherGrowthByteExact<ActivationPrecision::FP32>("FP32");
    runCapturedResidentGatherGrowthByteExact<ActivationPrecision::FP16>("FP16");
    runCapturedResidentGatherGrowthByteExact<ActivationPrecision::BF16>("BF16");
    runCapturedResidentGatherGrowthByteExact<ActivationPrecision::Q8_1>("Q8_1");
}

// =============================================================================
// Test: Basic Append and Retrieval
// =============================================================================

TEST(Test__CUDARingKVCache, BasicAppendRetrieve_FP32)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Parameters
    const int n_layers = 2;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    // Create cache
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    ScopedCudaStream stream;

    // Generate test data (10 tokens)
    const int num_tokens = 10;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);

    // Allocate device memory
    float *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * kv_dim * sizeof(float));
    cudaMalloc(&d_V, num_tokens * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Append to cache (layer 0)
    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, num_tokens, stream.stream()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);
    EXPECT_FALSE(cache->is_wrapped(0, 0)); // Should not be wrapped yet

    // Retrieve K/V
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back and verify
    std::vector<float> h_K_out(num_tokens * kv_dim);
    std::vector<float> h_V_out(num_tokens * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err_K = computeMaxError(h_K, h_K_out);
    float max_err_V = computeMaxError(h_V, h_V_out);

    LOG_INFO("[BasicAppendRetrieve] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f);
    EXPECT_EQ(max_err_V, 0.0f);

    // Cleanup
    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[BasicAppendRetrieve_FP32] PASSED");
}

/**
 * @brief Device publication leaves GPU-visible KV metadata as the sole owner.
 *
 * MTP publication updates the canonical device count/head rows on an explicit
 * stream. Host cache getters are diagnostic shadows and deliberately remain at
 * the pre-publication boundary; production has no adoption API that can turn
 * those stale values back into execution state.
 */
TEST(Test__CUDARingKVCache, DeviceResidentSequenceStatePublicationRemainsDeviceOwned)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr int n_layers = 1;
    constexpr int batch_size = 1;
    constexpr int max_seq_len = 8;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 16;
    constexpr int kv_dim = n_kv_heads * head_dim;
    constexpr int initial_tokens = 6;
    constexpr int target_cached_tokens = 5;
    constexpr int accepted_state_count = 1;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    ASSERT_TRUE(cache->supportsDeviceResidentSequenceStatePublication());
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(initial_tokens * kv_dim, 20260615);
    auto h_V = generateRandomFP32(initial_tokens * kv_dim, 20260616);

    float *d_K = nullptr;
    float *d_V = nullptr;
    ASSERT_EQ(cudaMalloc(&d_K, initial_tokens * kv_dim * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_V, initial_tokens * kv_dim * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  d_K,
                  h_K.data(),
                  initial_tokens * kv_dim * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream.stream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  d_V,
                  h_V.data(),
                  initial_tokens * kv_dim * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream.stream()),
              cudaSuccess);
    ASSERT_TRUE(cache->append(0, 0, d_K, d_V, initial_tokens, stream.stream()));
    stream.synchronize();

    const int initial_count = cache->get_cached_tokens(0, 0);
    const int initial_head = cache->ring_head(0, 0);
    ASSERT_EQ(initial_count, initial_tokens);

    int32_t *d_target = nullptr;
    int32_t *d_accepted = nullptr;
    int32_t *d_ok = nullptr;
    const int32_t h_target = target_cached_tokens;
    const int32_t h_accepted = accepted_state_count;
    const int32_t h_ok = 1;
    ASSERT_EQ(cudaMalloc(&d_target, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_accepted, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ok, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_target, &h_target, sizeof(int32_t), cudaMemcpyHostToDevice, stream.stream()), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_accepted, &h_accepted, sizeof(int32_t), cudaMemcpyHostToDevice, stream.stream()), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(d_ok, &h_ok, sizeof(int32_t), cudaMemcpyHostToDevice, stream.stream()), cudaSuccess);

    IKVCache::DeviceSequenceStatePublicationRequest device_request;
    device_request.request_count = 1;
    device_request.first_seq_idx = 0;
    device_request.target_cached_tokens_device = d_target;
    device_request.accepted_state_counts_device = d_accepted;
    device_request.publication_ok_flags_device = d_ok;
    device_request.basis =
        IKVCache::DeviceSequenceStatePublicationBasis::
            CurrentVisibleWindow;
    device_request.stream = stream.opaque();
    std::string device_error;
    ASSERT_TRUE(cache->publishSequenceStateFromDeviceMetadata(device_request, &device_error))
        << device_error;
    stream.synchronize();

    const int initial_tail =
        (initial_head - initial_count + max_seq_len) % max_seq_len;
    const int expected_device_head =
        (initial_tail + target_cached_tokens) % max_seq_len;
    int device_count = -1;
    int device_head = -1;
    ASSERT_NE(cache->deviceCachedTokenCountPtr(0, 0), nullptr);
    ASSERT_NE(cache->deviceRingHeadPtr(0, 0), nullptr);
    ASSERT_EQ(cudaMemcpyAsync(&device_count,
                              cache->deviceCachedTokenCountPtr(0, 0),
                              sizeof(int),
                              cudaMemcpyDeviceToHost,
                              stream.stream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&device_head,
                              cache->deviceRingHeadPtr(0, 0),
                              sizeof(int),
                              cudaMemcpyDeviceToHost,
                              stream.stream()),
              cudaSuccess);
    stream.synchronize();

    EXPECT_EQ(device_count, target_cached_tokens);
    EXPECT_EQ(device_head, expected_device_head);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), target_cached_tokens)
        << "Diagnostic observation must read the canonical device count.";
    EXPECT_EQ(cache->ring_head(0, 0), expected_device_head)
        << "Diagnostic observation must read the canonical device ring head.";

    cudaFree(d_ok);
    cudaFree(d_accepted);
    cudaFree(d_target);
    cudaFree(d_V);
    cudaFree(d_K);
}

/**
 * @brief Captured-base publication commits every cache layer, not only layer zero.
 *
 * The MTP verifier mutates all attention-layer rings before acceptance is known.
 * Publication therefore owns a two-dimensional request/layer commit and must
 * derive every canonical pair from one immutable checkpoint. A multi-layer
 * fixture makes incomplete launch geometry observable.
 */
TEST(Test__CUDARingKVCache, CapturedBasePublicationCommitsEveryLayer)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    constexpr int n_layers = 4;
    constexpr int batch_size = 1;
    constexpr int max_seq_len = 16;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 16;
    constexpr int kv_dim = n_kv_heads * head_dim;
    constexpr int base_tokens = 6;
    constexpr int verifier_rows = 2;
    constexpr int accepted_rows = 1;
    constexpr int target_tokens = base_tokens + accepted_rows;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(base_tokens * kv_dim, 20260730);
    auto h_V = generateRandomFP32(base_tokens * kv_dim, 20260731);
    float *d_K = nullptr;
    float *d_V = nullptr;
    ASSERT_EQ(
        cudaMalloc(&d_K, base_tokens * kv_dim * sizeof(float)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(&d_V, base_tokens * kv_dim * sizeof(float)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_K, h_K.data(), base_tokens * kv_dim * sizeof(float),
            cudaMemcpyHostToDevice, stream.stream()),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_V, h_V.data(), base_tokens * kv_dim * sizeof(float),
            cudaMemcpyHostToDevice, stream.stream()),
        cudaSuccess);
    for (int layer = 0; layer < n_layers; ++layer)
    {
        ASSERT_TRUE(cache->append(
            layer, 0, d_K, d_V, base_tokens, stream.stream()));
    }

    const size_t checkpoint_bytes =
        cache->deviceSequenceStateCheckpointBytes();
    ASSERT_EQ(
        checkpoint_bytes,
        static_cast<size_t>(2 * n_layers) * sizeof(int32_t));
    int32_t *d_checkpoint = nullptr;
    ASSERT_EQ(cudaMalloc(&d_checkpoint, checkpoint_bytes), cudaSuccess);
    std::string checkpoint_error;
    ASSERT_TRUE(cache->captureDeviceSequenceStateCheckpoint(
        0,
        d_checkpoint,
        checkpoint_bytes,
        stream.opaque(),
        &checkpoint_error))
        << checkpoint_error;

    for (int layer = 0; layer < n_layers; ++layer)
    {
        ASSERT_TRUE(cache->append(
            layer, 0, d_K, d_V, verifier_rows, stream.stream()));
    }

    int32_t *d_target = nullptr;
    int32_t *d_accepted = nullptr;
    int32_t *d_ok = nullptr;
    const int32_t h_target = target_tokens;
    const int32_t h_accepted = accepted_rows;
    const int32_t h_ok = 1;
    ASSERT_EQ(cudaMalloc(&d_target, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_accepted, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ok, sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_target, &h_target, sizeof(int32_t),
            cudaMemcpyHostToDevice, stream.stream()),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_accepted, &h_accepted, sizeof(int32_t),
            cudaMemcpyHostToDevice, stream.stream()),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_ok, &h_ok, sizeof(int32_t),
            cudaMemcpyHostToDevice, stream.stream()),
        cudaSuccess);

    IKVCache::DeviceSequenceStatePublicationRequest request;
    request.request_count = 1;
    request.first_seq_idx = 0;
    request.target_cached_tokens_device = d_target;
    request.accepted_state_counts_device = d_accepted;
    request.publication_ok_flags_device = d_ok;
    request.basis =
        IKVCache::DeviceSequenceStatePublicationBasis::CapturedBase;
    request.base_sequence_state_checkpoint_device = d_checkpoint;
    request.base_sequence_state_checkpoint_bytes = checkpoint_bytes;
    request.stream = stream.opaque();
    std::string publication_error;
    ASSERT_TRUE(cache->publishSequenceStateFromDeviceMetadata(
        request, &publication_error))
        << publication_error;

    std::array<int, n_layers> heads{};
    std::array<int, n_layers> counts{};
    for (int layer = 0; layer < n_layers; ++layer)
    {
        ASSERT_EQ(
            cudaMemcpyAsync(
                &heads[static_cast<size_t>(layer)],
                cache->deviceRingHeadPtr(layer, 0),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream()),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(
                &counts[static_cast<size_t>(layer)],
                cache->deviceCachedTokenCountPtr(layer, 0),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream()),
            cudaSuccess);
    }
    stream.synchronize();

    for (int layer = 0; layer < n_layers; ++layer)
    {
        EXPECT_EQ(heads[static_cast<size_t>(layer)], target_tokens)
            << "layer=" << layer;
        EXPECT_EQ(counts[static_cast<size_t>(layer)], target_tokens)
            << "layer=" << layer;
    }

    cudaFree(d_ok);
    cudaFree(d_accepted);
    cudaFree(d_target);
    cudaFree(d_checkpoint);
    cudaFree(d_V);
    cudaFree(d_K);
}

// =============================================================================
// Test: Ring Buffer Wrap-Around
// =============================================================================

TEST(Test__CUDARingKVCache, WrapAround_FP32)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Small buffer to force wrap-around
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 8; // Small!
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    ScopedCudaStream stream;

    // Phase 1: Fill buffer with 6 tokens [T0..T5]
    const int phase1_tokens = 6;
    auto h_K1 = generateRandomFP32(phase1_tokens * kv_dim, 100);
    auto h_V1 = generateRandomFP32(phase1_tokens * kv_dim, 200);

    float *d_K, *d_V;
    cudaMalloc(&d_K, max_seq_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V, max_seq_len * kv_dim * sizeof(float));

    cudaMemcpy(d_K, h_K1.data(), phase1_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V1.data(), phase1_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, phase1_tokens, stream.stream()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 6);
    EXPECT_EQ(cache->get_head_position(0, 0), 6);
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    // Phase 2: Append 4 more tokens [T6..T9] - causes wrap!
    // Buffer state: [T6,T7,T8,T9,T4,T5,_,_] after auto-evict
    // Actually since max=8 and we have 6, adding 4 means 10 > 8
    // So we evict 2 oldest (T0,T1), leaving T2..T9 in the buffer
    const int phase2_tokens = 4;
    auto h_K2 = generateRandomFP32(phase2_tokens * kv_dim, 300);
    auto h_V2 = generateRandomFP32(phase2_tokens * kv_dim, 400);

    cudaMemcpy(d_K, h_K2.data(), phase2_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V2.data(), phase2_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, phase2_tokens, stream.stream()));
    stream.synchronize();

    // The canonical count clamps at capacity, retaining T2 through T9.
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 8);

    // Head should have wrapped: 6 + 4 = 10 % 8 = 2
    EXPECT_EQ(cache->get_head_position(0, 0), 2);

    // Now buffer IS wrapped (tail=2, head=2, but count=8 so tail=(2-8+8)%8=2)
    // Actually: tail = (head - count + max) % max = (2 - 8 + 8) % 8 = 2
    // So tail=2, head=2, but count=8 means whole buffer is used
    // is_wrapped check: tail >= head && count > 0 --> 2 >= 2 && 8 > 0 --> TRUE
    EXPECT_TRUE(cache->is_wrapped(0, 0));

    // Retrieve and verify linearization happens
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, 8);
    EXPECT_EQ(cache->get_linearization_count(), 1); // Should have linearized

    // Verify content: should have T2,T3,T4,T5 from phase1 and T6,T7,T8,T9 from phase2
    std::vector<float> h_K_out(8 * kv_dim);
    std::vector<float> h_V_out(8 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 8 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, 8 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Check T2-T5 (indices 2-5 from phase1 data)
    for (int t = 0; t < 4; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = (t + 2) * kv_dim + d; // T2-T5 in original
            int dst_idx = t * kv_dim + d;       // T0-T3 in output
            EXPECT_FLOAT_EQ(h_K_out[dst_idx], h_K1[src_idx])
                << "K mismatch at output token " << t << " dim " << d;
        }
    }

    // Check T6-T9 (indices 0-3 from phase2 data)
    for (int t = 0; t < 4; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = t * kv_dim + d;       // T6-T9 in phase2
            int dst_idx = (t + 4) * kv_dim + d; // T4-T7 in output
            EXPECT_FLOAT_EQ(h_K_out[dst_idx], h_K2[src_idx])
                << "K mismatch at output token " << (t + 4) << " dim " << d;
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[WrapAround_FP32] PASSED - linearization_count=" << cache->get_linearization_count());
}

// =============================================================================
// Test: O(1) Eviction
// =============================================================================

TEST(Test__CUDARingKVCache, Eviction_O1)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 100;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ScopedCudaStream stream;

    // Fill with 50 tokens
    auto h_K = generateRandomFP32(50 * kv_dim);
    auto h_V = generateRandomFP32(50 * kv_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 50 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 50 * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), 50 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), 50 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, 50, stream.stream()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 50);

    // Evict 20 tokens - should be O(1), no kernel launch
    cache->evict_oldest(0, 0, 20);

    EXPECT_EQ(cache->get_cached_tokens(0, 0), 30);

    // Head position unchanged (eviction only affects tail)
    EXPECT_EQ(cache->get_head_position(0, 0), 50);

    // Retrieve remaining 30 tokens
    const void *d_K_out, *d_V_out;
    int kv_len;
    cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0);
    EXPECT_EQ(kv_len, 30);

    // Verify content: should have T20-T49
    std::vector<float> h_K_out(30 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 30 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    for (int t = 0; t < 30; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = (t + 20) * kv_dim + d; // T20-T49 in original
            int dst_idx = t * kv_dim + d;
            EXPECT_FLOAT_EQ(h_K_out[dst_idx], h_K[src_idx])
                << "K mismatch at token " << t;
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[Eviction_O1] PASSED");
}

// =============================================================================
// Test: Sliding Window Pattern
// =============================================================================

TEST(Test__CUDARingKVCache, SlidingWindow)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    // Simulate sliding window attention with window_size=32
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32; // Window size
    const int n_kv_heads = 4;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ScopedCudaStream stream;

    float *d_K, *d_V;
    cudaMalloc(&d_K, kv_dim * sizeof(float));
    cudaMalloc(&d_V, kv_dim * sizeof(float));

    // Simulate 100 decode steps
    for (int step = 0; step < 100; ++step)
    {
        auto h_K = generateRandomFP32(kv_dim, step);
        auto h_V = generateRandomFP32(kv_dim, step + 1000);

        cudaMemcpy(d_K, h_K.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Append 1 token
        ASSERT_TRUE(cache->append(
            0, 0, d_K, d_V, 1, stream.stream()));
        stream.synchronize();

        // Cache should never exceed window size (auto-evicts)
        EXPECT_LE(cache->get_cached_tokens(0, 0), max_seq_len);
    }

    // After 100 steps with window=32, should have exactly 32 tokens
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 32);

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[SlidingWindow] PASSED - retained=" << max_seq_len);
}

// =============================================================================
// Test: Batched Gather
// =============================================================================

TEST(Test__CUDARingKVCache, BatchedGather)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);
    auto workspace = bindRequiredWorkspace(workspace_consumer, batch_size);
    ASSERT_NE(workspace, nullptr);
    ScopedCudaStream stream;

    // Fill each sequence with different lengths
    int seq_lens[] = {10, 20, 15, 25};
    std::vector<std::vector<float>> h_Ks(batch_size);
    std::vector<std::vector<float>> h_Vs(batch_size);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 30 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 30 * kv_dim * sizeof(float));

    for (int seq = 0; seq < batch_size; ++seq)
    {
        h_Ks[seq] = generateRandomFP32(seq_lens[seq] * kv_dim, seq * 100);
        h_Vs[seq] = generateRandomFP32(seq_lens[seq] * kv_dim, seq * 100 + 1000);

        cudaMemcpy(d_K, h_Ks[seq].data(), seq_lens[seq] * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_Vs[seq].data(), seq_lens[seq] * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        cache->append(0, seq, d_K, d_V, seq_lens[seq], stream.stream());
    }
    stream.synchronize();

    // Verify individual sequence lengths
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(cache->get_cached_tokens(0, seq), seq_lens[seq]);
    }

    // Gather all sequences
    int max_kv_len = 25; // Max sequence length
    float *d_K_gathered, *d_V_gathered;
    cudaMalloc(&d_K_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));

    std::vector<int> kv_lens(batch_size);
    int actual_max = cache->gather_kv_batched(0, batch_size,
                                              d_K_gathered, d_V_gathered,
                                              kv_lens.data(), max_kv_len, stream.stream());
    stream.synchronize();

    EXPECT_EQ(actual_max, 25); // Max across sequences

    // Verify per-sequence lengths
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(kv_lens[seq], seq_lens[seq]);
    }

    // Verify content for sequence 0
    std::vector<float> h_K_gathered(batch_size * max_kv_len * kv_dim);
    cudaMemcpy(h_K_gathered.data(), d_K_gathered,
               batch_size * max_kv_len * kv_dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int t = 0; t < seq_lens[0]; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            int src_idx = t * kv_dim + d;
            int dst_idx = (0 * max_kv_len + t) * kv_dim + d;
            EXPECT_FLOAT_EQ(h_K_gathered[dst_idx], h_Ks[0][src_idx])
                << "Seq0 K mismatch at token " << t;
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_gathered);
    cudaFree(d_V_gathered);

    LOG_INFO("[BatchedGather] PASSED");
}

// =============================================================================
// Test: Contiguous Optimization
// =============================================================================

TEST(Test__CUDARingKVCache, ContiguousOptimization)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ScopedCudaStream stream;

    // Append tokens without wrapping
    auto h_K = generateRandomFP32(30 * kv_dim);
    auto h_V = generateRandomFP32(30 * kv_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 30 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 30 * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), 30 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), 30 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, 30, stream.stream()));
    stream.synchronize();

    // Should NOT be wrapped
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    // Get K/V - should return direct pointer (no linearization)
    const void *d_K_out, *d_V_out;
    int kv_len;
    cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0);

    // No linearizations should have occurred
    EXPECT_EQ(cache->get_linearization_count(), 0);

    // Multiple retrievals should still not linearize
    for (int i = 0; i < 10; ++i)
    {
        cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0);
    }
    EXPECT_EQ(cache->get_linearization_count(), 0);

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[ContiguousOptimization] PASSED - linearizations=0");
}

// =============================================================================
// Test: Clear Operations
// =============================================================================

TEST(Test__CUDARingKVCache, ClearOperations)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 3;
    const int batch_size = 2;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ScopedCudaStream stream;

    // Fill all layers and sequences
    auto h_K = generateRandomFP32(10 * kv_dim);
    auto h_V = generateRandomFP32(10 * kv_dim);

    float *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K.data(), 10 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), 10 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            ASSERT_TRUE(cache->append(
                layer, seq, d_K, d_V, 10, stream.stream()));
        }
    }
    stream.synchronize();

    // Verify all filled
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache->get_cached_tokens(layer, seq), 10);
        }
    }

    ScopedCudaStream reset_stream;

    // Reset one layer/sequence entry.
    ASSERT_TRUE(cache->resetLayerSequenceState(
        1,
        0,
        IKVCache::StateResetContext::testReinitialization(
            reset_stream.opaque())));
    EXPECT_EQ(cache->get_cached_tokens(1, 0), 0);
    EXPECT_EQ(cache->get_cached_tokens(1, 1), 10); // Other sequence unchanged

    // Clear entire layer
    ASSERT_TRUE(cache->resetLayerState(
        2,
        IKVCache::StateResetContext::testReinitialization(
            reset_stream.opaque())));
    EXPECT_EQ(cache->get_cached_tokens(2, 0), 0);
    EXPECT_EQ(cache->get_cached_tokens(2, 1), 0);

    // Clear all
    ASSERT_TRUE(cache->resetRequestState(
        IKVCache::StateResetContext::testReinitialization(
            reset_stream.opaque())));
    for (int layer = 0; layer < n_layers; ++layer)
    {
        for (int seq = 0; seq < batch_size; ++seq)
        {
            EXPECT_EQ(cache->get_cached_tokens(layer, seq), 0);
        }
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[ClearOperations] PASSED");
}

// =============================================================================
// Test: Multi-Precision (FP16)
// =============================================================================

TEST(Test__CUDARingKVCache, MultiPrecision_FP16)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);
    ScopedCudaStream stream;

    // Generate FP32 data and convert to FP16
    auto h_K_fp32 = generateRandomFP32(10 * kv_dim);
    auto h_V_fp32 = generateRandomFP32(10 * kv_dim);

    std::vector<__half> h_K_fp16(10 * kv_dim);
    std::vector<__half> h_V_fp16(10 * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_fp16[i] = __float2half(h_K_fp32[i]);
        h_V_fp16[i] = __float2half(h_V_fp32[i]);
    }

    __half *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(__half));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(__half));
    cudaMemcpy(d_K, h_K_fp16.data(), 10 * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_fp16.data(), 10 * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, 10, stream.stream()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, 10);

    // Verify content
    std::vector<__half> h_K_out(10 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 10 * kv_dim * sizeof(__half), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10 * kv_dim; ++i)
    {
        float expected = __half2float(h_K_fp16[i]);
        float actual = __half2float(h_K_out[i]);
        EXPECT_FLOAT_EQ(actual, expected) << "FP16 mismatch at " << i;
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[MultiPrecision_FP16] PASSED");
}

// =============================================================================
// REGRESSION TEST: FP32 ITensor → FP16 cache via appendWithStream
//
// Locks in the fix for the bug where appendWithStream() was never called
// because KVCacheAppendStage::Params.device_id defaulted to CPU.
//
// The bug path was: append(ITensor*) gets raw FP32 GPU pointer → passes it
// directly to ring buffer's append(void*) → raw FP32 bytes (4 bytes/elem)
// interpreted as __half (2 bytes/elem) → complete data corruption.
//
// The correct path: appendWithStream(ITensor*) detects FP32→FP16 mismatch →
// calls cuda_convert_tensor_to_fp16() on GPU → correct FP16 in ring buffer.
//
// This test validates the appendWithStream conversion path directly.
// =============================================================================

TEST(Test__CUDARingKVCache, AppendWithStream_FP32_to_FP16_Conversion)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    // Create FP16 precision cache
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::FP16);

    // Create FP32 tensors with known data (simulating projected K/V from GEMM)
    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});

    // Fill with recognizable values in [-1, 1] range
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = dist(rng);
        V_tensor->mutable_data()[i] = dist(rng);
    }

    // Upload to GPU (this populates gpu_data_ptr())
    DeviceId cuda_dev = DeviceId::cuda(0);
    ASSERT_TRUE(K_tensor->ensureOnDevice(cuda_dev));
    ASSERT_TRUE(V_tensor->ensureOnDevice(cuda_dev));
    ASSERT_NE(K_tensor->gpu_data_ptr(), nullptr);
    ASSERT_NE(V_tensor->gpu_data_ptr(), nullptr);

    // Use appendWithStream (the correct GPU path). This should detect
    // FP32→FP16 mismatch and convert on the caller's explicit stream.
    ScopedCudaStream stream;
    ASSERT_TRUE(cache->appendWithStream(0, 0,
                                        static_cast<const ITensor *>(K_tensor.get()),
                                        static_cast<const ITensor *>(V_tensor.get()),
                                        num_tokens, stream.opaque()));
    stream.synchronize();

    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve cached data
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, num_tokens);

    // Read back FP16 data from cache
    std::vector<uint16_t> h_K_out_fp16(num_tokens * kv_dim);
    std::vector<uint16_t> h_V_out_fp16(num_tokens * kv_dim);
    cudaMemcpy(h_K_out_fp16.data(), d_K_out,
               num_tokens * kv_dim * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out_fp16.data(), d_V_out,
               num_tokens * kv_dim * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Verify: FP16 round-trip should match FP32→FP16→FP32 within tolerance
    const float *k_src = K_tensor->data();
    const float *v_src = V_tensor->data();
    float max_k_err = 0.0f;
    float max_v_err = 0.0f;
    int k_zero_count = 0;

    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        float k_cached = fp16_to_fp32(h_K_out_fp16[i]);
        float v_cached = fp16_to_fp32(h_V_out_fp16[i]);

        // FP16 has ~3 decimal digits of precision; error should be < 1e-3 for [-1,1]
        float k_err = std::abs(k_cached - k_src[i]);
        float v_err = std::abs(v_cached - v_src[i]);
        max_k_err = std::max(max_k_err, k_err);
        max_v_err = std::max(max_v_err, v_err);

        if (h_K_out_fp16[i] == 0 && k_src[i] != 0.0f)
            ++k_zero_count;
    }

    LOG_INFO("[AppendWithStream_FP32_to_FP16] max_k_err=" << max_k_err
                                                          << " max_v_err=" << max_v_err
                                                          << " k_zero_count=" << k_zero_count);

    // FP16 in [-1,1] range should have error < 0.001
    EXPECT_LT(max_k_err, 0.001f)
        << "REGRESSION: FP16 K data in cache doesn't match FP32 source. "
           "If errors are very large (>1.0), appendWithStream() likely "
           "wasn't called and raw FP32 bytes were written to FP16 buffer.";
    EXPECT_LT(max_v_err, 0.001f)
        << "REGRESSION: FP16 V data in cache doesn't match FP32 source.";

    // Verify no spurious zeros (sign that raw FP32 bytes were misinterpreted)
    EXPECT_EQ(k_zero_count, 0)
        << "REGRESSION: Found " << k_zero_count << " unexpected zero values. "
                                                   "This suggests FP32→FP16 conversion was not performed.";

    LOG_INFO("[AppendWithStream_FP32_to_FP16] PASSED");
}

/**
 * @brief Proves grouped verifier KV publication is byte-identical to serial decode appends.
 *
 * The MTP verifier path appends multiple newly verified K/V rows at once.  Serial
 * decode appends those same rows one at a time through appendConvertedWithStream(),
 * so grouped publication is only decode-equivalent if the final FP16 cache bytes
 * match exactly.  This regression keeps both source layouts used by the graph:
 * position-major `[row][kv_dim]` and verifier head-major `[head][row][dim]`.
 */
TEST(Test__CUDARingKVCache, VerifierRowsFP32ToFP16AppendMatchesSerialDecodeForPositionAndHeadMajor)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr int n_layers = 1;
    constexpr int batch_size = 1;
    constexpr int max_seq_len = 32;
    constexpr int n_kv_heads = 3;
    constexpr int head_dim = 32;
    constexpr int kv_dim = n_kv_heads * head_dim;
    constexpr int history_tokens = 5;
    constexpr int verifier_rows = 3;
    constexpr int total_tokens = history_tokens + verifier_rows;

    ScopedCudaStream stream;

    auto history_k = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_tokens), static_cast<size_t>(kv_dim)});
    auto history_v = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(history_tokens), static_cast<size_t>(kv_dim)});
    auto verifier_k_position = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(verifier_rows), static_cast<size_t>(kv_dim)});
    auto verifier_v_position = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(verifier_rows), static_cast<size_t>(kv_dim)});
    auto verifier_k_head = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_kv_heads * verifier_rows), static_cast<size_t>(head_dim)});
    auto verifier_v_head = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_kv_heads * verifier_rows), static_cast<size_t>(head_dim)});

    for (size_t i = 0; i < static_cast<size_t>(history_tokens) * kv_dim; ++i)
    {
        history_k->mutable_data()[i] = 0.00390625f * static_cast<float>(static_cast<int>(i % 97) - 48);
        history_v->mutable_data()[i] = -0.0029296875f * static_cast<float>(static_cast<int>(i % 89) - 44);
    }

    for (int row = 0; row < verifier_rows; ++row)
    {
        for (int elem = 0; elem < kv_dim; ++elem)
        {
            const int head = elem / head_dim;
            const int lane = elem - head * head_dim;
            const size_t position_idx = static_cast<size_t>(row) * kv_dim + elem;
            const size_t head_idx = (static_cast<size_t>(head) * verifier_rows + row) * head_dim + lane;

            // Use non-trivial, exactly reproducible FP32 values.  The values are
            // intentionally not all FP16-exact so both serial and grouped paths
            // must perform the same round-to-nearest-even conversion.
            const float k_value =
                0.00137f * static_cast<float>((row + 1) * 101 + head * 17 + lane - 80);
            const float v_value =
                -0.00191f * static_cast<float>((row + 1) * 73 + head * 19 + lane - 64);

            verifier_k_position->mutable_data()[position_idx] = k_value;
            verifier_v_position->mutable_data()[position_idx] = v_value;
            verifier_k_head->mutable_data()[head_idx] = k_value;
            verifier_v_head->mutable_data()[head_idx] = v_value;
        }
    }

    ASSERT_TRUE(history_k->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(history_v->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(verifier_k_position->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(verifier_v_position->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(verifier_k_head->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(verifier_v_head->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    stream.synchronize();

    auto make_cache = [&]()
    {
        auto cache = createCUDARingKVCache(
            ActivationPrecision::FP16,
            n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
        EXPECT_NE(cache, nullptr);
        return cache;
    };

    auto append_history = [&](ICUDARingKVCache *cache)
    {
        ASSERT_NE(cache, nullptr);
        ASSERT_TRUE(cache->appendConvertedWithStream(
            0, 0,
            history_k->gpu_data_ptr(),
            history_v->gpu_data_ptr(),
            TensorType::FP32,
            history_tokens,
            stream.stream()));
        stream.synchronize();
        ASSERT_EQ(cache->get_cached_tokens(0, 0), history_tokens);
    };

    auto read_cache = [&](ICUDARingKVCache *cache,
                          std::vector<uint16_t> *out_k,
                          std::vector<uint16_t> *out_v)
    {
        ASSERT_NE(cache, nullptr);
        const void *d_k_out = nullptr;
        const void *d_v_out = nullptr;
        int kv_len = 0;
        ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_k_out, &d_v_out, &kv_len, stream.stream()));
        ASSERT_EQ(kv_len, total_tokens);
        ASSERT_NE(d_k_out, nullptr);
        ASSERT_NE(d_v_out, nullptr);

        out_k->assign(static_cast<size_t>(total_tokens) * kv_dim, uint16_t{0});
        out_v->assign(static_cast<size_t>(total_tokens) * kv_dim, uint16_t{0});
        ASSERT_EQ(cudaMemcpyAsync(out_k->data(), d_k_out,
                                  out_k->size() * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost, stream.stream()),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(out_v->data(), d_v_out,
                                  out_v->size() * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost, stream.stream()),
                  cudaSuccess);
        stream.synchronize();
    };

    auto run_case = [&](const char *case_name,
                        FP32Tensor *grouped_k,
                        FP32Tensor *grouped_v)
    {
        auto serial = make_cache();
        auto grouped = make_cache();
        ASSERT_NE(serial, nullptr);
        ASSERT_NE(grouped, nullptr);

        auto *serial_workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(serial.get());
        auto *grouped_workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(grouped.get());
        ASSERT_NE(serial_workspace_consumer, nullptr);
        ASSERT_NE(grouped_workspace_consumer, nullptr);
        auto serial_workspace = bindRequiredWorkspace(
            serial_workspace_consumer,
            verifier_rows,
            batch_size,
            head_dim);
        auto grouped_workspace = bindRequiredWorkspace(
            grouped_workspace_consumer,
            verifier_rows,
            batch_size,
            head_dim);
        ASSERT_NE(serial_workspace, nullptr);
        ASSERT_NE(grouped_workspace, nullptr);

        append_history(serial.get());
        append_history(grouped.get());

        const auto *serial_k_base = static_cast<const float *>(verifier_k_position->gpu_data_ptr());
        const auto *serial_v_base = static_cast<const float *>(verifier_v_position->gpu_data_ptr());
        ASSERT_NE(serial_k_base, nullptr);
        ASSERT_NE(serial_v_base, nullptr);
        for (int row = 0; row < verifier_rows; ++row)
        {
            ASSERT_TRUE(serial->appendConvertedWithStream(
                0, 0,
                serial_k_base + static_cast<size_t>(row) * kv_dim,
                serial_v_base + static_cast<size_t>(row) * kv_dim,
                TensorType::FP32,
                1,
                stream.stream()))
                << case_name << " serial row=" << row;
            stream.synchronize();
            ASSERT_EQ(serial->get_cached_tokens(0, 0), history_tokens + row + 1);
        }

        ASSERT_TRUE(grouped->appendVerifierRowsDecodeEquivalent(
            0, 0,
            static_cast<const ITensor *>(grouped_k),
            static_cast<const ITensor *>(grouped_v),
            verifier_rows,
            stream.opaque()))
            << case_name << " grouped verifier append failed";
        stream.synchronize();
        ASSERT_EQ(grouped->get_cached_tokens(0, 0), total_tokens);

        std::vector<uint16_t> serial_k;
        std::vector<uint16_t> serial_v;
        std::vector<uint16_t> grouped_k_bytes;
        std::vector<uint16_t> grouped_v_bytes;
        read_cache(serial.get(), &serial_k, &serial_v);
        read_cache(grouped.get(), &grouped_k_bytes, &grouped_v_bytes);

        ASSERT_EQ(grouped_k_bytes.size(), serial_k.size());
        ASSERT_EQ(grouped_v_bytes.size(), serial_v.size());
        EXPECT_EQ(grouped_k_bytes, serial_k)
            << case_name << " grouped K cache bytes diverged from serial decode";
        EXPECT_EQ(grouped_v_bytes, serial_v)
            << case_name << " grouped V cache bytes diverged from serial decode";

        /*
         * The workspace manager must outlive every bound consumer. Explicitly
         * sever the non-owning cache references before the local managers are
         * destroyed at the end of this case.
         */
        serial_workspace_consumer->unbindWorkspace();
        grouped_workspace_consumer->unbindWorkspace();
    };

    run_case("position-major", verifier_k_position.get(), verifier_v_position.get());
    run_case("head-major", verifier_k_head.get(), verifier_v_head.get());
}

// =============================================================================
// REGRESSION TEST: FP32 ITensor → FP16 cache via append (non-stream)
//
// Locks in the fail-fast behavior for the legacy non-stream GPU append path.
// GPU cache writes must use appendWithStream() so residency, format conversion,
// and graph-capture ordering all happen on the caller's explicit stream.
// =============================================================================

TEST(Test__CUDARingKVCache, Append_ITensor_FP32_to_FP16_RequiresExplicitStream)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 4;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Create FP32 tensor with known data on GPU
    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});

    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = 0.5f; // Easy to verify
        V_tensor->mutable_data()[i] = -0.25f;
    }

    DeviceId cuda_dev = DeviceId::cuda(0);
    ASSERT_TRUE(K_tensor->ensureOnDevice(cuda_dev));
    ASSERT_TRUE(V_tensor->ensureOnDevice(cuda_dev));

    // The non-stream ITensor append path used to reinterpret FP32 bytes as
    // FP16 and write corrupted cache rows. It now fails before touching cache
    // state, forcing callers onto appendWithStream().
    bool append_ok = cache->append(0, 0,
                                   static_cast<const ITensor *>(K_tensor.get()),
                                   static_cast<const ITensor *>(V_tensor.get()),
                                   num_tokens);
    ASSERT_FALSE(append_ok) << "GPU append(ITensor) must require appendWithStream()";
}

TEST(Test__CUDARingKVCache, AppendWithStream_RejectsNullAndAcceptsExplicitStream)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 4;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    for (size_t i = 0; i < num_tokens * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = 0.125f * static_cast<float>((i % 7) - 3);
        V_tensor->mutable_data()[i] = -0.0625f * static_cast<float>((i % 5) - 2);
    }

    EXPECT_THROW(
        cache->appendWithStream(
            0,
            0,
            static_cast<const ITensor *>(K_tensor.get()),
            static_cast<const ITensor *>(V_tensor.get()),
            num_tokens,
            nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        cache->resetRequestState(
            IKVCache::StateResetContext::testReinitialization(nullptr)),
        std::invalid_argument);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    ScopedCudaStream stream;
    ASSERT_TRUE(cache->appendWithStream(0, 0,
                                        static_cast<const ITensor *>(K_tensor.get()),
                                        static_cast<const ITensor *>(V_tensor.get()),
                                        num_tokens, stream.opaque()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);
}

/**
 * @brief Proves captured FP32-to-FP16 append uses the fused no-scratch kernel.
 *
 * The production append stage publishes head/count/real-row metadata before
 * capture. Once that state is resident, conversion happens in-register inside
 * the ring append kernel; requiring a temporary FP16 tensor would add two full
 * device writes and make small grouped verifier rows uneconomical.
 */
TEST(Test__CUDARingKVCache, GraphCapturedFP32ToFP16AppendUsesFusedConversionWithoutWorkspace)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 16;
    const int n_kv_heads = 1;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 4;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);
    ASSERT_FALSE(workspace_consumer->hasWorkspace());

    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    for (size_t i = 0; i < static_cast<size_t>(num_tokens) * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = 0.125f * static_cast<float>(static_cast<int>(i % 7) - 3);
        V_tensor->mutable_data()[i] = -0.0625f * static_cast<float>(static_cast<int>(i % 5) - 2);
    }

    ScopedCudaStream stream;
    ASSERT_TRUE(K_tensor->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(V_tensor->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    stream.synchronize();

    ASSERT_TRUE(cache->bindGraphAppendCountSource(
        0, 0, nullptr, num_tokens, stream.opaque()));
    stream.synchronize();
    {
        GraphCaptureGuard guard;
        EXPECT_TRUE(cache->appendWithStream(
            0, 0,
            static_cast<const ITensor *>(K_tensor.get()),
            static_cast<const ITensor *>(V_tensor.get()),
            num_tokens,
            stream.opaque()))
            << "The fused converted append must be graph-capturable without conversion scratch";
    }
    stream.synchronize();
    EXPECT_FALSE(workspace_consumer->hasWorkspace())
        << "Fused converted append must not allocate or bind an implicit workspace";
}

/**
 * @brief Replays fused FP32-to-FP16 append after a cache reset.
 *
 * The captured graph reads and advances canonical device sequence metadata.
 * Rebinding the exact captured width after clear must not upload or adopt a
 * host sequence-state copy.
 */
TEST(Test__CUDARingKVCache, GraphCapturedFP32ToFP16FusedAppendReplaysAfterClear)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 16;
    const int n_kv_heads = 1;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 6;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);
    auto workspace = bindRequiredWorkspace(workspace_consumer, num_tokens, batch_size, 0);
    ASSERT_NE(workspace, nullptr);

    auto K_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    auto V_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)});
    for (size_t i = 0; i < static_cast<size_t>(num_tokens) * kv_dim; ++i)
    {
        K_tensor->mutable_data()[i] = 0.03125f * static_cast<float>(static_cast<int>(i % 17) - 8);
        V_tensor->mutable_data()[i] = -0.015625f * static_cast<float>(static_cast<int>(i % 13) - 6);
    }

    ScopedCudaStream stream;
    ASSERT_TRUE(K_tensor->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(V_tensor->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    stream.synchronize();

    ASSERT_TRUE(cache->bindGraphAppendCountSource(
        0, 0, nullptr, num_tokens, stream.opaque()));
    stream.synchronize();

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal), cudaSuccess);
    bool capture_append_ok = false;
    {
        GraphCaptureGuard guard;
        capture_append_ok = cache->appendWithStream(0, 0,
                                                    static_cast<const ITensor *>(K_tensor.get()),
                                                    static_cast<const ITensor *>(V_tensor.get()),
                                                    num_tokens, stream.opaque());
    }
    ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
    ASSERT_TRUE(capture_append_ok);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);

    ASSERT_TRUE(cache->resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
    ASSERT_TRUE(cache->bindGraphAppendCountSource(
        0, 0, nullptr, num_tokens, stream.opaque()));
    stream.synchronize();

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    ITensor *K_out = nullptr;
    ITensor *V_out = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_snapshot_view(
        0, 0, num_tokens, &K_out, &V_out, &kv_len));
    ASSERT_EQ(kv_len, num_tokens);
    ASSERT_NE(K_out, nullptr);
    ASSERT_NE(V_out, nullptr);

    std::vector<uint16_t> h_K_out(static_cast<size_t>(num_tokens) * kv_dim);
    std::vector<uint16_t> h_V_out(static_cast<size_t>(num_tokens) * kv_dim);
    ASSERT_EQ(cudaMemcpyAsync(h_K_out.data(), K_out->gpu_data_ptr(),
                              h_K_out.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost, stream.stream()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(h_V_out.data(), V_out->gpu_data_ptr(),
                              h_V_out.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost, stream.stream()),
              cudaSuccess);
    stream.synchronize();

    for (size_t i = 0; i < h_K_out.size(); ++i)
    {
        EXPECT_NEAR(fp16_to_fp32(h_K_out[i]), K_tensor->data()[i], 0.001f) << "i=" << i;
        EXPECT_NEAR(fp16_to_fp32(h_V_out[i]), V_tensor->data()[i], 0.001f) << "i=" << i;
    }

    EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

/**
 * @brief Proves a padded captured FP32-to-Q8_1 append preserves every live row.
 *
 * Production prefill records a fixed 256-row graph even when only a short
 * prompt prefix is live.  The Q8_1 adapter first quantizes the fixed input
 * geometry into cache-owned scratch, then the append and resident gather must
 * both consume the device-owned real-row count.  This regression validates
 * that complete chain with non-zero padding so an incorrect full-bucket copy,
 * row stride, or scratch dependency cannot hide behind zero-filled inputs.
 */
TEST(Test__CUDARingKVCache, CapturedPaddedFP32ToQ8AppendAndGatherPreservesLiveRows)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr int captured_rows = 256;
    constexpr int live_rows = 9;
    constexpr int max_seq_len = 512;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int kv_dim = n_kv_heads * head_dim;
    constexpr int blocks_per_row =
        kv_dim / static_cast<int>(Q8_1Block::BLOCK_SIZE);

    auto cache = createCUDARingKVCache(
        ActivationPrecision::Q8_1,
        /*n_layers=*/1,
        /*batch_size=*/1,
        max_seq_len,
        n_kv_heads,
        head_dim);
    ASSERT_NE(cache, nullptr);
    auto *workspace_consumer =
        dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);
    auto workspace = bindRequiredWorkspace(
        workspace_consumer, captured_rows, /*batch_size=*/1, head_dim);
    ASSERT_NE(workspace, nullptr);

    auto K = std::make_unique<FP32Tensor>(std::vector<size_t>{
        captured_rows, static_cast<size_t>(kv_dim)});
    auto V = std::make_unique<FP32Tensor>(std::vector<size_t>{
        captured_rows, static_cast<size_t>(kv_dim)});
    for (int row = 0; row < captured_rows; ++row)
    {
        for (int col = 0; col < kv_dim; ++col)
        {
            const size_t index = static_cast<size_t>(row) * kv_dim + col;
            if (row < live_rows)
            {
                K->mutable_data()[index] =
                    0.015625f * static_cast<float>(((row * 37 + col * 13) % 97) - 48);
                V->mutable_data()[index] =
                    0.0078125f * static_cast<float>(((row * 19 + col * 29) % 89) - 44);
            }
            else
            {
                K->mutable_data()[index] = 37.0f + static_cast<float>(row);
                V->mutable_data()[index] = -41.0f - static_cast<float>(row);
            }
        }
    }

    ScopedCudaStream stream;
    ASSERT_TRUE(K->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
    ASSERT_TRUE(V->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));

    int32_t *device_live_rows = nullptr;
    ASSERT_EQ(cudaMalloc(&device_live_rows, sizeof(int32_t)), cudaSuccess);
    const int32_t host_live_rows = live_rows;
    ASSERT_EQ(
        cudaMemcpyAsync(
            device_live_rows,
            &host_live_rows,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream.stream()),
        cudaSuccess);
    ASSERT_TRUE(cache->bindGraphAppendCountSource(
        /*layer=*/0,
        /*seq_idx=*/0,
        device_live_rows,
        captured_rows,
        stream.opaque()));

    ITensor *gathered_k = nullptr;
    ITensor *gathered_v = nullptr;
    ASSERT_TRUE(cache->get_kv_batched_device_view(
        /*layer=*/0,
        /*first_seq_idx=*/0,
        /*request_count=*/1,
        &gathered_k,
        &gathered_v,
        stream.opaque()));
    stream.synchronize();

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(
        cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal),
        cudaSuccess);
    bool capture_ok = false;
    {
        GraphCaptureGuard guard;
        capture_ok = cache->appendWithStream(
            /*layer=*/0,
            /*seq_idx=*/0,
            K.get(),
            V.get(),
            captured_rows,
            stream.opaque());
        capture_ok = capture_ok && cache->get_kv_batched_device_view(
                                       /*layer=*/0,
                                       /*first_seq_idx=*/0,
                                       /*request_count=*/1,
                                       &gathered_k,
                                       &gathered_v,
                                       stream.opaque());
    }
    ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
    ASSERT_TRUE(capture_ok);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
    stream.synchronize();

    ASSERT_NE(gathered_k, nullptr);
    ASSERT_NE(gathered_v, nullptr);
    const size_t live_block_count =
        static_cast<size_t>(live_rows) * blocks_per_row;
    std::vector<Q8_1Block> actual_k(live_block_count);
    std::vector<Q8_1Block> actual_v(live_block_count);
    ASSERT_EQ(
        cudaMemcpyAsync(
            actual_k.data(), gathered_k->gpu_data_ptr(),
            actual_k.size() * sizeof(Q8_1Block),
            cudaMemcpyDeviceToHost, stream.stream()),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            actual_v.data(), gathered_v->gpu_data_ptr(),
            actual_v.size() * sizeof(Q8_1Block),
            cudaMemcpyDeviceToHost, stream.stream()),
        cudaSuccess);
    stream.synchronize();

    auto compare_dequantized = [&](const std::vector<Q8_1Block> &blocks,
                                   const FP32Tensor &reference,
                                   const char *label)
    {
        double dot = 0.0;
        double actual_norm = 0.0;
        double reference_norm = 0.0;
        float max_error = 0.0f;
        for (int row = 0; row < live_rows; ++row)
        {
            for (int col = 0; col < kv_dim; ++col)
            {
                const Q8_1Block &block = blocks[
                    static_cast<size_t>(row) * blocks_per_row +
                    col / static_cast<int>(Q8_1Block::BLOCK_SIZE)];
                const float scale = fp16_to_fp32(block.d);
                const float actual =
                    scale * static_cast<float>(block.qs[
                                col % static_cast<int>(Q8_1Block::BLOCK_SIZE)]);
                const float expected =
                    reference.data()[static_cast<size_t>(row) * kv_dim + col];
                dot += static_cast<double>(actual) * expected;
                actual_norm += static_cast<double>(actual) * actual;
                reference_norm += static_cast<double>(expected) * expected;
                max_error = std::max(max_error, std::abs(actual - expected));
            }
        }
        const double cosine = dot / std::sqrt(actual_norm * reference_norm);
        EXPECT_GT(cosine, 0.9999) << label << " cosine=" << cosine;
        EXPECT_LT(max_error, 0.01f) << label << " max_error=" << max_error;
    };
    compare_dequantized(actual_k, *K, "K");
    compare_dequantized(actual_v, *V, "V");

    EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cudaFree(device_live_rows), cudaSuccess);
    workspace_consumer->unbindWorkspace();
}

// =============================================================================
// Test: Multi-Precision (BF16)
// =============================================================================

TEST(Test__CUDARingKVCache, MultiPrecision_BF16)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::BF16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    EXPECT_EQ(cache->precision(), ActivationPrecision::BF16);
    ScopedCudaStream stream;

    // Generate FP32 data and convert to BF16
    auto h_K_fp32 = generateRandomFP32(10 * kv_dim);
    auto h_V_fp32 = generateRandomFP32(10 * kv_dim);

    std::vector<__nv_bfloat16> h_K_bf16(10 * kv_dim);
    std::vector<__nv_bfloat16> h_V_bf16(10 * kv_dim);
    for (size_t i = 0; i < h_K_fp32.size(); ++i)
    {
        h_K_bf16[i] = __float2bfloat16(h_K_fp32[i]);
        h_V_bf16[i] = __float2bfloat16(h_V_fp32[i]);
    }

    __nv_bfloat16 *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(__nv_bfloat16));
    cudaMemcpy(d_K, h_K_bf16.data(), 10 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_bf16.data(), 10 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cache->append(
        0, 0, d_K, d_V, 10, stream.stream()));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, 10);

    // Verify content
    std::vector<__nv_bfloat16> h_K_out(10 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 10 * kv_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10 * kv_dim; ++i)
    {
        float expected = __bfloat162float(h_K_bf16[i]);
        float actual = __bfloat162float(h_K_out[i]);
        EXPECT_FLOAT_EQ(actual, expected) << "BF16 mismatch at " << i;
    }

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[MultiPrecision_BF16] PASSED");
}

// =============================================================================
// Test: IWorkspaceConsumer Interface - getWorkspaceRequirements
// =============================================================================

TEST(Test__CUDARingKVCache, WorkspaceRequirements)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    // Cast to IWorkspaceConsumer to test the interface
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);

    // Get workspace requirements with default batch size
    auto reqs = workspace_consumer->getWorkspaceRequirements(0);
    EXPECT_EQ(reqs.buffers.size(), 2u);

    // Immutable cache pointer topology and mutable head/count rows are
    // cache-owned device allocations. The graph workspace therefore contains
    // only output materialization scratch; no replay uploads host metadata.
    bool found_conv_scratch_k = false, found_conv_scratch_v = false;
    const size_t expected_default_scratch =
        static_cast<size_t>(max_seq_len) * batch_size *
        n_kv_heads * head_dim * sizeof(float);

    for (const auto &buf : reqs.buffers)
    {
        if (buf.name == KVCacheWorkspaceBuffers::CONV_SCRATCH_K)
        {
            found_conv_scratch_k = true;
            EXPECT_GE(buf.size_bytes, expected_default_scratch);
            EXPECT_TRUE(buf.required);
        }
        else if (buf.name == KVCacheWorkspaceBuffers::CONV_SCRATCH_V)
        {
            found_conv_scratch_v = true;
            EXPECT_GE(buf.size_bytes, expected_default_scratch);
            EXPECT_TRUE(buf.required);
        }
        else
        {
            FAIL() << "Unexpected host-staging KV workspace buffer: " << buf.name;
        }
    }

    EXPECT_TRUE(found_conv_scratch_k) << "Missing CONV_SCRATCH_K buffer";
    EXPECT_TRUE(found_conv_scratch_v) << "Missing CONV_SCRATCH_V buffer";

    // Test with explicit batch size
    auto reqs2 = workspace_consumer->getWorkspaceRequirements(8);
    const size_t expected_explicit_scratch =
        static_cast<size_t>(max_seq_len) * 8u *
        n_kv_heads * head_dim * sizeof(float);
    for (const auto &buf : reqs2.buffers)
    {
        EXPECT_TRUE(buf.name == KVCacheWorkspaceBuffers::CONV_SCRATCH_K ||
                    buf.name == KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
        EXPECT_GE(buf.size_bytes, expected_explicit_scratch);
    }

    /*
     * A graph bucket bounds only the newly submitted rows. On the third 16-row
     * chunk, attention may already gather more than 16 cached rows. Prove the
     * production two-dimensional sizing API keeps the complete cache horizon
     * and configured batch capacity instead of shrinking to that first bucket.
     */
    const int graph_bucket_tokens = 16;
    const int first_request_batch = 1;
    const auto bucket_reqs = workspace_consumer->getWorkspaceRequirements(
        graph_bucket_tokens,
        first_request_batch,
        0);
    const size_t expected_resident_horizon_scratch =
        static_cast<size_t>(max_seq_len) * batch_size *
        n_kv_heads * head_dim * sizeof(float);
    for (const auto &buf : bucket_reqs.buffers)
    {
        EXPECT_TRUE(buf.name == KVCacheWorkspaceBuffers::CONV_SCRATCH_K ||
                    buf.name == KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
        EXPECT_GE(buf.size_bytes, expected_resident_horizon_scratch)
            << "Resident KV gather scratch must cover history beyond the active graph bucket";
    }

    LOG_INFO("[WorkspaceRequirements] PASSED");
}

// =============================================================================
// Test: IWorkspaceConsumer Interface - bindWorkspace/hasWorkspace
// =============================================================================

TEST(Test__CUDARingKVCache, WorkspaceBinding)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 4;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);

    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);

    // Initially no workspace bound
    EXPECT_FALSE(workspace_consumer->hasWorkspace());
    EXPECT_EQ(workspace_consumer->getWorkspace(), nullptr);

    // Bind nullptr (unbind)
    workspace_consumer->bindWorkspace(nullptr);
    EXPECT_FALSE(workspace_consumer->hasWorkspace());

    LOG_INFO("[WorkspaceBinding] PASSED");
}

// =============================================================================
// Test: Diagnostic batched gather uses caller-owned output without workspace
// =============================================================================

TEST(Test__CUDARingKVCache, BatchedGatherToCallerOwnedBuffersDoesNotRequireWorkspace)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    const int n_layers = 1;
    const int batch_size = 2;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int kv_dim = n_kv_heads * head_dim;

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cache, nullptr);
    ScopedCudaStream stream;

    // Verify no workspace bound
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
    ASSERT_NE(workspace_consumer, nullptr);
    EXPECT_FALSE(workspace_consumer->hasWorkspace());

    // Fill sequences
    float *d_K, *d_V;
    cudaMalloc(&d_K, 10 * kv_dim * sizeof(float));
    cudaMalloc(&d_V, 10 * kv_dim * sizeof(float));

    auto h_K = generateRandomFP32(10 * kv_dim, 42);
    auto h_V = generateRandomFP32(10 * kv_dim, 43);

    for (int seq = 0; seq < batch_size; ++seq)
    {
        int seq_len = 5 + seq * 3; // 5 and 8 tokens
        cudaMemcpy(d_K, h_K.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cache->append(0, seq, d_K, d_V, seq_len, stream.stream());
    }
    stream.synchronize();

    /*
     * This diagnostic API receives complete output buffers from its caller and
     * therefore needs no cache-owned scratch. Production graph attention uses
     * get_kv_batched_device_view(), which keeps lengths device-resident and
     * requires graph-stable workspace.
     */
    int max_kv_len = 10;
    float *d_K_gathered, *d_V_gathered;
    cudaMalloc(&d_K_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V_gathered, batch_size * max_kv_len * kv_dim * sizeof(float));

    std::vector<int> kv_lens(batch_size);
    int actual_max = cache->gather_kv_batched(0, batch_size,
                                              d_K_gathered, d_V_gathered,
                                              kv_lens.data(), max_kv_len, stream.stream());

    EXPECT_EQ(actual_max, 8);
    ASSERT_EQ(kv_lens.size(), static_cast<size_t>(batch_size));
    EXPECT_EQ(kv_lens[0], 5);
    EXPECT_EQ(kv_lens[1], 8);

    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_gathered);
    cudaFree(d_V_gathered);

    LOG_INFO("[BatchedGatherToCallerOwnedBuffersDoesNotRequireWorkspace] PASSED");
}

/**
 * @brief Proves captured request-local append counts govern metadata and payload writes.
 *
 * Request-batched prefill captures one fixed eight-row append per request, but
 * the second request owns only five real rows.  The same captured graph is then
 * replayed with one real continuation row per request while the physical ring
 * has only ten rows.  A kernel that writes all eight padded rows will wrap and
 * overwrite live history even if another kernel happens to publish the right
 * count.  Comparing the complete live prefix before and after continuation
 * therefore covers both halves of the contract: exact device head/count
 * advancement and suppression of every padded payload store.
 *
 * All standard CUDA KV storage formats run through KVCacheAppendStage so the
 * test exercises the production request slicing and device-count publication
 * path, not a cache-only test shim.
 */
TEST(Test__CUDARingKVCache, CapturedUnequalRequestLengthsPreserveContinuationAllFormats)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr int batch_size = 2;
    constexpr int captured_rows = 8;
    constexpr int max_seq_len = 10;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int kv_dim = n_kv_heads * head_dim;
    constexpr std::array<int, batch_size> initial_counts{captured_rows, 5};
    constexpr std::array<int, batch_size> final_counts{captured_rows + 1, 6};

    struct FormatCase
    {
        ActivationPrecision precision;
        const char *name;
    };
    constexpr std::array<FormatCase, 4> formats{{
        {ActivationPrecision::FP32, "FP32"},
        {ActivationPrecision::FP16, "FP16"},
        {ActivationPrecision::BF16, "BF16"},
        {ActivationPrecision::Q8_1, "Q8_1"},
    }};

    const size_t source_elements =
        static_cast<size_t>(batch_size) * captured_rows * kv_dim;
    auto k_fp32 = generateRandomFP32(source_elements, 701);
    auto v_fp32 = generateRandomFP32(source_elements, 907);
    for (int request = 0; request < batch_size; ++request)
    {
        const size_t request_begin =
            static_cast<size_t>(request) * captured_rows * kv_dim;
        for (size_t index = 0;
             index < static_cast<size_t>(captured_rows) * kv_dim;
             ++index)
        {
            k_fp32[request_begin + index] += 0.5f * request;
            v_fp32[request_begin + index] -= 0.375f * request;
        }
    }

    auto makeNativeTensor = [](
                                const std::vector<float> &values,
                                ActivationPrecision precision)
        -> std::shared_ptr<TensorBase>
    {
        const std::vector<size_t> shape{
            static_cast<size_t>(batch_size * captured_rows),
            static_cast<size_t>(kv_dim)};
        switch (precision)
        {
        case ActivationPrecision::FP32:
        {
            auto tensor = std::make_shared<FP32Tensor>(shape);
            std::copy(values.begin(), values.end(), tensor->mutable_data());
            return tensor;
        }
        case ActivationPrecision::FP16:
        {
            std::vector<uint16_t> encoded(values.size());
            for (size_t index = 0; index < values.size(); ++index)
                encoded[index] = fp32_to_fp16(values[index]);
            return std::make_shared<FP16Tensor>(shape, encoded);
        }
        case ActivationPrecision::BF16:
        {
            auto tensor = std::make_shared<BF16Tensor>(shape);
            tensor->from_fp32(values.data(), values.size());
            return tensor;
        }
        case ActivationPrecision::Q8_1:
            return Q8_1Tensor::quantize_from_fp32(values.data(), shape);
        default:
            return nullptr;
        }
    };

    auto rowBytes = [](ActivationPrecision precision) -> size_t
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return static_cast<size_t>(kv_dim) * sizeof(float);
        case ActivationPrecision::FP16:
        case ActivationPrecision::BF16:
            return static_cast<size_t>(kv_dim) * sizeof(uint16_t);
        case ActivationPrecision::Q8_1:
            return static_cast<size_t>(kv_dim / Q8_1Block::BLOCK_SIZE) *
                   sizeof(Q8_1Block);
        default:
            return 0;
        }
    };

    for (const FormatCase &format : formats)
    {
        SCOPED_TRACE(format.name);
        ScopedCudaStream stream;
        auto cache = createCUDARingKVCache(
            format.precision,
            /*n_layers=*/1,
            batch_size,
            max_seq_len,
            n_kv_heads,
            head_dim);
        ASSERT_NE(cache, nullptr);

        auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(cache.get());
        ASSERT_NE(workspace_consumer, nullptr);
        auto workspace = bindRequiredWorkspace(
            workspace_consumer, captured_rows, batch_size, head_dim);
        ASSERT_NE(workspace, nullptr);

        auto k_tensor = makeNativeTensor(k_fp32, format.precision);
        auto v_tensor = makeNativeTensor(v_fp32, format.precision);
        ASSERT_NE(k_tensor, nullptr);
        ASSERT_NE(v_tensor, nullptr);
        // Mirror the executor's external-input admission: residency alone is
        // not an authenticated producer-event join for a captured consumer.
        TransferEngine::prepareDeviceInput(k_tensor.get(), DeviceId::cuda(0), stream.opaque());
        TransferEngine::prepareDeviceInput(v_tensor.get(), DeviceId::cuda(0), stream.opaque());
        TransferEngine::requireDeviceInput(k_tensor.get(), DeviceId::cuda(0), stream.opaque());
        TransferEngine::requireDeviceInput(v_tensor.get(), DeviceId::cuda(0), stream.opaque());

        int32_t *device_lengths = nullptr;
        ASSERT_EQ(
            cudaMalloc(&device_lengths, batch_size * sizeof(int32_t)),
            cudaSuccess);
        const std::array<int32_t, batch_size> first_device_lengths{
            initial_counts[0], initial_counts[1]};
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_lengths,
                first_device_lengths.data(),
                batch_size * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream.stream()),
            cudaSuccess);

        KVCacheAppendStage append_stage({
            .device_id = DeviceId::cuda(0),
            .K = k_tensor.get(),
            .V = v_tensor.get(),
            .kv_cache = cache.get(),
            .layer_idx = 0,
            .seq_idx = 0,
            .num_tokens = batch_size * captured_rows,
            .batch_size = batch_size,
            .seq_len = captured_rows,
            .request_sequence_lengths_device = device_lengths,
            .head_dim = head_dim,
        });
        append_stage.setGPUStream(stream.opaque());
        append_stage.updateDynamicParams(/*pos_offset=*/0, captured_rows);
        stream.synchronize();

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal),
            cudaSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = append_stage.execute(nullptr);
        }
        ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();

        std::array<std::vector<uint8_t>, batch_size> initial_k;
        std::array<std::vector<uint8_t>, batch_size> initial_v;
        const size_t row_bytes = rowBytes(format.precision);
        ASSERT_GT(row_bytes, 0u);
        for (int request = 0; request < batch_size; ++request)
        {
            SCOPED_TRACE("initial request=" + std::to_string(request));
            int device_count = -1;
            int device_head = -1;
            ASSERT_NE(cache->deviceCachedTokenCountPtr(0, request), nullptr);
            ASSERT_NE(cache->deviceRingHeadPtr(0, request), nullptr);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    &device_count,
                    cache->deviceCachedTokenCountPtr(0, request),
                    sizeof(int), cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    &device_head,
                    cache->deviceRingHeadPtr(0, request),
                    sizeof(int), cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);

            ITensor *cache_k = nullptr;
            ITensor *cache_v = nullptr;
            int cache_rows = 0;
            ASSERT_TRUE(cache->get_kv_snapshot_view(
                0, request, initial_counts[request],
                &cache_k, &cache_v, &cache_rows));
            ASSERT_EQ(cache_rows, initial_counts[request]);
            ASSERT_NE(cache_k, nullptr);
            ASSERT_NE(cache_v, nullptr);
            const size_t live_bytes =
                static_cast<size_t>(cache_rows) * row_bytes;
            initial_k[request].resize(live_bytes);
            initial_v[request].resize(live_bytes);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    initial_k[request].data(), cache_k->gpu_data_ptr(),
                    live_bytes, cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    initial_v[request].data(), cache_v->gpu_data_ptr(),
                    live_bytes, cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            stream.synchronize();
            EXPECT_EQ(device_count, initial_counts[request]);
            EXPECT_EQ(device_head, initial_counts[request]);
        }

        /*
         * Keep the captured eight-row topology but publish one real row.  The
         * persistent device length row is restamped before launch; no graph
         * node or cache metadata is rebuilt on the host.
         */
        // Device-owned unequal sequence lengths, not the captured padded M,
        // determine each diagnostic prefix boundary. Observe before replay.
        PrefixProbeCapturePolicy probe_policy;
        probe_policy.hash_full_kv_payloads = true;
        probe_policy.capture_requested_kv_segment_payloads = true;
        PrefixRuntimeStateSnapshot seed;
        seed.initialized = true;
        seed.mtp_kv_caches = {inspectKVCacheForPrefixProbe(
            *cache, "mtp:0", DeviceId::cuda(0), batch_size, stream.stream(), probe_policy)};
        const auto continuation_policy = probe_policy.forKVContinuationOf(seed);
        constexpr std::array<int32_t, batch_size> continuation_lengths{1, 1};
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_lengths,
                continuation_lengths.data(),
                batch_size * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream.stream()),
            cudaSuccess);
        append_stage.updateDynamicParams(
            /*pos_offset=*/captured_rows,
            /*seq_len=*/captured_rows);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();
        for (int request = 0; request < batch_size; ++request)
        {
            SCOPED_TRACE("continuation request=" + std::to_string(request));
            int device_count = -1;
            int device_head = -1;
            ASSERT_EQ(
                cudaMemcpyAsync(
                    &device_count,
                    cache->deviceCachedTokenCountPtr(0, request),
                    sizeof(int), cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    &device_head,
                    cache->deviceRingHeadPtr(0, request),
                    sizeof(int), cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);

            ITensor *cache_k = nullptr;
            ITensor *cache_v = nullptr;
            int cache_rows = 0;
            ASSERT_TRUE(cache->get_kv(
                0, request, &cache_k, &cache_v, &cache_rows));
            ASSERT_EQ(cache_rows, final_counts[request]);
            const size_t final_bytes =
                static_cast<size_t>(cache_rows) * row_bytes;
            std::vector<uint8_t> final_k(final_bytes);
            std::vector<uint8_t> final_v(final_bytes);
            std::vector<uint8_t> source_k(row_bytes);
            std::vector<uint8_t> source_v(row_bytes);
            const size_t source_offset =
                static_cast<size_t>(request) * captured_rows * row_bytes;
            ASSERT_EQ(
                cudaMemcpyAsync(
                    final_k.data(), cache_k->gpu_data_ptr(), final_bytes,
                    cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    final_v.data(), cache_v->gpu_data_ptr(), final_bytes,
                    cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    source_k.data(),
                    static_cast<const uint8_t *>(k_tensor->gpu_data_ptr()) + source_offset,
                    row_bytes, cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    source_v.data(),
                    static_cast<const uint8_t *>(v_tensor->gpu_data_ptr()) + source_offset,
                    row_bytes, cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            stream.synchronize();

            EXPECT_EQ(device_count, final_counts[request]);
            EXPECT_EQ(device_head, final_counts[request]);
            EXPECT_EQ(
                std::memcmp(
                    final_k.data(), initial_k[request].data(),
                    initial_k[request].size()),
                0)
                << "captured padded K rows overwrote live " << format.name
                << " history for request " << request;
            EXPECT_EQ(
                std::memcmp(
                    final_v.data(), initial_v[request].data(),
                    initial_v[request].size()),
                0)
                << "captured padded V rows overwrote live " << format.name
                << " history for request " << request;
            EXPECT_EQ(
                std::memcmp(
                    final_k.data() + initial_k[request].size(),
                    source_k.data(), row_bytes),
                0)
                << "real continuation K row was not published for " << format.name
                << " request " << request;
            EXPECT_EQ(
                std::memcmp(
                    final_v.data() + initial_v[request].size(),
                    source_v.data(), row_bytes),
                0)
                << "real continuation V row was not published for " << format.name
                << " request " << request;
        }

        const auto continued = inspectKVCacheForPrefixProbe(
            *cache, "mtp:0", DeviceId::cuda(0), batch_size, stream.stream(), continuation_policy);
        ASSERT_EQ(continued.layers.size(), batch_size);
        for (int request = 0; request < batch_size; ++request)
        {
            const auto &layer = continued.layers[request];
            const auto &seed_layer = seed.mtp_kv_caches.front().layers[request];
            EXPECT_EQ(layer.leading_segment_tokens, initial_counts[request]);
            EXPECT_EQ(layer.leading_k_payload_hash, seed_layer.k_payload_hash);
            EXPECT_EQ(layer.leading_v_payload_hash, seed_layer.v_payload_hash);
            ASSERT_EQ(layer.segments.size(), 1u);
            const auto &suffix = layer.segments.front();
            EXPECT_EQ(suffix.token_start, initial_counts[request]);
            EXPECT_EQ(suffix.token_count, 1);
            EXPECT_EQ(suffix.k_payload.size(), row_bytes);
            EXPECT_EQ(suffix.v_payload.size(), row_bytes);
        }
        EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        EXPECT_EQ(cudaFree(device_lengths), cudaSuccess);
    }
}
