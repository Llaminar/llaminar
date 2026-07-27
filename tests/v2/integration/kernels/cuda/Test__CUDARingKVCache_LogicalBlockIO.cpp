#include <gtest/gtest.h>

#include "kernels/cuda/kvcache/CUDARingKVCache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    bool hasCUDADevice()
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count <= 0)
        {
            return false;
        }
        return cudaSetDevice(0) == cudaSuccess;
    }

    std::vector<float> taggedRows(int rows, int cols, float base)
    {
        std::vector<float> values(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                values[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)] =
                    base + static_cast<float>(row * 10 + col);
            }
        }
        return values;
    }

    void expectRows(const std::vector<float> &actual,
                    int rows,
                    int cols,
                    const std::vector<float> &first_values)
    {
        ASSERT_EQ(static_cast<size_t>(rows) * static_cast<size_t>(cols), actual.size());
        ASSERT_EQ(static_cast<size_t>(rows), first_values.size());
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                EXPECT_FLOAT_EQ(actual[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)],
                                first_values[static_cast<size_t>(row)] + static_cast<float>(col));
            }
        }
    }
} // namespace

TEST(Test__CUDARingKVCache_LogicalBlockIO, FP32WrappedExportImportAndTruncateWithStream)
{
    if (!hasCUDADevice())
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int KV_DIM = 2;
    CUDARingKVCacheFP32 source(/*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/4,
                               /*n_kv_heads=*/1, /*head_dim=*/KV_DIM, /*device_id=*/0);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    auto k0 = taggedRows(4, KV_DIM, 100.0f);
    auto v0 = taggedRows(4, KV_DIM, 200.0f);
    auto k1 = taggedRows(2, KV_DIM, 140.0f);
    auto v1 = taggedRows(2, KV_DIM, 240.0f);

    float *d_k0 = nullptr;
    float *d_v0 = nullptr;
    float *d_k1 = nullptr;
    float *d_v1 = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_k0, k0.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_v0, v0.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_k1, k1.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_v1, v1.size() * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(d_k0, k0.data(), k0.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(d_v0, v0.data(), v0.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(d_k1, k1.data(), k1.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(d_v1, v1.data(), v1.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_TRUE(source.append(0, 0, d_k0, d_v0, 4, stream));
    ASSERT_TRUE(source.append(0, 0, d_k1, d_v1, 2, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    auto state = source.sequenceState(0, 0);
    EXPECT_EQ(state.cached_tokens, 4);
    EXPECT_EQ(state.implementation_head, 2);
    EXPECT_TRUE(state.wrapped);

    const auto layout = source.logicalBlockLayout(0, 4);
    ASSERT_EQ(layout.layout, TensorLayout::KV_POS_HEAD_DIM);
    ASSERT_TRUE(layout.device_resident);
    ASSERT_EQ(layout.k_bytes, 4u * KV_DIM * sizeof(float));

    std::vector<float> exported_k(4 * KV_DIM, 0.0f);
    std::vector<float> exported_v(4 * KV_DIM, 0.0f);
    IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 4, stream};
    ASSERT_TRUE(source.exportLogicalBlock(desc, exported_k.data(), exported_v.data()));

    expectRows(exported_k, 4, KV_DIM, {120.0f, 130.0f, 140.0f, 150.0f});
    expectRows(exported_v, 4, KV_DIM, {220.0f, 230.0f, 240.0f, 250.0f});

    CUDARingKVCacheFP32 target(/*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/4,
                               /*n_kv_heads=*/1, /*head_dim=*/KV_DIM, /*device_id=*/0);
    ASSERT_TRUE(target.importLogicalBlock(desc, exported_k.data(), exported_v.data()));
    EXPECT_EQ(target.sequenceState(0, 0).cached_tokens, 4);
    EXPECT_EQ(target.sequenceState(0, 0).implementation_head, 0);

    std::vector<float> roundtrip_k(4 * KV_DIM, 0.0f);
    std::vector<float> roundtrip_v(4 * KV_DIM, 0.0f);
    ASSERT_TRUE(target.exportLogicalBlock(desc, roundtrip_k.data(), roundtrip_v.data()));
    EXPECT_EQ(roundtrip_k, exported_k);
    EXPECT_EQ(roundtrip_v, exported_v);

    ASSERT_TRUE(target.truncateSequence(0, 2, stream));
    EXPECT_EQ(target.sequenceState(0, 0).cached_tokens, 2);
    EXPECT_EQ(target.sequenceState(0, 0).implementation_head, 2);
    ASSERT_TRUE(target.truncateSequence(0, 3, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(target.sequenceState(0, 0).cached_tokens, 2)
        << "A device-only truncate request may enqueue asynchronously, but it "
           "must never extend canonical sequence state.";

    std::vector<float> truncated_k(2 * KV_DIM, 0.0f);
    std::vector<float> truncated_v(2 * KV_DIM, 0.0f);
    IKVCache::KVCacheLogicalBlockDescriptor prefix_desc{0, 0, 0, 2, stream};
    ASSERT_TRUE(target.exportLogicalBlock(prefix_desc, truncated_k.data(), truncated_v.data()));
    expectRows(truncated_k, 2, KV_DIM, {120.0f, 130.0f});
    expectRows(truncated_v, 2, KV_DIM, {220.0f, 230.0f});

    cudaFree(d_k0);
    cudaFree(d_v0);
    cudaFree(d_k1);
    cudaFree(d_v1);
    cudaStreamDestroy(stream);
}

TEST(Test__CUDARingKVCache_LogicalBlockIO, Q8ShardedLayoutReportsDeviceResidentBytes)
{
    if (!hasCUDADevice())
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    CUDARingKVCacheQ8_1 cache(/*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/8,
                              /*n_kv_heads=*/4, /*local_n_kv_heads=*/2,
                              /*kv_head_start=*/2, /*head_dim=*/64, /*device_id=*/0);

    const auto layout = cache.logicalBlockLayout(0, 3);
    EXPECT_EQ(layout.k_precision, ActivationPrecision::Q8_1);
    EXPECT_EQ(layout.v_precision, ActivationPrecision::Q8_1);
    EXPECT_EQ(layout.local_kv_heads, 2);
    EXPECT_EQ(layout.kv_head_start, 2);
    EXPECT_EQ(layout.head_dim, 64);
    EXPECT_EQ(layout.k_bytes, 3u * 4u * sizeof(Q8_1Block));
    EXPECT_EQ(layout.v_bytes, layout.k_bytes);
    EXPECT_TRUE(layout.device_resident);
}

/**
 * @brief Prove oldest-row eviction is one device-owned metadata mutation.
 *
 * The production call returns after enqueueing work. The test synchronizes only
 * at its observation boundary, then verifies that the ring head still names the
 * end of the original append while the visible payload starts two rows later.
 */
TEST(Test__CUDARingKVCache_LogicalBlockIO,
     EvictOldestPreservesHeadAndPublishesExactDeviceTail)
{
    if (!hasCUDADevice())
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int KV_DIM = 2;
    CUDARingKVCacheFP32 cache(
        /*n_layers=*/1,
        /*batch_size=*/1,
        /*max_seq_len=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/KV_DIM,
        /*device_id=*/0);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    const auto host_k = taggedRows(5, KV_DIM, 100.0f);
    const auto host_v = taggedRows(5, KV_DIM, 200.0f);
    float *device_k = nullptr;
    float *device_v = nullptr;
    ASSERT_EQ(
        cudaSuccess,
        cudaMalloc(&device_k, host_k.size() * sizeof(float)));
    ASSERT_EQ(
        cudaSuccess,
        cudaMalloc(&device_v, host_v.size() * sizeof(float)));
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            device_k,
            host_k.data(),
            host_k.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            device_v,
            host_v.data(),
            host_v.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
    ASSERT_TRUE(cache.append(0, 0, device_k, device_v, 5, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    cache.evict_oldest(0, 0, 2);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    const auto state = cache.sequenceState(0, 0);
    EXPECT_EQ(state.cached_tokens, 3);
    EXPECT_EQ(state.implementation_head, 5);
    EXPECT_FALSE(state.wrapped);

    std::vector<float> visible_k(3 * KV_DIM, 0.0f);
    std::vector<float> visible_v(3 * KV_DIM, 0.0f);
    IKVCache::KVCacheLogicalBlockDescriptor desc{0, 0, 0, 3, stream};
    ASSERT_TRUE(cache.exportLogicalBlock(
        desc,
        visible_k.data(),
        visible_v.data()));
    expectRows(visible_k, 3, KV_DIM, {120.0f, 130.0f, 140.0f});
    expectRows(visible_v, 3, KV_DIM, {220.0f, 230.0f, 240.0f});

    ASSERT_EQ(cudaSuccess, cudaFree(device_k));
    ASSERT_EQ(cudaSuccess, cudaFree(device_v));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

/**
 * @brief Prove opaque sequence rollback restores every layer without host replay.
 *
 * The checkpoint captures only canonical ring metadata. Payload rows remain in
 * the cache while speculative appends advance the live heads; restoring the
 * checkpoint makes the original three-row prefixes visible again. All capture,
 * mutation, and restore operations are ordered on one explicit CUDA stream.
 */
TEST(Test__CUDARingKVCache_LogicalBlockIO,
     DeviceSequenceCheckpointRestoresAllLayersAndRequestsExactly)
{
    if (!hasCUDADevice())
    {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    constexpr int LAYERS = 2;
    constexpr int REQUESTS = 2;
    constexpr int KV_DIM = 2;
    constexpr int INITIAL_ROWS = 3;
    constexpr int SPECULATIVE_ROWS = 2;
    CUDARingKVCacheFP32 cache(
        LAYERS,
        REQUESTS,
        /*max_seq_len=*/8,
        /*n_kv_heads=*/1,
        KV_DIM,
        /*device_id=*/0);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    float *device_k = nullptr;
    float *device_v = nullptr;
    const size_t staging_elements =
        static_cast<size_t>(INITIAL_ROWS) * KV_DIM;
    ASSERT_EQ(
        cudaSuccess,
        cudaMalloc(&device_k, staging_elements * sizeof(float)));
    ASSERT_EQ(
        cudaSuccess,
        cudaMalloc(&device_v, staging_elements * sizeof(float)));

    std::vector<std::vector<float>> expected_k(LAYERS * REQUESTS);
    std::vector<std::vector<float>> expected_v(LAYERS * REQUESTS);
    for (int layer = 0; layer < LAYERS; ++layer)
    {
        for (int request = 0; request < REQUESTS; ++request)
        {
            const size_t index =
                static_cast<size_t>(layer * REQUESTS + request);
            expected_k[index] = taggedRows(
                INITIAL_ROWS,
                KV_DIM,
                1000.0f + static_cast<float>(layer * 100 + request * 10));
            expected_v[index] = taggedRows(
                INITIAL_ROWS,
                KV_DIM,
                2000.0f + static_cast<float>(layer * 100 + request * 10));
            ASSERT_EQ(
                cudaSuccess,
                cudaMemcpyAsync(
                    device_k,
                    expected_k[index].data(),
                    expected_k[index].size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
            ASSERT_EQ(
                cudaSuccess,
                cudaMemcpyAsync(
                    device_v,
                    expected_v[index].data(),
                    expected_v[index].size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
            ASSERT_TRUE(cache.append(
                layer,
                request,
                device_k,
                device_v,
                INITIAL_ROWS,
                stream));
        }
    }

    const size_t checkpoint_bytes =
        cache.deviceSequenceStateCheckpointBytes();
    ASSERT_GT(checkpoint_bytes, 0u);
    std::vector<void *> checkpoints(REQUESTS, nullptr);
    for (int request = 0; request < REQUESTS; ++request)
    {
        ASSERT_EQ(
            cudaSuccess,
            cudaMalloc(&checkpoints[static_cast<size_t>(request)],
                       checkpoint_bytes));
        std::string error;
        ASSERT_TRUE(cache.captureDeviceSequenceStateCheckpoint(
            request,
            checkpoints[static_cast<size_t>(request)],
            checkpoint_bytes,
            stream,
            &error))
            << error;
    }

    auto speculative_k = taggedRows(
        SPECULATIVE_ROWS,
        KV_DIM,
        9000.0f);
    auto speculative_v = taggedRows(
        SPECULATIVE_ROWS,
        KV_DIM,
        10000.0f);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            device_k,
            speculative_k.data(),
            speculative_k.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            device_v,
            speculative_v.data(),
            speculative_v.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
    for (int layer = 0; layer < LAYERS; ++layer)
    {
        for (int request = 0; request < REQUESTS; ++request)
        {
            ASSERT_TRUE(cache.append(
                layer,
                request,
                device_k,
                device_v,
                SPECULATIVE_ROWS,
                stream));
        }
    }
    for (int request = 0; request < REQUESTS; ++request)
    {
        std::string error;
        ASSERT_TRUE(cache.restoreDeviceSequenceStateCheckpoint(
            request,
            checkpoints[static_cast<size_t>(request)],
            checkpoint_bytes,
            stream,
            &error))
            << error;
    }
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int layer = 0; layer < LAYERS; ++layer)
    {
        for (int request = 0; request < REQUESTS; ++request)
        {
            const auto state = cache.sequenceState(layer, request);
            EXPECT_EQ(state.cached_tokens, INITIAL_ROWS);
            EXPECT_EQ(state.implementation_head, INITIAL_ROWS);
            EXPECT_FALSE(state.wrapped);

            std::vector<float> restored_k(
                static_cast<size_t>(INITIAL_ROWS) * KV_DIM,
                0.0f);
            std::vector<float> restored_v(
                static_cast<size_t>(INITIAL_ROWS) * KV_DIM,
                0.0f);
            IKVCache::KVCacheLogicalBlockDescriptor descriptor{
                layer,
                request,
                /*logical_token_start=*/0,
                INITIAL_ROWS,
                stream};
            ASSERT_TRUE(cache.exportLogicalBlock(
                descriptor,
                restored_k.data(),
                restored_v.data()));
            const size_t index =
                static_cast<size_t>(layer * REQUESTS + request);
            EXPECT_EQ(restored_k, expected_k[index]);
            EXPECT_EQ(restored_v, expected_v[index]);
        }
    }

    for (void *checkpoint : checkpoints)
        ASSERT_EQ(cudaSuccess, cudaFree(checkpoint));
    ASSERT_EQ(cudaSuccess, cudaFree(device_k));
    ASSERT_EQ(cudaSuccess, cudaFree(device_v));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

/**
 * @brief Reproduce the LocalTP sibling-device checkpoint launch boundary.
 *
 * LocalTP participant runners share a coordinator thread, so CUDA's current
 * device can still name the last sibling when an earlier participant begins a
 * live MTP checkpoint. Capture and restore own enough information to establish
 * their cache's allocation device and must never depend on ambient host-thread
 * state inherited from the previous participant.
 */
TEST(Test__CUDARingKVCache_LogicalBlockIO,
     DeviceCheckpointReactivatesOwningDeviceAfterSiblingExecution)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2)
    {
        GTEST_SKIP() << "Two CUDA devices are required";
    }

    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    CUDARingKVCacheFP32 cache(
        /*n_layers=*/2,
        /*batch_size=*/1,
        /*max_seq_len=*/8,
        /*n_kv_heads=*/1,
        /*head_dim=*/2,
        /*device_id=*/0);

    cudaStream_t owner_stream = nullptr;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&owner_stream));
    const size_t checkpoint_bytes =
        cache.deviceSequenceStateCheckpointBytes();
    ASSERT_GT(checkpoint_bytes, 0u);

    void *checkpoint = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&checkpoint, checkpoint_bytes));

    ASSERT_EQ(cudaSuccess, cudaSetDevice(1));
    std::string capture_error;
    ASSERT_TRUE(cache.captureDeviceSequenceStateCheckpoint(
        /*seq_idx=*/0,
        checkpoint,
        checkpoint_bytes,
        owner_stream,
        &capture_error))
        << capture_error;

    int current_device = -1;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, 0);

    ASSERT_EQ(cudaSuccess, cudaSetDevice(1));
    std::string restore_error;
    ASSERT_TRUE(cache.restoreDeviceSequenceStateCheckpoint(
        /*seq_idx=*/0,
        checkpoint,
        checkpoint_bytes,
        owner_stream,
        &restore_error))
        << restore_error;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, 0);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(owner_stream));

    ASSERT_EQ(cudaSuccess, cudaFree(checkpoint));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(owner_stream));
}
