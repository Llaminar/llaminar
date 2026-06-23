#include <gtest/gtest.h>

#include "kernels/rocm/kvcache/ROCmRingKVCache.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace llaminar2;

namespace
{
    bool hasROCmDevice()
    {
        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err != hipSuccess || device_count <= 0)
        {
            return false;
        }
        return hipSetDevice(0) == hipSuccess;
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

TEST(Test__ROCmRingKVCache_LogicalBlockIO, FP32WrappedExportImportAndTruncateWithStream)
{
    if (!hasROCmDevice())
    {
        GTEST_SKIP() << "ROCm device unavailable";
    }

    constexpr int KV_DIM = 2;
    ROCmRingKVCacheFP32 source(/*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/4,
                               /*n_kv_heads=*/1, /*head_dim=*/KV_DIM, /*device_id=*/0);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&stream));

    auto k0 = taggedRows(4, KV_DIM, 100.0f);
    auto v0 = taggedRows(4, KV_DIM, 200.0f);
    auto k1 = taggedRows(2, KV_DIM, 140.0f);
    auto v1 = taggedRows(2, KV_DIM, 240.0f);

    float *d_k0 = nullptr;
    float *d_v0 = nullptr;
    float *d_k1 = nullptr;
    float *d_v1 = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&d_k0, k0.size() * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_v0, v0.size() * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_k1, k1.size() * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMalloc(&d_v1, v1.size() * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(d_k0, k0.data(), k0.size() * sizeof(float),
                                         hipMemcpyHostToDevice, stream));
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(d_v0, v0.data(), v0.size() * sizeof(float),
                                         hipMemcpyHostToDevice, stream));
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(d_k1, k1.data(), k1.size() * sizeof(float),
                                         hipMemcpyHostToDevice, stream));
    ASSERT_EQ(hipSuccess, hipMemcpyAsync(d_v1, v1.data(), v1.size() * sizeof(float),
                                         hipMemcpyHostToDevice, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(source.append(0, 0, d_k0, d_v0, 4, stream));
    ASSERT_TRUE(source.append(0, 0, d_k1, d_v1, 2, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

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

    ROCmRingKVCacheFP32 target(/*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/4,
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
    EXPECT_FALSE(target.truncateSequence(0, 3, stream));

    std::vector<float> truncated_k(2 * KV_DIM, 0.0f);
    std::vector<float> truncated_v(2 * KV_DIM, 0.0f);
    IKVCache::KVCacheLogicalBlockDescriptor prefix_desc{0, 0, 0, 2, stream};
    ASSERT_TRUE(target.exportLogicalBlock(prefix_desc, truncated_k.data(), truncated_v.data()));
    expectRows(truncated_k, 2, KV_DIM, {120.0f, 130.0f});
    expectRows(truncated_v, 2, KV_DIM, {220.0f, 230.0f});

    (void)hipFree(d_k0);
    (void)hipFree(d_v0);
    (void)hipFree(d_k1);
    (void)hipFree(d_v1);
    (void)hipStreamDestroy(stream);
}

TEST(Test__ROCmRingKVCache_LogicalBlockIO, Q8ShardedLayoutReportsDeviceResidentBytes)
{
    if (!hasROCmDevice())
    {
        GTEST_SKIP() << "ROCm device unavailable";
    }

    ROCmRingKVCacheQ8_1 cache(/*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/8,
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
