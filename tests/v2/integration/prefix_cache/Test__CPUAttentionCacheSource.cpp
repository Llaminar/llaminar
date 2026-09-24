/**
 * @file Test__CPUAttentionCacheSource.cpp
 * @brief Prove that cold and restored attention consume the same native cache.
 *
 * This model-free regression composes production cache append, logical-block
 * export/import and attention. Transient projection buffers are deliberately
 * hostile: a cache-backed reader must never select them just because the
 * request starts at position zero. Every supported CPU cache precision owns
 * the same contract, including grouped and ordinary continuation widths.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/AttentionComputeStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/CPUKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "utils/MPIContext.h"
#include "../../utils/TestTensorFactory.h"
#include "../../utils/VerifierRowTestInventory.h"
#include "../../utils/AttentionKeyOutlierFixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llaminar2::test
{
namespace
{
using llaminar::v2::kernels::KernelFactory;
using llaminar::v2::kernels::KVCacheConfig;

/** @brief All-format production append/restore/attention fixture, without weights. */
class CPUAttentionCacheSource : public ::testing::TestWithParam<ActivationPrecision>
{
protected:
    /**
     * @brief Append one FP32 projection range through production conversion.
     * @param cache Destination owning the post-append sequence state.
     * @param rows Number of consecutive input positions to publish.
     * @param seed Deterministic fixture input identity.
     */
    void append(ICPUKVCache &cache, int rows, uint32_t seed)
    {
        const std::vector<size_t> shape{static_cast<size_t>(rows), kv_dim};
        auto key = TestTensorFactory::createFP32Random(shape, -3.0f, 3.0f, seed);
        auto value = TestTensorFactory::createFP32Random(shape, 0.1f, 1.0f, seed + 1);
        // Exercise BF16's native-operand append contract directly. This source
        // ownership test does not depend on a separate FP32-to-BF16 producer.
        auto key_bf16 = TestTensorFactory::createBF16Random(shape, -3.0f, 3.0f, seed);
        auto value_bf16 = TestTensorFactory::createBF16Random(shape, 0.1f, 1.0f, seed + 1);
        KVCacheAppendStage::Params params;
        params.K = GetParam() == ActivationPrecision::BF16
                       ? static_cast<ITensor *>(key_bf16.get()) : key.get();
        params.V = GetParam() == ActivationPrecision::BF16
                       ? static_cast<ITensor *>(value_bf16.get()) : value.get();
        params.kv_cache = &cache;
        params.num_tokens = rows;
        params.seq_len = rows;
        params.head_dim = head_dim;
        params.turboquant_ctx = &turboquant;
        KVCacheAppendStage stage(params);
        ASSERT_TRUE(stage.execute(nullptr));
    }

    /**
     * @brief Execute cache-backed attention with deliberately stale projections.
     * @return Production output, whose bytes may only depend on Q and the cache.
     */
    std::unique_ptr<FP32Tensor> attend(ICPUKVCache &cache, FP32Tensor &query)
    {
        const int rows = static_cast<int>(query.shape()[0]);
        auto stale = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(rows), kv_dim});
        auto output = TestTensorFactory::createFP32(query.shape());
        AttentionComputeStage::Params params;
        params.Q = &query;
        params.K = stale.get();
        params.V = stale.get();
        params.output = output.get();
        params.seq_len = rows;
        params.kv_len = rows; // Deliberately not an authoritative cache length.
        params.n_heads = query_heads;
        params.n_kv_heads = kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.kv_cache = &cache;
        params.layer_idx = 0;
        params.turboquant_ctx = &turboquant;
        AttentionComputeStage stage(params);
        EXPECT_TRUE(stage.execute(nullptr));
        return output;
    }

    static constexpr int head_dim = 64;
    static constexpr int kv_heads = 2;
    static constexpr int query_heads = 14;
    static constexpr size_t kv_dim = kv_heads * head_dim;
    static constexpr size_t query_dim = query_heads * head_dim;
    TurboQuantContext turboquant{head_dim, 42};
};

/** @brief Every suffix row must equal its cold-prefill counterpart byte for byte. */
TEST_P(CPUAttentionCacheSource, ColdAndRestoredRowsUseIdenticalNativeOperands)
{
    constexpr int prefix_rows = 139;
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    std::vector<int> widths(kGroupedVerifierRuntimeRows.begin(), kGroupedVerifierRuntimeRows.end());
    widths.insert(widths.end(), {1, 19, 64});
    for (const int suffix_rows : widths)
    {
        SCOPED_TRACE(suffix_rows);
        const int total_rows = prefix_rows + suffix_rows;
        KVCacheConfig config;
        config.precision = GetParam();
        config.num_layers = 1;
        config.max_seq_len = total_rows + 7;
        config.n_kv_heads = kv_heads;
        config.head_dim = head_dim;
        config.mpi_ctx = &mpi;
        config.turboquant_ctx = &turboquant;
        auto cold = KernelFactory::createCPUKVCache(config);
        auto restored = KernelFactory::createCPUKVCache(config);
        ASSERT_NO_FATAL_FAILURE(append(*cold, prefix_rows, 47));

        const IKVCache::KVCacheLogicalBlockDescriptor prefix{0, 0, 0, prefix_rows, nullptr};
        const auto layout = cold->logicalBlockLayout(0, prefix_rows);
        std::vector<uint8_t> key_bytes(layout.k_bytes), value_bytes(layout.v_bytes);
        ASSERT_TRUE(cold->exportLogicalBlock(prefix, key_bytes.data(), value_bytes.data()));
        ASSERT_TRUE(restored->importLogicalBlock(prefix, key_bytes.data(), value_bytes.data()));
        ASSERT_NO_FATAL_FAILURE(append(*cold, suffix_rows, 73));
        ASSERT_NO_FATAL_FAILURE(append(*restored, suffix_rows, 73));

        auto query = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(total_rows), query_dim}, -2.0f, 2.0f, 97);
        auto suffix = TestTensorFactory::createFP32(
            {static_cast<size_t>(suffix_rows), query_dim});
        std::memcpy(suffix->mutable_data(), query->data() + prefix_rows * query_dim,
                    suffix->numel() * sizeof(float));
        const auto whole_output = attend(*cold, *query);
        const auto suffix_output = attend(*restored, *suffix);
        ASSERT_FALSE(HasFailure());
        EXPECT_EQ(std::memcmp(whole_output->data() + prefix_rows * query_dim,
                              suffix_output->data(), suffix_output->numel() * sizeof(float)), 0)
            << "Cold prefill must not bypass the configured native KV precision";
    }
}

INSTANTIATE_TEST_SUITE_P(AllCPUFormats, CPUAttentionCacheSource,
    ::testing::Values(ActivationPrecision::FP32, ActivationPrecision::BF16,
                      ActivationPrecision::FP16, ActivationPrecision::Q8_1,
                      ActivationPrecision::Q16_1, ActivationPrecision::TQ4,
                      ActivationPrecision::TQ8),
    [](const auto &info) { return activationPrecisionToString(info.param); });

/**
 * @brief Production Q8 cache must preserve score-relevant small key coordinates.
 *
 * The scalar codec already has this quality gate. Exercise the public cache
 * selector, real append and native attention too: a correct unused reference
 * implementation cannot certify a backend's different physical key codec.
 */
TEST(CPUAttentionKeyQuality, Q8KeysPreserveScoreRelevantSmallCoordinates)
{
    const AttentionKeyOutlierFixture fixture;
    using F = AttentionKeyOutlierFixture;
    MPIContext mpi(0, 1, MPI_COMM_WORLD);
    KVCacheConfig config;
    config.precision = ActivationPrecision::Q8_1;
    config.num_layers = 1;
    config.max_seq_len = 32;
    config.n_kv_heads = F::kv_heads;
    config.head_dim = F::head_dim;
    config.mpi_ctx = &mpi;
    auto cache = KernelFactory::createCPUKVCache(config);
    auto key = TestTensorFactory::createFP32({F::tokens, F::kv_heads * F::head_dim});
    auto value = TestTensorFactory::createFP32(key->shape());
    auto query = TestTensorFactory::createFP32({F::tokens, F::query_heads * F::head_dim});
    auto output = TestTensorFactory::createFP32(query->shape());
    std::memcpy(key->mutable_data(), fixture.keys.data(), fixture.keys.size() * sizeof(float));
    std::memcpy(value->mutable_data(), fixture.values.data(), fixture.values.size() * sizeof(float));
    std::memcpy(query->mutable_data(), fixture.queries.data(), fixture.queries.size() * sizeof(float));
    KVCacheAppendStage::Params append_params;
    append_params.K = key.get();
    append_params.V = value.get();
    append_params.kv_cache = cache.get();
    append_params.num_tokens = F::tokens;
    append_params.seq_len = F::tokens;
    append_params.head_dim = F::head_dim;
    KVCacheAppendStage append(append_params);
    ASSERT_TRUE(append.execute(nullptr));

    AttentionComputeStage::Params attention_params;
    attention_params.Q = query.get();
    attention_params.output = output.get();
    attention_params.seq_len = F::tokens;
    attention_params.n_heads = F::query_heads;
    attention_params.n_kv_heads = F::kv_heads;
    attention_params.head_dim = F::head_dim;
    attention_params.causal = true;
    attention_params.kv_cache = cache.get();
    attention_params.layer_idx = 0;
    AttentionComputeStage attention(attention_params);
    ASSERT_TRUE(attention.execute(nullptr));

    // Independent FP64 causal attention, not the native codec's own decoder.
    // This measures whether compression preserved the useful computation, not
    // merely whether two readers agree on the same lossy cache bytes.
    double dot = 0.0, reference_norm = 0.0, actual_norm = 0.0, squared_error = 0.0;
    for (int row = 0; row < F::tokens; ++row)
    {
        for (int head = 0; head < F::query_heads; ++head)
        {
            const int kv_head = head / (F::query_heads / F::kv_heads);
            std::array<double, F::tokens> scores{};
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key_row = 0; key_row <= row; ++key_row)
            {
                for (int dim = 0; dim < F::head_dim; ++dim)
                {
                    scores[key_row] += static_cast<double>(fixture.queries[F::offset(row, head, dim, F::query_heads)]) *
                                       fixture.keys[F::offset(key_row, kv_head, dim, F::kv_heads)];
                }
                scores[key_row] /= std::sqrt(static_cast<double>(F::head_dim));
                maximum = std::max(maximum, scores[key_row]);
            }
            double sum = 0.0;
            for (int key_row = 0; key_row <= row; ++key_row)
            {
                scores[key_row] = std::exp(scores[key_row] - maximum);
                sum += scores[key_row];
            }
            for (int dim = 0; dim < F::head_dim; ++dim)
            {
                double expected = 0.0;
                for (int key_row = 0; key_row <= row; ++key_row)
                    expected += scores[key_row] / sum * fixture.values[F::offset(key_row, kv_head, dim, F::kv_heads)];
                const double actual = output->data()[F::offset(row, head, dim, F::query_heads)];
                ASSERT_TRUE(std::isfinite(actual));
                dot += expected * actual;
                reference_norm += expected * expected;
                actual_norm += actual * actual;
                squared_error += (expected - actual) * (expected - actual);
            }
        }
    }
    // Same established quality budget as the scalar AQ8 regression. No new
    // tolerance is inferred from the currently failing production result.
    const double cosine = dot / std::sqrt(reference_norm * actual_norm);
    const double relative_error = std::sqrt(squared_error / reference_norm);
    RecordProperty("attention_cosine", std::to_string(cosine));
    RecordProperty("attention_relative_l2", std::to_string(relative_error));
    EXPECT_GT(cosine, 0.995);
    EXPECT_LT(relative_error, 0.06);
}
} // namespace
} // namespace llaminar2::test
