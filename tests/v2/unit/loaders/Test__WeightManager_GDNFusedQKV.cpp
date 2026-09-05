/**
 * @file Test__WeightManager_GDNFusedQKV.cpp
 * @brief Byte-exact production-loader tests for linked GDN tensor parallelism.
 *
 * A Gated Delta Net value head consumes key head @c value_head % n_k_heads.
 * Contiguously sharding V while replicating Q/K is correct but wastes most of
 * the projection work at small TP degrees.  The production CPU layout instead
 * owns a contiguous Q/K interval and packs every modulo-linked V interval next
 * to it.  Every value-associated tensor must use that same packed order or the
 * recurrence will silently combine weights from different global heads.
 *
 * These tests exercise WeightManager's real native slicing entrypoints.  They
 * prove TP=2/4/8 totality, exact FP32 row/column placement, parity between the
 * LocalTP-assignment and equal-rank MPI entrypoints, and native Q8_0 byte
 * preservation.  No dequantized or synthetic row-replay path is involved.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config/GDNHeadAssignment.h"
#include "config/TensorParallelConfig.h"
#include "loaders/WeightManager.h"
#include "mocks/MockModelLoader.h"
#include "models/qwen35/Qwen35Schema.h"
#include "tensors/TensorSlice.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

namespace llaminar2::test
{
    namespace
    {
        // Keep both attention head counts at least TP=16.  This prevents the
        // ordinary GQA-replication policy from introducing zero-head synthetic
        // participants, allowing this fixture to isolate GDN ownership itself.
        constexpr int kAttentionHeads = 32;
        constexpr int kAttentionKVHeads = 16;
        constexpr int kAttentionHeadDim = 4;
        constexpr int kGDNKeyHeads = 8;
        constexpr int kGDNValueHeads = 16;
        constexpr int kGDNState = 4;
        constexpr int kHidden = 16;
        constexpr int kConvWidth = 4;
        constexpr int kDFF = 64;
        constexpr int kVocab = 128;

        constexpr std::size_t kKeyElements =
            static_cast<std::size_t>(kGDNKeyHeads * kGDNState);
        constexpr std::size_t kValueElements =
            static_cast<std::size_t>(kGDNValueHeads * kGDNState);
        constexpr std::size_t kFusedElements =
            2 * kKeyElements + kValueElements;

        /** @brief Build a matrix whose complete rows are byte-distinguishable. */
        std::shared_ptr<FP32Tensor> makeTaggedMatrix(
            std::size_t rows,
            std::size_t columns,
            int tag)
        {
            auto tensor = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{rows, columns});
            for (std::size_t row = 0; row < rows; ++row)
            {
                for (std::size_t column = 0; column < columns; ++column)
                {
                    tensor->mutable_data()[row * columns + column] =
                        static_cast<float>(tag + row * columns + column);
                }
            }
            return tensor;
        }

        /** @brief Build a vector whose elements are byte-distinguishable. */
        std::shared_ptr<FP32Tensor> makeTaggedVector(
            std::size_t elements,
            int tag)
        {
            auto tensor = std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{elements});
            for (std::size_t index = 0; index < elements; ++index)
                tensor->mutable_data()[index] = static_cast<float>(tag + index);
            return tensor;
        }

        /** @brief Expand a set of source intervals into explicit element indices. */
        std::vector<std::size_t> expandSpans(
            const std::vector<GDNHeadSpan> &spans,
            std::size_t base = 0)
        {
            std::vector<std::size_t> indices;
            for (const GDNHeadSpan span : spans)
            {
                for (int index = 0; index < span.count; ++index)
                {
                    indices.push_back(
                        base + static_cast<std::size_t>(span.start + index));
                }
            }
            return indices;
        }

        /** @brief Return source rows for packed local [Q|K|V]. */
        std::vector<std::size_t> expectedFusedRows(
            const GDNHeadAssignment &assignment,
            int elements_per_head)
        {
            std::vector<std::size_t> rows;
            const GDNHeadSpan key =
                assignment.keyElementSpan(elements_per_head);
            const auto append = [&rows](GDNHeadSpan span, std::size_t base)
            {
                for (int index = 0; index < span.count; ++index)
                {
                    rows.push_back(
                        base + static_cast<std::size_t>(span.start + index));
                }
            };

            append(key, 0);
            append(key, kKeyElements);
            for (const GDNHeadSpan value :
                 assignment.valueElementSpans(elements_per_head))
            {
                append(value, 2 * kKeyElements);
            }
            return rows;
        }

        /** @brief Assert exact row selection, including every payload byte. */
        void expectFP32Rows(
            const TensorBase &actual,
            const FP32Tensor &source,
            const std::vector<std::size_t> &source_rows)
        {
            ASSERT_EQ(actual.shape().size(), 2u);
            ASSERT_EQ(source.shape().size(), 2u);
            ASSERT_EQ(actual.shape()[0], source_rows.size());
            ASSERT_EQ(actual.shape()[1], source.shape()[1]);
            ASSERT_NE(actual.raw_data(), nullptr);

            const std::size_t row_bytes =
                source.shape()[1] * sizeof(float);
            const auto *actual_bytes =
                static_cast<const std::uint8_t *>(actual.raw_data());
            const auto *source_bytes =
                static_cast<const std::uint8_t *>(source.raw_data());
            for (std::size_t local_row = 0;
                 local_row < source_rows.size(); ++local_row)
            {
                EXPECT_EQ(
                    std::memcmp(
                        actual_bytes + local_row * row_bytes,
                        source_bytes + source_rows[local_row] * row_bytes,
                        row_bytes),
                    0)
                    << "local row " << local_row
                    << " expected source row " << source_rows[local_row];
            }
        }

        /** @brief Assert exact column selection for every source row. */
        void expectFP32Columns(
            const TensorBase &actual,
            const FP32Tensor &source,
            const std::vector<std::size_t> &source_columns)
        {
            ASSERT_EQ(actual.shape().size(), 2u);
            ASSERT_EQ(source.shape().size(), 2u);
            ASSERT_EQ(actual.shape()[0], source.shape()[0]);
            ASSERT_EQ(actual.shape()[1], source_columns.size());

            const auto *actual_data = static_cast<const float *>(actual.raw_data());
            const float *source_data = source.data();
            for (std::size_t row = 0; row < source.shape()[0]; ++row)
            {
                for (std::size_t local_column = 0;
                     local_column < source_columns.size(); ++local_column)
                {
                    const float *actual_element =
                        actual_data + row * source_columns.size() + local_column;
                    const float *source_element =
                        source_data + row * source.shape()[1] +
                        source_columns[local_column];
                    EXPECT_EQ(
                        std::memcmp(actual_element, source_element, sizeof(float)),
                        0)
                        << "row " << row << " local column " << local_column
                        << " expected source column "
                        << source_columns[local_column];
                }
            }
        }

        /** @brief Assert exact element selection from a one-dimensional tensor. */
        void expectFP32Elements(
            const TensorBase &actual,
            const FP32Tensor &source,
            const std::vector<std::size_t> &source_elements)
        {
            ASSERT_EQ(actual.numel(), source_elements.size());
            const auto *actual_data = static_cast<const float *>(actual.raw_data());
            for (std::size_t local = 0; local < source_elements.size(); ++local)
            {
                EXPECT_EQ(
                    std::memcmp(
                        actual_data + local,
                        source.data() + source_elements[local],
                        sizeof(float)),
                    0)
                    << "local element " << local
                    << " expected source element " << source_elements[local];
            }
        }

        /**
         * @brief Mock loader with native block-aligned Q8_0 row/column slices.
         *
         * MockModelLoader intentionally implements only FP32 slicing.  This
         * narrow extension mirrors GGUF's native Q8_0 byte slicing so the
         * regression can prove that linked packing never dequantizes/requantizes
         * a production quantized tensor.
         */
        class NativeQ8SliceLoader final : public MockModelLoader
        {
        public:
            std::shared_ptr<TensorBase> loadTensorRowSlice(
                const std::string &name,
                std::size_t row_start,
                std::size_t row_end,
                DeviceId device = DeviceId::cpu(),
                WeightPrecision precision = WeightPrecision::NATIVE) override
            {
                auto source = MockModelLoader::loadTensor(name, device, precision);
                if (!source || source->native_type() != TensorType::Q8_0 ||
                    source->shape().size() != 2 ||
                    row_start >= row_end || row_end > source->shape()[0])
                {
                    return nullptr;
                }

                const std::size_t row_bytes =
                    source->size_bytes() / source->shape()[0];
                std::vector<std::uint8_t> bytes(
                    (row_end - row_start) * row_bytes);
                std::memcpy(
                    bytes.data(),
                    static_cast<const std::uint8_t *>(source->raw_data()) +
                        row_start * row_bytes,
                    bytes.size());
                return std::make_shared<Q8_0Tensor>(
                    std::vector<std::size_t>{
                        row_end - row_start, source->shape()[1]},
                    bytes);
            }

            std::shared_ptr<TensorBase> loadTensorColumnSlice(
                const std::string &name,
                std::size_t column_start,
                std::size_t column_end,
                DeviceId device = DeviceId::cpu(),
                WeightPrecision precision = WeightPrecision::NATIVE) override
            {
                auto source = MockModelLoader::loadTensor(name, device, precision);
                if (!source || source->native_type() != TensorType::Q8_0 ||
                    source->shape().size() != 2 || column_start >= column_end ||
                    column_end > source->shape()[1] ||
                    column_start % 32 != 0 || column_end % 32 != 0)
                {
                    return nullptr;
                }

                const std::size_t source_row_bytes =
                    source->size_bytes() / source->shape()[0];
                const std::size_t byte_start =
                    source_row_bytes * column_start / source->shape()[1];
                const std::size_t slice_row_bytes =
                    source_row_bytes * (column_end - column_start) /
                    source->shape()[1];
                std::vector<std::uint8_t> bytes(
                    source->shape()[0] * slice_row_bytes);
                const auto *source_bytes =
                    static_cast<const std::uint8_t *>(source->raw_data());
                for (std::size_t row = 0; row < source->shape()[0]; ++row)
                {
                    std::memcpy(
                        bytes.data() + row * slice_row_bytes,
                        source_bytes + row * source_row_bytes + byte_start,
                        slice_row_bytes);
                }
                return std::make_shared<Q8_0Tensor>(
                    std::vector<std::size_t>{
                        source->shape()[0], column_end - column_start},
                    bytes);
            }
        };
    } // namespace

    /**
     * @brief Fixture containing every value-associated GDN production weight.
     */
    class WeightManagerLinkedGDNTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            loader_ = std::make_shared<MockModelLoader>();
            loader_->setLoaded(true);
            loader_->setArchitecture("qwen3.5");
            loader_->setBlockCount(1);
            loader_->setEmbeddingLength(kHidden);
            loader_->setHeadCount(kAttentionHeads);
            loader_->setHeadCountKV(kAttentionKVHeads);
            loader_->setFeedForwardLength(kDFF);
            loader_->setVocabSize(kVocab);

            qkv_ = makeTaggedMatrix(kFusedElements, kHidden, 1000);
            conv_ = makeTaggedMatrix(kFusedElements, kConvWidth, 2000);
            gate_ = makeTaggedMatrix(kValueElements, kHidden, 3000);
            alpha_ = makeTaggedMatrix(kGDNValueHeads, kHidden, 4000);
            beta_ = makeTaggedMatrix(kGDNValueHeads, kHidden, 5000);
            dt_bias_ = makeTaggedVector(kGDNValueHeads, 6000);
            decay_a_ = makeTaggedVector(kGDNValueHeads, 7000);
            output_ = makeTaggedMatrix(kHidden, kValueElements, 8000);

            loader_->addTensor("blk.0.attn_qkv.weight", qkv_);
            loader_->addTensor("blk.0.ssm_conv1d.weight", conv_);
            loader_->addTensor("blk.0.attn_gate.weight", gate_);
            loader_->addTensor("blk.0.ssm_alpha.weight", alpha_);
            loader_->addTensor("blk.0.ssm_beta.weight", beta_);
            loader_->addTensor("blk.0.ssm_dt.bias", dt_bias_);
            loader_->addTensor("blk.0.ssm_a", decay_a_);
            loader_->addTensor("blk.0.ssm_out.weight", output_);

            Qwen35SchemaFactory schema_factory;
            sharding_ = schema_factory.getWeightShardingConfig();
        }

        /** @brief Build the canonical equal TP partition used by production. */
        std::shared_ptr<TensorParallelConfig> makeTPConfig(int world_size) const
        {
            return std::make_shared<TensorParallelConfig>(
                TensorParallelConfig::equalSplit(
                    world_size,
                    kAttentionHeads,
                    kAttentionKVHeads,
                    kDFF,
                    kVocab));
        }

        /** @brief Build one rank-local manager with assignment-aware loading. */
        std::unique_ptr<WeightManager> makeManager(
            int rank,
            int world_size,
            const std::shared_ptr<TensorParallelConfig> &tp_config) const
        {
            auto manager = std::make_unique<WeightManager>(
                *loader_,
                MPIContextFactory::create_mock(rank, world_size),
                nullptr,
                WeightDistributionStrategy::SHARDED,
                WeightPrecision::NATIVE);
            manager->setWeightShardingConfig(sharding_);
            manager->setModelDimensions(
                kAttentionHeads, kAttentionKVHeads, kAttentionHeadDim);
            manager->setGDNDimensions(
                kGDNKeyHeads, kGDNValueHeads, kGDNState);
            if (tp_config)
                manager->setTensorParallelConfig(tp_config);
            return manager;
        }

        /** @brief Load through the explicit assignment production entrypoint. */
        std::shared_ptr<TensorBase> loadAssigned(
            WeightManager &manager,
            const TensorParallelConfig &config,
            int rank,
            const std::string &name) const
        {
            return manager.getShardedWeightForAssignment(
                name, DeviceId::cpu(), config.forRank(rank), 0);
        }

        /** @brief Resolve the expected linked GDN assignment for a TP rank. */
        GDNHeadAssignment linkedAssignment(
            const TensorParallelConfig &config,
            int rank) const
        {
            const auto &participant = config.forRank(rank);
            return GDNHeadAssignment::fromPartition(
                kGDNKeyHeads,
                kGDNValueHeads,
                participant.head_start,
                participant.head_count,
                config.totalHeads());
        }

        std::shared_ptr<MockModelLoader> loader_;
        WeightShardingConfig sharding_;
        std::shared_ptr<FP32Tensor> qkv_;
        std::shared_ptr<FP32Tensor> conv_;
        std::shared_ptr<FP32Tensor> gate_;
        std::shared_ptr<FP32Tensor> alpha_;
        std::shared_ptr<FP32Tensor> beta_;
        std::shared_ptr<FP32Tensor> dt_bias_;
        std::shared_ptr<FP32Tensor> decay_a_;
        std::shared_ptr<FP32Tensor> output_;
    };

    /**
     * @brief Fused projection and short-convolution channels share linked order.
     */
    TEST_F(WeightManagerLinkedGDNTest, FusedProjectionAndConvAreByteExactForTP2TP4TP8)
    {
        for (const int world_size : {2, 4, 8})
        {
            auto config = makeTPConfig(world_size);
            for (int rank = 0; rank < world_size; ++rank)
            {
                auto manager = makeManager(rank, world_size, config);
                const GDNHeadAssignment assignment =
                    linkedAssignment(*config, rank);
                const std::vector<std::size_t> expected =
                    expectedFusedRows(assignment, kGDNState);

                auto qkv = loadAssigned(
                    *manager, *config, rank, "blk.0.attn_qkv.weight");
                auto conv = loadAssigned(
                    *manager, *config, rank, "blk.0.ssm_conv1d.weight");
                ASSERT_NE(qkv, nullptr) << "TP=" << world_size << " rank=" << rank;
                ASSERT_NE(conv, nullptr) << "TP=" << world_size << " rank=" << rank;
                expectFP32Rows(*qkv, *qkv_, expected);
                expectFP32Rows(*conv, *conv_, expected);

                EXPECT_EQ(qkv->shape()[0], kFusedElements / world_size);
                EXPECT_EQ(conv->shape()[0], kFusedElements / world_size);
            }
        }
    }

    /**
     * @brief CUDA/ROCm-tagged production loading honors uneven TP=2..8 plans.
     *
     * MockModelLoader keeps this a fast, device-free unit test while the
     * DeviceId forces WeightManager through the same non-CPU materialization
     * branch used by CUDA and ROCm.  Qwen3.5-122B geometry has 32 GDN heads
     * and only two attention KV heads: TP=3 therefore owns 12/10/10 query/GDN
     * heads and replicates K/V.  Every source row is checked byte-for-byte so
     * an equal-row fallback cannot masquerade as a valid shard.
     */
    TEST(WeightManagerUnevenFusedQKVTest,
         CUDAAndROCmTP2ThroughTP8ConsumeTypedAssignments)
    {
        constexpr int attention_heads = 32;
        constexpr int attention_kv_heads = 2;
        constexpr int attention_head_dim = 4;
        constexpr int gdn_heads = 32;
        constexpr int gdn_state = 4;
        constexpr int hidden = 16;
        constexpr int d_ff = 256;
        constexpr int vocab = 128;
        constexpr size_t attention_q_rows =
            attention_heads * attention_head_dim;
        constexpr size_t attention_kv_rows =
            attention_kv_heads * attention_head_dim;
        constexpr size_t attention_fused_rows =
            attention_q_rows + 2 * attention_kv_rows;
        constexpr size_t gdn_block_rows = gdn_heads * gdn_state;
        constexpr size_t gdn_fused_rows = 3 * gdn_block_rows;

        auto loader = std::make_shared<MockModelLoader>();
        loader->setLoaded(true);
        loader->setArchitecture("qwen3.5");
        auto attention_source = makeTaggedMatrix(
            attention_fused_rows, hidden, 11000);
        auto gdn_source = makeTaggedMatrix(
            gdn_fused_rows, hidden, 21000);
        loader->addTensor("blk.1.attn_qkv.weight", attention_source);
        loader->addTensor("blk.0.attn_qkv.weight", gdn_source);

        Qwen35SchemaFactory schema_factory;
        const WeightShardingConfig sharding =
            schema_factory.getWeightShardingConfig();

        for (const bool rocm_backend : {false, true})
        {
            for (int degree = 2; degree <= 8; ++degree)
            {
                std::vector<DeviceId> devices;
                devices.reserve(static_cast<size_t>(degree));
                for (int rank = 0; rank < degree; ++rank)
                {
                    devices.push_back(
                        rocm_backend ? DeviceId::rocm(rank)
                                     : DeviceId::cuda(rank));
                }
                auto config = std::make_shared<TensorParallelConfig>(
                    TensorParallelConfig::equalSplit(
                        degree,
                        attention_heads,
                        attention_kv_heads,
                        d_ff,
                        vocab,
                        devices));

                for (int rank = 0; rank < degree; ++rank)
                {
                    WeightManager manager(
                        *loader,
                        MPIContextFactory::create_mock(rank, degree),
                        nullptr,
                        WeightDistributionStrategy::SHARDED,
                        WeightPrecision::NATIVE);
                    manager.setWeightShardingConfig(sharding);
                    manager.setModelDimensions(
                        attention_heads,
                        attention_kv_heads,
                        attention_head_dim);
                    manager.setGDNDimensions(
                        gdn_heads, gdn_heads, gdn_state);
                    manager.setTensorParallelConfig(config);

                    const auto &assignment = config->forRank(rank);
                    auto gdn = manager.getShardedWeightForAssignment(
                        "blk.0.attn_qkv.weight",
                        devices[static_cast<size_t>(rank)],
                        assignment,
                        0);
                    auto attention = manager.getShardedWeightForAssignment(
                        "blk.1.attn_qkv.weight",
                        devices[static_cast<size_t>(rank)],
                        assignment,
                        0);
                    ASSERT_NE(gdn, nullptr)
                        << "backend=" << (rocm_backend ? "ROCm" : "CUDA")
                        << " TP=" << degree << " rank=" << rank;
                    ASSERT_NE(attention, nullptr)
                        << "backend=" << (rocm_backend ? "ROCm" : "CUDA")
                        << " TP=" << degree << " rank=" << rank;

                    std::vector<size_t> expected_gdn_rows;
                    std::vector<size_t> expected_attention_rows;
                    const auto append = [](
                                            std::vector<size_t> &rows,
                                            size_t base,
                                            size_t start,
                                            size_t count)
                    {
                        for (size_t row = 0; row < count; ++row)
                            rows.push_back(base + start + row);
                    };
                    const size_t local_q_start =
                        static_cast<size_t>(assignment.head_start) *
                        gdn_state;
                    const size_t local_q_count =
                        static_cast<size_t>(assignment.head_count) *
                        gdn_state;
                    append(
                        expected_gdn_rows,
                        0u,
                        local_q_start,
                        local_q_count);
                    append(
                        expected_gdn_rows,
                        gdn_block_rows,
                        local_q_start,
                        local_q_count);
                    append(
                        expected_gdn_rows,
                        2 * gdn_block_rows,
                        local_q_start,
                        local_q_count);

                    const size_t attention_q_start =
                        static_cast<size_t>(assignment.head_start) *
                        attention_head_dim;
                    const size_t attention_q_count =
                        static_cast<size_t>(assignment.head_count) *
                        attention_head_dim;
                    const size_t attention_kv_start =
                        static_cast<size_t>(assignment.kv_head_start) *
                        attention_head_dim;
                    const size_t attention_kv_count =
                        static_cast<size_t>(assignment.kv_head_count) *
                        attention_head_dim;
                    append(
                        expected_attention_rows,
                        0u,
                        attention_q_start,
                        attention_q_count);
                    append(
                        expected_attention_rows,
                        attention_q_rows,
                        attention_kv_start,
                        attention_kv_count);
                    append(
                        expected_attention_rows,
                        attention_q_rows + attention_kv_rows,
                        attention_kv_start,
                        attention_kv_count);

                    expectFP32Rows(
                        *gdn, *gdn_source, expected_gdn_rows);
                    expectFP32Rows(
                        *attention,
                        *attention_source,
                        expected_attention_rows);
                }
            }
        }
    }

    /**
     * @brief Every V-associated row, scalar, and column uses one packed order.
     */
    TEST_F(WeightManagerLinkedGDNTest, ValueWeightsShareOneByteExactOrderForTP2TP4TP8)
    {
        for (const int world_size : {2, 4, 8})
        {
            auto config = makeTPConfig(world_size);
            for (int rank = 0; rank < world_size; ++rank)
            {
                auto manager = makeManager(rank, world_size, config);
                const GDNHeadAssignment assignment =
                    linkedAssignment(*config, rank);
                const auto value_rows = expandSpans(
                    assignment.valueElementSpans(kGDNState));
                const auto value_heads = expandSpans(
                    assignment.valueElementSpans(1));

                expectFP32Rows(
                    *loadAssigned(*manager, *config, rank,
                                  "blk.0.attn_gate.weight"),
                    *gate_, value_rows);
                expectFP32Rows(
                    *loadAssigned(*manager, *config, rank,
                                  "blk.0.ssm_alpha.weight"),
                    *alpha_, value_heads);
                expectFP32Rows(
                    *loadAssigned(*manager, *config, rank,
                                  "blk.0.ssm_beta.weight"),
                    *beta_, value_heads);
                expectFP32Elements(
                    *loadAssigned(*manager, *config, rank, "blk.0.ssm_dt.bias"),
                    *dt_bias_, value_heads);
                expectFP32Elements(
                    *loadAssigned(*manager, *config, rank, "blk.0.ssm_a"),
                    *decay_a_, value_heads);
                expectFP32Columns(
                    *loadAssigned(*manager, *config, rank,
                                  "blk.0.ssm_out.weight"),
                    *output_, value_rows);
            }
        }
    }

    /**
     * @brief Scalar GDN shards retain participant identity across cache hits.
     *
     * The scalar time-step and decay tensors are physically packed into a
     * participant-local FP32 vector.  That payload must remain wrapped in a
     * TensorSlice: otherwise the second production lookup cannot distinguish
     * the packed shard from an accidental full-tensor clone and must fail the
     * cache contract.  Exercise every supported TP degree and participant.
     */
    TEST_F(WeightManagerLinkedGDNTest, ScalarValueCacheHitsPreserveTPSliceIdentity)
    {
        for (const int world_size : {2, 4, 8})
        {
            auto config = makeTPConfig(world_size);
            for (int rank = 0; rank < world_size; ++rank)
            {
                auto manager = makeManager(rank, world_size, config);
                for (const std::string &name : {
                         "blk.0.ssm_dt.bias",
                         "blk.0.ssm_a"})
                {
                    auto first = loadAssigned(*manager, *config, rank, name);
                    ASSERT_NE(first, nullptr) << name;

                    const auto *slice = dynamic_cast<const TensorSlice *>(first.get());
                    ASSERT_NE(slice, nullptr)
                        << "Packed scalar GDN weight lost TP ownership: " << name;
                    EXPECT_TRUE(slice->is_column_parallel()) << name;
                    EXPECT_EQ(slice->metadata().rank, rank) << name;
                    EXPECT_EQ(slice->metadata().world_size, world_size) << name;
                    EXPECT_TRUE(slice->metadata().inner_is_presliced) << name;

                    auto cached = loadAssigned(*manager, *config, rank, name);
                    ASSERT_NE(cached, nullptr) << name;
                    EXPECT_EQ(cached.get(), first.get()) << name;
                }
            }
        }
    }

    /**
     * @brief Equal-rank MPI loading and assignment-aware loading cannot diverge.
     */
    TEST_F(WeightManagerLinkedGDNTest, EqualRankMPIEntryPointMatchesAssignmentEntryPoint)
    {
        for (const int world_size : {2, 4, 8})
        {
            auto config = makeTPConfig(world_size);
            for (int rank = 0; rank < world_size; ++rank)
            {
                auto assigned_manager = makeManager(rank, world_size, config);
                auto mpi_manager = makeManager(rank, world_size, nullptr);

                for (const std::string &name : {
                         "blk.0.attn_qkv.weight",
                         "blk.0.ssm_conv1d.weight",
                         "blk.0.attn_gate.weight",
                         "blk.0.ssm_alpha.weight",
                         "blk.0.ssm_dt.bias",
                         "blk.0.ssm_out.weight"})
                {
                    auto assigned = loadAssigned(
                        *assigned_manager, *config, rank, name);
                    auto equal_rank = mpi_manager->getWeightForDevice(
                        name, DeviceId::cpu());
                    ASSERT_NE(assigned, nullptr) << name;
                    ASSERT_NE(equal_rank, nullptr) << name;
                    ASSERT_EQ(assigned->shape(), equal_rank->shape()) << name;
                    ASSERT_EQ(assigned->size_bytes(), equal_rank->size_bytes())
                        << name;
                    EXPECT_EQ(
                        std::memcmp(
                            assigned->raw_data(),
                            equal_rank->raw_data(),
                            assigned->size_bytes()),
                        0)
                        << "TP=" << world_size << " rank=" << rank
                        << " weight=" << name;
                }
            }
        }
    }

    /**
     * @brief A TP topology that cuts through a key head fails before loading.
     */
    TEST_F(WeightManagerLinkedGDNTest, NonIntegralKeyHeadPartitionFailsHard)
    {
        constexpr int world_size = 16;
        auto config = makeTPConfig(world_size);
        auto manager = makeManager(0, world_size, config);

        EXPECT_THROW(
            loadAssigned(
                *manager, *config, 0, "blk.0.attn_qkv.weight"),
            std::invalid_argument);

        auto mpi_manager = makeManager(0, world_size, nullptr);
        EXPECT_THROW(
            mpi_manager->getWeightForDevice(
                "blk.0.attn_qkv.weight", DeviceId::cpu()),
            std::invalid_argument);
    }

    /**
     * @brief Native Q8_0 linked packing copies source blocks byte for byte.
     */
    TEST(WeightManagerLinkedGDNQ8Test, NativeRowsAndColumnsAreNotRequantized)
    {
        constexpr int state = 32;
        constexpr std::size_t key_elements = kGDNKeyHeads * state;
        constexpr std::size_t value_elements = kGDNValueHeads * state;
        constexpr std::size_t fused_elements =
            2 * key_elements + value_elements;
        constexpr std::size_t hidden = 32;
        constexpr int world_size = 8;
        constexpr int rank = 3;

        auto loader = std::make_shared<NativeQ8SliceLoader>();
        loader->setLoaded(true);
        loader->setArchitecture("qwen3.5");

        const auto make_q8 = [](std::size_t rows, std::size_t columns, int seed)
        {
            const std::size_t blocks = rows * columns / 32;
            std::vector<std::uint8_t> bytes(blocks * 34);
            for (std::size_t index = 0; index < bytes.size(); ++index)
            {
                bytes[index] = static_cast<std::uint8_t>(
                    (index * 29 + static_cast<std::size_t>(seed)) & 0xffu);
            }
            return std::make_shared<Q8_0Tensor>(
                std::vector<std::size_t>{rows, columns}, bytes);
        };

        auto qkv_source = make_q8(fused_elements, hidden, 17);
        auto output_source = make_q8(hidden, value_elements, 43);
        loader->addTensor("blk.0.attn_qkv.weight", qkv_source);
        loader->addTensor("blk.0.ssm_out.weight", output_source);

        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(
                world_size,
                kAttentionHeads,
                kAttentionKVHeads,
                kDFF,
                kVocab));
        auto manager = std::make_unique<WeightManager>(
            *loader,
            MPIContextFactory::create_mock(rank, world_size),
            nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        Qwen35SchemaFactory schema_factory;
        manager->setWeightShardingConfig(
            schema_factory.getWeightShardingConfig());
        manager->setModelDimensions(
            kAttentionHeads, kAttentionKVHeads, kAttentionHeadDim);
        manager->setGDNDimensions(
            kGDNKeyHeads, kGDNValueHeads, state);
        manager->setTensorParallelConfig(config);

        const auto &participant = config->forRank(rank);
        const GDNHeadAssignment assignment =
            GDNHeadAssignment::fromPartition(
                kGDNKeyHeads,
                kGDNValueHeads,
                participant.head_start,
                participant.head_count,
                config->totalHeads());
        auto qkv = manager->getShardedWeightForAssignment(
            "blk.0.attn_qkv.weight", DeviceId::cpu(), participant, 0);
        auto output = manager->getShardedWeightForAssignment(
            "blk.0.ssm_out.weight", DeviceId::cpu(), participant, 0);
        ASSERT_NE(qkv, nullptr);
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(qkv->native_type(), TensorType::Q8_0);
        ASSERT_EQ(output->native_type(), TensorType::Q8_0);

        const auto fused_rows = [&assignment]()
        {
            std::vector<std::size_t> rows;
            const auto append = [&rows](GDNHeadSpan span, std::size_t base)
            {
                for (int index = 0; index < span.count; ++index)
                {
                    rows.push_back(
                        base + static_cast<std::size_t>(span.start + index));
                }
            };
            const GDNHeadSpan key = assignment.keyElementSpan(state);
            append(key, 0);
            append(key, key_elements);
            for (const GDNHeadSpan value : assignment.valueElementSpans(state))
                append(value, 2 * key_elements);
            return rows;
        }();

        const std::size_t source_qkv_row_bytes =
            qkv_source->size_bytes() / qkv_source->shape()[0];
        const auto *source_qkv_bytes =
            static_cast<const std::uint8_t *>(qkv_source->raw_data());
        const auto *actual_qkv_bytes =
            static_cast<const std::uint8_t *>(qkv->raw_data());
        for (std::size_t local_row = 0; local_row < fused_rows.size(); ++local_row)
        {
            EXPECT_EQ(
                std::memcmp(
                    actual_qkv_bytes + local_row * source_qkv_row_bytes,
                    source_qkv_bytes + fused_rows[local_row] * source_qkv_row_bytes,
                    source_qkv_row_bytes),
                0);
        }

        const auto value_columns = expandSpans(
            assignment.valueElementSpans(state));
        const std::size_t source_output_row_bytes =
            output_source->size_bytes() / output_source->shape()[0];
        const std::size_t actual_output_row_bytes =
            output->size_bytes() / output->shape()[0];
        const std::size_t bytes_per_block = 34;
        const auto *source_output_bytes =
            static_cast<const std::uint8_t *>(output_source->raw_data());
        const auto *actual_output_bytes =
            static_cast<const std::uint8_t *>(output->raw_data());
        ASSERT_EQ(value_columns.size() % 32, 0u);
        for (std::size_t row = 0; row < output_source->shape()[0]; ++row)
        {
            for (std::size_t local_block = 0;
                 local_block < value_columns.size() / 32; ++local_block)
            {
                const std::size_t source_block =
                    value_columns[local_block * 32] / 32;
                EXPECT_EQ(
                    std::memcmp(
                        actual_output_bytes + row * actual_output_row_bytes +
                            local_block * bytes_per_block,
                        source_output_bytes + row * source_output_row_bytes +
                            source_block * bytes_per_block,
                        bytes_per_block),
                    0);
            }
        }
    }

} // namespace llaminar2::test
