/**
 * @file Test__VocabParallelEmbeddingSharding.cpp
 * @brief Unit tests for vocabulary-parallel embedding sharding infrastructure
 *
 * Tests the non-GPU logic for Megatron-style column-parallel embedding sharding:
 * 1. WeightShardingConfig recognizes token_embd.weight as ColumnParallel + Vocab
 * 2. TensorParallelConfig produces correct vocab_start/vocab_count for each rank
 * 3. WeightSlicer computes correct slice boundaries for token_embd.weight
 * 4. EmbedQ8Repack vocab-range overload produces correct partial repacks
 * 5. PreparedEmbeddingWeights carries vocab_offset and total_vocab metadata
 * 6. Schema embedding stage has TPMode::RowParallel annotation
 * 7. Edge cases: odd vocab sizes, 3-way splits, proportional splits
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

#include "config/TensorParallelConfig.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "kernels/KernelFactory.h"
#include "kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "kernels/common/EmbedQ8Repack.h"
#include "kernels/common/PreparedEmbeddingWeights.h"
#include "loaders/WeightSlicer.h"
#include "models/qwen/Qwen2Schema.h"
#include "models/qwen3/Qwen3Schema.h"
#include "models/qwen35/Qwen35Schema.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "backends/BackendManager.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar::v2::kernels;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__VocabParallelEmbeddingSharding : public ::testing::Test
{
protected:
    // Qwen3.5-4B model parameters (real-world test case)
    static constexpr int QWEN35_HEADS = 16;
    static constexpr int QWEN35_KV_HEADS = 4;
    static constexpr int QWEN35_D_FF = 9216;
    static constexpr int QWEN35_VOCAB = 248320;
    static constexpr int QWEN35_D_MODEL = 2560;
    static constexpr int QWEN35_HEAD_DIM = 160; // d_model / n_heads

    // Smaller params for fast unit tests
    static constexpr int TEST_HEADS = 16;
    static constexpr int TEST_KV_HEADS = 4;
    static constexpr int TEST_D_FF = 1024;
    static constexpr int TEST_VOCAB = 32000;
    static constexpr int TEST_D_MODEL = 64;
    static constexpr int TEST_HEAD_DIM = 4; // 64 / 16

    void SetUp() override
    {
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        if (!getCPUBackend())
            initCPUBackend(0);

        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    ModelDimensions makeDimensions(int n_heads = TEST_HEADS, int n_kv = TEST_KV_HEADS,
                                   int hd = TEST_HEAD_DIM) const
    {
        return ModelDimensions{
            .n_heads = n_heads,
            .n_kv_heads = n_kv,
            .head_dim = static_cast<size_t>(hd)};
    }
};

// ============================================================================
// WeightShardingConfig — token_embd.weight Classification
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen2Schema_EmbeddingIsColumnParallel)
{
    Qwen2SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    auto mode = config.getMode("token_embd.weight");
    EXPECT_EQ(mode, WeightShardingMode::ColumnParallel)
        << "token_embd.weight should be ColumnParallel for vocab-parallel sharding";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen2Schema_EmbeddingDimensionIsVocab)
{
    Qwen2SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    auto dim = config.getDimensionType("token_embd.weight");
    EXPECT_EQ(dim, WeightDimensionType::Vocab)
        << "token_embd.weight should use Vocab dimension for slicing";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen2Schema_EmbeddingModeAndDimension)
{
    Qwen2SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    auto [mode, dim] = config.getModeAndDimension("token_embd.weight");
    EXPECT_EQ(mode, WeightShardingMode::ColumnParallel);
    EXPECT_EQ(dim, WeightDimensionType::Vocab);
}

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen3Schema_EmbeddingIsColumnParallel)
{
    Qwen3SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    auto [mode, dim] = config.getModeAndDimension("token_embd.weight");
    EXPECT_EQ(mode, WeightShardingMode::ColumnParallel);
    EXPECT_EQ(dim, WeightDimensionType::Vocab);
}

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen35Schema_EmbeddingIsColumnParallel)
{
    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    auto [mode, dim] = config.getModeAndDimension("token_embd.weight");
    EXPECT_EQ(mode, WeightShardingMode::ColumnParallel);
    EXPECT_EQ(dim, WeightDimensionType::Vocab);
}

TEST_F(Test__VocabParallelEmbeddingSharding, AllSchemas_LMHeadMatchesEmbedding)
{
    // Both token_embd.weight and output.weight should use Vocab dimension
    for (const auto &schema_name : {"qwen2", "qwen35", "qwen3"})
    {
        WeightShardingConfig config;
        if (std::string(schema_name) == "qwen2")
        {
            Qwen2SchemaFactory f;
            config = f.getWeightShardingConfig();
        }
        else if (std::string(schema_name) == "qwen35")
        {
            Qwen35SchemaFactory f;
            config = f.getWeightShardingConfig();
        }
        else
        {
            Qwen3SchemaFactory f;
            config = f.getWeightShardingConfig();
        }

        auto [emb_mode, emb_dim] = config.getModeAndDimension("token_embd.weight");
        auto [lm_mode, lm_dim] = config.getModeAndDimension("output.weight");

        EXPECT_EQ(emb_dim, WeightDimensionType::Vocab)
            << schema_name << ": token_embd.weight should use Vocab dimension";
        EXPECT_EQ(lm_dim, WeightDimensionType::Vocab)
            << schema_name << ": output.weight should use Vocab dimension";
        EXPECT_EQ(emb_mode, WeightShardingMode::ColumnParallel)
            << schema_name << ": token_embd.weight should be ColumnParallel";
        EXPECT_EQ(lm_mode, WeightShardingMode::ColumnParallel)
            << schema_name << ": output.weight should be ColumnParallel";
    }
}

// ============================================================================
// Schema Embedding Stage — TPMode Annotation
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen2Schema_EmbeddingStageHasRowParallel)
{
    Qwen2SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    EXPECT_EQ(schema.embedding.tp_mode, TPMode::RowParallel)
        << "Embedding stage needs RowParallel annotation for allreduce after lookup";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen35Schema_EmbeddingStageHasRowParallel)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    EXPECT_EQ(schema.embedding.tp_mode, TPMode::RowParallel)
        << "Embedding stage needs RowParallel annotation for allreduce after lookup";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Qwen3Schema_EmbeddingStageHasRowParallel)
{
    Qwen3SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    EXPECT_EQ(schema.embedding.tp_mode, TPMode::RowParallel)
        << "Embedding stage needs RowParallel annotation for allreduce after lookup";
}

// ============================================================================
// TensorParallelConfig — Vocab Split Arithmetic
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, EqualSplit_2Way_VocabRanges)
{
    auto config = TensorParallelConfig::equalSplit(
        2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

    const auto &r0 = config.forRank(0);
    const auto &r1 = config.forRank(1);

    // Each device gets half the vocab
    EXPECT_EQ(r0.vocab_start, 0);
    EXPECT_EQ(r0.vocab_count, TEST_VOCAB / 2);
    EXPECT_EQ(r1.vocab_start, TEST_VOCAB / 2);
    EXPECT_EQ(r1.vocab_count, TEST_VOCAB / 2);

    // Ranges are contiguous
    EXPECT_EQ(r0.vocabEnd(), r1.vocab_start);

    // Total covers full vocab
    EXPECT_EQ(r0.vocab_count + r1.vocab_count, TEST_VOCAB);
    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, EqualSplit_3Way_VocabCoverage)
{
    // 32000 / 3 = 10666.67, so remainder must be distributed
    auto config = TensorParallelConfig::equalSplit(
        3, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

    const auto &r0 = config.forRank(0);
    const auto &r1 = config.forRank(1);
    const auto &r2 = config.forRank(2);

    // Total coverage
    int total = r0.vocab_count + r1.vocab_count + r2.vocab_count;
    EXPECT_EQ(total, TEST_VOCAB);

    // Contiguous ranges
    EXPECT_EQ(r0.vocab_start, 0);
    EXPECT_EQ(r1.vocab_start, r0.vocabEnd());
    EXPECT_EQ(r2.vocab_start, r1.vocabEnd());

    // Each rank has at least floor(32000/3) = 10666
    EXPECT_GE(r0.vocab_count, TEST_VOCAB / 3);
    EXPECT_GE(r1.vocab_count, TEST_VOCAB / 3);
    EXPECT_GE(r2.vocab_count, TEST_VOCAB / 3);
}

TEST_F(Test__VocabParallelEmbeddingSharding, EqualSplit_RealVocab248320_2Way)
{
    // 248320 / 2 = 124160 each (exact division)
    auto config = TensorParallelConfig::equalSplit(
        2, QWEN35_HEADS, QWEN35_KV_HEADS, QWEN35_D_FF, QWEN35_VOCAB);

    const auto &r0 = config.forRank(0);
    const auto &r1 = config.forRank(1);

    EXPECT_EQ(r0.vocab_start, 0);
    EXPECT_EQ(r0.vocab_count, 124160);
    EXPECT_EQ(r1.vocab_start, 124160);
    EXPECT_EQ(r1.vocab_count, 124160);
    EXPECT_EQ(config.totalVocab(), QWEN35_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, ProportionalSplit_VocabFollsFraction)
{
    std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
    std::vector<float> fractions = {0.73f, 0.27f};

    auto config = TensorParallelConfig::proportionalSplit(
        devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

    const auto &cuda = config.forDevice(DeviceId::cuda(0));
    const auto &rocm = config.forDevice(DeviceId::rocm(0));

    // Total coverage
    EXPECT_EQ(cuda.vocab_count + rocm.vocab_count, TEST_VOCAB);
    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);

    // CUDA should have majority
    EXPECT_GT(cuda.vocab_count, rocm.vocab_count);

    // Contiguous ranges
    EXPECT_EQ(cuda.vocab_start, 0);
    EXPECT_EQ(rocm.vocab_start, cuda.vocabEnd());
}

TEST_F(Test__VocabParallelEmbeddingSharding, SingleDevice_FullVocab)
{
    auto config = TensorParallelConfig::singleDevice(
        DeviceId::cuda(0), TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

    const auto &r0 = config.forRank(0);
    EXPECT_EQ(r0.vocab_start, 0);
    EXPECT_EQ(r0.vocab_count, TEST_VOCAB);
    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, EqualSplit_DuplicateCPUDevices_PreservesRankVocabOffsets)
{
    auto config = TensorParallelConfig::equalSplit(
        2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB,
        std::vector<DeviceId>{DeviceId::cpu(), DeviceId::cpu()});

    const auto &r0 = config.forRank(0);
    const auto &r1 = config.forRank(1);

    EXPECT_EQ(r0.device, DeviceId::cpu());
    EXPECT_EQ(r1.device, DeviceId::cpu());
    EXPECT_EQ(r0.vocab_start, 0);
    EXPECT_EQ(r1.vocab_start, r0.vocabEnd());
    EXPECT_EQ(r0.vocab_count + r1.vocab_count, TEST_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, CPUEmbeddingKernel_ZeroesTokensOutsideLocalShard)
{
    constexpr int kLocalVocab = 4;
    constexpr int kDModel = 3;
    constexpr int kVocabOffset = 4;
    const int token_ids[] = {5, 1, 7};

    FP32Tensor embed_table({kLocalVocab, kDModel});
    for (int row = 0; row < kLocalVocab; ++row)
    {
        for (int col = 0; col < kDModel; ++col)
        {
            embed_table.mutable_data()[row * kDModel + col] = 10.0f * row + static_cast<float>(col + 1);
        }
    }

    FP32Tensor output({3, kDModel});
    std::fill(output.mutable_data(), output.mutable_data() + output.numel(), -1.0f);

    CPUEmbeddingKernelT<FP32Tensor> kernel;
    kernel.setVocabRange(kVocabOffset, kLocalVocab);

    ASSERT_TRUE(kernel.apply_tensor(&embed_table, token_ids, 3, kDModel, &output));

    const float *out = output.data();
    EXPECT_FLOAT_EQ(out[0], 11.0f);
    EXPECT_FLOAT_EQ(out[1], 12.0f);
    EXPECT_FLOAT_EQ(out[2], 13.0f);

    EXPECT_FLOAT_EQ(out[3], 0.0f);
    EXPECT_FLOAT_EQ(out[4], 0.0f);
    EXPECT_FLOAT_EQ(out[5], 0.0f);

    EXPECT_FLOAT_EQ(out[6], 31.0f);
    EXPECT_FLOAT_EQ(out[7], 32.0f);
    EXPECT_FLOAT_EQ(out[8], 33.0f);
}

// ============================================================================
// WeightSlicer — computeSliceForAssignment for token_embd.weight
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, Slicer_TokenEmbdSlice_2Way)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB));

    Qwen2SchemaFactory factory;
    WeightSlicer slicer(makeDimensions(), factory.getWeightShardingConfig(), tp);

    // Rank 0: first half
    auto s0 = slicer.computeSliceForAssignment("token_embd.weight", TEST_VOCAB, tp->forRank(0));
    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, static_cast<size_t>(TEST_VOCAB / 2));

    // Rank 1: second half
    auto s1 = slicer.computeSliceForAssignment("token_embd.weight", TEST_VOCAB, tp->forRank(1));
    EXPECT_EQ(s1.start, static_cast<size_t>(TEST_VOCAB / 2));
    EXPECT_EQ(s1.count, static_cast<size_t>(TEST_VOCAB / 2));

    // Full coverage
    EXPECT_EQ(s0.count + s1.count, static_cast<size_t>(TEST_VOCAB));
}

TEST_F(Test__VocabParallelEmbeddingSharding, Slicer_TokenEmbdSlice_MatchesOutputWeight)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB));

    Qwen2SchemaFactory factory;
    WeightSlicer slicer(makeDimensions(), factory.getWeightShardingConfig(), tp);

    // Both token_embd.weight and output.weight use Vocab dimension
    auto emb = slicer.computeSliceForAssignment("token_embd.weight", TEST_VOCAB, tp->forRank(0));
    auto lm = slicer.computeSliceForAssignment("output.weight", TEST_VOCAB, tp->forRank(0));

    // Same vocab dimension → same slice
    EXPECT_EQ(emb.start, lm.start);
    EXPECT_EQ(emb.count, lm.count);
}

TEST_F(Test__VocabParallelEmbeddingSharding, Slicer_TokenEmbdSlice_Qwen35Schema)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, QWEN35_HEADS, QWEN35_KV_HEADS, QWEN35_D_FF, QWEN35_VOCAB));

    Qwen35SchemaFactory factory;

    ModelDimensions dims{
        .n_heads = QWEN35_HEADS,
        .n_kv_heads = QWEN35_KV_HEADS,
        .head_dim = QWEN35_HEAD_DIM};

    WeightSlicer slicer(dims, factory.getWeightShardingConfig(), tp);

    auto s0 = slicer.computeSliceForAssignment("token_embd.weight", QWEN35_VOCAB, tp->forRank(0));
    auto s1 = slicer.computeSliceForAssignment("token_embd.weight", QWEN35_VOCAB, tp->forRank(1));

    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, 124160u);
    EXPECT_EQ(s1.start, 124160u);
    EXPECT_EQ(s1.count, 124160u);
}

// ============================================================================
// EmbedQ8Repack — Vocab-Range Partial Repack
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_FullVocab_MatchesStandard)
{
    constexpr size_t kVocab = 128;
    constexpr size_t kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});

    // Full repack via standard path
    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);

    // Full repack via vocab-range path (start=0, count=full)
    auto ranged = repackEmbeddingToQ8(tensor.get(), kDModel, 0, kVocab);

    EXPECT_EQ(full.vocab_size, ranged.vocab_size);
    EXPECT_EQ(full.blocks_per_row, ranged.blocks_per_row);
    EXPECT_EQ(full.total_blocks, ranged.total_blocks);
    EXPECT_EQ(full.byte_size, ranged.byte_size);
    EXPECT_EQ(full.data, ranged.data) << "Full range should produce identical bytes";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_HalfVocab_CorrectSize)
{
    constexpr size_t kVocab = 256;
    constexpr size_t kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});

    auto first_half = repackEmbeddingToQ8(tensor.get(), kDModel, 0, kVocab / 2);
    auto second_half = repackEmbeddingToQ8(tensor.get(), kDModel, kVocab / 2, kVocab / 2);

    // Each half should have half the vocab
    EXPECT_EQ(first_half.vocab_size, kVocab / 2);
    EXPECT_EQ(second_half.vocab_size, kVocab / 2);

    // Blocks per row should be the same
    EXPECT_EQ(first_half.blocks_per_row, second_half.blocks_per_row);

    // Total blocks should be half
    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);
    EXPECT_EQ(first_half.total_blocks + second_half.total_blocks, full.total_blocks);
    EXPECT_EQ(first_half.byte_size + second_half.byte_size, full.byte_size);
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_HalfVocab_DataMatchesFullRows)
{
    constexpr size_t kVocab = 128;
    constexpr size_t kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});

    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);
    auto first_half = repackEmbeddingToQ8(tensor.get(), kDModel, 0, kVocab / 2);

    // First half data should match the first half of full data
    size_t half_bytes = first_half.byte_size;
    ASSERT_LE(half_bytes, full.byte_size);
    EXPECT_EQ(
        std::vector<uint8_t>(full.data.begin(), full.data.begin() + half_bytes),
        first_half.data)
        << "First half repack should match first half of full repack";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_SecondHalf_DataMatchesFullRows)
{
    constexpr size_t kVocab = 128;
    constexpr size_t kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});

    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);
    auto second_half = repackEmbeddingToQ8(tensor.get(), kDModel, kVocab / 2, kVocab / 2);

    // Second half data should match the second half of full data
    size_t half_bytes = second_half.byte_size;
    ASSERT_LE(half_bytes * 2, full.byte_size + 1); // Allow rounding
    EXPECT_EQ(
        std::vector<uint8_t>(full.data.begin() + half_bytes, full.data.end()),
        second_half.data)
        << "Second half repack should match second half of full repack";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_OutOfRange_Throws)
{
    constexpr size_t kVocab = 128;
    constexpr size_t kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});

    // Exceeding total vocab should throw
    EXPECT_THROW(
        repackEmbeddingToQ8(tensor.get(), kDModel, 64, 128),
        std::runtime_error)
        << "Vocab range [64, 192) exceeds total vocab 128";

    EXPECT_THROW(
        repackEmbeddingToQ8(tensor.get(), kDModel, 129, 1),
        std::runtime_error)
        << "Start beyond vocab should throw";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_SingleRow_Works)
{
    constexpr size_t kVocab = 128;
    constexpr size_t kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});

    auto single = repackEmbeddingToQ8(tensor.get(), kDModel, 42, 1);
    EXPECT_EQ(single.vocab_size, 1u);
    EXPECT_GT(single.byte_size, 0u);
}

// ============================================================================
// PreparedEmbeddingWeights — Vocab Offset & Total Vocab Metadata
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, PreparedWeights_UnshardedDefaults)
{
    constexpr size_t kVocab = 128;
    constexpr int kDModel = 64;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, static_cast<size_t>(kDModel)});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModel, DeviceId::cpu());

    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->weights, nullptr);

    // Default (unsharded): vocab_offset=0, total_vocab matches vocab_size
    EXPECT_EQ(handle->weights->vocab_offset, 0u);
    // total_vocab should equal vocab_size when unsharded
    EXPECT_EQ(handle->weights->total_vocab, handle->weights->vocab_size);
}

TEST_F(Test__VocabParallelEmbeddingSharding, PreparedWeights_ShardedMetadata)
{
    constexpr size_t kLocalVocab = 64; // Each shard holds 64 rows
    constexpr int kDModel = 64;
    constexpr size_t kVocabOffset = 64;
    constexpr size_t kTotalVocab = 128;

    auto tensor = TestTensorFactory::createQ8_0Random({kLocalVocab, static_cast<size_t>(kDModel)});
    auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor.get(), kDModel, DeviceId::cpu(), kVocabOffset, kTotalVocab);

    ASSERT_NE(handle, nullptr);
    ASSERT_NE(handle->weights, nullptr);

    EXPECT_EQ(handle->weights->vocab_offset, kVocabOffset);
    EXPECT_EQ(handle->weights->total_vocab, kTotalVocab);
    EXPECT_EQ(handle->weights->vocab_size, kLocalVocab);
}

TEST_F(Test__VocabParallelEmbeddingSharding, PreparedWeights_TwoShards_ComplementaryRanges)
{
    constexpr size_t kTotalVocab = 128;
    constexpr size_t kLocalVocab = 64;
    constexpr int kDModel = 64;

    auto tensor0 = TestTensorFactory::createQ8_0Random({kLocalVocab, static_cast<size_t>(kDModel)}, 42);
    auto tensor1 = TestTensorFactory::createQ8_0Random({kLocalVocab, static_cast<size_t>(kDModel)}, 99);

    auto *handle0 = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor0.get(), kDModel, DeviceId::cpu(), 0, kTotalVocab);
    auto *handle1 = KernelFactory::getOrCreatePreparedEmbeddingWeights(
        tensor1.get(), kDModel, DeviceId::cpu(), kLocalVocab, kTotalVocab);

    ASSERT_NE(handle0, nullptr);
    ASSERT_NE(handle1, nullptr);

    // Shard 0: [0, 64)
    EXPECT_EQ(handle0->weights->vocab_offset, 0u);
    EXPECT_EQ(handle0->weights->vocab_size, kLocalVocab);
    EXPECT_EQ(handle0->weights->total_vocab, kTotalVocab);

    // Shard 1: [64, 128)
    EXPECT_EQ(handle1->weights->vocab_offset, kLocalVocab);
    EXPECT_EQ(handle1->weights->vocab_size, kLocalVocab);
    EXPECT_EQ(handle1->weights->total_vocab, kTotalVocab);

    // Ranges are complementary
    EXPECT_EQ(handle0->weights->vocab_offset + handle0->weights->vocab_size,
              handle1->weights->vocab_offset);
    EXPECT_EQ(handle0->weights->vocab_size + handle1->weights->vocab_size,
              kTotalVocab);
}

// ============================================================================
// PreparedEmbeddingWeights — Move Semantics
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, PreparedWeights_MovePreservesShardingMetadata)
{
    PreparedEmbeddingWeights w;
    w.vocab_size = 64;
    w.vocab_offset = 128;
    w.total_vocab = 256;
    w.d_model = 2560;
    w.blocks_per_row = 80;

    PreparedEmbeddingWeights moved(std::move(w));

    EXPECT_EQ(moved.vocab_size, 64u);
    EXPECT_EQ(moved.vocab_offset, 128u);
    EXPECT_EQ(moved.total_vocab, 256u);
    EXPECT_EQ(moved.d_model, 2560);
    EXPECT_EQ(moved.blocks_per_row, 80u);
}

// ============================================================================
// WeightShardingConfig — Unmatched Weights Default to Replicate
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, ShardingConfig_NormsStillReplicate)
{
    Qwen2SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    // Norms should still be Replicate (not affected by embedding change)
    EXPECT_EQ(config.getMode("blk.0.attn_norm.weight"), WeightShardingMode::Replicate);
    EXPECT_EQ(config.getMode("blk.0.ffn_norm.weight"), WeightShardingMode::Replicate);
    EXPECT_EQ(config.getMode("output_norm.weight"), WeightShardingMode::Replicate);
}

TEST_F(Test__VocabParallelEmbeddingSharding, ShardingConfig_AttentionUnchanged)
{
    Qwen2SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    // Attention weights should be unaffected
    auto [q_mode, q_dim] = config.getModeAndDimension("blk.0.attn_q.weight");
    EXPECT_EQ(q_mode, WeightShardingMode::ColumnParallel);
    EXPECT_EQ(q_dim, WeightDimensionType::Heads);

    auto [wo_mode, wo_dim] = config.getModeAndDimension("blk.0.attn_output.weight");
    EXPECT_EQ(wo_mode, WeightShardingMode::InputParallel);
    EXPECT_EQ(wo_dim, WeightDimensionType::Heads);
}

// ============================================================================
// Edge Cases — Odd Vocab Sizes
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, EqualSplit_OddVocab_NoCoverage)
{
    // 32001 / 2 = 16000 + 16001 (one rank gets the extra)
    constexpr int ODD_VOCAB = 32001;
    auto config = TensorParallelConfig::equalSplit(
        2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, ODD_VOCAB);

    const auto &r0 = config.forRank(0);
    const auto &r1 = config.forRank(1);

    EXPECT_EQ(r0.vocab_count + r1.vocab_count, ODD_VOCAB);
    EXPECT_EQ(r0.vocabEnd(), r1.vocab_start) << "No gap between vocab ranges";
    EXPECT_LE(std::abs(r0.vocab_count - r1.vocab_count), 1)
        << "Imbalance should be at most 1";
}

TEST_F(Test__VocabParallelEmbeddingSharding, EqualSplit_PrimeVocab_3Way)
{
    constexpr int PRIME_VOCAB = 32003; // Prime number
    auto config = TensorParallelConfig::equalSplit(
        3, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, PRIME_VOCAB);

    int total = 0;
    for (int i = 0; i < 3; ++i)
    {
        const auto &r = config.forRank(i);
        total += r.vocab_count;
        if (i > 0)
        {
            EXPECT_EQ(config.forRank(i - 1).vocabEnd(), r.vocab_start)
                << "Gap between rank " << (i - 1) << " and " << i;
        }
    }
    EXPECT_EQ(total, PRIME_VOCAB);
}

// ============================================================================
// Slicer — Qwen3Schema (inherits Qwen2 sharding behavior)
// ============================================================================

TEST_F(Test__VocabParallelEmbeddingSharding, Slicer_TokenEmbdSlice_Qwen3Schema)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB));

    Qwen3SchemaFactory factory;
    WeightSlicer slicer(makeDimensions(), factory.getWeightShardingConfig(), tp);

    auto s0 = slicer.computeSliceForAssignment("token_embd.weight", TEST_VOCAB, tp->forRank(0));
    auto s1 = slicer.computeSliceForAssignment("token_embd.weight", TEST_VOCAB, tp->forRank(1));

    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, static_cast<size_t>(TEST_VOCAB / 2));
    EXPECT_EQ(s1.start, static_cast<size_t>(TEST_VOCAB / 2));
    EXPECT_EQ(s1.count, static_cast<size_t>(TEST_VOCAB / 2));
}

// ============================================================================
// Multi-Degree Sharding — TP=2,4,8,16 Vocab Split & Slicer Verification
// ============================================================================

// Parameters for multi-degree tests: need enough heads/KV-heads for 16-way.
// 32 Q heads / 16 KV heads allows clean division up to TP=16.
static constexpr int MD_HEADS = 32;
static constexpr int MD_KV_HEADS = 16;
static constexpr int MD_D_FF = 2048;
static constexpr int MD_HEAD_DIM = 8;

// ---- TP=2 ----

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP2_VocabSplit)
{
    constexpr int N = 2;
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB);

    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);

    int running_start = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        EXPECT_EQ(a.vocab_start, running_start)
            << "Rank " << r << " vocab_start mismatch";
        EXPECT_EQ(a.vocab_count, TEST_VOCAB / N)
            << "Rank " << r << " vocab_count mismatch (expected " << TEST_VOCAB / N << ")";
        running_start += a.vocab_count;
    }
    EXPECT_EQ(running_start, TEST_VOCAB) << "Vocab ranges did not cover full vocab";
}

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP2_SlicerBoundaries)
{
    constexpr int N = 2;
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB));

    Qwen2SchemaFactory factory;
    ModelDimensions dims{.n_heads = MD_HEADS, .n_kv_heads = MD_KV_HEADS, .head_dim = MD_HEAD_DIM};
    WeightSlicer slicer(dims, factory.getWeightShardingConfig(), tp);

    size_t running = 0;
    for (int r = 0; r < N; ++r)
    {
        auto s = slicer.computeSliceForAssignment(
            "token_embd.weight", TEST_VOCAB, tp->forRank(r));
        EXPECT_EQ(s.start, running) << "Rank " << r;
        EXPECT_EQ(s.count, static_cast<size_t>(TEST_VOCAB / N)) << "Rank " << r;
        running += s.count;
    }
    EXPECT_EQ(running, static_cast<size_t>(TEST_VOCAB));
}

// ---- TP=4 ----

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP4_VocabSplit)
{
    constexpr int N = 4;
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB);

    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);
    EXPECT_EQ(config.worldSize(), N);

    int total_vocab = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        if (r > 0)
        {
            EXPECT_EQ(config.forRank(r - 1).vocabEnd(), a.vocab_start)
                << "Gap between rank " << (r - 1) << " and " << r;
        }
        EXPECT_EQ(a.vocab_count, TEST_VOCAB / N)
            << "Rank " << r << " vocab_count (expected " << TEST_VOCAB / N << ")";
        total_vocab += a.vocab_count;
    }
    EXPECT_EQ(total_vocab, TEST_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP4_SlicerBoundaries)
{
    constexpr int N = 4;
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB));

    Qwen2SchemaFactory factory;
    ModelDimensions dims{.n_heads = MD_HEADS, .n_kv_heads = MD_KV_HEADS, .head_dim = MD_HEAD_DIM};
    WeightSlicer slicer(dims, factory.getWeightShardingConfig(), tp);

    size_t running = 0;
    for (int r = 0; r < N; ++r)
    {
        auto s = slicer.computeSliceForAssignment(
            "token_embd.weight", TEST_VOCAB, tp->forRank(r));
        EXPECT_EQ(s.start, running) << "Rank " << r;
        EXPECT_EQ(s.count, static_cast<size_t>(TEST_VOCAB / N)) << "Rank " << r;

        // Also verify output.weight gets same slice (tied weights, same Vocab dim)
        auto lm = slicer.computeSliceForAssignment(
            "output.weight", TEST_VOCAB, tp->forRank(r));
        EXPECT_EQ(lm.start, s.start) << "output.weight mismatch at rank " << r;
        EXPECT_EQ(lm.count, s.count) << "output.weight mismatch at rank " << r;

        running += s.count;
    }
    EXPECT_EQ(running, static_cast<size_t>(TEST_VOCAB));
}

// ---- TP=8 ----

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP8_VocabSplit)
{
    constexpr int N = 8;
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB);

    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);
    EXPECT_EQ(config.worldSize(), N);

    // 32000 / 8 = 4000 each — divides evenly
    int total_vocab = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        EXPECT_EQ(a.vocab_start, r * (TEST_VOCAB / N))
            << "Rank " << r << " vocab_start";
        EXPECT_EQ(a.vocab_count, TEST_VOCAB / N)
            << "Rank " << r << " vocab_count";
        total_vocab += a.vocab_count;
    }
    EXPECT_EQ(total_vocab, TEST_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP8_SlicerBoundaries)
{
    constexpr int N = 8;
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB));

    Qwen2SchemaFactory factory;
    ModelDimensions dims{.n_heads = MD_HEADS, .n_kv_heads = MD_KV_HEADS, .head_dim = MD_HEAD_DIM};
    WeightSlicer slicer(dims, factory.getWeightShardingConfig(), tp);

    size_t running = 0;
    for (int r = 0; r < N; ++r)
    {
        auto s = slicer.computeSliceForAssignment(
            "token_embd.weight", TEST_VOCAB, tp->forRank(r));
        EXPECT_EQ(s.start, running) << "Rank " << r;
        EXPECT_EQ(s.count, static_cast<size_t>(TEST_VOCAB / N)) << "Rank " << r;
        running += s.count;
    }
    EXPECT_EQ(running, static_cast<size_t>(TEST_VOCAB));
}

// ---- TP=16 ----

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP16_VocabSplit)
{
    constexpr int N = 16;
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB);

    EXPECT_EQ(config.totalVocab(), TEST_VOCAB);
    EXPECT_EQ(config.worldSize(), N);

    // 32000 / 16 = 2000 each — divides evenly
    int total_vocab = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        EXPECT_EQ(a.vocab_start, r * (TEST_VOCAB / N))
            << "Rank " << r << " vocab_start";
        EXPECT_EQ(a.vocab_count, TEST_VOCAB / N)
            << "Rank " << r << " vocab_count";
        total_vocab += a.vocab_count;
    }
    EXPECT_EQ(total_vocab, TEST_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_TP16_SlicerBoundaries)
{
    constexpr int N = 16;
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(N, MD_HEADS, MD_KV_HEADS, MD_D_FF, TEST_VOCAB));

    Qwen2SchemaFactory factory;
    ModelDimensions dims{.n_heads = MD_HEADS, .n_kv_heads = MD_KV_HEADS, .head_dim = MD_HEAD_DIM};
    WeightSlicer slicer(dims, factory.getWeightShardingConfig(), tp);

    size_t running = 0;
    for (int r = 0; r < N; ++r)
    {
        auto s = slicer.computeSliceForAssignment(
            "token_embd.weight", TEST_VOCAB, tp->forRank(r));
        EXPECT_EQ(s.start, running) << "Rank " << r;
        EXPECT_EQ(s.count, static_cast<size_t>(TEST_VOCAB / N)) << "Rank " << r;
        running += s.count;
    }
    EXPECT_EQ(running, static_cast<size_t>(TEST_VOCAB));
}

// ---- Real Qwen3.5-4B vocab=248320 at various TP degrees ----

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_RealVocab_TP4)
{
    constexpr int N = 4;
    // 248320 / 4 = 62080 each
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, QWEN35_VOCAB);

    int total = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        EXPECT_EQ(a.vocab_start, r * 62080) << "Rank " << r;
        EXPECT_EQ(a.vocab_count, 62080) << "Rank " << r;
        total += a.vocab_count;
    }
    EXPECT_EQ(total, QWEN35_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_RealVocab_TP8)
{
    constexpr int N = 8;
    // 248320 / 8 = 31040 each
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, QWEN35_VOCAB);

    int total = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        EXPECT_EQ(a.vocab_start, r * 31040) << "Rank " << r;
        EXPECT_EQ(a.vocab_count, 31040) << "Rank " << r;
        total += a.vocab_count;
    }
    EXPECT_EQ(total, QWEN35_VOCAB);
}

TEST_F(Test__VocabParallelEmbeddingSharding, MultiDegree_RealVocab_TP16)
{
    constexpr int N = 16;
    // 248320 / 16 = 15520 each
    auto config = TensorParallelConfig::equalSplit(
        N, MD_HEADS, MD_KV_HEADS, MD_D_FF, QWEN35_VOCAB);

    int total = 0;
    for (int r = 0; r < N; ++r)
    {
        const auto &a = config.forRank(r);
        EXPECT_EQ(a.vocab_start, r * 15520) << "Rank " << r;
        EXPECT_EQ(a.vocab_count, 15520) << "Rank " << r;
        total += a.vocab_count;
    }
    EXPECT_EQ(total, QWEN35_VOCAB);
}

// ---- EmbedQ8Repack with N-way splits ----

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_4WaySplit_ReconstructsFull)
{
    constexpr size_t kVocab = 256;
    constexpr size_t kDModel = 64;
    constexpr int N = 4;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});
    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);

    // Repack each quarter and concatenate
    std::vector<uint8_t> reconstructed;
    for (int r = 0; r < N; ++r)
    {
        size_t start = r * (kVocab / N);
        size_t count = kVocab / N;
        auto shard = repackEmbeddingToQ8(tensor.get(), kDModel, start, count);

        EXPECT_EQ(shard.vocab_size, kVocab / N) << "Shard " << r;
        EXPECT_EQ(shard.blocks_per_row, full.blocks_per_row) << "Shard " << r;
        reconstructed.insert(reconstructed.end(), shard.data.begin(), shard.data.end());
    }

    EXPECT_EQ(reconstructed.size(), full.byte_size);
    EXPECT_EQ(reconstructed, full.data)
        << "Concatenation of 4 shards should reconstruct the full repack";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_8WaySplit_ReconstructsFull)
{
    constexpr size_t kVocab = 256;
    constexpr size_t kDModel = 64;
    constexpr int N = 8;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});
    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);

    std::vector<uint8_t> reconstructed;
    for (int r = 0; r < N; ++r)
    {
        size_t start = r * (kVocab / N);
        size_t count = kVocab / N;
        auto shard = repackEmbeddingToQ8(tensor.get(), kDModel, start, count);

        EXPECT_EQ(shard.vocab_size, kVocab / N) << "Shard " << r;
        reconstructed.insert(reconstructed.end(), shard.data.begin(), shard.data.end());
    }

    EXPECT_EQ(reconstructed.size(), full.byte_size);
    EXPECT_EQ(reconstructed, full.data)
        << "Concatenation of 8 shards should reconstruct the full repack";
}

TEST_F(Test__VocabParallelEmbeddingSharding, Repack_16WaySplit_ReconstructsFull)
{
    constexpr size_t kVocab = 256;
    constexpr size_t kDModel = 64;
    constexpr int N = 16;

    auto tensor = TestTensorFactory::createQ8_0Random({kVocab, kDModel});
    auto full = repackEmbeddingToQ8(tensor.get(), kDModel);

    std::vector<uint8_t> reconstructed;
    for (int r = 0; r < N; ++r)
    {
        size_t start = r * (kVocab / N);
        size_t count = kVocab / N;
        auto shard = repackEmbeddingToQ8(tensor.get(), kDModel, start, count);

        EXPECT_EQ(shard.vocab_size, kVocab / N) << "Shard " << r;
        reconstructed.insert(reconstructed.end(), shard.data.begin(), shard.data.end());
    }

    EXPECT_EQ(reconstructed.size(), full.byte_size);
    EXPECT_EQ(reconstructed, full.data)
        << "Concatenation of 16 shards should reconstruct the full repack";
}

// ---- PreparedEmbeddingWeights metadata at various TP degrees ----

TEST_F(Test__VocabParallelEmbeddingSharding, PreparedWeights_4Shards_Metadata)
{
    constexpr size_t kTotalVocab = 256;
    constexpr int kDModel = 64;
    constexpr int N = 4;
    constexpr size_t kLocalVocab = kTotalVocab / N;

    // Keep all tensors alive to prevent allocator pointer reuse (cache key = raw pointer)
    std::vector<std::unique_ptr<TensorBase>> tensors;
    for (int r = 0; r < N; ++r)
    {
        tensors.push_back(TestTensorFactory::createQ8_0Random(
            {kLocalVocab, static_cast<size_t>(kDModel)}, 42 + r));
        size_t offset = r * kLocalVocab;

        auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
            tensors.back().get(), kDModel, DeviceId::cpu(), offset, kTotalVocab);

        ASSERT_NE(handle, nullptr) << "Shard " << r;
        ASSERT_NE(handle->weights, nullptr) << "Shard " << r;
        EXPECT_EQ(handle->weights->vocab_size, kLocalVocab) << "Shard " << r;
        EXPECT_EQ(handle->weights->vocab_offset, offset) << "Shard " << r;
        EXPECT_EQ(handle->weights->total_vocab, kTotalVocab) << "Shard " << r;
        EXPECT_EQ(handle->weights->d_model, kDModel) << "Shard " << r;
    }
}

TEST_F(Test__VocabParallelEmbeddingSharding, PreparedWeights_16Shards_ContiguousRanges)
{
    constexpr size_t kTotalVocab = 256;
    constexpr int kDModel = 64;
    constexpr int N = 16;
    constexpr size_t kLocalVocab = kTotalVocab / N; // 16 rows each

    struct ShardInfo
    {
        size_t offset;
        size_t size;
    };
    std::vector<ShardInfo> shards;

    // Keep all tensors alive to prevent allocator pointer reuse (cache key = raw pointer)
    std::vector<std::unique_ptr<TensorBase>> tensors;
    for (int r = 0; r < N; ++r)
    {
        tensors.push_back(TestTensorFactory::createQ8_0Random(
            {kLocalVocab, static_cast<size_t>(kDModel)}, 100 + r));
        size_t offset = r * kLocalVocab;

        auto *handle = KernelFactory::getOrCreatePreparedEmbeddingWeights(
            tensors.back().get(), kDModel, DeviceId::cpu(), offset, kTotalVocab);

        ASSERT_NE(handle, nullptr) << "Shard " << r;
        ShardInfo si;
        si.offset = handle->weights->vocab_offset;
        si.size = handle->weights->vocab_size;
        shards.push_back(si);
    }

    // Verify contiguous coverage
    for (int r = 1; r < N; ++r)
    {
        EXPECT_EQ(shards[r - 1].offset + shards[r - 1].size, shards[r].offset)
            << "Gap between shard " << (r - 1) << " and " << r;
    }

    // Verify total coverage
    size_t total = 0;
    for (const auto &s : shards)
        total += s.size;
    EXPECT_EQ(total, kTotalVocab);
}
