/**
 * @file Test__WeightManagerProportional.cpp
 * @brief Integration tests for proportional weight slicing with TensorParallelConfig
 *
 * Tests WeightManager's ability to use TensorParallelConfig for proportional
 * weight slicing instead of the default 1/world_size calculation.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "../../src/v2/loaders/WeightManager.h"
#include "../../src/v2/loaders/ModelLoader.h"
#include "../../src/v2/config/TensorParallelConfig.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/tensors/TensorSlice.h"
#include "../../src/v2/tensors/TensorFactory.h"
#include "../../src/v2/models/qwen/Qwen2Schema.h"
#include <memory>
#include <filesystem>

namespace llaminar2::test
{

    /**
     * @brief Test fixture for proportional weight slicing tests
     *
     * Uses a real model (Qwen2.5-0.5B) to test weight loading with
     * both equal and proportional splits.
     */
    class WeightManagerProportionalTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

            // Model dimensions for Qwen2.5-0.5B
            n_heads_ = 14;
            n_kv_heads_ = 2;
            d_model_ = 896;
            d_ff_ = 4864;
            vocab_size_ = 151936;
            head_dim_ = d_model_ / n_heads_;
        }

        std::string model_path_;
        int n_heads_ = 14;
        int n_kv_heads_ = 2;
        int d_model_ = 896;
        int d_ff_ = 4864;
        int vocab_size_ = 151936;
        int head_dim_ = 64;

        /**
         * @brief Create WeightManager with sharding config for testing
         */
        std::unique_ptr<WeightManager> createWeightManager(
            ModelLoader &loader,
            std::shared_ptr<MPIContext> mpi_ctx,
            WeightDistributionStrategy strategy = WeightDistributionStrategy::SHARDED)
        {
            auto mgr = std::make_unique<WeightManager>(loader, mpi_ctx, nullptr, strategy);
            Qwen2SchemaFactory schema_factory;
            mgr->setWeightShardingConfig(schema_factory.getWeightShardingConfig());
            return mgr;
        }
    };

    /**
     * @brief Test that TensorParallelConfig can be set and retrieved
     */
    TEST_F(WeightManagerProportionalTest, ConfigSetterGetter)
    {
        auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
        TensorFactory factory(*mpi_ctx);
        ModelLoader loader(&factory);

        if (!loader.loadModel(model_path_))
        {
            GTEST_SKIP() << "Model file not found: " << model_path_;
        }

        auto mgr = createWeightManager(loader, mpi_ctx, WeightDistributionStrategy::REPLICATED);

        // Initially no config
        EXPECT_EQ(mgr->tensorParallelConfig(), nullptr);

        // Set config
        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(2, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        mgr->setTensorParallelConfig(config);

        // Config should be set
        EXPECT_NE(mgr->tensorParallelConfig(), nullptr);
        EXPECT_EQ(mgr->tensorParallelConfig()->worldSize(), 2);
    }

    /**
     * @brief Test that equal split produces consistent slices
     */
    TEST_F(WeightManagerProportionalTest, EqualSplit_ProducesConsistentSlices)
    {
        // Create equal split config for 2 devices
        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(2, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        // Verify slices sum to total
        const auto &r0 = config->forRank(0);
        const auto &r1 = config->forRank(1);

        EXPECT_EQ(r0.head_count + r1.head_count, n_heads_);
        EXPECT_EQ(r0.kv_head_count + r1.kv_head_count, n_kv_heads_);
        EXPECT_EQ(r0.d_ff_count + r1.d_ff_count, d_ff_);
        EXPECT_EQ(r0.vocab_count + r1.vocab_count, vocab_size_);

        // For equal split, both should get same amount (or differ by 1 for odd counts)
        EXPECT_LE(std::abs(r0.head_count - r1.head_count), 1);
    }

    /**
     * @brief Test 73%/27% proportional split for attention Q weights
     */
    TEST_F(WeightManagerProportionalTest, ProportionalSplit_AttentionQ_73_27)
    {
        // Create 73%/27% split
        std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
        std::vector<float> fractions = {0.73f, 0.27f};

        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::proportionalSplit(
                devices, fractions, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        EXPECT_TRUE(config->isProportional());

        const auto &r0 = config->forRank(0);
        const auto &r1 = config->forRank(1);

        // Check heads distribution
        EXPECT_EQ(r0.head_count + r1.head_count, n_heads_);

        // CUDA (73%) should get more heads than ROCm (27%)
        EXPECT_GT(r0.head_count, r1.head_count);

        // Verify work fraction reflects actual distribution
        float actual_r0_fraction = static_cast<float>(r0.head_count) / n_heads_;
        EXPECT_NEAR(actual_r0_fraction, 0.73f, 0.1f); // Allow 10% tolerance due to integer rounding

        // Log the actual distribution
        std::cout << "73/27 split: CUDA heads=" << r0.head_count
                  << " (" << (actual_r0_fraction * 100) << "%)"
                  << ", ROCm heads=" << r1.head_count
                  << " (" << ((1.0f - actual_r0_fraction) * 100) << "%)" << std::endl;
    }

    /**
     * @brief Test FFN d_ff alignment to 32 elements
     */
    TEST_F(WeightManagerProportionalTest, SliceAlignment_DFF32)
    {
        std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
        std::vector<float> fractions = {0.73f, 0.27f};

        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::proportionalSplit(
                devices, fractions, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        const auto &r0 = config->forRank(0);
        const auto &r1 = config->forRank(1);

        // d_ff slices should be 32-aligned for quantization blocks
        EXPECT_EQ(r0.d_ff_count % 32, 0) << "CUDA d_ff_count should be 32-aligned";
        // Note: Last rank may not be aligned if total doesn't allow
        // r1.d_ff_count might not be 32-aligned if total - r0 isn't aligned

        // But the sum must equal total
        EXPECT_EQ(r0.d_ff_count + r1.d_ff_count, d_ff_);

        std::cout << "FFN d_ff split: CUDA=" << r0.d_ff_count
                  << " (aligned=" << (r0.d_ff_count % 32 == 0 ? "yes" : "no") << ")"
                  << ", ROCm=" << r1.d_ff_count
                  << " (aligned=" << (r1.d_ff_count % 32 == 0 ? "yes" : "no") << ")" << std::endl;
    }

    /**
     * @brief Test attention slices are head_dim aligned
     */
    TEST_F(WeightManagerProportionalTest, SliceAlignment_HeadDim)
    {
        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(2, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        const auto &r0 = config->forRank(0);
        const auto &r1 = config->forRank(1);

        // Attention dimension = head_count * head_dim
        int r0_attn_dim = r0.head_count * head_dim_;
        int r1_attn_dim = r1.head_count * head_dim_;

        // Sum should equal total attention dimension
        EXPECT_EQ(r0_attn_dim + r1_attn_dim, n_heads_ * head_dim_);

        // Each slice is automatically head_dim-aligned by definition (integer heads)
        EXPECT_EQ(r0_attn_dim % head_dim_, 0);
        EXPECT_EQ(r1_attn_dim % head_dim_, 0);
    }

    /**
     * @brief Test GQA KV heads maintain Q-to-KV ratio
     */
    TEST_F(WeightManagerProportionalTest, GQA_KVHeadsRatio)
    {
        // For Qwen2.5-0.5B: 14 Q heads, 2 KV heads (7:1 ratio)
        // Each KV head serves 7 Q heads

        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(2, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        const auto &r0 = config->forRank(0);
        const auto &r1 = config->forRank(1);

        // Each rank should have proportional KV heads
        EXPECT_EQ(r0.kv_head_count + r1.kv_head_count, n_kv_heads_);

        // For 2-way equal split of 14 Q heads: 7 each
        // For 2-way equal split of 2 KV heads: 1 each
        // This maintains the 7:1 ratio per rank

        int q_to_kv_ratio = n_heads_ / n_kv_heads_; // 7
        EXPECT_EQ(r0.head_count, q_to_kv_ratio * r0.kv_head_count);
        EXPECT_EQ(r1.head_count, q_to_kv_ratio * r1.kv_head_count);
    }

    /**
     * @brief Test weight categorization for different weight types
     */
    TEST_F(WeightManagerProportionalTest, WeightCategorization)
    {
        auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
        TensorFactory factory(*mpi_ctx);
        ModelLoader loader(&factory);

        if (!loader.loadModel(model_path_))
        {
            GTEST_SKIP() << "Model file not found: " << model_path_;
        }

        auto mgr = createWeightManager(loader, mpi_ctx);

        // Q/K/V should be column-parallel
        EXPECT_EQ(mgr->getShardingMode("blk.0.attn_q.weight"), ShardingMode::COLUMN_PARALLEL);
        EXPECT_EQ(mgr->getShardingMode("blk.0.attn_k.weight"), ShardingMode::COLUMN_PARALLEL);
        EXPECT_EQ(mgr->getShardingMode("blk.0.attn_v.weight"), ShardingMode::COLUMN_PARALLEL);

        // Wo should be input-parallel (split input dim, allreduce after)
        // Note: This is INPUT_PARALLEL in Megatron-style TP, not ROW_PARALLEL
        EXPECT_EQ(mgr->getShardingMode("blk.0.attn_output.weight"), ShardingMode::INPUT_PARALLEL);

        // FFN gate/up should be column-parallel
        EXPECT_EQ(mgr->getShardingMode("blk.0.ffn_gate.weight"), ShardingMode::COLUMN_PARALLEL);
        EXPECT_EQ(mgr->getShardingMode("blk.0.ffn_up.weight"), ShardingMode::COLUMN_PARALLEL);

        // FFN down should be input-parallel
        EXPECT_EQ(mgr->getShardingMode("blk.0.ffn_down.weight"), ShardingMode::INPUT_PARALLEL);

        // Norms should be replicated
        EXPECT_EQ(mgr->getShardingMode("blk.0.attn_norm.weight"), ShardingMode::REPLICATE);
    }

    /**
     * @brief Test validation of TensorParallelConfig
     */
    TEST_F(WeightManagerProportionalTest, ConfigValidation)
    {
        auto config = TensorParallelConfig::equalSplit(2, n_heads_, n_kv_heads_, d_ff_, vocab_size_);

        EXPECT_TRUE(config.validate()) << config.validationError();
        EXPECT_TRUE(config.validationError().empty());
    }

    /**
     * @brief Test invalid config detection
     */
    TEST_F(WeightManagerProportionalTest, ConfigValidation_DetectsInvalid)
    {
        // Try to split more ways than we have heads
        EXPECT_THROW(
            TensorParallelConfig::equalSplit(100, n_heads_, n_kv_heads_, d_ff_, vocab_size_),
            std::invalid_argument);

        // Empty devices list
        EXPECT_THROW(
            TensorParallelConfig::proportionalSplit({}, {}, n_heads_, n_kv_heads_, d_ff_, vocab_size_),
            std::invalid_argument);

        // Mismatched devices and fractions
        EXPECT_THROW(
            TensorParallelConfig::proportionalSplit(
                {DeviceId::cuda(0)},
                {0.5f, 0.5f},
                n_heads_, n_kv_heads_, d_ff_, vocab_size_),
            std::invalid_argument);

        // Negative fractions
        EXPECT_THROW(
            TensorParallelConfig::proportionalSplit(
                {DeviceId::cuda(0), DeviceId::rocm(0)},
                {0.7f, -0.3f},
                n_heads_, n_kv_heads_, d_ff_, vocab_size_),
            std::invalid_argument);
    }

    /**
     * @brief Test single device configuration
     */
    TEST_F(WeightManagerProportionalTest, SingleDevice)
    {
        auto config = TensorParallelConfig::singleDevice(
            DeviceId::cuda(0), n_heads_, n_kv_heads_, d_ff_, vocab_size_);

        EXPECT_EQ(config.worldSize(), 1);
        EXPECT_FALSE(config.isProportional());

        const auto &r0 = config.forRank(0);
        EXPECT_EQ(r0.head_count, n_heads_);
        EXPECT_EQ(r0.kv_head_count, n_kv_heads_);
        EXPECT_EQ(r0.d_ff_count, d_ff_);
        EXPECT_EQ(r0.vocab_count, vocab_size_);
    }

    /**
     * @brief Test config toString for debugging
     */
    TEST_F(WeightManagerProportionalTest, ConfigToString)
    {
        std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
        std::vector<float> fractions = {0.73f, 0.27f};

        auto config = TensorParallelConfig::proportionalSplit(
            devices, fractions, n_heads_, n_kv_heads_, d_ff_, vocab_size_);

        std::string str = config.toString();
        EXPECT_FALSE(str.empty());
        EXPECT_NE(str.find("proportional=true"), std::string::npos);
        // Note: DeviceId::toString() uses capitalized names (CUDA:0, ROCm:0)
        EXPECT_NE(str.find("CUDA:0"), std::string::npos);
        EXPECT_NE(str.find("ROCm:0"), std::string::npos);

        std::cout << "Config toString:\n"
                  << str << std::endl;
    }

    /**
     * @brief Test that setting TensorParallelConfig clears cache
     */
    TEST_F(WeightManagerProportionalTest, ConfigClearsCache)
    {
        auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
        TensorFactory factory(*mpi_ctx);
        ModelLoader loader(&factory);

        if (!loader.loadModel(model_path_))
        {
            GTEST_SKIP() << "Model file not found: " << model_path_;
        }

        auto mgr = createWeightManager(loader, mpi_ctx, WeightDistributionStrategy::REPLICATED);

        // Load a weight to populate cache
        auto w1 = mgr->getWeight("blk.0.attn_norm.weight");
        ASSERT_NE(w1, nullptr);
        EXPECT_EQ(mgr->cacheSize(), 1);

        // Set TP config - should clear cache
        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::singleDevice(DeviceId::cpu(), n_heads_, n_kv_heads_, d_ff_, vocab_size_));
        mgr->setTensorParallelConfig(config);

        EXPECT_EQ(mgr->cacheSize(), 0) << "Setting TensorParallelConfig should clear cache";
    }

    /**
     * @brief Test proportional slicing with actual model weights
     */
    TEST_F(WeightManagerProportionalTest, ProportionalSlice_FFN_WithModel)
    {
        // Create 2-rank setup with 73/27 split
        auto mpi_ctx_r0 = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
        auto mpi_ctx_r1 = std::make_shared<MPIContext>(1, 2, MPI_COMM_NULL);

        TensorFactory factory0(*mpi_ctx_r0);
        TensorFactory factory1(*mpi_ctx_r1);

        ModelLoader loader0(&factory0);
        ModelLoader loader1(&factory1);

        if (!loader0.loadModel(model_path_) || !loader1.loadModel(model_path_))
        {
            GTEST_SKIP() << "Model file not found: " << model_path_;
        }

        auto mgr0 = createWeightManager(loader0, mpi_ctx_r0);
        auto mgr1 = createWeightManager(loader1, mpi_ctx_r1);

        // Create proportional config: 73% for rank 0, 27% for rank 1
        std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
        std::vector<float> fractions = {0.73f, 0.27f};

        auto config = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::proportionalSplit(
                devices, fractions, n_heads_, n_kv_heads_, d_ff_, vocab_size_));

        mgr0->setTensorParallelConfig(config);
        mgr1->setTensorParallelConfig(config);

        // Load FFN gate weight (column-parallel)
        const std::string weight_name = "blk.0.ffn_gate.weight";
        auto w0 = mgr0->getWeight(weight_name, DeviceId::cpu());
        auto w1 = mgr1->getWeight(weight_name, DeviceId::cpu());

        ASSERT_NE(w0, nullptr) << "Rank 0 failed to load weight";
        ASSERT_NE(w1, nullptr) << "Rank 1 failed to load weight";

        // Both should be TensorSlice
        auto *slice0 = dynamic_cast<TensorSlice *>(w0.get());
        auto *slice1 = dynamic_cast<TensorSlice *>(w1.get());

        ASSERT_NE(slice0, nullptr) << "Rank 0 weight should be TensorSlice";
        ASSERT_NE(slice1, nullptr) << "Rank 1 weight should be TensorSlice";

        // Check that slices are proportional (not equal)
        // Rank 0 should have more rows since it has 73%
        const auto &shape0 = slice0->inner()->shape();
        const auto &shape1 = slice1->inner()->shape();

        // d_ff = 4864, so 73% ≈ 3550, 27% ≈ 1314 (32-aligned)
        size_t expected_r0_rows = config->forRank(0).d_ff_count;
        size_t expected_r1_rows = config->forRank(1).d_ff_count;

        EXPECT_EQ(shape0[0], expected_r0_rows) << "Rank 0 should have proportional rows";
        EXPECT_EQ(shape1[0], expected_r1_rows) << "Rank 1 should have proportional rows";
        EXPECT_EQ(shape0[0] + shape1[0], d_ff_) << "Total rows should equal d_ff";

        // Rank 0 (73%) should have more than rank 1 (27%)
        EXPECT_GT(shape0[0], shape1[0]) << "Rank 0 (73%) should have more rows than rank 1 (27%)";

        std::cout << "FFN gate weight proportional slicing: "
                  << "rank0=" << shape0[0] << " rows (" << (100.0 * shape0[0] / d_ff_) << "%), "
                  << "rank1=" << shape1[0] << " rows (" << (100.0 * shape1[0] / d_ff_) << "%)"
                  << std::endl;
    }

} // namespace llaminar2::test
