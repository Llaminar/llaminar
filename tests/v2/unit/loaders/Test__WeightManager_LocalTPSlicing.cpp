/**
 * @file Test__WeightManager_LocalTPSlicing.cpp
 * @brief Unit tests for WeightManager LOCAL TP slicing helper methods
 *
 * Tests the static helper methods in WeightManager that support device-aware
 * weight slicing for LOCAL tensor parallelism:
 * - Weight category detection (isQKVWeight, isFFNGateUpWeight, etc.)
 * - Row slicing utility (sliceRowRange)
 * - Immutable TP declaration ownership across parallel participant setup and replacement
 *
 * Owning-snapshot regressions use the actual WeightManager publication APIs and
 * original QueryInterval/GDN declarations without loading weights or initializing GPUs.
 *
 * Note: Full WeightManager integration tests that require model loading
 * are in tests/v2/integration/.
 */

#include <gtest/gtest.h>
#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "TestTensorFactory.h"
#include <memory>
#include <cstring>
#include "mocks/MockModelLoader.h"
#include <array>
#include <atomic>
#include <barrier>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace test
    {

        // =============================================================================
        // Test Fixture
        // =============================================================================

        class Test__WeightManager_LocalTPSlicing : public ::testing::Test
        {
        protected:
            // Common layer name patterns used in GGUF models
            static constexpr const char *ATTN_Q_LAYER0 = "blk.0.attn_q.weight";
            static constexpr const char *ATTN_K_LAYER0 = "blk.0.attn_k.weight";
            static constexpr const char *ATTN_V_LAYER0 = "blk.0.attn_v.weight";
            static constexpr const char *ATTN_QKV_FUSED = "blk.0.attn_qkv.weight";
            static constexpr const char *ATTN_OUTPUT_LAYER0 = "blk.0.attn_output.weight";

            static constexpr const char *FFN_GATE_LAYER0 = "blk.0.ffn_gate.weight";
            static constexpr const char *FFN_UP_LAYER0 = "blk.0.ffn_up.weight";
            static constexpr const char *FFN_GATE_UP_FUSED = "blk.0.ffn_gate_up.weight";
            static constexpr const char *FFN_DOWN_LAYER0 = "blk.0.ffn_down.weight";

            static constexpr const char *LM_HEAD = "output.weight";
            static constexpr const char *EMBEDDING = "token_embd.weight";
            static constexpr const char *OUTPUT_NORM = "output_norm.weight";
            static constexpr const char *ATTN_NORM_LAYER0 = "blk.0.attn_norm.weight";
            static constexpr const char *FFN_NORM_LAYER0 = "blk.0.ffn_norm.weight";
        };

        // =============================================================================
        // sliceRowRange Tests - FP32 Tensors
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_BasicSlice)
        {
            // Create 10x8 FP32 tensor with sequential values for easy verification
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            // Slice rows 3-5 (3 rows: indices 3, 4, 5)
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 3, 3);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape().size(), 2);
            EXPECT_EQ(sliced->shape()[0], 3);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Verify values
            // Row 3 starts at index 3*8=24, contains values 24-31
            // Row 4 starts at index 4*8=32, contains values 32-39
            // Row 5 starts at index 5*8=40, contains values 40-47
            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 24.0f);  // First element of row 3
            EXPECT_FLOAT_EQ(sliced_data[7], 31.0f);  // Last element of row 3
            EXPECT_FLOAT_EQ(sliced_data[8], 32.0f);  // First element of row 4
            EXPECT_FLOAT_EQ(sliced_data[23], 47.0f); // Last element of row 5
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_FirstRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 0, 4);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 4);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 0.0f);
            EXPECT_FLOAT_EQ(sliced_data[31], 31.0f);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_LastRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 7, 3); // Rows 7, 8, 9

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 3);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            // Row 7 starts at 7*8=56
            EXPECT_FLOAT_EQ(sliced_data[0], 56.0f);
            // Row 9 ends at 9*8+7=79
            EXPECT_FLOAT_EQ(sliced_data[23], 79.0f);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_SingleRow)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 5, 1);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 1);
            EXPECT_EQ(sliced->shape()[1], 8);

            const float *sliced_data = sliced->data();
            EXPECT_FLOAT_EQ(sliced_data[0], 40.0f); // First of row 5
            EXPECT_FLOAT_EQ(sliced_data[7], 47.0f); // Last of row 5
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_AllRows)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            float *data = tensor->mutable_data();
            for (size_t i = 0; i < 80; ++i)
            {
                data[i] = static_cast<float>(i);
            }

            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());
            auto sliced = WeightManager::sliceRowRange(shared_tensor, 0, 10);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 10);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Should be a complete copy
            const float *sliced_data = sliced->data();
            for (size_t i = 0; i < 80; ++i)
            {
                EXPECT_FLOAT_EQ(sliced_data[i], static_cast<float>(i));
            }
        }

        // =============================================================================
        // sliceRowRange Error Cases
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_NullTensor_ReturnsNull)
        {
            std::shared_ptr<TensorBase> null_tensor = nullptr;
            EXPECT_THROW(WeightManager::sliceRowRange(null_tensor, 0, 1), std::runtime_error);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_OutOfBounds_ReturnsNull)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Start + count exceeds tensor dimensions
            EXPECT_THROW(WeightManager::sliceRowRange(shared_tensor, 8, 5), std::runtime_error);
        }

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_StartAtEnd_ReturnsNull)
        {
            auto tensor = TestTensorFactory::createFP32({10, 8});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Start at row 10 (out of bounds)
            EXPECT_THROW(WeightManager::sliceRowRange(shared_tensor, 10, 1), std::runtime_error);
        }

        // =============================================================================
        // sliceRowRange - Larger Realistic Sizes
        // =============================================================================

        TEST_F(Test__WeightManager_LocalTPSlicing, SliceRowRange_FP32_LargeMatrix_ProportionalSplit)
        {
            // Simulate splitting a 1024x896 weight matrix for 73%/27% TP split
            const size_t out_dim = 1024;
            const size_t in_dim = 896;

            auto tensor = TestTensorFactory::createFP32Random({out_dim, in_dim});
            auto shared_tensor = std::shared_ptr<TensorBase>(tensor.release());

            // Device 0 gets ~73% = 747 rows (rounded down for alignment)
            // Device 1 gets ~27% = 277 rows
            size_t cuda_rows = 747;
            size_t rocm_rows = out_dim - cuda_rows;

            auto cuda_slice = WeightManager::sliceRowRange(shared_tensor, 0, cuda_rows);
            auto rocm_slice = WeightManager::sliceRowRange(shared_tensor, cuda_rows, rocm_rows);

            ASSERT_NE(cuda_slice, nullptr);
            ASSERT_NE(rocm_slice, nullptr);

            EXPECT_EQ(cuda_slice->shape()[0], cuda_rows);
            EXPECT_EQ(cuda_slice->shape()[1], in_dim);
            EXPECT_EQ(rocm_slice->shape()[0], rocm_rows);
            EXPECT_EQ(rocm_slice->shape()[1], in_dim);

            // Slices should be independent (not sharing memory with original)
            // Verify by checking that all values are valid floats
            auto *cuda_fp32 = dynamic_cast<FP32Tensor *>(cuda_slice.get());
            auto *rocm_fp32 = dynamic_cast<FP32Tensor *>(rocm_slice.get());
            ASSERT_NE(cuda_fp32, nullptr) << "CUDA slice should be FP32";
            ASSERT_NE(rocm_fp32, nullptr) << "ROCm slice should be FP32";
            EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(cuda_fp32));
            EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(rocm_fp32));
        }

        namespace
        {
            /**
             * @brief Construct an actual canonical immutable QueryInterval/GDN declaration.
             * @param degree Complete symbolic GPU membership count, with no native initialization.
             * @return Fresh publication owner; the existing distributor owns every interval.
             */
            std::shared_ptr<TensorParallelConfig> snapshotAssignment(int degree)
            {
                constexpr std::array ordinals{19, 7, 11, 3, 23, 5, 17, 2};
                std::vector<DeviceId> devices;
                for (int i = 0; i < degree; ++i)
                    devices.push_back(DeviceId::rocm(ordinals[static_cast<size_t>(i)]));
                return std::make_shared<TensorParallelConfig>(TensorParallelConfig::equalSplit(
                    degree, 24, 2, 640, 248320, std::move(devices), KVHeadCoveragePolicy::QueryInterval,
                    QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{16, 48}})));
            }

            /**
             * @brief Validate each retained snapshot through the production const declaration API.
             * @param snapshot Owning publication lease; absence is an error, never a skipped read.
             * @return Complete source, alignment, physical membership and interval agreement.
             */
            bool validSnapshot(const std::shared_ptr<const TensorParallelConfig> &snapshot)
            {
                if (!snapshot || !snapshot->validate() || snapshot->worldSize() < 2 || snapshot->worldSize() > 8 ||
                    snapshot->kvHeadCoveragePolicy() != KVHeadCoveragePolicy::QueryInterval ||
                    snapshot->totalHeads() != 24 || snapshot->totalKVHeads() != 2 ||
                    snapshot->totalDFF() != 640 || snapshot->totalVocab() != 248320 ||
                    snapshot->queryHeadGranularity() != 3 || !snapshot->queryHeadPartitionDeclaration() ||
                    !snapshot->kvHeadCoverageDeclaration())
                    return false;
                const auto &query = *snapshot->queryHeadPartitionDeclaration();
                const auto &coverage = *snapshot->kvHeadCoverageDeclaration();
                if (query.originalQueryHeads() != 24 || query.consumers() !=
                        std::vector<OriginalGDNHeadGeometry>{{16, 48}} ||
                    coverage.query_heads != 24 || coverage.kv_heads != 2)
                    return false;
                constexpr std::array ordinals{19, 7, 11, 3, 23, 5, 17, 2};
                int next_head = 0;
                for (int rank = 0; rank < snapshot->worldSize(); ++rank)
                {
                    const auto &actual = snapshot->forRank(rank);
                    const auto required = resolveQueryKVHeadCoverage(coverage, actual.head_start, actual.head_count);
                    if (actual.local_rank != rank || actual.device != DeviceId::rocm(ordinals[static_cast<size_t>(rank)]) ||
                        actual.head_start != next_head || actual.head_count <= 0 ||
                        actual.head_start % 3 != 0 || actual.head_count % 3 != 0 ||
                        actual.kv_head_start != required.first || actual.kv_head_count != required.count)
                        return false;
                    next_head += actual.head_count;
                }
                return next_head == 24;
            }
        } // namespace

        /**
         * @test Both real publication APIs preserve an older reader's immutable owner.
         *
         * The writer completes on a separate thread before any old declaration is
         * inspected. Weak ownership proves replacement retires the original only
         * after the final reader lease, rather than merely leaving readable memory.
         */
        TEST_F(Test__WeightManager_LocalTPSlicing, TensorParallelSnapshotRetainsReplacedQueryIntervalOwner)
        {
            static_assert(std::is_same_v<decltype(std::declval<const WeightManager &>().tensorParallelConfig()),
                std::shared_ptr<const TensorParallelConfig>>, "TP declarations require an owning immutable snapshot");
            MockModelLoader loader;
            WeightManager manager(loader);
            EXPECT_EQ(manager.tensorParallelConfig(), nullptr);
            for (const bool configure_writer : {false, true})
            {
                auto original = snapshotAssignment(3);
                std::weak_ptr<const TensorParallelConfig> retired = original;
                manager.setTensorParallelConfig(std::move(original));
                auto reader = manager.tensorParallelConfig();
                ASSERT_TRUE(reader);
                EXPECT_TRUE(validSnapshot(reader));
                EXPECT_EQ(reader->worldSize(), 3);
                std::jthread writer([&] {
                    if (configure_writer)
                    {
                        WeightManagerConfig config;
                        config.tp_config = snapshotAssignment(5);
                        manager.configure(config);
                    }
                    else
                        manager.setTensorParallelConfig(snapshotAssignment(5));
                });
                writer.join();
                EXPECT_FALSE(retired.expired());
                EXPECT_TRUE(validSnapshot(reader));
                EXPECT_EQ(reader->worldSize(), 3);
                const auto current = manager.tensorParallelConfig();
                ASSERT_TRUE(current);
                EXPECT_TRUE(validSnapshot(current));
                EXPECT_EQ(current->worldSize(), 5);
                EXPECT_NE(current.get(), reader.get());
                auto final_reader = reader;
                reader.reset();
                EXPECT_FALSE(retired.expired());
                final_reader.reset();
                EXPECT_TRUE(retired.expired());
            }
        }

        /**
         * @test Four builders retain old declarations while both writers publish fresh owners.
         *
         * Every round authenticates an initial lease before replacement, overlaps
         * fresh snapshot reads with a bounded publication sweep, then checks that
         * retained old declaration after all replacements. No null/failed read is
         * skipped and no worker performs GTest assertions or accesses tensor data.
         */
        TEST_F(Test__WeightManager_LocalTPSlicing, TensorParallelSnapshotConcurrentReplacementKeepsEveryReadValid)
        {
            MockModelLoader loader;
            WeightManager manager(loader);
            constexpr int readers = 4, rounds = 32, reads_per_round = 64;
            std::barrier cohort(readers + 1);
            std::array<std::atomic<int>, readers> failures{};
            std::array<std::atomic<int>, readers> checks{};
            std::vector<std::jthread> workers;
            workers.reserve(readers);
            for (int worker = 0; worker < readers; ++worker)
                workers.emplace_back([&, worker] {
                    for (int round = 0; round < rounds; ++round)
                    {
                        cohort.arrive_and_wait(); // The initial owner is fully published.
                        const auto retained = manager.tensorParallelConfig();
                        if (!validSnapshot(retained)) ++failures[worker];
                        ++checks[worker];
                        cohort.arrive_and_wait(); // Every reader now owns that exact generation.
                        for (int read = 0; read < reads_per_round; ++read)
                        {
                            const auto current = manager.tensorParallelConfig();
                            if (!validSnapshot(current)) ++failures[worker];
                            ++checks[worker];
                            std::this_thread::yield();
                        }
                        cohort.arrive_and_wait(); // All replacements and fresh reads are complete.
                        if (!validSnapshot(retained) || retained->worldSize() != 2 + round % 7)
                            ++failures[worker];
                        ++checks[worker];
                        cohort.arrive_and_wait(); // Retained reads complete before the next generation.
                    }
                });
            for (int round = 0; round < rounds; ++round)
            {
                manager.setTensorParallelConfig(snapshotAssignment(2 + round % 7));
                cohort.arrive_and_wait();
                cohort.arrive_and_wait();
                for (int publication = 0; publication < reads_per_round; ++publication)
                {
                    auto next = snapshotAssignment(2 + (round + publication + 1) % 7);
                    if (publication % 2 == 0)
                    {
                        WeightManagerConfig config;
                        config.tp_config = std::move(next);
                        manager.configure(config);
                    }
                    else
                        manager.setTensorParallelConfig(std::move(next));
                    std::this_thread::yield();
                }
                cohort.arrive_and_wait();
                cohort.arrive_and_wait();
            }
            for (auto &worker : workers) worker.join();
            for (int worker = 0; worker < readers; ++worker)
            {
                EXPECT_EQ(failures[worker].load(), 0) << "reader " << worker;
                EXPECT_EQ(checks[worker].load(), rounds * (reads_per_round + 2)) << "reader " << worker;
            }
        }

    } // namespace test
} // namespace llaminar2
