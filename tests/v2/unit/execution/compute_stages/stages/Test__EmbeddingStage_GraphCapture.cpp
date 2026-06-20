/**
 * @file Test__EmbeddingStage_GraphCapture.cpp
 * @brief Unit tests for EmbeddingStage graph-capture support
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "backends/DeviceId.h"
#include "execution/compute_stages/ComputeStages.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "../../../../utils/TestTensorFactory.h"

namespace llaminar2
{
    namespace
    {
        class Test__EmbeddingStage_GraphCapture : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                vocab_size_ = 4096;
                d_model_ = 128;
            }

            std::unique_ptr<FP32Tensor> createEmbeddingTable() const
            {
                auto table = std::make_unique<FP32Tensor>(std::vector<size_t>{
                    static_cast<size_t>(vocab_size_), static_cast<size_t>(d_model_)});

                float *data = table->mutable_data();
                std::mt19937 generator(42);
                std::normal_distribution<float> distribution(0.0f, 0.02f);
                for (int i = 0; i < vocab_size_ * d_model_; ++i)
                {
                    data[i] = distribution(generator);
                }

                return table;
            }

            static double cosineSimilarity(const float *lhs, const float *rhs, size_t count)
            {
                double dot = 0.0;
                double lhs_norm = 0.0;
                double rhs_norm = 0.0;
                for (size_t i = 0; i < count; ++i)
                {
                    dot += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
                    lhs_norm += static_cast<double>(lhs[i]) * static_cast<double>(lhs[i]);
                    rhs_norm += static_cast<double>(rhs[i]) * static_cast<double>(rhs[i]);
                }

                if (lhs_norm < 1e-12 || rhs_norm < 1e-12)
                {
                    return 0.0;
                }

                return dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm));
            }

            static double maxAbsDiff(const float *lhs, const float *rhs, size_t count)
            {
                double max_diff = 0.0;
                for (size_t i = 0; i < count; ++i)
                {
                    max_diff = std::max(max_diff, std::abs(static_cast<double>(lhs[i] - rhs[i])));
                }
                return max_diff;
            }

            int vocab_size_ = 0;
            int d_model_ = 0;
        };

        TEST_F(Test__EmbeddingStage_GraphCapture, AdvertisesGraphCaptureSupport)
        {
            auto embed_table = createEmbeddingTable();
            auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d_model_)});
            std::vector<int> token_ids = {42};

            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = output.get();
            params.num_tokens = 1;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            EmbeddingStage stage(params);

            EXPECT_TRUE(stage.isGraphCapturable());
            EXPECT_TRUE(stage.hasDynamicParams());
        }

        TEST_F(Test__EmbeddingStage_GraphCapture, UsesUpdatedTokenBufferAcrossExecutions)
        {
            auto embed_table = createEmbeddingTable();
            auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d_model_)});
            std::vector<int> token_ids = {7};

            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = output.get();
            params.num_tokens = 1;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            EmbeddingStage stage(params);

            ASSERT_TRUE(stage.execute(nullptr));

            std::vector<float> first_run(static_cast<size_t>(d_model_));
            std::memcpy(first_run.data(), output->data(), static_cast<size_t>(d_model_) * sizeof(float));

            token_ids[0] = 99;
            stage.updateDynamicParams(/*pos_offset=*/0, /*seq_len=*/1);
            ASSERT_TRUE(stage.execute(nullptr));

            const float *second_run = output->data();
            const float *expected_second = embed_table->data() + static_cast<size_t>(token_ids[0]) * d_model_;

            EXPECT_GT(cosineSimilarity(expected_second, second_run, static_cast<size_t>(d_model_)), 0.999999);
            EXPECT_GT(maxAbsDiff(first_run.data(), second_run, static_cast<size_t>(d_model_)), 1e-5);
        }

        TEST_F(Test__EmbeddingStage_GraphCapture, QuantizedGpuEmbeddingRequiresPreparedStoreRef)
        {
            auto embed_table = test::TestTensorFactory::createQ8_0Random({64, static_cast<size_t>(d_model_)});
            auto output = std::make_unique<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d_model_)});
            std::vector<int> token_ids = {7};

            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = output.get();
            params.num_tokens = 1;
            params.d_model = d_model_;
            params.vocab_size = 64;
            params.device_id = DeviceId::cuda(0);

            EmbeddingStage missing(params);
            std::string error;
            EXPECT_FALSE(missing.validatePreparedWeights(&error));
            EXPECT_NE(error.find("PreparedWeightStore"), std::string::npos);

            PreparedWeightStore store(ModelContextId{99});
            WeightBinding binding;
            binding.binding_id = 21;
            binding.identity = makeSourceWeightIdentity("token_embd.weight", ModelContextId{99}, 21);
            binding.identity.role = WeightRole::Embedding;
            binding.residency.home_device = DeviceId::cuda(0);
            binding.residency.resident_device = DeviceId::cuda(0);
            binding.tensor = embed_table.get();
            binding.immutable = true;

            PreparedEmbeddingHandle handle;
            handle.tensor = embed_table.get();
            handle.device_id = DeviceId::cuda(0);

            params.prepared_ref = store.registerPreparedEmbeddingFromPipeline(
                binding, DeviceId::cuda(0), &handle);
            params.prepared_store = &store;

            EmbeddingStage prepared(params);
            EXPECT_TRUE(prepared.validatePreparedWeights(&error));
        }
    }
}