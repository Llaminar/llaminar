/**
 * @file Test__MTPStochasticSeedBinding.cpp
 * @brief Device-free proof that MTP verification consumes admitted seed rows.
 *
 * Recording backends observe the production stage's exact pointer and stream
 * arguments. Host arrays stand in for opaque device addresses; no GPU is opened
 * and no kernel runs. Seed contents may change without changing graph identity,
 * while absent, misaligned and relocated banks exercise admission boundaries.
 */
#include "execution/compute_stages/stages/MTPStochasticOutcomeStage.h"
#include "mocks/MockBackend.h"
#include "mocks/MockComputeStage.h"

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::sampling_math;
using namespace llaminar2::test;
using llaminar2::testing::MockDeviceContext;

namespace
{
/** @brief Observe backend submissions without reading the pretend device rows. */
class SeedRecordingBackend final : public MockBackend
{
public:
    std::vector<const uint64_t *> seeds;
    std::vector<uint64_t> scalar_seeds;
    std::vector<void *> streams;
    int summaries = 0;

    /** @brief Record the fused serial-law seed and stream exactly as submitted. */
    bool enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
        const void *, const void *, int, int, int, uint64_t scalar,
        const void *, int, const void *, const void *, const void *, int,
        void *stream, int, void *, void *, void *, void *,
        const uint64_t *seed) override
    {
        seeds.push_back(seed);
        scalar_seeds.push_back(scalar);
        streams.push_back(stream);
        return true;
    }

    /** @brief Record the probability-law seed without substituting scalar data. */
    bool enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
        const void *, const void *, const void *, const void *, int, int,
        const void *, const float *, const float *, int, int, void *stream,
        void *, void *, void *, void *, const void *, uint64_t scalar, int,
        int, const void *, int, const int *, const uint64_t *seed) override
    {
        seeds.push_back(seed);
        scalar_seeds.push_back(scalar);
        streams.push_back(stream);
        return true;
    }

    /** @brief Preserve the probability law's second, complete reduction node. */
    bool enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
        const void *, const void *, const void *, int, const void *,
        const void *, const void *, bool, const void *, int, void *, int,
        void *, void *) override
    {
        ++summaries;
        return true;
    }
};

/** @brief Supply well-shaped, opaque addresses shared by both backend identities. */
class MTPStochasticSeedBinding : public ::testing::Test
{
protected:
    SeedRecordingBackend backend;
    std::array<uint64_t, 4> seed_bank{1, 0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull, 99};
    std::array<int32_t, 512> words{};
    std::array<float, 32> probabilities{};

    /** @return Three admitted request rows; none of these addresses is executed. */
    MTPStochasticOutcomeStage::Params params(DeviceId device,
        MTPStochasticOutcomeStage::Verification law)
    {
        MTPStochasticOutcomeStage::Params p;
        p.device_id = device;
        p.backend = &backend;
        p.verification = law;
        p.vocabulary_size = 32;
        p.target_token_ids_device = words.data();
        p.target_probs_device = probabilities.data();
        p.target_distribution_row_stride = p.top_k = 2;
        p.verifier_input_tokens_device = words.data();
        p.verifier_input_token_stride = 3;
        p.stop_tokens_device = words.data();
        p.threshold_base_positions_device = words.data();
        p.threshold_seeds_device = seed_bank.data();
        p.generation_control_device = words.data();
        p.generation_control_stride = kDeviceGenerationControlCount;
        p.verifier_row_capacity = 3;
        p.sampled_target_tokens_device = words.data();
        p.sampled_target_token_stride = 3;
        p.accepted_rows_device = words.data();
        p.accepted_row_stride = 3;
        p.output_tokens_device = words.data();
        p.output_token_stride = kSpeculativeBatchMaxOutputTokens;
        p.output_meta_device = words.data();
        p.output_meta_stride = kSpeculativeBatchMetaCount;
        p.request_count = 3;
        p.comparison_rows_per_request = 2;
        return p;
    }
};

/** @test Both laws bind every row to admitted storage on the exact graph stream. */
TEST_F(MTPStochasticSeedBinding, ChangedSeedBytesRetainIdentityAndExactRequestRows)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const auto law : {MTPStochasticOutcomeStage::Verification::SerialEquivalent,
                           MTPStochasticOutcomeStage::Verification::OneHotProbabilityRejection})
    {
        const auto p = params(device, law);
        MTPStochasticOutcomeStage stage(p);
        MockDeviceContext context(device, device == DeviceId::cuda(0)
            ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
        void *const stream = words.data() + 400;
        stage.setGPUStream(stream);
        for (int reset = 0; reset < 20; ++reset)
        {
            seed_bank[1] ^= (uint64_t{1} << 32) | (uint64_t{1} << reset);
            EXPECT_TRUE(stage.hasSameCaptureIdentity(p));
            backend.seeds.clear(); backend.scalar_seeds.clear(); backend.streams.clear();
            backend.summaries = 0;
            ASSERT_TRUE(stage.execute(&context));
            ASSERT_EQ(backend.seeds.size(), 3u);
            for (size_t row = 0; row < 3; ++row)
            {
                EXPECT_EQ(backend.seeds[row], seed_bank.data() + row);
                EXPECT_EQ(backend.scalar_seeds[row], 0u);
                EXPECT_EQ(backend.streams[row], stream);
            }
            EXPECT_EQ(backend.summaries,
                law == MTPStochasticOutcomeStage::Verification::OneHotProbabilityRejection ? 3 : 0);
        }
        auto relocated = p;
        ++relocated.threshold_seeds_device;
        EXPECT_FALSE(stage.hasSameCaptureIdentity(relocated));
    }
}

/** @test Invalid banks fail before any backend work; device contents stay opaque. */
TEST_F(MTPStochasticSeedBinding, AbsentAndUnalignedBanksRejectBeforeSubmission)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const auto law : {MTPStochasticOutcomeStage::Verification::SerialEquivalent,
                           MTPStochasticOutcomeStage::Verification::OneHotProbabilityRejection})
    for (const auto *seed : {static_cast<const uint64_t *>(nullptr),
        reinterpret_cast<const uint64_t *>(reinterpret_cast<const char *>(seed_bank.data()) + 1)})
    {
        auto p = params(device, law);
        p.threshold_seeds_device = seed;
        MTPStochasticOutcomeStage stage(p);
        stage.setGPUStream(words.data() + 400);
        MockDeviceContext context(device, device == DeviceId::cuda(0)
            ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
        EXPECT_FALSE(stage.execute(&context));
        EXPECT_TRUE(backend.seeds.empty());
        EXPECT_EQ(backend.summaries, 0);
    }
}
} // namespace
