/**
 * @file Test__PredictorGlobalBindingsNativeCPU.cpp
 * @brief Real frozen binding admission for CPU-original versus GPU-prepared embeddings.
 *
 * These tests construct tiny original Q8_0 tensors and invoke the production
 * fromFrozen builder. They never prepare weights, allocate device storage or
 * execute inference. CPU source lifetime remains the frozen binding's owner.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorGlobalBindings.h"
#include "loaders/PreparedWeightStore.h"
#include "models/GraphTypes.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;
namespace {
/** @brief One complete original native CPU embedding and an empty prepared store. */
struct NativeCPUFixture
{
    GraphConfig config;
    WeightBinding embedding;
    ModelWeightBindings primary, replicated;
    PreparedWeightStore store{ModelContextId{7}};

    /** @brief Materialize only a tiny native source matching production frozen CPU metadata. */
    NativeCPUFixture()
    {
        config.vocab_size = 8;
        config.d_model = 32;
        embedding.binding_id = 91;
        embedding.identity = {{7}, 31, 40, "token_embd.weight", WeightRole::Embedding};
        embedding.slice = {.source_rows=8, .source_cols=32, .row_count=8, .col_count=32};
        embedding.residency.home_device = DeviceId::cpu();
        embedding.residency.resident_device = DeviceId::cpu();
        embedding.residency.host_policy = WeightHostPolicy::RequiredForCPUExecution;
        embedding.tensor_owner = std::make_shared<Q8_0Tensor>(
            std::vector<size_t>{8,32}, std::vector<uint8_t>(8*34, 0));
        embedding.tensor = embedding.tensor_owner.get();
        embedding.immutable = true;
        primary.embedding_table = &embedding;
    }

    /** @return Native production builder result; shifted-prefill consumes no LM head. */
    Qwen4ExpPredictorGlobalBindings build(DeviceId device = DeviceId::cpu())
    {
        return Qwen4ExpPredictorGlobalBindings::fromFrozen(config, primary, replicated,
            store, device, Qwen4ExpPredictorInvocation::ShiftedPrefill);
    }
};
}

TEST(PredictorGlobalBindingsNativeCPU, ActualBuilderBorrowsOriginalWithoutPreparingOrRetargeting)
{
    NativeCPUFixture f;
    const auto *source = f.embedding.tensor;
    const auto before = source->raw_data();
    const auto original_identity = f.embedding.identity;
    const auto admitted = f.build();
    EXPECT_EQ(&admitted.embedding(), &f.embedding);
    EXPECT_EQ(admitted.embedding().tensor, source);
    EXPECT_EQ(admitted.embedding().tensor->raw_data(), before);
    EXPECT_EQ(admitted.embedding().tensor->native_type(), TensorType::Q8_0);
    EXPECT_EQ(admitted.embedding().identity, original_identity);
    EXPECT_EQ(admitted.embeddingOffset(), 0);
    EXPECT_EQ(admitted.embeddingRows(), 8);
    EXPECT_EQ(admitted.vocabulary(), 8);
    EXPECT_FALSE(admitted.reconstructEmbedding());
    EXPECT_FALSE(admitted.embedding().prepared.has_value());
    EXPECT_EQ(f.store.size(), 0);
    EXPECT_THROW((void)admitted.head(), std::logic_error);
}

TEST(PredictorGlobalBindingsNativeCPU, EveryGPUStillRequiresItsExistingPreparedNativeOwner)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId(DeviceType::ROCm,0)})
    {
        NativeCPUFixture f;
        EXPECT_THROW((void)f.build(device), std::invalid_argument);
        EXPECT_EQ(f.store.size(), 0);
        EXPECT_FALSE(f.embedding.prepared.has_value());
    }
}

TEST(PredictorGlobalBindingsNativeCPU, InvalidCPUStorageLifetimeOrSliceCannotBeAdmitted)
{
    for (int mutation = 0; mutation < 14; ++mutation)
    {
        NativeCPUFixture f;
        switch (mutation)
        {
        case 0: f.embedding.residency.home_device = DeviceId::cuda(0); break;
        case 1: f.embedding.residency.resident_device.reset(); break;
        case 2: f.embedding.residency.host_policy = WeightHostPolicy::ReleasableAfterPreparation; break;
        case 3: f.embedding.residency.raw_host_data_available = false; break;
        case 4: f.embedding.slice.source_rows = 9; break;
        case 5: f.embedding.slice.source_cols = 33; break;
        case 6: f.embedding.slice.row_count = 0; break;
        case 7: f.embedding.slice.row_count = 9; break;
        case 8: f.embedding.slice.row_start = std::numeric_limits<size_t>::max(); break;
        case 9: f.embedding.slice.col_start = 1; break;
        case 10: f.embedding.slice.col_count = 31; break;
        case 11: f.embedding.slice.expert_count = 1; break;
        case 12: f.embedding.tensor->release_raw_data(); break;
        case 13: f.embedding.prepared = PreparedWeightRef{{7},91,PreparedWeightKind::PreparedEmbedding,DeviceId::cpu()}; break;
        }
        EXPECT_THROW((void)f.build(), std::invalid_argument) << mutation;
        EXPECT_EQ(f.store.size(), 0) << mutation;
    }
}

TEST(PredictorGlobalBindingsNativeCPU, CPUOriginalStillAuthenticatesSourceRoleShapeAndTransport)
{
    for (int mutation = 0; mutation < 7; ++mutation)
    {
        NativeCPUFixture f;
        switch (mutation)
        {
        case 0: f.embedding.identity.layer = 48; break;
        case 1: f.embedding.identity.role = WeightRole::Other; break;
        case 2: f.embedding.identity.canonical_name = "nextn.token_embd.weight"; break;
        case 3: f.embedding.binding_id = 0; break;
        case 4: f.config.d_model = 64; break;
        case 5: f.config.vocab_size = 7; break;
        case 6: f.embedding.slice.row_count = 4; break;
        }
        EXPECT_THROW((void)f.build(), std::invalid_argument) << mutation;
    }
}

TEST(PredictorGlobalBindingsNativeCPU, DraftStillRequiresItsOriginalPreparedTerminalHead)
{
    NativeCPUFixture f;
    EXPECT_THROW((void)Qwen4ExpPredictorGlobalBindings::fromFrozen(f.config,
        f.primary, f.replicated, f.store, DeviceId::cpu(),
        Qwen4ExpPredictorInvocation::Draft), std::invalid_argument);
    EXPECT_EQ(f.store.size(), 0);
    EXPECT_FALSE(f.embedding.prepared.has_value());
}
