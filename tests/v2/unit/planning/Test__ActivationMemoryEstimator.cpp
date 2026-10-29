/**
 * @file Test__ActivationMemoryEstimator.cpp
 * @brief Verifies generic and production model graph-arena byte accounting.
 * CPU arena roots use an explicit model-free fixture admission authority.
 */

#include "../../utils/CPUActivationArenaFixture.h"
#include <gtest/gtest.h>
#include "planning/ActivationMemoryEstimator.h"
#include "execution/config/RuntimeConfig.h"
#include "backends/DeviceId.h"
#include "config/TensorParallelAssignmentDeclaration.h"
#include "kernels/gdn/GDNPreparedQK.h"
#include "memory/BufferArena.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"

#include <algorithm>
#include <array>
#include <string>

using namespace llaminar2;

TEST(Test__ActivationMemoryEstimator, ReturnsNonZeroForValidInput)
{
    size_t bytes = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes, 0u);
}

TEST(Test__ActivationMemoryEstimator, ReturnsZeroForInvalidInput)
{
    EXPECT_EQ(ActivationMemoryEstimator::estimate(0, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0)), 0u);
    EXPECT_EQ(ActivationMemoryEstimator::estimate(1, 0, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0)), 0u);
    EXPECT_EQ(ActivationMemoryEstimator::estimate(1, 4096, 0, 4864, 14, 2, 64, 151936, DeviceId::cuda(0)), 0u);
}

TEST(Test__ActivationMemoryEstimator, ScalesWithSeqLen)
{
    size_t bytes_2k = ActivationMemoryEstimator::estimate(
        1, 2048, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));
    size_t bytes_4k = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));

    // Larger seq_len should mean more activation memory
    EXPECT_GT(bytes_4k, bytes_2k);
}

TEST(Test__ActivationMemoryEstimator, LargeVocabDoesNotReserveAllPositionLogits)
{
    constexpr size_t B = 1, S = 4096, D = 896, F = 4864;
    constexpr size_t H = 14, HK = 2, HD = 64, V = 151936;
    constexpr size_t FP32 = 4;

    const size_t actual = ActivationMemoryEstimator::estimate(
        static_cast<int>(B), static_cast<int>(S), static_cast<int>(D), static_cast<int>(F),
        static_cast<int>(H), static_cast<int>(HK), static_cast<int>(HD), static_cast<int>(V),
        DeviceId::cuda(0));

    const size_t hidden_state = B * S * D * FP32;
    const size_t terminal_logits = B * V * FP32;
    const size_t all_position_logits = B * S * V * FP32;

    EXPECT_LT(actual, hidden_state + all_position_logits);
    EXPECT_GE(actual, hidden_state + terminal_logits);
}

TEST(Test__ActivationMemoryEstimator, CPUPricesEachPhysicalRootSeparately)
{
    // Activation estimate should be similar for CPU and GPU (same buffers needed)
    size_t gpu = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cuda(0));
    size_t cpu = ActivationMemoryEstimator::estimate(
        1, 4096, 896, 4864, 14, 2, 64, 151936, DeviceId::cpu());

    const auto logits_payload = size_t{151936} * sizeof(float);
    EXPECT_EQ(cpu, gpu - logits_payload +
        AlignedVector<float>::requiredAllocationBytes(151936));
}

TEST(Test__ActivationMemoryEstimator, LogicalArenaOwnership_MatchesManualComputation)
{
    // Manually compute the physical BufferId owners for known inputs:
    // B=1, S=512, D=256, F=1024, H=4, HK=2, HD=64, V=1000
    constexpr size_t B = 1, S = 512, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 1000;
    constexpr size_t FP32 = 4;

    const size_t Q = H * HD;
    const size_t KV = HK * HD;
    const size_t expected_owned =
        B * S * (5 * D + 2 * Q + 2 * KV + 2 * F + S) * FP32 +
        B * V * FP32;

    size_t actual = ActivationMemoryEstimator::estimate(
        1, 512, 256, 1024, 4, 2, 64, 1000, DeviceId::cuda(0));

    EXPECT_EQ(actual, expected_owned);
}

TEST(Test__ActivationMemoryEstimator, OneRowPrefill_StillOwnsEveryRegisteredBuffer)
{
    // A large vocabulary dominates this case, but registered one-row graph
    // buffers remain physically owned and must not disappear from the estimate.
    constexpr size_t B = 1, S = 1, D = 256, F = 1024;
    constexpr size_t H = 4, HK = 2, HD = 64, V = 500000;
    constexpr size_t FP32 = 4;

    const size_t Q = H * HD;
    const size_t KV = HK * HD;
    const size_t expected =
        B * S * (5 * D + 2 * Q + 2 * KV + 2 * F + S) * FP32 +
        B * V * FP32;

    size_t actual = ActivationMemoryEstimator::estimate(
        static_cast<int>(B), static_cast<int>(S), static_cast<int>(D), static_cast<int>(F),
        static_cast<int>(H), static_cast<int>(HK), static_cast<int>(HD), static_cast<int>(V),
        DeviceId::cuda(0));

    EXPECT_EQ(actual, expected);
}

TEST(Test__ActivationMemoryEstimator,
     Qwen35MoE122BLocalTP_MatchesDeclarativeArenaByteForByte)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 49;
    profile.mtp_layer_count = 1;
    profile.d_model = 3072;
    profile.d_ff = 1024;
    profile.n_heads = 32;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.vocab_size = 248320;
    profile.max_seq_len = 4096;
    profile.gdn_conv_kernel_size = 4;
    profile.gdn_state_size = 128;
    profile.gdn_inner_size = 8192;
    profile.gdn_group_count = 16;
    profile.gdn_time_step_rank = 64;
    profile.expert_count = 256;
    profile.expert_used_count = 8;
    profile.expert_feed_forward_length = 1024;

    const auto add_projection = [&](
        std::string name,
        size_t output_columns,
        size_t input_columns)
    {
        TensorSizeInfo tensor;
        tensor.name = std::move(name);
        tensor.K = input_columns;
        tensor.elements = output_columns * input_columns;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add_projection("blk.0.attn_q.weight", 16384, 3072);
    add_projection("blk.0.attn_gate.weight", 8192, 3072);
    add_projection("blk.0.attn_qkv.weight", 12288, 3072);
    add_projection("blk.0.ssm_out.weight", 3072, 8192);
    add_projection("blk.0.ssm_alpha.weight", 64, 3072);
    add_projection("blk.0.ssm_beta.weight", 64, 3072);

    ActivationGraphMemoryGeometry geometry{
            .batch_size = 1,
            .resident_graph_rows = 4096,
            .local_d_ff = 512,
            .local_n_heads = 16,
            .local_n_kv_heads = 1,
            .first_layer = 0,
            .last_layer = 47,
            .total_shards = 2,
            .mtp_target_query_rows = 2,
            .mtp_terminal_logits_layout =
                MTPTerminalLogitsLayout::FullVocabularyPerParticipant,
        };
    const size_t actual = ActivationMemoryEstimator::estimate(
        profile, geometry, DeviceId::cuda(0));

    /*
     * The historical independent owner sum was 2,253,085,700 bytes before
     * the separately admitted UINT64 seed. Its synthetic merged width was
     * 16,384, while the source tensor is [12,288, 3,072]. At TP2 this corrects
     * each of the two distinct FP32 raw-QKV banks from 8,192 to 6,144 columns:
     * 2 * 4,096 rows * 2,048 columns * 4 bytes = 67,108,864 bytes removed.
     * All other owner terms are unchanged, including the 503,316,480-byte
     * canonical route tensor. This is an explicit schema-width correction,
     * not a claim that a new production BufferArena probe was executed.
     */
    constexpr size_t kExpectedDeclarativeArenaBytes = 2185976836ULL + sizeof(uint64_t);
    EXPECT_EQ(actual, kExpectedDeclarativeArenaBytes);

    // Actual TP2 ownership has eight key heads of width128, hence2,048
    // FP32 Q/K words (8,192 bytes), independent of the retained row capacity.
    // CPU/CUDA have no prepared activation role.
    for(const int resident_rows:{1,4096})
    {
        auto scalar_geometry=geometry;
        scalar_geometry.resident_graph_rows=resident_rows;
        const auto raw=ActivationMemoryEstimator::estimate(profile,scalar_geometry,DeviceId::cuda(0));
        EXPECT_EQ(ActivationMemoryEstimator::estimate(profile,scalar_geometry,DeviceId::rocm(0)),raw+8192u);
    }

    // A sharded projection replaces the full local bank with its vocabulary
    // slice AND needs the complete gathered bank. Previously admission only
    // charged a 1x1 placeholder, even though condition/verifier graphs bind
    // the full output. Keep this independent sum valid on every backend.
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int degree : {2, 4, 8})
    for (const int rows : {2, 4, 16, 31})
    {
        geometry.total_shards = degree;
        geometry.local_vocab = profile.vocab_size / degree;
        geometry.mtp_target_query_rows = rows;
        geometry.mtp_terminal_logits_layout = MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
        const auto mirrored = ActivationMemoryEstimator::estimate(profile, geometry, device);
        geometry.mtp_terminal_logits_layout = MTPTerminalLogitsLayout::VocabularyShardPerParticipant;
        const auto sharded = ActivationMemoryEstimator::estimate(profile, geometry, device);
        const auto root = [device](size_t words) {
            return device.is_cpu() ? AlignedVector<float>::requiredAllocationBytes(words) : words * sizeof(float);
        };
        EXPECT_EQ(sharded, mirrored + root(static_cast<size_t>(rows) * geometry.local_vocab) - root(1));
    }
}

/** @test GPU request seeds use real retained row capacity, not MTP enablement or TP division. */
TEST(Test__ActivationMemoryEstimator, GPUSeedBankUsesExactRequestCapacity)
{
    ModelMemoryProfile profile;
    profile.d_model = 32;
    profile.head_dim = 32;
    profile.vocab_size = 64;
    ActivationGraphMemoryGeometry geometry{
        .batch_size = 1, .resident_graph_rows = 32, .local_d_ff = 64,
        .local_n_heads = 1, .local_n_kv_heads = 1};
    const auto cpu = ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cpu());
    const auto gpu_without_seeds = ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cuda(0)) -
        static_cast<size_t>(geometry.generation_request_capacity) * sizeof(uint64_t);
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const int requests : {1, 3, 8, 15})
        {
            geometry.generation_request_capacity = requests;
            EXPECT_EQ(ActivationMemoryEstimator::estimate(profile, geometry, device),
                gpu_without_seeds + static_cast<size_t>(requests) * sizeof(uint64_t));
            EXPECT_EQ(ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cpu()), cpu);
        }
    geometry.generation_request_capacity = 0;
    EXPECT_THROW(ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cuda(0)),
        std::invalid_argument);
}

/** @test Prepared capacity uses exact whole heads for TP1..8, including uneven TP5. */
TEST(Test__ActivationMemoryEstimator, PreparedQKCapacityMatchesCanonicalUnevenHeadOwnership)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 1;
    profile.d_model = 2560;
    profile.d_ff = 1024;
    profile.n_heads = 24;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.vocab_size = 320;
    profile.max_seq_len = 128;
    profile.gdn_group_count = 16;
    profile.gdn_time_step_rank = 48;
    profile.gdn_state_size = 128;
    profile.gdn_inner_size = 6144;
    profile.gdn_conv_kernel_size = 4;
    const auto add = [&](std::string name, std::size_t n, std::size_t k)
    {
        TensorSizeInfo tensor;
        tensor.name = std::move(name);
        tensor.K = k;
        tensor.elements = n * k;
        tensor.layer_index = 0;
        profile.tensors.push_back(std::move(tensor));
    };
    add("blk.0.attn_q.weight", 12288, 2560);
    add("blk.0.attn_gate.weight", 6144, 2560);
    add("blk.0.attn_qkv.weight", 10240, 2560);
    add("blk.0.ssm_out.weight", 2560, 6144);
    add("blk.0.ssm_alpha.weight", 48, 2560);
    add("blk.0.ssm_beta.weight", 48, 2560);
    const auto declaration = TensorParallelAssignmentDeclaration::fromOriginal(
        {24, 2}, KVHeadCoveragePolicy::QueryInterval,
        QueryHeadPartitionDeclaration::fromOriginalConsumers(24, {{16, 48}}));
    bool saw_uneven_five = false;
    for (int degree = 1; degree <= 8; ++degree)
    {
        std::vector<DeviceId> devices;
        for (int rank = 0; rank < degree; ++rank) devices.push_back(DeviceId::rocm(rank));
        const auto assignment = declaration.partition(devices, {}, profile.d_ff, profile.vocab_size);
        for (const auto &local : assignment.assignments())
        {
            SCOPED_TRACE(::testing::Message() << "degree=" << degree
                << " head_start=" << local.head_start << " head_count=" << local.head_count);
            const auto actual = GDNHeadAssignment::fromPartition(16, 48,
                local.head_start, local.head_count, 24);
            const auto expected = gdn::ScalarPreparedQKLayout::fromHeads(
                actual.localKeyHeads(), actual.localValueHeads(), 128, 128);
            const auto capacity = gdn::ScalarPreparedQKLayout::fromHeadPartition(
                16, 48, 128, local.head_count, 24);
            EXPECT_EQ(capacity.bytes, expected.bytes);
            ActivationGraphMemoryGeometry geometry;
            geometry.resident_graph_rows = 32;
            geometry.context_tokens = 128;
            geometry.local_d_ff = local.d_ff_count;
            geometry.local_n_heads = local.head_count;
            geometry.local_n_kv_heads = local.kv_head_count;
            geometry.local_vocab = local.vocab_count;
            geometry.first_layer = geometry.last_layer = 0;
            geometry.total_shards = degree;
            for (bool replicated : {false, true})
            {
                geometry.replicated_dense_decode = replicated;
                const auto prepared = replicated
                    ? gdn::ScalarPreparedQKLayout::fromHeads(16, 48, 128, 128) : expected;
                EXPECT_EQ(ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::rocm(0)),
                    ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cuda(0)) +
                        prepared.bytes);
            }
            if (degree == 5 && local.head_count != 24 / degree) saw_uneven_five = true;
        }
    }
    EXPECT_TRUE(saw_uneven_five);
    for (int dk : {64, 128})
        EXPECT_TRUE(gdn::ScalarPreparedQKLayout::supportsNativeGeometry(dk, 4));
    for (int dk : {2, 8, 32, 96, 256})
        EXPECT_FALSE(gdn::ScalarPreparedQKLayout::supportsNativeGeometry(dk, 4));
    EXPECT_FALSE(gdn::ScalarPreparedQKLayout::supportsNativeGeometry(128, 3));
    EXPECT_THROW((void)gdn::ScalarPreparedQKLayout::fromHeadPartition(
        16, 48, 128, 1, 24), std::invalid_argument);
}

/** @test The actual CPU full-row arena pair is priced independently from sharded projection capacity. */
TEST(Test__ActivationMemoryEstimator, CPUPhaseSplitKVHandoffMatchesActualArenaOwners)
{
    ModelMemoryProfile profile;
    profile.n_layers=1;profile.d_model=32;profile.d_ff=64;
    profile.n_heads=4;profile.n_kv_heads=2;profile.head_dim=8;
    profile.vocab_size=64;profile.max_seq_len=8;
    MPIContext mpi(0,1,MPI_COMM_NULL);
    TensorFactory factory(mpi);
    for (const int rows : {1,3,8})
    {
        ActivationGraphMemoryGeometry geometry;
        geometry.resident_graph_rows=rows;geometry.context_tokens=8;
        geometry.local_d_ff=32;geometry.local_n_heads=2;geometry.local_n_kv_heads=1;
        geometry.local_vocab=32;geometry.total_shards=2;geometry.first_layer=geometry.last_layer=0;
        BufferArena arena(::llaminar2::test::cpuArenaFixtureConfig(&factory));
        ASSERT_TRUE(arena.registerBuffer(BufferId::K_FULL_PREFILL,rows,16,"FP32",DeviceId::cpu()));
        ASSERT_TRUE(arena.registerBuffer(BufferId::V_FULL_PREFILL,rows,16,"FP32",DeviceId::cpu()));
        ASSERT_TRUE(arena.allocate());
        const size_t actual = arena.getTensor(BufferId::K_FULL_PREFILL)->size_bytes() +
            arena.getTensor(BufferId::V_FULL_PREFILL)->size_bytes();
        ASSERT_EQ(actual, size_t(2*rows*2*8)*sizeof(float));
        for (const auto device : {DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        {
            geometry.replicated_dense_decode=false;
            const auto sharded = ActivationMemoryEstimator::estimate(profile,geometry,device);
            geometry.replicated_dense_decode=true;
            EXPECT_EQ(ActivationMemoryEstimator::estimate(profile,geometry,device)-sharded,
                device.is_cpu() ? 2 * AlignedVector<float>::requiredAllocationBytes(size_t(rows) * 16) : actual);
        }
    }
}

/** @test An admitted CPU BOM equals the physical claims of the declared common roots. */
TEST(Test__ActivationMemoryEstimator, CpuTinyRootsMatchLivePhysicalArena)
{
    constexpr size_t rows = 3, width = 7, q = 5, kv = 2, ffn = 11, vocab = 13;
    const auto estimated = ActivationMemoryEstimator::estimate(1, rows, width, ffn, q, kv, 1, vocab, DeviceId::cpu());
    auto config = test::cpuArenaFixtureConfig(nullptr, estimated);
    auto authority = config.physical_memory_authority;
    BufferArena arena(config);
    const std::array widths{width,width,width,width,width,q,q,kv,kv,ffn,ffn,rows};
    const std::array ids{BufferId::HIDDEN_STATE,BufferId::NORMALIZED,BufferId::RESIDUAL,
        BufferId::ATTN_PROJ,BufferId::FFN_OUTPUT,BufferId::Q_PROJ,BufferId::ATTN_OUTPUT,
        BufferId::K_PROJ,BufferId::V_PROJ,BufferId::GATE_PROJ,BufferId::UP_PROJ,BufferId::GEMM_WORKSPACE};
    for(size_t i=0;i<ids.size();++i)
        ASSERT_TRUE(arena.registerBuffer(ids[i],rows,widths[i],"FP32",DeviceId::cpu()));
    ASSERT_TRUE(arena.registerBuffer(BufferId::LOGITS,1,vocab,"FP32",DeviceId::cpu()));
    ASSERT_TRUE(arena.allocate());
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ActivationArena,
        PhysicalMemoryMaterializationKind::NewAllocation),estimated);
}

/** @test CPU retained MTP admission includes the exact independently owned terminal-hidden mailbox. */
TEST(Test__ActivationMemoryEstimator, CpuRetainedMtpTerminalHiddenMatchesActualRoot)
{
    ModelMemoryProfile profile;
    profile.d_model = 7; profile.head_dim = 1; profile.vocab_size = 13;
    for (const bool enabled : {false, true})
    for (const int requests : {1, 3})
    for (const int configured_batch : {1, 17})
    {
        MTPRuntimeConfig mtp;
        mtp.enabled = enabled;
        mtp.graph_capacity_draft_tokens = 3;
        mtp.max_request_batch = requests;
        ActivationGraphMemoryGeometry geometry;
        geometry.batch_size = resolveRuntimeBatchSizeForMTP(configured_batch, mtp);
        geometry.resident_graph_rows = 1;
        geometry.local_d_ff = 11;
        geometry.local_n_heads = 5;
        geometry.local_n_kv_heads = 2;
        geometry.mtp_target_query_rows = resolveMTPRetainedTargetQueryRows(mtp);
        geometry.retained_mtp_candidate_rows = geometry.mtp_target_query_rows;
        const auto retained = ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cpu());
        const auto gpu_retained = ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cuda(0));
        geometry.retained_mtp_candidate_rows = 0;
        const auto absent = ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cpu());
        EXPECT_EQ(gpu_retained, ActivationMemoryEstimator::estimate(profile, geometry, DeviceId::cuda(0)));
        const auto rows = static_cast<size_t>(resolveMTPTerminalHiddenRowCapacity(geometry.batch_size, mtp));
        const auto bytes = AlignedVector<float>::requiredAllocationBytes(rows * profile.d_model);
        EXPECT_EQ(retained - absent, bytes);
        auto config = test::cpuArenaFixtureConfig(nullptr, bytes);
        auto authority = config.physical_memory_authority;
        {
            BufferArena arena(config);
            ASSERT_TRUE(arena.registerBuffer(BufferId::PREFIX_TERMINAL_HIDDEN, rows,
                profile.d_model, "FP32", DeviceId::cpu()));
            ASSERT_TRUE(arena.allocate());
            EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
                PhysicalMemoryMaterializationKind::NewAllocation), retained - absent);
        }
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }
}
