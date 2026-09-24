/**
 * @file Test__PlanningForwardStateWork.cpp
 * @brief Device-free context, ownership, codec and arithmetic-proxy regressions.
 *
 * Analytic identities are checked independently of any measured latency. No
 * fixture opens a device or loads weights. Increasing allocated capacity must
 * not increase work; increasing live context must. Replicated decode, uneven
 * TP, hybrid layer mapping and expert-only roles keep their distinct semantics.
 */
#include "planning/PlanningForwardStateWork.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/MemoryPlanner.h"
#include "utils/CPUFeatures.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @return Metadata with four main layers and one retained, unexecuted predictor. */
    PlanningModelMetadata model(bool hybrid = false)
    {
        ModelMemoryProfile profile;
        profile.architecture = "qwen35";
        profile.n_layers = 5;
        profile.mtp_layer_count = 1;
        profile.n_heads = 32;
        profile.n_kv_heads = 4;
        profile.head_dim = 64;
        profile.d_model = 2048;
        profile.d_ff = 4096;
        profile.vocab_size = 4096;
        profile.max_seq_len = 32768;
        if (hybrid)
        {
            profile.full_attention_interval = 2;
            profile.gdn_group_count = 8;
            profile.gdn_time_step_rank = 32;
            profile.gdn_state_size = 64;
            profile.gdn_inner_size = 2048;
            profile.gdn_conv_kernel_size = 4;
        }
        return PlanningModelMetadata(std::move(profile), 4);
    }

    /** @return Exact admitted-like coordinates; this fixture never allocates their capacities. */
    DevicePlanConfig participant(DeviceId device)
    {
        DevicePlanConfig result;
        result.device = device;
        result.max_seq_len = 32768;
        result.activation_seq_len = 256;
        result.kv_precision = "fp16";
        return result;
    }

    /** @return One typed layer, failing the test if classification changes. */
    PlanningAttentionStateWork attention(const PlanningModelMetadata &metadata, const DevicePlanConfig &device,
        int rows, double context, PlanningMainForwardPhase phase = PlanningMainForwardPhase::Prefill)
    {
        const auto work = compilePlanningForwardStateWork(metadata, device, phase, {rows, context});
        if (work.empty()) throw std::logic_error("Fixture lost attention work");
        return std::get<PlanningAttentionStateWork>(work.front().operation);
    }
}

TEST(PlanningForwardStateWork, LiveContextNotAllocatedCapacityDeterminesAttentionWork)
{
    for (const auto endpoint : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto device = participant(endpoint);
        const auto short_work = attention(model(), device, 1, 0, PlanningMainForwardPhase::Decode);
        const auto long_work = attention(model(), device, 1, 8192, PlanningMainForwardPhase::Decode);
        EXPECT_DOUBLE_EQ(short_work.causal_pairs, 1);
        EXPECT_DOUBLE_EQ(long_work.causal_pairs, 8193);
        EXPECT_DOUBLE_EQ(long_work.operations, 8193 * short_work.operations);
        EXPECT_DOUBLE_EQ(long_work.kv_read_bytes, 8193 * short_work.kv_read_bytes);
        EXPECT_DOUBLE_EQ(long_work.kv_write_bytes, short_work.kv_write_bytes);
        EXPECT_DOUBLE_EQ(long_work.activation_bytes, short_work.activation_bytes);
        device.max_seq_len = 16384;
        device.activation_seq_len = 64;
        const auto less_capacity = attention(model(), device, 1, 8192, PlanningMainForwardPhase::Decode);
        EXPECT_DOUBLE_EQ(less_capacity.operations, long_work.operations);
        EXPECT_DOUBLE_EQ(less_capacity.kv_read_bytes, long_work.kv_read_bytes);
        EXPECT_DOUBLE_EQ(short_work.kv_write_bytes, 2 * 4 * 64 * sizeof(uint16_t));
        const auto all = compilePlanningForwardStateWork(model(), device, PlanningMainForwardPhase::Decode, {1, 8192});
        EXPECT_EQ(all.size(), 4u);
        EXPECT_EQ(all.back().layer, 3); // The retained predictor is not main-forward execution.
    }
}

TEST(PlanningForwardStateWork, MeanDecodeAndCausalChunkArithmeticMatchExplicitTokenSums)
{
    const auto device = participant(DeviceId::cpu());
    double serial = 0;
    for (int row = 0; row < 127; ++row)
        serial += attention(model(), device, 1, 4096 + row).operations;
    EXPECT_DOUBLE_EQ(attention(model(), device, 127, 4096).operations, serial);
    EXPECT_DOUBLE_EQ(attention(model(), device, 1, 4096 + 126.0 / 2).operations * 127, serial);
    const auto first = attention(model(), device, 64, 4096);
    const auto second = attention(model(), device, 63, 4160);
    EXPECT_DOUBLE_EQ(first.operations + second.operations, serial);
    // Query-row reuse is an explicit lower-bound traffic model, not repeated
    // full-cache reads for every prefill query or reserved graph row.
    EXPECT_DOUBLE_EQ(first.kv_read_bytes, (4096 + 64) * first.kv_write_bytes / 64);
}

TEST(PlanningForwardStateWork, ExactUnevenHeadsAndReplicatedDecodeStayDifferent)
{
    for (const auto endpoint : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto device = participant(endpoint);
        device.total_shards = 3;
        device.shard_index = 1;
        DeviceShardingAssignment slice;
        slice.device = endpoint;
        slice.local_rank = 1;
        slice.head_start = 11;
        slice.head_count = 11;
        slice.kv_head_start = 1;
        slice.kv_head_count = 2;
        slice.d_ff_count = 1365;
        slice.vocab_count = 1365;
        device.bindTensorParallelAssignment(slice);
        device.additional_weight_sets = {AdditionalPersistentWeightSet::ReplicatedDenseDecode};
        const auto prefill = attention(model(), device, 1, 5);
        const auto decode = attention(model(), device, 1, 5, PlanningMainForwardPhase::Decode);
        EXPECT_EQ(prefill.query_heads, 11);
        EXPECT_EQ(prefill.kv_heads, 2);
        EXPECT_EQ(decode.query_heads, 32);
        EXPECT_EQ(decode.kv_heads, 4);
        EXPECT_DOUBLE_EQ(prefill.operations * 32, decode.operations * 11);
        EXPECT_DOUBLE_EQ(prefill.kv_read_bytes * 2, decode.kv_read_bytes);
    }
}

TEST(PlanningForwardStateWork, UniformTPThroughDegreeEightPreservesArithmeticAndReplicatedGQA)
{
    for (const auto endpoint : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (int degree : {1, 2, 4, 8})
    {
        auto device = participant(endpoint);
        const auto complete = attention(model(), device, 32, 2048);
        device.total_shards = degree;
        device.local_kv_heads = std::max(1, 4 / degree);
        double operations = 0, bytes = 0;
        for (int shard = 0; shard < degree; ++shard)
        {
            device.shard_index = shard;
            const auto local = attention(model(), device, 32, 2048);
            operations += local.operations;
            bytes += local.kv_read_bytes;
        }
        EXPECT_DOUBLE_EQ(operations, complete.operations);
        EXPECT_DOUBLE_EQ(bytes, complete.kv_read_bytes * std::max(1, degree / 4));
    }
}

TEST(PlanningForwardStateWork, HybridUsesCanonicalMappingAndOnlyLocalLiveBanks)
{
    const auto metadata = model(true);
    for (const auto endpoint : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto device = participant(endpoint);
        device.total_shards = 2;
        device.shard_index = 1;
        device.local_kv_heads = 2;
        const auto early = compilePlanningForwardStateWork(metadata, device, PlanningMainForwardPhase::Decode, {1, 0});
        const auto late = compilePlanningForwardStateWork(metadata, device, PlanningMainForwardPhase::Decode, {1, 8192});
        ASSERT_EQ(early.size(), 4u);
        for (size_t layer = 0; layer < 4; ++layer)
        {
            if (layer % 2)
            {
                EXPECT_TRUE(std::holds_alternative<PlanningAttentionStateWork>(early[layer].operation));
                continue;
            }
            const auto &a = std::get<PlanningGDNStateWork>(early[layer].operation);
            const auto &b = std::get<PlanningGDNStateWork>(late[layer].operation);
            const auto geometry = HybridGDNStateGeometry::resolve(32, 16, 16, 8, 32, 64, 2048, 4);
            EXPECT_EQ(a.geometry.local_recurrence_state_floats, geometry.local_recurrence_state_floats);
            EXPECT_DOUBLE_EQ(a.state_bytes, 2 * geometry.localPayloadBytes(1));
            EXPECT_DOUBLE_EQ(a.operations, 7 * 65536 + 3 * 1024 + 2 * 4 * 1536 + 10 * 4 * 64);
            EXPECT_DOUBLE_EQ(a.operations, b.operations);
            EXPECT_DOUBLE_EQ(a.state_bytes, b.state_bytes);
        }
        const auto grouped = compilePlanningForwardStateWork(metadata, device, PlanningMainForwardPhase::Prefill, {64, 512});
        const auto &serial_state = std::get<PlanningGDNStateWork>(early[0].operation);
        const auto &grouped_state = std::get<PlanningGDNStateWork>(grouped[0].operation);
        EXPECT_DOUBLE_EQ(grouped_state.operations, 64 * serial_state.operations);
        EXPECT_DOUBLE_EQ(grouped_state.state_bytes, serial_state.state_bytes);
        EXPECT_DOUBLE_EQ(grouped_state.activation_bytes, 64 * serial_state.activation_bytes);
        device.first_layer = 1;
        device.last_layer = 1;
        const auto pp = compilePlanningForwardStateWork(metadata, device, PlanningMainForwardPhase::Prefill, {64, 512});
        ASSERT_EQ(pp.size(), 1u);
        EXPECT_TRUE(std::holds_alternative<PlanningAttentionStateWork>(pp[0].operation));
    }
}

TEST(PlanningForwardStateWork, ExpertOnlyHasNoStateAndInvalidInvocationsFailClosed)
{
    auto device = participant(DeviceId::cpu());
    device.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant;
    EXPECT_TRUE(compilePlanningForwardStateWork(model(), device, PlanningMainForwardPhase::Decode, {1, 32}).empty());
    device.execution_role = DeviceExecutionMemoryRole::ContinuationGraph;
    EXPECT_THROW(PlanningForwardInvocation(0, 0), std::invalid_argument);
    EXPECT_THROW(PlanningForwardInvocation(1, -1), std::invalid_argument);
    EXPECT_THROW(PlanningForwardInvocation(1, std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(PlanningForwardInvocation(1, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(compilePlanningForwardStateWork(model(), device, PlanningMainForwardPhase::Decode, {2, 32}), std::invalid_argument);
    EXPECT_THROW(compilePlanningForwardStateWork(model(), device, PlanningMainForwardPhase::Prefill, {512, 32}), std::invalid_argument);
    EXPECT_THROW(compilePlanningForwardStateWork(model(), device, PlanningMainForwardPhase::Decode, {1, 32768}), std::invalid_argument);
    device.total_shards = 3;
    device.local_kv_heads = 1;
    EXPECT_THROW(compilePlanningForwardStateWork(model(), device, PlanningMainForwardPhase::Decode, {1, 0}), std::invalid_argument);
}

TEST(PlanningForwardStateWork, QualifiedProxyUsesIndependentBandwidthAndExactCPUWorkshare)
{
    const auto work = compilePlanningForwardStateWork(model(), participant(DeviceId::cpu()), PlanningMainForwardPhase::Decode, {1, 8192})[0];
    RankInventory rank;
    rank.rank = 0;
    rank.cpu_cores = 16;
    rank.cpu_worker_threads = 3;
    rank.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    rank.cpu.numa_node = 0;
    rank.cpu.last_level_cache_bytes = 32u << 20;
    CPUSocketInfo socket;
    socket.socket_id = socket.numa_node = 0;
    socket.physical_cores = {0, 1, 2};
    rank.cpu_socket_info.push_back(socket);
    const auto request = PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu());
    PlanningMemoryBandwidthObservation memory{request, 0, {PlanningWorkUnit::Bytes, 1000, 1, "synthetic streaming fixture"}};
    PlanningFP32ArithmeticObservations arithmetic{DeviceId::cpu(), 0,
        PlanningProjectionCPUObservation{3, ISALevel::AVX512, ISALevel::AVX512, rank.cpu_execution},
        {{1, 0, {PlanningWorkUnit::ArithmeticOperations, 1e12, 1, "synthetic FP32 fixture"}},
         {64, 0, {PlanningWorkUnit::ArithmeticOperations, 1e12, 1, "synthetic FP32 fixture"}}}};
    const auto &attention = std::get<PlanningAttentionStateWork>(work.operation);
    EXPECT_DOUBLE_EQ(planningStateServiceSeconds(work, arithmetic, memory),
        (attention.kv_read_bytes + attention.kv_write_bytes + attention.activation_bytes) / 1000);
    memory.service = {PlanningWorkUnit::Bytes, 1e15, 1, "synthetic fast streaming fixture"};
    EXPECT_DOUBLE_EQ(planningStateServiceSeconds(work, arithmetic, memory), attention.operations / 1e12);
    auto malformed = work;
    auto &demand = std::get<PlanningAttentionStateWork>(malformed.operation);
    demand.kv_write_bytes = -1; // A positive read sum must not hide a malformed component.
    EXPECT_THROW(planningStateServiceSeconds(malformed, arithmetic, memory), std::invalid_argument);
    malformed = work;
    std::get<PlanningAttentionStateWork>(malformed.operation).operations = 0;
    EXPECT_THROW(planningStateServiceSeconds(malformed, arithmetic, memory), std::invalid_argument);
    arithmetic.cpu->workers = 16;
    EXPECT_THROW(planningStateServiceSeconds(work, arithmetic, memory), std::invalid_argument);
    arithmetic.cpu->workers = 3;
    arithmetic.device = DeviceId::cuda(0);
    EXPECT_THROW(planningStateServiceSeconds(work, arithmetic, memory), std::invalid_argument);
}
