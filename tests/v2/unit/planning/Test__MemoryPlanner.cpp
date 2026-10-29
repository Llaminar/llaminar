/**
 * @file Test__MemoryPlanner.cpp
 * @brief Unit coverage for metadata-only device memory admission.
 *
 * These tests pin the boundary between a declarative execution placement and
 * the bytes that its real graph allocates.  In particular, mixed ExpertOverlay
 * continuation endpoints must reserve both dense graph state and their own
 * stable sparse-route buffers before captured graph construction begins.
 * Native FP16 admission also authenticates the configured codec and declared
 * schema visibility against exact local and replicated attention geometry.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */

#include <gtest/gtest.h>
#include "planning/MemoryPlanner.h"
#include "memory/CPUCompactPacketStorageLayout.h"
#include "planning/ServingSnapshotArenaInventory.h"
#include "planning/RankMemoryPlanInputs.h"
#include "loaders/ImmutableRowTable.h"
#include "planning/PlanningModelMetadata.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "memory/CPUGroupedMoEStorageLayout.h"
#include "memory/CPURouterQ8StorageLayout.h"
#include "utils/DebugEnv.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "models/qwen35/Qwen35Schema.h"
#include "collective/DeviceCountedAllGather.h"
#include "collective/VocabularyGatherWorkspaceContract.h"
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/ActivationBufferSizing.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "kernels/attention/AttentionWorkspaceContract.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "kernels/HybridGDNStateGeometry.h"
#include "backends/DeviceId.h"

#include <memory>
#include <set>
#include "execution/compute_stages/stages/GDNLiveStateAllGatherStage.h"
#include "../../mocks/MockLocalTPContext.h"
#include <string>
#include <stdexcept>
#include <utility>

using namespace llaminar2;

namespace
{

    ModelMemoryProfile createTestProfile()
    {
        ModelMemoryProfile p;
        p.architecture = "qwen2";
        p.n_layers = 24;
        p.d_model = 896;
        p.d_ff = 4864;
        p.n_heads = 14;
        p.n_kv_heads = 2;
        p.head_dim = 64;
        p.vocab_size = 151936;
        p.max_seq_len = 4096;
        p.total_native_bytes = 0;

        auto addTensor = [&](const std::string &name, size_t rows, size_t cols,
                             const std::string &quant, int layer = -1)
        {
            TensorSizeInfo t;
            t.name = name;
            t.elements = rows * cols;
            t.K = cols;
            t.quant_type = quant;
            if (quant == "Q8_0")
                t.native_bytes = rows * cols * 34 / 32;
            else if (quant == "F32")
                t.native_bytes = rows * cols * 4;
            else
                t.native_bytes = rows * cols;
            t.layer_index = layer;
            p.total_native_bytes += t.native_bytes;
            p.tensors.push_back(t);
        };

        addTensor("token_embd.weight", 151936, 896, "Q8_0");
        addTensor("output.weight", 151936, 896, "Q8_0");
        addTensor("output_norm.weight", 1, 896, "F32");

        for (int layer = 0; layer < 24; ++layer)
        {
            std::string prefix = "blk." + std::to_string(layer) + ".";
            addTensor(prefix + "attn_q.weight", 896, 896, "Q8_0", layer);
            addTensor(prefix + "attn_k.weight", 128, 896, "Q8_0", layer);
            addTensor(prefix + "attn_v.weight", 128, 896, "Q8_0", layer);
            addTensor(prefix + "attn_output.weight", 896, 896, "Q8_0", layer);
            addTensor(prefix + "ffn_gate.weight", 4864, 896, "Q8_0", layer);
            addTensor(prefix + "ffn_up.weight", 4864, 896, "Q8_0", layer);
            addTensor(prefix + "ffn_down.weight", 896, 4864, "Q8_0", layer);
            addTensor(prefix + "attn_norm.weight", 1, 896, "F32", layer);
            addTensor(prefix + "ffn_norm.weight", 1, 896, "F32", layer);
        }

        return p;
    }

    ModelMemoryProfile createQwen36DenseLikeProfile(size_t device_weight_bytes)
    {
        ModelMemoryProfile p;
        p.architecture = "qwen35";
        p.n_layers = 65;
        p.d_model = 5120;
        p.d_ff = 27648;
        p.n_heads = 40;
        p.n_kv_heads = 8;
        p.head_dim = 128;
        p.vocab_size = 151936;
        p.max_seq_len = 4096;
        p.total_native_bytes = device_weight_bytes;

        TensorSizeInfo t;
        t.name = "synthetic_dense_weights";
        t.quant_type = "F32";
        t.elements = device_weight_bytes / 4;
        t.K = 1;
        t.native_bytes = t.elements * 4;
        t.layer_index = -1;
        p.tensors.push_back(t);

        return p;
    }

    ModelMemoryProfile createQwen36HybridMTPStateProfile()
    {
        ModelMemoryProfile profile;
        profile.architecture = "qwen35";
        profile.n_layers = 65;
        profile.mtp_layer_count = 1;
        profile.d_model = 5120;
        profile.d_ff = 27648;
        profile.n_heads = 40;
        profile.n_kv_heads = 8;
        profile.head_dim = 128;
        profile.vocab_size = 248320;
        profile.max_seq_len = 4096;
        profile.full_attention_interval = 4;
        profile.gdn_conv_kernel_size = 4;
        profile.gdn_state_size = 128;
        profile.gdn_inner_size = 6144;
        profile.gdn_group_count = 16;
        profile.gdn_time_step_rank = 48;

        for (int layer = 0; layer < profile.n_layers; ++layer)
        {
            TensorSizeInfo attention_marker;
            attention_marker.name =
                "blk." + std::to_string(layer) +
                (layer >= 64 || (layer + 1) % 4 == 0
                     ? ".attn_q.weight"
                     : ".attn_qkv.weight");
            attention_marker.layer_index = layer;
            profile.tensors.push_back(std::move(attention_marker));
        }
        return profile;
    }

    ModelMemoryProfile createMoEOverlayProfile()
    {
        ModelMemoryProfile profile;
        profile.architecture = "qwen35moe";
        profile.n_layers = 2;
        profile.d_model = 32;
        profile.d_ff = 64;
        profile.n_heads = 4;
        profile.n_kv_heads = 2;
        profile.head_dim = 8;
        profile.vocab_size = 100;
        profile.max_seq_len = 64;
        profile.expert_count = 8;
        profile.expert_used_count = 2;
        profile.expert_feed_forward_length = 64;
        profile.expert_shared_feed_forward_length = 64;

        auto addF32 = [&](const std::string &name,
                          size_t elements,
                          size_t k,
                          int layer)
        {
            TensorSizeInfo tensor;
            tensor.name = name;
            tensor.elements = elements;
            tensor.K = k;
            tensor.quant_type = "F32";
            tensor.native_bytes = elements * sizeof(float);
            tensor.layer_index = layer;
            profile.total_native_bytes += tensor.native_bytes;
            profile.tensors.push_back(std::move(tensor));
        };

        addF32("token_embd.weight", 100u * 32u, 32, -1);
        addF32("output.weight", 100u * 32u, 32, -1);
        for (int layer = 0; layer < profile.n_layers; ++layer)
        {
            const std::string prefix = "blk." + std::to_string(layer);
            addF32(prefix + ".attn_q.weight", 32u * 32u, 32, layer);
            addF32(prefix + ".ffn_gate_shexp.weight", 64u * 32u, 32, layer);
            addF32(
                prefix + ".ffn_gate_exps.weight",
                8u * 64u * 32u,
                32,
                layer);
            addF32(
                prefix + ".ffn_up_exps.weight",
                8u * 64u * 32u,
                32,
                layer);
            addF32(
                prefix + ".ffn_down_exps.weight",
                8u * 32u * 64u,
                64,
                layer);
        }
        return profile;
    }

    DevicePlanConfig overlayDeviceConfig(DeviceId device)
    {
        DevicePlanConfig config;
        config.device = device;
        config.cpu_execution = device.is_cpu() ? test::kSyntheticCPUExecutionGeometry : CPUExecutionGeometry{};
        config.device_compute_units = device.is_cuda() ? 82 : 60;
        config.device_total_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        config.device_free_bytes = config.device_total_bytes;
        config.first_layer = 0;
        config.last_layer = 1;
        config.batch_size = 1;
        config.max_seq_len = 64;
        config.activation_seq_len = 32;
        config.kv_precision = "fp16";
        return config;
    }

/** @brief Author this fixture's exact CPU endpoint/controller and bounded family policy. */
    void declareCPUSerialFixture(DevicePlanConfig &config, const ModelMemoryProfile &profile)
    {
        if (!config.device.is_cpu()) return;
        const int rows = config.serial_routed_expert_compact_rows > 0
            ? config.serial_routed_expert_compact_rows : config.batch_size * config.activation_seq_len;
        auto admission = std::make_shared<CPUSerialExpertStorageAdmission>();
        admission->allocator = {config.world_rank, config.device};
        for (int participant = 0; participant < config.serial_routed_expert_participant_count; ++participant)
            admission->participant_ids.push_back(participant);
        admission->controller_devices = {config.device};
        admission->envelope = std::make_shared<const CPUSerialExpertStorageEnvelope>(
            config.device, rows, 1, profile.d_model, profile.expert_used_count,
            std::vector<int>{1,2,4,8,16,32,64,128,256,512,1024,2048,4096}, 1, 16,
            true, CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,
            profile.expert_feed_forward_length, profile.expert_count);
        config.cpu_serial_expert_storage = std::move(admission);
    }
} // anonymous namespace

/** @test Canonical physical admission charges each compact domain edge once on CPU and GPU. */
TEST(Test__MemoryPlanner, CountedProjectionTransportHasOneCanonicalPhysicalBOM)
{
    auto profile = createMoEOverlayProfile();
    // TP8 needs eight semantic query heads. Keep the original matrix width by
    // shrinking each synthetic head, rather than bypassing sharding validation.
    profile.n_heads = 8;
    profile.head_dim = 4;
    constexpr auto staging_owner = PhysicalMemoryOwner::ActivationTransportStaging;
    for (const auto first : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int degree : {2, 4, 8})
    for (const auto coverage : {PeerAccessCoverage::None, PeerAccessCoverage::Partial, PeerAccessCoverage::Complete})
    {
        std::vector<DevicePlanConfig> configs;
        for (int participant = 0; participant < degree; ++participant)
        {
            auto config = overlayDeviceConfig(first.is_cuda() ? DeviceId::cuda(participant) : DeviceId::rocm(participant));
            config.shard_index = participant; config.total_shards = degree;
            config.local_tp_backend = first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            config.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
            config.projection_peer_access = coverage;
            config.associated_host_memory = PhysicalMemoryResource{.world_rank = -1, .device = DeviceId::cpu(),
                .total_bytes = 1ull << 40, .admission_available_bytes = 1ull << 40};
            configs.push_back(config);
        }
        const auto plan = MemoryPlanner::plan(profile, configs);
        const auto geometry = DeviceCountedAllGather::memoryFor(degree,
            MoEProjectionArenaGeometry::resolve(profile, degree).packetCapacityBytes(32));
        std::size_t host_bytes = 0, device_bytes = 0;
        for (const auto &row : plan.devices)
            (row.device().is_cpu() ? host_bytes : device_bytes) += row.bom().bytes(staging_owner);
        EXPECT_EQ(host_bytes, coverage == PeerAccessCoverage::None ? degree * geometry.host_bytes : 0u);
        EXPECT_EQ(device_bytes, coverage == PeerAccessCoverage::None ? degree * geometry.device_bytes : 0u);
    }
}

/**
 * @brief A weight shard does not imply a rank-local collective allocation.
 *
 * Node/global TP can install one device on each rank. The weight geometry is
 * still sharded, but no LocalTP context exists on that rank. Exercise the same
 * metadata-only admission contract on CPU, CUDA and ROCm without using devices.
 */
TEST(Test__MemoryPlanner, DistributedTPDoesNotInventLocalCollective)
{
    const auto profile = createMoEOverlayProfile();
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(device.toString());
        auto config = overlayDeviceConfig(device);
        config.total_shards = 2;
        config.shard_index = 0;
        const auto plan = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(plan.devices.size(), 1u);
        EXPECT_EQ(plan.devices.front().collective_bytes(), 0u);
        EXPECT_GT(plan.devices.front().weight_bytes(), 0u);
        EXPECT_GT(plan.devices.front().activation_bytes(), 0u);
    }
}

/** A present-but-unresolved local collective is not the absent NodeTP case. */
TEST(Test__MemoryPlanner, UnresolvedLocalCollectiveStillFailsAdmission)
{
    const auto profile = createMoEOverlayProfile();
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(device.toString());
        auto config = overlayDeviceConfig(device);
        config.total_shards = 2;
        config.local_tp_backend = CollectiveBackendType::AUTO;
        EXPECT_THROW((void)MemoryPlanner::plan(profile, {config}), std::invalid_argument);
    }
}

TEST(Test__MemoryPlanner, SingleGPU_Fits)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024; // 24 GB
    cfg.device_free_bytes = 23ULL * 1024 * 1024 * 1024;  // 23 GB free
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";

    auto plan = MemoryPlanner::plan(profile, {cfg});

    EXPECT_EQ(plan.devices.size(), 1u);
    EXPECT_GT(plan.devices[0].weight_bytes(), 0u);
    EXPECT_GT(plan.devices[0].kv_cache_bytes(), 0u);
    EXPECT_GT(plan.devices[0].activation_bytes(), 0u);
    EXPECT_GT(plan.devices[0].workspace_bytes(), 0u);
    EXPECT_TRUE(plan.devices[0].fits());
    EXPECT_TRUE(plan.fits());
    ASSERT_EQ(plan.physicalPlan().resources().size(), 1u);
    EXPECT_EQ(
        &plan.physicalPlan().resources().front(),
        plan.physicalPlan().find({-1, DeviceId::cuda(0)}));
    EXPECT_NO_THROW((void)plan.admit());
}

TEST(Test__MemoryPlanner, Qwen36DenseLike_UsesTerminalLogitsAndPreparedEmbeddingWorkspace)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    const size_t qwen36_dense_weight_bytes = static_cast<size_t>(16.3 * static_cast<double>(GiB));
    auto profile = createQwen36DenseLikeProfile(qwen36_dense_weight_bytes);

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 24ULL * GiB;
    cfg.device_free_bytes = static_cast<size_t>(23.3 * static_cast<double>(GiB));
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";

    auto plan = MemoryPlanner::plan(profile, {cfg});

    ASSERT_EQ(plan.devices.size(), 1u);
    const auto &device_plan = plan.devices.front();
    EXPECT_TRUE(device_plan.fits()) << plan.renderTable();
    EXPECT_LT(device_plan.activation_bytes(), 2ULL * GiB)
        << "normal prefill must not reserve full-context all-position logits";
    EXPECT_GT(device_plan.workspace_bytes(), 768ULL * 1024ULL * 1024ULL)
        << "large grouped-prefill accumulators must exceed the minimum floor";
    EXPECT_LT(
        device_plan.workspace_bytes(),
        size_t{151936} * size_t{5120} * sizeof(float))
        << "prepared embedding must not reserve a vocab-by-hidden fallback table";
}

TEST(Test__MemoryPlanner, Qwen36HybridMTP_AccountsExactKVAndPersistentState)
{
    auto profile = createQwen36HybridMTPStateProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";
    cfg.mtp_enabled = true;
    cfg.prefix_cache = PrefixCacheRuntimeConfig{};
    cfg.associated_host_memory = PhysicalMemoryResource{
        .world_rank = -1,
        .device = DeviceId::cpu(),
        .total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL,
        .admission_available_bytes =
            64ULL * 1024ULL * 1024ULL * 1024ULL,
    };

    const auto plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan.devices.size(), 2u);
    const auto gpu = std::find_if(
        plan.devices.begin(), plan.devices.end(), [](const auto &candidate)
        { return candidate.device().is_cuda(); });
    const auto host = std::find_if(
        plan.devices.begin(), plan.devices.end(), [](const auto &candidate)
        { return candidate.device().is_cpu(); });
    ASSERT_NE(gpu, plan.devices.end());
    ASSERT_NE(host, plan.devices.end());
    const auto &device = *gpu;
    EXPECT_EQ(
        host->bom().bytes(PhysicalMemoryOwner::PrefixHostTier),
        cfg.prefix_cache.ram_budget_bytes);
    EXPECT_EQ(
        host->bom().bytes(PhysicalMemoryOwner::PrefixArchiveStaging),
        PrefixArchiveIOGeometry::scratchBytes())
        << "Tiered disk restore owns one bounded host I/O window.";

    const HybridGDNStateGeometry gdn_geometry =
        HybridGDNStateGeometry::resolve(
            profile.n_heads,
            /*local_head_start=*/0,
            /*local_attention_heads=*/profile.n_heads,
            profile.gdn_group_count,
            profile.gdn_time_step_rank,
            profile.gdn_state_size,
            profile.gdn_inner_size,
            profile.gdn_conv_kernel_size);
    const size_t expected_main_gdn_payload =
        gdn_geometry.deviceSerializedPayloadBytes(/*gdn_layers=*/48);
    const size_t expected_live_gdn_arena =
        gdn_geometry.deviceArenaBytes(
            /*gdn_layers=*/48,
            cfg.batch_size);
    constexpr size_t expected_terminal_hidden =
        5120ULL * sizeof(float);
    const size_t expected_kv = KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
        /*n_layers=*/17,
        /*batch_size=*/1,
        /*max_seq_len=*/4096,
        /*n_kv_heads=*/8,
        /*head_dim=*/128,
        "fp16",
        DeviceId::cuda(0));

    EXPECT_EQ(device.kv_cache_bytes(), expected_kv)
        << "Only 16 main FA layers plus one shifted MTP FA layer own KV.";
    EXPECT_EQ(
        device.live_recurrent_state_bytes(),
        expected_live_gdn_arena);
    EXPECT_EQ(
        device.checkpoint_state_bytes(),
        4ULL *
            (expected_main_gdn_payload +
             expected_terminal_hidden))
        << "The shifted FA cache must not reserve a recurrent payload slot.";
    EXPECT_GT(
        device.persistent_state_bytes(),
        device.live_recurrent_state_bytes() +
            device.checkpoint_state_bytes())
        << "Device sequence metadata must remain explicitly accounted.";

    const auto logical_prefix =
        KVCacheMemoryEstimator::estimateGPULogicalBlock(KVCacheFamily::AttentionOnly,
            cfg.prefix_cache.block_size,
            profile.n_kv_heads,
            profile.head_dim,
            cfg.kv_precision,
            cfg.device);
    EXPECT_EQ(
        device.prefix_cache_staging_bytes(),
        expected_main_gdn_payload +
            2ULL * logical_prefix.totalBytes())
        << "Main KV, hybrid state, and shifted-MTP KV each own one archive slot.";

    const size_t expected_prefix_block =
        17ULL * logical_prefix.totalBytes() +
        expected_main_gdn_payload + expected_terminal_hidden +
        static_cast<size_t>(profile.vocab_size) * sizeof(float);
    EXPECT_EQ(
        device.prefix_cache_device_hot_bytes(),
        prefixCacheWholeBlockReservationBytes(
            cfg.prefix_cache.device_budget_bytes,
            expected_prefix_block))
        << "Admission must charge the same whole-block reservation that the "
           "device-hot backend can physically allocate.";

    DevicePlanConfig rocm_cfg = cfg;
    rocm_cfg.device = DeviceId::rocm(0);
    const auto rocm_plan = MemoryPlanner::plan(profile, {rocm_cfg});
    ASSERT_EQ(rocm_plan.devices.size(), 2u);
    const auto rocm = std::find_if(
        rocm_plan.devices.begin(),
        rocm_plan.devices.end(),
        [](const auto &candidate) { return candidate.device().is_rocm(); });
    ASSERT_NE(rocm, rocm_plan.devices.end());
    EXPECT_EQ(
        rocm->prefix_cache_staging_bytes(),
        device.prefix_cache_staging_bytes());
    EXPECT_EQ(
        rocm->prefix_cache_device_hot_bytes(),
        device.prefix_cache_device_hot_bytes());
}

/**
 * @brief Qwen 3.8 charges KV slots only for its sixteen full-attention layers.
 *
 * A Qwen 3.8 27B GGUF has 64 main layers plus one trailing MTP block, with
 * one main full-attention layer every four layers. Its Gated DeltaNet layers
 * own bounded recurrent state rather than a context-sized K/V history. This
 * deliberately uses the reported FP16 64k ROCm configuration: charging all
 * 64 main layers would turn the expected four-gibibyte cache into a
 * sixteen-gibibyte allocation and incorrectly reject a 32 GiB MI50.
 */
TEST(Test__MemoryPlanner,
     Qwen38Hybrid64K_ROCmChargesOnlyFullAttentionLayersForKV)
{
    auto profile = createQwen36HybridMTPStateProfile();
    profile.n_heads = 24;
    profile.n_kv_heads = 4;
    profile.head_dim = 256;

    DevicePlanConfig cfg;
    cfg.device = DeviceId::rocm(0);
    cfg.device_compute_units = 60;
    cfg.device_total_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 1;
    cfg.max_seq_len = 65536;
    cfg.kv_precision = "fp16";
    cfg.mtp_enabled = false;

    const auto plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan.devices.size(), 1u);
    const auto &device = plan.devices.front();
    const size_t expected_hybrid_kv = KVCacheMemoryEstimator::estimate(
        KVCacheFamily::Hybrid,
        /*n_layers=*/16,
        cfg.batch_size,
        cfg.max_seq_len,
        profile.n_kv_heads,
        profile.head_dim,
        cfg.kv_precision,
        cfg.device);
    const size_t incorrect_all_attention_kv = KVCacheMemoryEstimator::estimate(
        KVCacheFamily::Hybrid,
        /*n_layers=*/64,
        cfg.batch_size,
        cfg.max_seq_len,
        profile.n_kv_heads,
        profile.head_dim,
        cfg.kv_precision,
        cfg.device);

    EXPECT_EQ(device.kv_cache_bytes(), expected_hybrid_kv);
    EXPECT_EQ(expected_hybrid_kv, 4ULL * 1024ULL * 1024ULL * 1024ULL + 384ULL);
    EXPECT_EQ(incorrect_all_attention_kv, 4ULL * expected_hybrid_kv)
        << "K/V payload and its per-layer sequence metadata scale together.";
    EXPECT_GT(incorrect_all_attention_kv, 15ULL * 1024ULL * 1024ULL * 1024ULL);
}

/**
 * @brief A replicated TP predictor owns a full-width shifted cache.
 *
 * The committed Qwen hybrid cache remains TP-sharded, but the learned MTP
 * predictor is a complete per-participant block and consequently projects all
 * KV heads. This reproduces the 2xROCm ExpertOverlay admission failure where
 * runtime requested an 8 MiB shifted cache after planning only its 4 MiB
 * sharded counterpart.
 */
TEST(Test__MemoryPlanner,
     ReplicatedMTPShiftedCacheUsesFullKVHeadsBesideShardedMain)
{
    const auto profile = createQwen36HybridMTPStateProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::rocm(0);
    cfg.device_compute_units = 60;
    cfg.device_total_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";
    cfg.total_shards = 2;
    cfg.shard_index = 0;
    cfg.local_kv_heads = 4;
    cfg.local_tp_backend = CollectiveBackendType::RCCL;
    cfg.mtp_enabled = true;
    cfg.mtp_shifted_kv_head_layout =
        MTPShiftedKVHeadLayout::FullModelPerParticipant;
    cfg.prefix_cache = PrefixCacheRuntimeConfig{};
    cfg.associated_host_memory = PhysicalMemoryResource{
        .world_rank = -1,
        .device = DeviceId::cpu(),
        .total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL,
        .admission_available_bytes =
            64ULL * 1024ULL * 1024ULL * 1024ULL,
    };

    const auto replicated_plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(replicated_plan.devices.size(), 2u);
    const auto replicated = std::find_if(
        replicated_plan.devices.begin(),
        replicated_plan.devices.end(),
        [](const auto &candidate) { return candidate.device().is_rocm(); });
    ASSERT_NE(replicated, replicated_plan.devices.end());

    const size_t expected_main = KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
        /*n_layers=*/16,
        cfg.batch_size,
        cfg.max_seq_len,
        /*n_kv_heads=*/4,
        profile.head_dim,
        cfg.kv_precision,
        cfg.device);
    const size_t expected_shifted = KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
        /*n_layers=*/1,
        cfg.batch_size,
        cfg.max_seq_len,
        /*n_kv_heads=*/8,
        profile.head_dim,
        cfg.kv_precision,
        cfg.device);
    EXPECT_EQ(
        replicated->kv_cache_bytes(),
        expected_main + expected_shifted);

    const auto main_prefix =
        KVCacheMemoryEstimator::estimateGPULogicalBlock(KVCacheFamily::AttentionOnly,
            cfg.prefix_cache.block_size,
            /*n_kv_heads=*/4,
            profile.head_dim,
            cfg.kv_precision,
            cfg.device);
    const auto shifted_prefix =
        KVCacheMemoryEstimator::estimateGPULogicalBlock(KVCacheFamily::AttentionOnly,
            cfg.prefix_cache.block_size,
            /*n_kv_heads=*/8,
            profile.head_dim,
            cfg.kv_precision,
            cfg.device);
    const HybridGDNStateGeometry gdn_geometry =
        HybridGDNStateGeometry::resolve(
            profile.n_heads,
            /*local_head_start=*/0,
            /*local_attention_heads=*/20,
            profile.gdn_group_count,
            profile.gdn_time_step_rank,
            profile.gdn_state_size,
            profile.gdn_inner_size,
            profile.gdn_conv_kernel_size);
    EXPECT_EQ(
        replicated->prefix_cache_staging_bytes(),
        main_prefix.totalBytes() + shifted_prefix.totalBytes() +
            gdn_geometry.deviceSerializedPayloadBytes(
                /*gdn_layers=*/48));

    DevicePlanConfig sharded_cfg = cfg;
    sharded_cfg.mtp_shifted_kv_head_layout =
        MTPShiftedKVHeadLayout::PrimaryTensorParallelShard;
    const auto sharded_plan = MemoryPlanner::plan(profile, {sharded_cfg});
    const auto sharded = std::find_if(
        sharded_plan.devices.begin(),
        sharded_plan.devices.end(),
        [](const auto &candidate) { return candidate.device().is_rocm(); });
    ASSERT_NE(sharded, sharded_plan.devices.end());
    EXPECT_EQ(
        sharded->kv_cache_bytes(),
        expected_main + KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
                            /*n_layers=*/1,
                            cfg.batch_size,
                            cfg.max_seq_len,
                            /*n_kv_heads=*/4,
                            profile.head_dim,
                            cfg.kv_precision,
                            cfg.device));
    EXPECT_LT(
        sharded->kv_cache_bytes(),
        replicated->kv_cache_bytes());
}

TEST(Test__MemoryPlanner,
     HybridGDNStateUsesTheExactModuloLinkedParticipantGeometry)
{
    const auto profile = createQwen36HybridMTPStateProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::rocm(0);
    cfg.device_compute_units = 60;
    cfg.device_total_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 2;
    cfg.generation_request_capacity = 2; // The two declared request banks own the retained rows.
    cfg.max_seq_len = 128;
    cfg.kv_precision = "fp16";
    cfg.total_shards = 4;
    cfg.local_tp_backend = CollectiveBackendType::RCCL;
    cfg.shard_index = 1;
    cfg.mtp_enabled = true;

    const auto plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan.devices.size(), 1u);

    const HybridGDNStateGeometry geometry =
        HybridGDNStateGeometry::resolve(
            profile.n_heads,
            /*local_head_start=*/10,
            /*local_attention_heads=*/10,
            profile.gdn_group_count,
            profile.gdn_time_step_rank,
            profile.gdn_state_size,
            profile.gdn_inner_size,
            profile.gdn_conv_kernel_size);
    EXPECT_EQ(geometry.local_key_heads, 4);
    EXPECT_EQ(geometry.local_value_heads, 12);
    EXPECT_EQ(
        plan.devices.front().live_recurrent_state_bytes(),
        geometry.deviceArenaBytes(
            /*gdn_layers=*/48,
            cfg.batch_size));

    const size_t device_checkpoint_payload =
        geometry.deviceSerializedPayloadBytes(/*gdn_layers=*/48);
    const size_t host_checkpoint_payload =
        geometry.localPayloadBytes(/*gdn_layers=*/48);
    const size_t terminal_hidden =
        static_cast<size_t>(profile.d_model) * sizeof(float);
    ASSERT_GT(device_checkpoint_payload, host_checkpoint_payload)
        << "TP-local and full banks must differ or this regression cannot "
           "detect under-admission.";
    EXPECT_EQ(
        plan.devices.front().checkpoint_state_bytes(),
        4ULL * (device_checkpoint_payload + terminal_hidden))
        << "ROCm rollback admission must include every distinct device bank.";

    DevicePlanConfig cuda_cfg = cfg;
    cuda_cfg.device = DeviceId::cuda(0);
    const auto cuda_plan = MemoryPlanner::plan(profile, {cuda_cfg});
    ASSERT_EQ(cuda_plan.devices.size(), 1u);
    EXPECT_EQ(
        cuda_plan.devices.front().checkpoint_state_bytes(),
        plan.devices.front().checkpoint_state_bytes())
        << "CUDA and ROCm serialize the same packed recurrent-state shape.";

    DevicePlanConfig cpu_cfg = cfg;
    cpu_cfg.device = DeviceId::cpu();
    cpu_cfg.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    const auto cpu_plan = MemoryPlanner::plan(profile, {cpu_cfg});
    ASSERT_EQ(cpu_plan.devices.size(), 1u);
    EXPECT_EQ(
        cpu_plan.devices.front().checkpoint_state_bytes(),
        4ULL * (host_checkpoint_payload + terminal_hidden))
        << "CPU rollback owns only its host-local recurrent vectors.";
}

TEST(Test__MemoryPlanner, TerminalParticipantOwnsTrailingMTPWeights)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen35";
    profile.n_layers = 3;
    profile.mtp_layer_count = 1;
    profile.d_model = 32;
    profile.d_ff = 64;
    profile.n_heads = 1;
    profile.n_kv_heads = 1;
    profile.head_dim = 32;
    profile.vocab_size = 64;
    profile.max_seq_len = 16;

    for (int layer = 0; layer < profile.n_layers; ++layer)
    {
        TensorSizeInfo tensor;
        tensor.name =
            "blk." + std::to_string(layer) + ".ffn_gate.weight";
        tensor.elements = 32 * 32;
        tensor.K = 32;
        tensor.quant_type = "F32";
        tensor.native_bytes = tensor.elements * sizeof(float);
        tensor.layer_index = layer;
        profile.total_native_bytes += tensor.native_bytes;
        profile.tensors.push_back(std::move(tensor));
    }

    DevicePlanConfig config;
    config.device = DeviceId::cuda(0);
    config.device_compute_units = 82;
    config.device_total_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    config.device_free_bytes = config.device_total_bytes;
    config.first_layer = 0;
    config.last_layer = 1;
    config.batch_size = 1;
    config.max_seq_len = 16;
    config.activation_seq_len = 16;

    const auto without_mtp = MemoryPlanner::plan(profile, {config});
    config.mtp_enabled = true;
    const auto with_mtp = MemoryPlanner::plan(profile, {config});

    ASSERT_EQ(without_mtp.devices.size(), 1u);
    ASSERT_EQ(with_mtp.devices.size(), 1u);
    EXPECT_EQ(
        with_mtp.devices.front().weight_bytes() -
            without_mtp.devices.front().weight_bytes(),
        32ULL * 32ULL * sizeof(float))
        << "The terminal main-graph participant must reserve the trailing "
           "predictor block loaded by the MTP sidecar.";
}

TEST(Test__MemoryPlanner, GPUActivationBufferSizing_UsesLargestPrefillBucket)
{
    EXPECT_EQ(resolveActivationBufferSeqLen(131072, DeviceId::cuda(0)), 512);
    EXPECT_EQ(resolveActivationBufferSeqLen(2048, DeviceId::cuda(0)), 512);
    EXPECT_EQ(resolveActivationBufferSeqLen(256, DeviceId::cuda(0)), 256);
    EXPECT_EQ(resolveActivationBufferSeqLen(131072, DeviceId::rocm(0)), 512);
    EXPECT_EQ(resolveActivationBufferSeqLen(131072, DeviceId::cpu()), 131072);
}

TEST(Test__MemoryPlanner, LongContext_KeepsKVFullContextButCapsActivationWorkspace)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    const size_t qwen36_dense_weight_bytes = static_cast<size_t>(16.3 * static_cast<double>(GiB));
    auto profile = createQwen36DenseLikeProfile(qwen36_dense_weight_bytes);

    DevicePlanConfig short_context;
    short_context.device = DeviceId::cuda(0);
    short_context.device_compute_units = 82;
    short_context.device_total_bytes = 48ULL * GiB;
    short_context.device_free_bytes = 48ULL * GiB;
    short_context.batch_size = 1;
    short_context.max_seq_len = 4096;
    short_context.activation_seq_len = 4096;
    short_context.kv_precision = "fp16";

    DevicePlanConfig long_context = short_context;
    long_context.max_seq_len = 16384;
    long_context.activation_seq_len = 4096;

    auto short_plan = MemoryPlanner::plan(profile, {short_context});
    auto long_plan = MemoryPlanner::plan(profile, {long_context});

    ASSERT_EQ(short_plan.devices.size(), 1u);
    ASSERT_EQ(long_plan.devices.size(), 1u);

    const auto &short_device = short_plan.devices.front();
    const auto &long_device = long_plan.devices.front();

    EXPECT_GT(long_device.kv_cache_bytes(), short_device.kv_cache_bytes())
        << "KV cache must still reserve the requested long context";
    EXPECT_EQ(long_device.activation_bytes(), short_device.activation_bytes())
        << "Activation arena must be sized to graph chunk capacity, not context capacity";
    EXPECT_GT(long_device.workspace_bytes(), short_device.workspace_bytes())
        << "FA2 context-summary capacity follows the full KV horizon even "
           "when ordinary activation rows remain bucketed";
    EXPECT_EQ(long_device.max_seq_len(), 16384);
    EXPECT_EQ(long_device.activation_seq_len(), 4096);
    EXPECT_TRUE(long_plan.fits()) << long_plan.renderTable();
}

TEST(Test__MemoryPlanner, ResidentGraphSelection_ChoosesLargestFittingBucket)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    auto profile = createQwen36DenseLikeProfile(16ULL * GiB);

    DevicePlanConfig probe;
    probe.device = DeviceId::cuda(0);
    probe.device_compute_units = 82;
    probe.device_total_bytes = 24ULL * GiB;
    probe.device_free_bytes = 24ULL * GiB;
    probe.batch_size = 1;
    probe.max_seq_len = 16384;
    probe.activation_seq_len = 2048;
    probe.kv_precision = "q8_1";

    const auto plan_2k = MemoryPlanner::plan(profile, {probe});
    ASSERT_TRUE(plan_2k.fits()) << plan_2k.renderTable();

    probe.device_free_bytes =
        plan_2k.devices.front().total_bytes() +
        1ULL * 1024ULL * 1024ULL;

    const auto selected =
        MemoryPlanner::planLargestFittingResidentGraphRows(
            profile,
            {probe},
            {1024, 2048, 4096});

    ASSERT_TRUE(selected.fits()) << selected.memory_plan.renderTable();
    EXPECT_EQ(selected.resident_graph_rows, 2048);
    ASSERT_EQ(selected.memory_plan.devices.size(), 1u);
    EXPECT_EQ(selected.memory_plan.devices.front().activation_seq_len(), 2048);
    EXPECT_EQ(selected.memory_plan.devices.front().max_seq_len(), 16384)
        << "resident graph rows must not shrink full KV context capacity";
}

TEST(Test__MemoryPlanner,
     CapturedServingFamilyIsPricedPerAdmittedBucketAndFixedExecutable)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 64ULL * GiB;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.activation_seq_len = 64;
    cfg.kv_precision = "fp16";
    cfg.captured_serving_graphs = {
        .prefill_bucket_rows = {32, 64, 128},
        .fixed_executable_count = 2u,
        .verifier_graph_family = MTPVerifierGraphFamily::BoundedBuckets,
        .generation_domain = DeviceGenerationExecutionDomain::NativeDomain,
    };

    const auto plan_64 = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan_64.devices.size(), 1u);
    EXPECT_EQ(
        plan_64.devices.front().captured_graph_bytes(),
        estimateCapturedGraphExecutableBytes(
            cfg.device,
            /*two prefill + decode + prefix bridge=*/4u));

    cfg.activation_seq_len = 128;
    const auto plan_128 = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan_128.devices.size(), 1u);
    EXPECT_EQ(
        plan_128.devices.front().captured_graph_bytes(),
        estimateCapturedGraphExecutableBytes(
            cfg.device,
            /*three prefill + decode + prefix bridge=*/5u));
    EXPECT_GT(
        plan_128.devices.front().captured_graph_bytes(),
        plan_64.devices.front().captured_graph_bytes());

    cfg.device_free_bytes =
        plan_64.devices.front().incremental_bytes();
    const auto selected =
        MemoryPlanner::planLargestFittingResidentGraphRows(
            profile,
            {cfg},
            {64, 128});
    ASSERT_TRUE(selected.fits()) << selected.memory_plan.renderTable();
    EXPECT_EQ(selected.resident_graph_rows, 64);
    EXPECT_EQ(
        selected.memory_plan.devices.front().captured_graph_bytes(),
        plan_64.devices.front().captured_graph_bytes());
}

TEST(Test__MemoryPlanner,
     CapturedServingFamilyPricesRuntimeOwnerSlotsNotSemanticFragments)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 64ULL * GiB;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.activation_seq_len = 64;
    cfg.kv_precision = "fp16";
    MTPRuntimeConfig retained_mtp;
    retained_mtp.enabled = false;
    retained_mtp.graph_capacity_draft_tokens = 15;
    cfg.captured_serving_graphs =
        resolveCapturedServingGraphMemoryInventory(
            {32, 64, 128},
            retained_mtp,
            MTPVerifierGraphFamily::BoundedBuckets,
            MTPGenerationParentRole::Controller,
            DeviceGenerationExecutionDomain::NativeDomain);
    EXPECT_EQ(cfg.captured_serving_graphs.fixed_executable_count, 12u);
    const MTPGraphOwnerPlan owner_plan(retained_mtp, MTPVerifierGraphFamily::BoundedBuckets, MTPGenerationParentRole::Controller);
    EXPECT_EQ(owner_plan.sidecarGraphSlots(), 21u);
    EXPECT_EQ(owner_plan.terminalHiddenGraphSlots(), 35u);
    EXPECT_EQ(owner_plan.draftPublicationGraphSlots(), 15u);
    EXPECT_EQ(owner_plan.verifierPreparationGraphSlots(), 32u);
    EXPECT_EQ(owner_plan.controllerGraphSlots(), 5u);
    EXPECT_EQ(owner_plan.boundedHelperExecutableSlotCount(), 83u);
    EXPECT_EQ(owner_plan.generalAuxiliaryExecutableSlotCount(), 25u);
    EXPECT_EQ(
        cfg.captured_serving_graphs.mtp_graph_owners
            .auxiliaryExecutableSlotCount(),
        108u);
    EXPECT_EQ(
        resolveMTPRetainedDeviceGenerationFragmentCapacity(15),
        375u)
        << "Semantic parent descriptors are a separate host-scratch inventory.";

    const auto plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan.devices.size(), 1u);
    EXPECT_EQ(
        plan.devices.front().captured_graph_bytes(),
        estimateCapturedGraphExecutableBytes(
            cfg.device,
            CapturedGraphExecutableInventory{
                .model_graph_identity_count =
                    /*two prefill + decode + prefix bridge + ten MTP forwards=*/14u,
                .model_graph_topology_variant_count = 1u,
                .auxiliary_executable_count =
                    owner_plan.generalAuxiliaryExecutableSlotCount() +
                    OrdinaryGenerationGraphPlan::additionalExecutableCount(
                        cfg.device,
                        true,
                        declaredDeviceGenerationExecutionPolicy(cfg.device.type, *cfg.captured_serving_graphs.generation_domain),
                        cfg.captured_serving_graphs.generation_parent_role),
                .bounded_helper_executable_count =
                    owner_plan.boundedHelperExecutableSlotCount(),
            }));
    EXPECT_GT(
        plan.devices.front().captured_graph_bytes(),
        estimateCapturedGraphExecutableBytes(
            cfg.device,
            /*two prefill + decode + prefix bridge + ten MTP forwards=*/14u));
}

/** @test Exact hosted width owners are admitted before any request, with ordinary M1 charged once. */
TEST(Test__MemoryPlanner, ExactHostedVerifierFamilyPricesEveryModelAndPublicationOwner)
{
    constexpr std::size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    const auto profile = createTestProfile();
    for (bool initially_enabled : {false, true})
    for (auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (auto parent_role : {MTPGenerationParentRole::Controller, MTPGenerationParentRole::PipelineFollower})
    {
        MTPRuntimeConfig retained;
        retained.enabled = initially_enabled;
        retained.draft_tokens = 3;
        retained.graph_capacity_draft_tokens = 15;
        const auto exact_family = resolveMTPVerifierGraphFamily(MTPVerifierExecutionRole::HostedCapturedTransactions);
        const bool follower = parent_role == MTPGenerationParentRole::PipelineFollower;
        const auto exact = resolveCapturedServingGraphMemoryInventory(
            {32, 64, 128},
            retained,
            exact_family,
            parent_role,
            DeviceGenerationExecutionDomain::HostedBoundary,
            1u);
        const auto bounded = resolveCapturedServingGraphMemoryInventory(
            {32, 64, 128},
            retained,
            MTPVerifierGraphFamily::BoundedBuckets,
            parent_role,
            DeviceGenerationExecutionDomain::NativeDomain);
        EXPECT_EQ(exact.fixed_executable_count, 34u); // Ordinary M1 + prefix bridge + 32 MTP forwards.
        EXPECT_EQ(bounded.fixed_executable_count, 12u);
        const auto &owners = exact.mtp_graph_owners;
        EXPECT_EQ(owners.speculativePublicationWidthCount(), 15u);
        EXPECT_EQ(owners.speculativePublicationGraphSlots(), 15u);
        EXPECT_EQ(owners.generationParentWidthCount(), follower ? 15u : 1u);
        EXPECT_EQ(owners.generationParentGraphSlots(), follower ? 15u : 1u);
        EXPECT_EQ(owners.samplerHelperWidthCount(), 15u);
        EXPECT_EQ(owners.samplerHelperGraphSlots(), follower ? 0u : 30u);
        EXPECT_EQ(owners.controllerGraphSlots(), follower ? 30u : 47u);
        EXPECT_EQ(owners.auxiliaryExecutableSlotCount(), follower ? 30u : 150u);
        EXPECT_EQ(owners.generalAuxiliaryExecutableSlotCount(), follower ? 30u : 67u);
        EXPECT_EQ(owners.boundedHelperExecutableSlotCount(), follower ? 0u : 83u);
        EXPECT_EQ(owners.sidecarGraphSlots(), bounded.mtp_graph_owners.sidecarGraphSlots());
        EXPECT_EQ(owners.verifierPreparationGraphSlots(), bounded.mtp_graph_owners.verifierPreparationGraphSlots());
        EXPECT_EQ(owners.terminalHiddenGraphSlots(), bounded.mtp_graph_owners.terminalHiddenGraphSlots());
        const auto two_programs = resolveCapturedServingGraphMemoryInventory(
            {32, 64, 128},
            retained,
            exact_family,
            parent_role,
            DeviceGenerationExecutionDomain::HostedBoundary,
            2u);
        EXPECT_EQ(two_programs.fixed_executable_count, 68u);
        EXPECT_EQ(two_programs.mtp_graph_owners.speculativePublicationGraphSlots(), 30u);
        EXPECT_EQ(two_programs.mtp_graph_owners.generationParentGraphSlots(), follower ? 30u : 2u);
        EXPECT_EQ(two_programs.mtp_graph_owners.samplerHelperWidthCount(), 15u);
        EXPECT_EQ(two_programs.mtp_graph_owners.samplerHelperGraphSlots(), follower ? 0u : 30u);
        EXPECT_EQ(two_programs.mtp_graph_owners.controllerGraphSlots(), follower ? 60u : 63u);
        EXPECT_EQ(two_programs.mtp_graph_owners.boundedHelperExecutableSlotCount(), follower ? 0u : 83u);

        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.device_compute_units = 82;
        cfg.device_total_bytes = 64u * GiB;
        cfg.device_free_bytes = cfg.device_total_bytes;
        cfg.batch_size = 1;
        cfg.max_seq_len = 4096;
        cfg.activation_seq_len = 64;
        cfg.kv_precision = "fp16";
        cfg.captured_serving_graphs = exact;
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        ASSERT_EQ(plan.devices.size(), 1u);
        EXPECT_EQ(plan.devices.front().captured_graph_bytes(), estimateCapturedGraphExecutableBytes(
            device, CapturedGraphExecutableInventory{
                .model_graph_identity_count = 36u, // Two admitted prefills plus the 34 fixed owners.
                .model_graph_topology_variant_count = 1u,
                .auxiliary_executable_count = owners.generalAuxiliaryExecutableSlotCount() +
                    OrdinaryGenerationGraphPlan::additionalExecutableCount(
                        device,
                        true,
                        declaredDeviceGenerationExecutionPolicy(device.type, *exact.generation_domain),
                        parent_role),
                .bounded_helper_executable_count = owners.boundedHelperExecutableSlotCount(),
            }));
    }
    const auto empty = resolveCapturedServingGraphMemoryInventory(
        {},
        {},
        MTPVerifierGraphFamily::ExactRows,
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::HostedBoundary);
    EXPECT_EQ(empty.fixed_executable_count, 1u); // Existing ordinary M1 alone.
    EXPECT_EQ(empty.mtp_graph_owners.speculativePublicationWidthCount(), 0u);
    EXPECT_EQ(empty.mtp_graph_owners.samplerHelperWidthCount(), 0u);
    EXPECT_EQ(empty.mtp_graph_owners.samplerHelperGraphSlots(), 0u);
    EXPECT_EQ(empty.mtp_graph_owners.auxiliaryExecutableSlotCount(), 0u);
    EXPECT_THROW((void)resolveCapturedServingGraphMemoryInventory(
        {},
        {},
        static_cast<MTPVerifierGraphFamily>(99),
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::NativeDomain), std::invalid_argument);
    EXPECT_THROW((void)resolveCapturedServingGraphMemoryInventory(
        {},
        {},
        MTPVerifierGraphFamily::ExactRows,
        static_cast<MTPGenerationParentRole>(99),
        DeviceGenerationExecutionDomain::HostedBoundary), std::invalid_argument);
}

/** @test Every exact sampler geometry is admitted once and shared across immutable query programs. */
TEST(Test__MemoryPlanner, ExactHostedSamplerHelpersRemainSharedAcrossQueryPrograms)
{
    for (int capacity = 1; capacity <= kMaximumSupportedMTPDraftDepth; ++capacity)
    for (int initial_depth = 1; initial_depth <= capacity; ++initial_depth)
    for (auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
    for (bool initially_enabled : {false, true})
    for (auto family : {MTPVerifierGraphFamily::BoundedBuckets, MTPVerifierGraphFamily::ExactRows})
    for (auto role : {MTPGenerationParentRole::Controller, MTPGenerationParentRole::PipelineFollower})
    {
        MTPRuntimeConfig config;
        config.enabled = initially_enabled;
        config.draft_tokens = initial_depth;
        config.graph_capacity_draft_tokens = capacity;
        config.depth_policy.mode = mode;
        config.depth_policy.min_depth = 1;
        config.depth_policy.initial_depth = initial_depth;
        config.depth_policy.max_depth = initial_depth;
        const MTPGraphOwnerPlan one(config, family, role);
        const MTPGraphOwnerPlan two(config, family, role, 2u);
        const std::size_t expected_widths = family == MTPVerifierGraphFamily::ExactRows
            ? mtpVerifierGraphFamilyWidthCount(capacity + 1, family) : 1u;
        EXPECT_EQ(one.samplerHelperWidthCount(), expected_widths);
        EXPECT_EQ(two.samplerHelperWidthCount(), expected_widths);
        EXPECT_EQ(one.samplerHelperGraphSlots(), role == MTPGenerationParentRole::Controller ? 2u * expected_widths : 0u);
        EXPECT_EQ(two.samplerHelperGraphSlots(), one.samplerHelperGraphSlots());
        EXPECT_EQ(one.controllerGraphSlots(), one.samplerHelperGraphSlots() +
            one.generationParentGraphSlots() + one.speculativePublicationGraphSlots() + one.directTokenHistoryGraphSlots());
        EXPECT_EQ(two.controllerGraphSlots(), two.samplerHelperGraphSlots() +
            two.generationParentGraphSlots() + two.speculativePublicationGraphSlots() + two.directTokenHistoryGraphSlots());
        EXPECT_EQ(two.controllerGraphSlots() - one.controllerGraphSlots(),
            one.generationParentGraphSlots() + one.speculativePublicationGraphSlots());
        EXPECT_EQ(two.boundedHelperExecutableSlotCount(), one.boundedHelperExecutableSlotCount());
    }
    // Capacity belongs to setup even if the first request only selects rows4.
    MTPRuntimeConfig shallow;
    shallow.enabled = true;
    shallow.draft_tokens = 3;
    shallow.graph_capacity_draft_tokens = 15;
    const MTPGraphOwnerPlan admitted(shallow, MTPVerifierGraphFamily::ExactRows, MTPGenerationParentRole::Controller);
    EXPECT_EQ(admitted.samplerHelperWidthCount(), 15u);
    EXPECT_EQ(admitted.samplerHelperGraphSlots(), 30u);
    EXPECT_EQ(admitted.controllerGraphSlots(), 47u);
    shallow.graph_capacity_draft_tokens = 0;
    EXPECT_EQ(MTPGraphOwnerPlan(shallow, MTPVerifierGraphFamily::ExactRows,
        MTPGenerationParentRole::Controller).samplerHelperWidthCount(), 3u);
}

TEST(Test__MemoryPlanner,
     CapturedGraphEstimatorPricesSimultaneousTopologyVariants)
{
    constexpr std::size_t kIdentitiesPerVariant = 8u;
    constexpr std::size_t kTopologyVariants = 2u;
    constexpr std::size_t kAuxiliaryExecutables = 13u;
    const CapturedGraphExecutableInventory inventory{
        .model_graph_identity_count = kIdentitiesPerVariant,
        .model_graph_topology_variant_count = kTopologyVariants,
        .auxiliary_executable_count = kAuxiliaryExecutables,
    };

    const std::size_t estimated =
        estimateCapturedGraphExecutableBytes(DeviceId::cuda(0), inventory);
    const std::size_t physical_model_graphs =
        kIdentitiesPerVariant * kTopologyVariants;
    EXPECT_EQ(
        estimated,
        (physical_model_graphs + kAuxiliaryExecutables) *
            GPUGraphMemoryContract::kCUDAExecutableReservationBytes);
    EXPECT_GT(
        estimated,
        estimateCapturedGraphExecutableBytes(
            DeviceId::cuda(0),
            CapturedGraphExecutableInventory{
                .model_graph_identity_count = kIdentitiesPerVariant,
                .model_graph_topology_variant_count = 1u,
                .auxiliary_executable_count =
                    kAuxiliaryExecutables,
            }))
        << "Prepared diagnostic and lean graphs coexist and must not share one driver-memory charge.";
}

/** @test Helper classification changes bytes, never physical owner cardinality. */
TEST(Test__MemoryPlanner, BoundedHelperFamilySharesBackendAdmissionContract)
{
    const CapturedGraphExecutableInventory inventory{
        .model_graph_identity_count = 7u,
        .model_graph_topology_variant_count = 1u,
        .auxiliary_executable_count = 25u,
        .bounded_helper_executable_count = 82u,
    };
    EXPECT_EQ(inventory.residentExecutableCount(), 114u);
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto general = GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        const auto helper = GPUGraphMemoryContract::reservationBytesPerExecutable(
            device, GPUGraphExecutableClass::BoundedFlatHelper);
        EXPECT_EQ(estimateCapturedGraphExecutableBytes(device, inventory),
                  32u * general + 82u * helper);
        if (device.is_cuda())
            EXPECT_EQ(114u * general - estimateCapturedGraphExecutableBytes(device, inventory),
                      1640u * 1024u * 1024u);
        else
            EXPECT_EQ(estimateCapturedGraphExecutableBytes(device, inventory), 114u * general);
    }
    EXPECT_THROW((void)GPUGraphMemoryContract::reservationBytesPerExecutable(
        DeviceId::cpu(), GPUGraphExecutableClass::BoundedFlatHelper), std::invalid_argument);
    EXPECT_THROW((void)GPUGraphMemoryContract::requiresBoundedFlatShape(
        static_cast<GPUGraphExecutableClass>(99)), std::invalid_argument);
    auto overflowing = inventory;
    overflowing.bounded_helper_executable_count = std::numeric_limits<size_t>::max();
    EXPECT_THROW((void)overflowing.residentExecutableCount(), std::overflow_error);
    EXPECT_THROW((void)estimateCapturedGraphExecutableBytes(DeviceId::cuda(0), overflowing),
                 std::overflow_error);
}

/**
 * @brief A shared driver-pool growth event belongs to the complete family.
 *
 * CUDA and HIP may grow their opaque graph pool while instantiating one graph
 * and then charge zero bytes to several later siblings. The event can therefore
 * exceed one slot unit without violating the family reservation. This is the
 * focused regression for the production ExpertOverlay prefill parent that
 * observed 30 MiB against a three-or-more-slot admitted family.
 */
TEST(Test__MemoryPlanner,
     CapturedGraphPoolGrowthIsCertifiedAgainstTheCompleteFamily)
{
    for (const DeviceId device :
         {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        constexpr std::size_t kRetainedSlots = 3u;
        const std::size_t slot_bytes =
            GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        const std::size_t observed_pool_growth = slot_bytes + 1u;
        const std::size_t admitted_family_bytes =
            estimateCapturedGraphExecutableBytes(
                device,
                CapturedGraphExecutableInventory::monolithic(
                    kRetainedSlots));

        EXPECT_GT(observed_pool_growth, slot_bytes);
        EXPECT_TRUE(
            GPUGraphMemoryContract::acceptsFamilyObservation(
                device,
                observed_pool_growth,
                admitted_family_bytes));
        EXPECT_FALSE(
            GPUGraphMemoryContract::acceptsFamilyObservation(
                device,
                admitted_family_bytes + 1u,
                admitted_family_bytes));
    }
}

TEST(Test__MemoryPlanner,
     PrefixDeviceHotAdmissionUsesTheRuntimeWholeBlockReservation)
{
    EXPECT_EQ(prefixCacheWholeBlockReservationBytes(0u, 128u), 0u);
    EXPECT_EQ(prefixCacheWholeBlockReservationBytes(1024u, 0u), 0u);
    EXPECT_EQ(prefixCacheWholeBlockReservationBytes(127u, 128u), 0u);
    EXPECT_EQ(prefixCacheWholeBlockReservationBytes(128u, 128u), 128u);
    EXPECT_EQ(prefixCacheWholeBlockReservationBytes(1024u, 300u), 900u);
    EXPECT_EQ(
        prefixCacheWholeBlockReservationBytes(1024u, 256u),
        1024u);
}

/**
 * @brief Pin prefix-terminal ownership across a two-stage GPU pipeline.
 *
 * The first stage owns only its logical K/V blocks; the last stage additionally
 * owns the terminal FP32 logits row.  This exact distinction must feed the
 * whole-slot rounding performed by both admission and runtime materialization.
 */
TEST(Test__MemoryPlanner,
     PipelinePrefixDeviceHotAdmissionMatchesTerminalOwnership)
{
    const auto profile = createTestProfile();

    DevicePlanConfig first_stage;
    first_stage.device = DeviceId::rocm(0);
    first_stage.device_compute_units = 60;
    first_stage.device_total_bytes =
        32ULL * 1024ULL * 1024ULL * 1024ULL;
    first_stage.device_free_bytes = first_stage.device_total_bytes;
    first_stage.first_layer = 0;
    first_stage.last_layer = 11;
    first_stage.batch_size = 1;
    first_stage.max_seq_len = 4096;
    first_stage.kv_precision = "fp16";
    first_stage.prefix_cache = PrefixCacheRuntimeConfig{};
    first_stage.associated_host_memory = PhysicalMemoryResource{
        .world_rank = -1,
        .device = DeviceId::cpu(),
        .total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL,
        .admission_available_bytes =
            64ULL * 1024ULL * 1024ULL * 1024ULL,
    };

    DevicePlanConfig terminal_stage = first_stage;
    terminal_stage.device = DeviceId::rocm(1);
    terminal_stage.first_layer = 12;
    terminal_stage.last_layer = 23;

    const auto plan = MemoryPlanner::plan(
        profile,
        {first_stage, terminal_stage});
    ASSERT_EQ(plan.devices.size(), 3u);

    const auto first = std::find_if(
        plan.devices.begin(),
        plan.devices.end(),
        [](const auto &candidate)
        { return candidate.device() == DeviceId::rocm(0); });
    const auto terminal = std::find_if(
        plan.devices.begin(),
        plan.devices.end(),
        [](const auto &candidate)
        { return candidate.device() == DeviceId::rocm(1); });
    ASSERT_NE(first, plan.devices.end());
    ASSERT_NE(terminal, plan.devices.end());

    const auto logical_block =
        KVCacheMemoryEstimator::estimateGPULogicalBlock(KVCacheFamily::AttentionOnly,
            first_stage.prefix_cache.block_size,
            profile.n_kv_heads,
            profile.head_dim,
            first_stage.kv_precision,
            first_stage.device);
    const size_t stage_kv_bytes =
        12ULL * logical_block.totalBytes();
    const size_t terminal_block_bytes =
        stage_kv_bytes +
        static_cast<size_t>(profile.vocab_size) * sizeof(float);

    EXPECT_EQ(
        first->prefix_cache_device_hot_bytes(),
        prefixCacheWholeBlockReservationBytes(
            first_stage.prefix_cache.device_budget_bytes,
            stage_kv_bytes))
        << "A non-terminal PP stage must not price a logits payload it cannot archive.";
    EXPECT_EQ(
        terminal->prefix_cache_device_hot_bytes(),
        prefixCacheWholeBlockReservationBytes(
            terminal_stage.prefix_cache.device_budget_bytes,
            terminal_block_bytes))
        << "The terminal PP stage must retain the exact full-vocabulary logits slot.";
}

/**
 * @brief Hybrid main and shifted MTP caches retain distinct physical Q8 codecs.
 *
 * A hybrid model's FA-only pipeline slice still selects the hybrid factory.
 * The predictor instead selects the attention-only factory even beside a
 * hybrid main cache. Both admission and prefix staging must retain that fact.
 */
TEST(Test__MemoryPlanner, HybridQ8MainAndMTPUseTheirOwnCacheFamilies)
{
    const auto profile = createQwen36HybridMTPStateProfile();
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const bool fa_only_slice : {false, true})
    {
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.device_total_bytes = cfg.device_free_bytes = 64ULL << 30;
        cfg.device_compute_units = 60;
        cfg.batch_size = 1;
        cfg.max_seq_len = 4096;
        cfg.activation_seq_len = 16;
        cfg.kv_precision = "q8_1";
        cfg.mtp_enabled = !fa_only_slice;
        cfg.prefix_cache = PrefixCacheRuntimeConfig{};
        cfg.associated_host_memory = PhysicalMemoryResource{
            .world_rank = -1, .device = DeviceId::cpu(),
            .total_bytes = 64ULL << 30, .admission_available_bytes = 64ULL << 30};
        if (fa_only_slice)
            cfg.first_layer = cfg.last_layer = 3;
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        const auto entry = std::find_if(plan.devices.begin(), plan.devices.end(),
            [&](const auto &candidate) { return candidate.device() == device; });
        ASSERT_NE(entry, plan.devices.end());
        const int fa_layers = fa_only_slice ? 1 : 16;
        const auto main = KVCacheMemoryEstimator::estimateGPULogicalBlock(
            KVCacheFamily::Hybrid, cfg.prefix_cache.block_size,
            profile.n_kv_heads, profile.head_dim, cfg.kv_precision, device);
        const auto shifted = KVCacheMemoryEstimator::estimateGPULogicalBlock(
            KVCacheFamily::AttentionOnly, cfg.prefix_cache.block_size,
            profile.n_kv_heads, profile.head_dim, cfg.kv_precision, device);
        EXPECT_NE(main.k_bytes, shifted.k_bytes);
        size_t prefix_bytes = fa_layers * main.totalBytes();
        size_t kv_bytes = KVCacheMemoryEstimator::estimate(KVCacheFamily::Hybrid,
            fa_layers, 1, cfg.max_seq_len, profile.n_kv_heads,
            profile.head_dim, cfg.kv_precision, device);
        size_t staging_bytes = main.totalBytes();
        if (!fa_only_slice)
        {
            const auto gdn = HybridGDNStateGeometry::resolve(profile.n_heads,
                0, profile.n_heads, profile.gdn_group_count,
                profile.gdn_time_step_rank, profile.gdn_state_size,
                profile.gdn_inner_size, profile.gdn_conv_kernel_size);
            const size_t recurrent = gdn.deviceSerializedPayloadBytes(48);
            prefix_bytes += recurrent + shifted.totalBytes() +
                (profile.d_model + profile.vocab_size) * sizeof(float);
            staging_bytes += recurrent + shifted.totalBytes();
            kv_bytes += KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
                1, 1, cfg.max_seq_len, profile.n_kv_heads,
                profile.head_dim, cfg.kv_precision, device);
        }
        EXPECT_EQ(entry->kv_cache_bytes(), kv_bytes);
        EXPECT_EQ(entry->prefix_cache_staging_bytes(), staging_bytes);
        EXPECT_EQ(entry->prefix_cache_device_hot_bytes(),
            prefixCacheWholeBlockReservationBytes(
                cfg.prefix_cache.device_budget_bytes, prefix_bytes));
    }
}

TEST(Test__MemoryPlanner,
     ResidentGraphSelection_CoversRetainedMTPRowsOnCUDAAndROCm)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    auto profile = createTestProfile();

    DevicePlanConfig cuda;
    cuda.device = DeviceId::cuda(0);
    cuda.device_compute_units = 82;
    cuda.device_total_bytes = 64ULL * GiB;
    cuda.device_free_bytes = cuda.device_total_bytes;
    cuda.batch_size = 1;
    cuda.max_seq_len = 4096;
    cuda.activation_seq_len = 9;
    cuda.kv_precision = "fp16";
    cuda.mtp_enabled = true;
    cuda.mtp_target_query_rows = 16;

    DevicePlanConfig rocm = cuda;
    rocm.device = DeviceId::rocm(0);
    rocm.device_compute_units = 60;

    const auto selected =
        MemoryPlanner::planLargestFittingResidentGraphRows(
            profile,
            {cuda, rocm},
            {1, 9});

    ASSERT_TRUE(selected.fits()) << selected.memory_plan.renderTable();
    EXPECT_EQ(selected.resident_graph_rows, 16)
        << "The graph identity must cover depth-fifteen's sixteen target rows, "
           "not only the nine-row exact prefill bucket";
    ASSERT_EQ(selected.memory_plan.devices.size(), 2u);
    EXPECT_EQ(selected.memory_plan.devices[0].activation_seq_len(), 16);
    EXPECT_EQ(selected.memory_plan.devices[1].activation_seq_len(), 16)
        << "CUDA and ROCm participants must receive the same retained shape";
}

/**
 * @brief Several local GPU caches share one physical archive-I/O window.
 *
 * The runtime registry keys durable archives by rank and normalized path.
 * Admission must use that same identity: RAM cache tiers remain participant
 * owned, while archive I/O scratch is one allocation shared by every
 * local participant opening the archive.
 */
TEST(Test__MemoryPlanner,
     HeterogeneousLocalGPUsChargeOneSharedPrefixArchiveScratch)
{
    auto profile = createQwen36HybridMTPStateProfile();

    DevicePlanConfig cuda;
    cuda.world_rank = 0;
    cuda.device = DeviceId::cuda(0);
    cuda.device_compute_units = 82;
    cuda.device_total_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
    cuda.device_free_bytes = cuda.device_total_bytes;
    cuda.batch_size = 1;
    cuda.max_seq_len = 4096;
    cuda.kv_precision = "fp16";
    cuda.prefix_cache = PrefixCacheRuntimeConfig{};
    cuda.associated_host_memory = PhysicalMemoryResource{
        .world_rank = 0,
        .device = DeviceId::cpu(),
        .total_bytes = 128ULL * 1024ULL * 1024ULL * 1024ULL,
        .admission_available_bytes =
            128ULL * 1024ULL * 1024ULL * 1024ULL,
    };

    DevicePlanConfig rocm = cuda;
    rocm.device = DeviceId::rocm(0);
    rocm.device_compute_units = 60;

    const auto plan = MemoryPlanner::plan(profile, {cuda, rocm});
    ASSERT_EQ(plan.devices.size(), 3u);
    const auto host = std::find_if(
        plan.devices.begin(), plan.devices.end(), [](const auto &candidate)
        { return candidate.device().is_cpu(); });
    ASSERT_NE(host, plan.devices.end());
    EXPECT_EQ(
        host->bom().bytes(PhysicalMemoryOwner::PrefixHostTier),
        2u * cuda.prefix_cache.ram_budget_bytes)
        << "Each participant retains its own bounded RAM cache";
    EXPECT_EQ(
        host->bom().bytes(PhysicalMemoryOwner::PrefixArchiveStaging),
        PrefixArchiveIOGeometry::scratchBytes())
        << "One rank/path archive owns exactly one reusable I/O scratch";
}

/** @test Stage grouping cannot duplicate common backing or erase private RAM tiers. */
TEST(Test__MemoryPlanner, PipelineGroupsMatchOneCompletePhysicalPlan)
{
    const auto profile = createTestProfile();
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        std::vector<std::vector<DevicePlanConfig>> groups(2);
        std::vector<DevicePlanConfig> flat;
        for (int stage = 0; stage < 2; ++stage)
        {
            DevicePlanConfig cfg;
            cfg.world_rank = 0;
            cfg.device = DeviceId(backend, backend == DeviceType::CPU ? 0 : stage);
            cfg.device_total_bytes = cfg.device_free_bytes = 256ULL << 30;
            cfg.device_compute_units = 8;
            if (backend == DeviceType::CPU) cfg.cpu_execution = test::kSyntheticCPUExecutionGeometry;
            cfg.first_layer = stage * 12;
            cfg.last_layer = cfg.first_layer + 11;
            cfg.owns_embedding = stage == 0;
            cfg.max_seq_len = 256;
            cfg.activation_seq_len = 32;
            cfg.prefix_cache = PrefixCacheRuntimeConfig{};
            cfg.prefix_cache.ram_budget_bytes = 16ULL << 30;
            cfg.prefix_cache.disk_budget_bytes = 32ULL << 30;
            cfg.prefix_cache.disk_dir = stage ? "/cache/./shared" : "/cache/shared";
            if (cfg.device.is_gpu()) cfg.associated_host_memory = PhysicalMemoryResource{
                .world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = 256ULL << 30, .admission_available_bytes = 256ULL << 30};
            groups[stage].push_back(cfg);
            flat.push_back(cfg);
        }
        const auto planned = MemoryPlanner::planGroups(profile, groups);
        const auto together = MemoryPlanner::plan(profile, flat);
        ASSERT_EQ(planned.size(), 2u);
        PhysicalMemoryPlanBuilder combined;
        for (const auto &group : planned)
            for (const auto &bom : group.physicalPlan().resources()) combined.add(bom);
        const auto aggregate = combined.build();
        EXPECT_EQ(aggregate.totalBytes(), together.physicalPlan().totalBytes());
        for (const auto &bom : together.physicalPlan().resources())
        {
            const auto *actual = aggregate.find({bom.resource().world_rank, bom.resource().device});
            ASSERT_NE(actual, nullptr);
            for (size_t owner = 0; owner < PhysicalMemoryBOM::ownerCount(); ++owner)
                EXPECT_EQ(actual->bytes(static_cast<PhysicalMemoryOwner>(owner)), bom.bytes(static_cast<PhysicalMemoryOwner>(owner)));
        }
        const auto *host = aggregate.find({0, DeviceId::cpu()});
        ASSERT_NE(host, nullptr);
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::PrefixHostTier), 32ULL << 30);
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), PrefixArchiveIOGeometry::scratchBytes());
        const auto *second_host = planned[1].physicalPlan().find({0, DeviceId::cpu()});
        ASSERT_NE(second_host, nullptr);
        EXPECT_EQ(second_host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), 0u);
    }
}

/** @test Failed or completed planning cannot leave a process-global archive credit. */
TEST(Test__MemoryPlanner, PipelineGroupSharedOwnersAreTransactionLocal)
{
    const auto profile = createTestProfile();
    DevicePlanConfig cfg;
    cfg.world_rank = 0;
    cfg.device = DeviceId::cuda(0);
    cfg.device_total_bytes = cfg.device_free_bytes = 64ULL << 30;
    cfg.device_compute_units = 80;
    cfg.max_seq_len = 256;
    cfg.activation_seq_len = 32;
    cfg.prefix_cache = PrefixCacheRuntimeConfig{};
    cfg.associated_host_memory = PhysicalMemoryResource{
        .world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = 256ULL << 30, .admission_available_bytes = 256ULL << 30};
    std::vector<std::vector<DevicePlanConfig>> groups{{cfg}, {cfg}};
    groups[1][0].concurrent_workspace_owners = 0;
    EXPECT_THROW((void)MemoryPlanner::planGroups(profile, groups), std::invalid_argument);
    groups[1][0].concurrent_workspace_owners = 1;
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        const auto planned = MemoryPlanner::planGroups(profile, groups);
        ASSERT_EQ(planned.size(), 2u);
        const auto *first = planned.front().physicalPlan().find({0, DeviceId::cpu()});
        const auto *last = planned.back().physicalPlan().find({0, DeviceId::cpu()});
        ASSERT_NE(first, nullptr);
        ASSERT_NE(last, nullptr);
        EXPECT_EQ(first->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), PrefixArchiveIOGeometry::scratchBytes());
        EXPECT_EQ(last->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), 0u);
    }
    EXPECT_THROW((void)MemoryPlanner::planGroups(profile, {}), std::invalid_argument);
    const std::vector<std::vector<DevicePlanConfig>> empty_owner{{}};
    EXPECT_TRUE(MemoryPlanner::planGroups(profile, empty_owner).front().physicalPlan().resources().empty());
    groups.front().front().prefix_cache.enabled = false;
    auto enabled_tail = MemoryPlanner::planGroups(profile, groups);
    const auto *tail_host = enabled_tail.back().physicalPlan().find({0, DeviceId::cpu()});
    ASSERT_NE(tail_host, nullptr);
    EXPECT_EQ(tail_host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), PrefixArchiveIOGeometry::scratchBytes());
    groups.back().front().prefix_cache.disk_budget_bytes = 0;
    const auto ram_only = MemoryPlanner::planGroups(profile, groups);
    tail_host = ram_only.back().physicalPlan().find({0, DeviceId::cpu()});
    ASSERT_NE(tail_host, nullptr);
    EXPECT_EQ(tail_host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), 0u);
    EXPECT_EQ(tail_host->bytes(PhysicalMemoryOwner::PrefixHostTier), cfg.prefix_cache.ram_budget_bytes);
}

TEST(Test__MemoryPlanner, SingleGPU_DoesNotFit)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 1ULL * 1024 * 1024 * 1024; // 1 GB
    cfg.device_free_bytes = 512ULL * 1024 * 1024;       // 512 MB free
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";

    auto plan = MemoryPlanner::plan(profile, {cfg});

    EXPECT_FALSE(plan.fits());
    EXPECT_GT(plan.devices[0].deficit(), 0u);
    EXPECT_FALSE(plan.diagnostics.empty());
}

TEST(Test__MemoryPlanner, CertifiedRetainedWeightsUseIncrementalAdmissionOnCPUAndGPU)
{
    auto profile = createTestProfile();

    for (const DeviceId device : {DeviceId::cpu(), DeviceId::cuda(0)})
    {
        SCOPED_TRACE(device.toString());
        DevicePlanConfig config;
        config.device = device;
        config.device_compute_units = device.is_cpu() ? 32 : 82;
        config.cpu_execution = device.is_cpu() ? test::kSyntheticCPUExecutionGeometry : CPUExecutionGeometry{};
        config.device_total_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
        config.device_free_bytes = config.device_total_bytes;
        config.batch_size = 1;
        config.max_seq_len = 4096;
        config.kv_precision = "fp16";

        const auto complete = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(complete.devices.size(), 1u);
        const auto &complete_device = complete.devices.front();
        ASSERT_GT(complete_device.weight_bytes(), 0u);
        ASSERT_EQ(complete_device.retained_weight_bytes(), 0u);

        const size_t non_weight_bytes =
            complete_device.total_bytes() - complete_device.weight_bytes();
        config.device_free_bytes =
            non_weight_bytes + 1ULL * 1024ULL * 1024ULL;

        const auto fresh = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(fresh.devices.size(), 1u);
        EXPECT_FALSE(fresh.fits())
            << "A fresh runner must still reserve the complete weight allocation";

        config.prepared_weight_admission =
            PreparedWeightAdmission::ReuseCertifiedCompleteSet;
        const auto retained = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(retained.devices.size(), 1u);
        const auto &retained_device = retained.devices.front();
        EXPECT_TRUE(retained.fits()) << retained.renderTable();
        EXPECT_EQ(retained_device.weight_bytes(), complete_device.weight_bytes())
            << "The final capacity BOM must retain the complete model footprint";
        EXPECT_EQ(
            retained_device.retained_weight_bytes(),
            retained_device.weight_bytes());
        EXPECT_EQ(retained_device.incremental_weight_bytes(), 0u);
        EXPECT_EQ(retained_device.incremental_bytes(), non_weight_bytes);
        EXPECT_NE(retained.renderTable().find("Retained"), std::string::npos);
    }
}

TEST(Test__MemoryPlanner, RetainedWorkspaceReducesOnlyIncrementalAdmission)
{
    auto profile = createTestProfile();

    DevicePlanConfig config;
    config.device = DeviceId::rocm(0);
    config.device_compute_units = 60;
    config.device_total_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    config.device_free_bytes = config.device_total_bytes;
    config.batch_size = 1;
    config.max_seq_len = 4096;
    config.kv_precision = "fp16";

    const auto fresh = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(fresh.devices.size(), 1u);
    const DeviceMemoryPlan fresh_device = fresh.devices.front();
    ASSERT_GT(fresh_device.workspace_bytes(), 0u);
    ASSERT_EQ(fresh_device.retained_workspace_bytes(), 0u);

    config.retained_workspace_bytes = fresh_device.workspace_bytes();
    const auto reused = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(reused.devices.size(), 1u);
    const DeviceMemoryPlan &reused_device = reused.devices.front();

    EXPECT_EQ(reused_device.total_bytes(), fresh_device.total_bytes())
        << "Retained workspace remains part of the final device BOM";
    EXPECT_EQ(
        reused_device.retained_workspace_bytes(),
        fresh_device.workspace_bytes());
    EXPECT_EQ(
        fresh_device.incremental_bytes() - reused_device.incremental_bytes(),
        fresh_device.workspace_bytes())
        << "Only bytes proven reusable by the typed lease reduce admission";
    EXPECT_NE(reused.renderTable().find("Ret.Wksp"), std::string::npos);
    EXPECT_NE(
        reused_device.summary().find("retained_workspace="),
        std::string::npos);
}

TEST(Test__MemoryPlanner, TP2_ReducesPerDeviceWeight)
{
    auto profile = createTestProfile();

    // Single device
    DevicePlanConfig single;
    single.device = DeviceId::cuda(0);
    single.device_compute_units = 82;
    single.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    single.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    single.batch_size = 1;
    single.max_seq_len = 4096;
    single.kv_precision = "fp16";

    auto single_plan = MemoryPlanner::plan(profile, {single});

    // TP-2
    DevicePlanConfig shard0 = single;
    shard0.shard_index = 0;
    shard0.total_shards = 2;
    shard0.local_tp_backend = CollectiveBackendType::NCCL;

    DevicePlanConfig shard1 = single;
    shard1.device = DeviceId::cuda(1);
    shard1.shard_index = 1;
    shard1.total_shards = 2;
    shard1.local_tp_backend = CollectiveBackendType::NCCL;

    auto tp_plan = MemoryPlanner::plan(profile, {shard0, shard1});

    EXPECT_EQ(tp_plan.devices.size(), 2u);
    // Each shard should have less weight than single
    EXPECT_LT(tp_plan.devices[0].weight_bytes(), single_plan.devices[0].weight_bytes());
    EXPECT_LT(tp_plan.devices[1].weight_bytes(), single_plan.devices[0].weight_bytes());
}

TEST(Test__MemoryPlanner,
     PhaseSplitDensePolicyPricesConcurrentReplicatedDecodeWeights)
{
    const auto profile = createMoEOverlayProfile();
    auto config = overlayDeviceConfig(DeviceId::cuda(0));
    config.shard_index = 0;
    config.total_shards = 2;
    config.local_tp_backend = CollectiveBackendType::NCCL;
    const std::vector<int> no_routed_experts(
        static_cast<std::size_t>(profile.n_layers), 0);
    config.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            no_routed_experts);
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
            config.total_shards,
            MTPRuntimeConfig{});

    const auto planned = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(planned.devices.size(), 1u);
    const auto primary = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        config.shard_index,
        config.total_shards,
        config.first_layer,
        config.last_layer,
        config.weight_residency);
    const auto replicated_decode = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        /*shard_index=*/0,
        /*total_shards=*/1,
        config.first_layer,
        config.last_layer,
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            no_routed_experts));

    EXPECT_EQ(planned.devices.front().weight_bytes(), primary.device_bytes);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes(),
        replicated_decode.device_bytes);
    EXPECT_EQ(
        planned.devices.front().total_weight_bytes(),
        primary.device_bytes + replicated_decode.device_bytes);

    config.additional_weight_sets.push_back(
        AdditionalPersistentWeightSet::ReplicatedDenseDecode);
    EXPECT_THROW(
        (void)MemoryPlanner::plan(profile, {config}),
        std::invalid_argument);
}

/**
 * @test A replicated decode weight view also changes the physical state BOM.
 *
 * Both backends retain full attention K/V and full-width arena/workspace rows
 * alongside TP-sharded prefill. Reserving only the extra weights lets automatic
 * expert capacity consume memory required by the first captured graph.
 */
TEST(Test__MemoryPlanner,
     PhaseSplitDenseDecodePricesFullKVAndCapturedActivationFamily)
{
    const auto profile = createMoEOverlayProfile();
    for (const DeviceId device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto sharded = overlayDeviceConfig(device);
        sharded.shard_index = 0;
        sharded.total_shards = 2;
        sharded.local_kv_heads = profile.n_kv_heads / 2;
        sharded.local_tp_backend = device.is_cuda()
            ? CollectiveBackendType::NCCL
            : CollectiveBackendType::RCCL;

        auto phase_split = sharded;
        phase_split.additional_weight_sets =
            resolveAdditionalPersistentWeightSets(
                DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
                phase_split.total_shards,
                MTPRuntimeConfig{});

        const auto sharded_plan = MemoryPlanner::plan(profile, {sharded});
        const auto split_plan = MemoryPlanner::plan(profile, {phase_split});
        ASSERT_EQ(sharded_plan.devices.size(), 1u);
        ASSERT_EQ(split_plan.devices.size(), 1u);
        const auto &tp = sharded_plan.devices.front();
        const auto &split = split_plan.devices.front();
        EXPECT_GT(split.kv_cache_bytes(), tp.kv_cache_bytes());
        EXPECT_GT(split.activation_bytes(), tp.activation_bytes());
        EXPECT_GT(split.workspace_bytes(), tp.workspace_bytes());
        EXPECT_GT(split.total_bytes(), tp.total_bytes());
    }
}

TEST(Test__MemoryPlanner,
     MirroredMTPVocabularyPricesPrimaryShardAndBothFullViews)
{
    const auto profile = createTestProfile();
    DevicePlanConfig config;
    config.device = DeviceId::cuda(0);
    config.device_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    config.device_free_bytes = config.device_total_bytes;
    config.device_compute_units = 82;
    config.shard_index = 0;
    config.total_shards = 2;
    config.local_tp_backend = CollectiveBackendType::NCCL;
    config.first_layer = 0;
    config.last_layer = profile.n_layers - 1;
    config.batch_size = 1;
    config.max_seq_len = 4096;
    config.mtp_enabled = true;
    config.mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 2;
    mtp.sidecar_dense_policy = MTPSidecarDensePolicy::TensorParallel;
    mtp.terminal_head_policy =
        MTPTerminalHeadPolicy::MirroredFullVocabulary;
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding,
            config.total_shards,
            mtp);

    ASSERT_EQ(config.additional_weight_sets.size(), 2u);
    EXPECT_EQ(
        config.additional_weight_sets[0],
        AdditionalPersistentWeightSet::MirroredDecodeEmbedding);
    EXPECT_EQ(
        config.additional_weight_sets[1],
        AdditionalPersistentWeightSet::MirroredMTPTerminalHead);

    const auto primary = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        config.shard_index,
        config.total_shards,
        config.first_layer,
        config.last_layer,
        config.weight_residency);
    const auto mirrored = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        /*shard_index=*/0,
        /*total_shards=*/1,
        config.first_layer,
        config.last_layer,
        config.weight_residency);
    const auto planned = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(planned.devices.size(), 1u);
    EXPECT_EQ(planned.devices.front().weight_bytes(), primary.device_bytes);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes(),
        mirrored.prepared_embedding_bytes + mirrored.lm_head_bytes);
    EXPECT_EQ(
        planned.devices.front().collective_bytes(),
        CollectiveMemoryEstimator::localTP(
            config.max_seq_len,
            profile.d_model,
            config.local_tp_backend.value())
            .perDeviceBytes());
}

/**
 * @brief Logical collective payload capacity must not masquerade as VRAM.
 *
 * Homogeneous NCCL/RCCL retain only the guarded FP16 conversion scratch. The
 * wider FP32 payload value is still passed to the backend as protocol capacity,
 * but no allocation exists for admission to charge.
 */
TEST(Test__MemoryPlanner,
     LocalTPCollectiveBomSeparatesLogicalCapacityFromPhysicalScratch)
{
    constexpr int max_rows = 10;
    constexpr int hidden = 20;
    constexpr std::size_t logical_elements = 200u;
    constexpr std::size_t margined_elements = 220u;

    const auto bom = CollectiveMemoryEstimator::localTP(
        max_rows, hidden, CollectiveBackendType::RCCL);

    EXPECT_EQ(
        bom.backend_payload_capacity_bytes,
        logical_elements * sizeof(float) * 11u / 10u);
    EXPECT_EQ(bom.fp16_scratch_elements, margined_elements);
    EXPECT_EQ(
        bom.fp16_scratch_bytes,
        margined_elements * sizeof(std::uint16_t) +
            CollectiveMemoryEstimator::kFP16ScratchGuardBytes);
    EXPECT_EQ(
        bom.perDeviceBytes(),
        bom.fp16_scratch_bytes + bom.graph_capture_boundary_bytes)
        << "Protocol payload capacity is not a second physical GPU buffer";
}

/** @test Native PP owns only its fence; it does not borrow TP conversion capacity. */
TEST(Test__MemoryPlanner, NativePipelineFenceAccountingIsBackendSymmetric)
{
    const auto profile = createTestProfile();
    for (auto backend : {CollectiveBackendType::NCCL, CollectiveBackendType::RCCL})
    {
        DevicePlanConfig config;
        config.device = backend == CollectiveBackendType::NCCL ? DeviceId::cuda(0) : DeviceId::rocm(0);
        config.device_total_bytes = config.device_free_bytes = 64ull << 30;
        config.device_compute_units = 128;
        config.max_seq_len = 256;
        const auto before = MemoryPlanner::plan(profile, {config}).physicalPlan().totalBytes();
        config.local_pipeline_backend = backend;
        const auto after = MemoryPlanner::plan(profile, {config}).physicalPlan().totalBytes();
        EXPECT_EQ(after - before, CollectiveMemoryEstimator::nativePipelineBoundaryBytes(backend));
        EXPECT_EQ(after - before, sizeof(int32_t));
    }
    EXPECT_THROW((void)CollectiveMemoryEstimator::nativePipelineBoundaryBytes(CollectiveBackendType::AUTO), std::invalid_argument);
    EXPECT_THROW((void)CollectiveMemoryEstimator::nativePipelineBoundaryBytes(CollectiveBackendType::HOST), std::invalid_argument);
}

/** @test Channel admission charges each physical owner once, independent of GPU family. */
TEST(Test__MemoryPlanner, CapturedPipelineChannelAccountingIsExactAndBackendSymmetric)
{
    const auto profile = createTestProfile();
    constexpr auto owner = PhysicalMemoryOwner::ActivationTransportStaging;
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(3)})
    for (const int verifier_rows : {1, 2, 4, 16})
    for (const auto sides : {std::vector{PipelineBoundarySide::EarlierDomain},
             std::vector{PipelineBoundarySide::LaterDomain},
             std::vector{PipelineBoundarySide::LaterDomain, PipelineBoundarySide::EarlierDomain}})
    {
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.device_total_bytes = cfg.device_free_bytes = 64ull << 30;
        cfg.device_compute_units = 128;
        cfg.max_seq_len = 256;
        cfg.activation_seq_len = 8; // A deeper verifier, not prefill, can set the slot capacity.
        cfg.mtp_enabled = verifier_rows > 1;
        cfg.mtp_target_query_rows = verifier_rows;
        cfg.associated_host_memory = PhysicalMemoryResource{.world_rank = -1, .device = DeviceId::cpu(),
            .total_bytes = 64ull << 30, .admission_available_bytes = 64ull << 30};
        const auto before = MemoryPlanner::plan(profile, {cfg}).physicalPlan().totalBytes();
        cfg.captured_pipeline_boundaries = sides;
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        const auto geometry = PipelineTransferMemory::forRows(profile.d_model, 8, verifier_rows);
        const bool owns_host = std::find(sides.begin(), sides.end(), PipelineBoundarySide::EarlierDomain) != sides.end();
        size_t host_bytes = 0, device_bytes = 0;
        for (const auto &row : plan.devices)
            (row.device().is_cpu() ? host_bytes : device_bytes) += row.bom().bytes(owner);
        EXPECT_EQ(device_bytes, geometry.device_bytes_per_boundary * sides.size());
        EXPECT_EQ(host_bytes, owns_host ? geometry.host_bytes_per_boundary : 0);
        EXPECT_EQ(plan.physicalPlan().totalBytes() - before, device_bytes + host_bytes);
        EXPECT_EQ(geometry.activation_capacity_bytes, size_t(profile.d_model) * std::max(8, verifier_rows) * sizeof(float));
        EXPECT_EQ(geometry.metadata_capacity_bytes, size_t(verifier_rows) * sizeof(int32_t));

        auto invalid = cfg;
        invalid.captured_pipeline_boundaries.push_back(sides.front());
        EXPECT_THROW(MemoryPlanner::plan(profile, {invalid}), std::invalid_argument);
        invalid = cfg; invalid.shard_index = 1;
        EXPECT_THROW(MemoryPlanner::plan(profile, {invalid}), std::invalid_argument);
        invalid = cfg; invalid.local_pipeline_backend = device.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        EXPECT_THROW(MemoryPlanner::plan(profile, {invalid}), std::invalid_argument);
        if (owns_host)
        {
            invalid = cfg; invalid.associated_host_memory.reset();
            EXPECT_THROW(MemoryPlanner::plan(profile, {invalid}), std::invalid_argument);
        }
    }
    EXPECT_THROW(PipelineTransferMemory::forRows(0, 1, 1), std::invalid_argument);
    EXPECT_THROW(PipelineTransferMemory::forRows(1, 0, 1), std::invalid_argument);
    EXPECT_THROW(PipelineTransferMemory::forRows(1, 1, 0), std::invalid_argument);
    EXPECT_THROW(PipelineTransferMemory::forRows(std::numeric_limits<size_t>::max(), 2, 1), std::overflow_error);
}

TEST(Test__MemoryPlanner,
     MirroredTerminalHeadIsPricedForSerialOracleWhenMTPIsOff)
{
    const auto profile = createTestProfile();
    DevicePlanConfig config;
    config.device = DeviceId::cuda(0);
    config.device_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    config.device_free_bytes = config.device_total_bytes;
    config.device_compute_units = 82;
    config.shard_index = 0;
    config.total_shards = 2;
    config.local_tp_backend = CollectiveBackendType::NCCL;
    config.first_layer = 0;
    config.last_layer = profile.n_layers - 1;
    config.batch_size = 1;
    config.max_seq_len = 64;
    config.mtp_enabled = false;
    MTPRuntimeConfig mtp;
    mtp.enabled = false;
    mtp.terminal_head_policy =
        MTPTerminalHeadPolicy::MirroredFullVocabulary;
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding,
            config.total_shards,
            mtp);

    ASSERT_EQ(config.additional_weight_sets.size(), 2u);
    EXPECT_EQ(
        config.additional_weight_sets[0],
        AdditionalPersistentWeightSet::MirroredDecodeEmbedding);
    EXPECT_EQ(
        config.additional_weight_sets[1],
        AdditionalPersistentWeightSet::MirroredMTPTerminalHead);

    const auto mirrored = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        /*shard_index=*/0,
        /*total_shards=*/1,
        config.first_layer,
        config.last_layer,
        config.weight_residency);
    const auto planned = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(planned.devices.size(), 1u);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes(),
        mirrored.prepared_embedding_bytes + mirrored.lm_head_bytes)
        << "Serial decode and grouped MTP must retain the same mirrored terminal surface.";
}

/**
 * @brief Admit native mirrored tables beside primary shards through the one BOM.
 *
 * The old embedding component counted only EmbedQ8 and silently contributed
 * zero for native floating tables. Both GPU backends must price the full
 * physical view at every TP degree, independent of how small its primary
 * vocabulary shard is. This is device-free metadata admission, not a ledger.
 */
TEST(Test__MemoryPlanner, NativeMirroredVocabularyIsNotFreeAtAnyTPDegree)
{
    for (const auto &format : {"F32", "F16", "BF16"})
    {
        auto profile = createTestProfile();
        const size_t element_bytes = std::string(format) == "F32" ? 4u : 2u;
        for (auto &tensor : profile.tensors)
        {
            if (tensor.name != "token_embd.weight" && tensor.name != "output.weight")
                continue;
            profile.total_native_bytes -= tensor.native_bytes;
            tensor.quant_type = format;
            tensor.native_bytes = tensor.elements * element_bytes;
            profile.total_native_bytes += tensor.native_bytes;
        }
        for (const DeviceType backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            for (int degree = 2; degree <= 8; ++degree)
            {
                SCOPED_TRACE(std::string(format) + " " + DeviceId(backend, 0).toString() +
                             " degree=" + std::to_string(degree));
                DevicePlanConfig config;
                config.device = DeviceId(backend, 0);
                config.device_total_bytes = 64ULL << 30u;
                config.device_free_bytes = config.device_total_bytes;
                config.device_compute_units = 64;
                config.total_shards = degree;
                config.local_tp_backend = backend == DeviceType::CUDA
                    ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                config.first_layer = 0;
                config.last_layer = profile.n_layers - 1;
                config.max_seq_len = 64;
                config.additional_weight_sets = {
                    AdditionalPersistentWeightSet::MirroredDecodeEmbedding,
                    AdditionalPersistentWeightSet::MirroredMTPTerminalHead};
                const auto planned = MemoryPlanner::plan(profile, {config});
                ASSERT_EQ(planned.devices.size(), 1u);
                EXPECT_EQ(planned.devices.front().additional_weight_bytes(),
                          2u * static_cast<size_t>(profile.vocab_size) * profile.d_model * element_bytes);
            }
        }
    }
}

/**
 * @brief Ordinary TP must admit its mirrored terminal head before loading it.
 *
 * The ordinary LocalTP graph keeps its embedding vocabulary-sharded, while
 * the default MTP terminal-head policy gives each participant a complete Q8
 * terminal projection for serial decode and grouped verification.  Treating
 * terminal mirroring as embedding mirroring both overstates admission and
 * obscures the exact allocation that PhysicalMemoryAuthority must authorize.
 */
TEST(Test__MemoryPlanner,
     OrdinaryTensorParallelMTPOffPricesOnlyMirroredTerminalHead)
{
    const auto profile = createTestProfile();
    DevicePlanConfig config;
    config.device = DeviceId::rocm(0);
    config.device_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    config.device_free_bytes = config.device_total_bytes;
    config.device_compute_units = 60;
    config.shard_index = 0;
    config.total_shards = 2;
    config.local_tp_backend = CollectiveBackendType::RCCL;
    config.first_layer = 0;
    config.last_layer = profile.n_layers - 1;
    config.batch_size = 1;
    config.max_seq_len = 64;

    MTPRuntimeConfig mtp;
    mtp.enabled = false;
    mtp.terminal_head_policy =
        MTPTerminalHeadPolicy::MirroredFullVocabulary;
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallel,
            config.total_shards,
            mtp);

    ASSERT_EQ(config.additional_weight_sets.size(), 1u);
    EXPECT_EQ(
        config.additional_weight_sets.front(),
        AdditionalPersistentWeightSet::MirroredMTPTerminalHead);

    const auto mirrored = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        /*shard_index=*/0,
        /*total_shards=*/1,
        config.first_layer,
        config.last_layer,
        config.weight_residency);
    const auto planned = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(planned.devices.size(), 1u);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes(),
        mirrored.lm_head_bytes);
}

TEST(Test__MemoryPlanner,
     ReplicatedMTPSidecarPricesTrailingDenseAndSharedWeights)
{
    auto profile = createMoEOverlayProfile();
    profile.mtp_layer_count = 1;

    DevicePlanConfig config = overlayDeviceConfig(DeviceId::rocm(0));
    config.shard_index = 0;
    config.total_shards = 2;
    config.local_tp_backend = CollectiveBackendType::RCCL;
    config.mtp_enabled = true;
    config.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            std::vector<int>(
                static_cast<std::size_t>(profile.n_layers),
                0));

    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 2;
    mtp.sidecar_dense_policy =
        MTPSidecarDensePolicy::ReplicatedPerParticipant;
    mtp.terminal_head_policy =
        MTPTerminalHeadPolicy::VocabularySharded;
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallel,
            config.total_shards,
            mtp);

    ASSERT_EQ(config.additional_weight_sets.size(), 1u);
    EXPECT_EQ(
        config.additional_weight_sets.front(),
        AdditionalPersistentWeightSet::ReplicatedMTPSidecarDense);

    const auto sidecar_and_globals = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        /*shard_index=*/0,
        /*total_shards=*/1,
        /*first_layer=*/profile.n_layers - profile.mtp_layer_count,
        /*last_layer=*/profile.n_layers - 1,
        config.weight_residency);
    const auto globals_only = WeightMemoryEstimator::estimate(
        profile,
        config.device,
        /*shard_index=*/0,
        /*total_shards=*/1,
        /*first_layer=*/profile.n_layers,
        /*last_layer=*/profile.n_layers - 1,
        config.weight_residency);
    ASSERT_GT(sidecar_and_globals.device_bytes, globals_only.device_bytes);

    const auto planned = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(planned.devices.size(), 1u);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes(),
        sidecar_and_globals.device_bytes - globals_only.device_bytes)
        << "The auxiliary BOM must include the complete unsharded MTP "
           "predictor while ExpertOverlay retains routed-parent authority.";
}

TEST(Test__MemoryPlanner, TP2_ReducesKVCachePerDevice)
{
    auto profile = createTestProfile();

    // Single device — all KV heads
    DevicePlanConfig single;
    single.device = DeviceId::cuda(0);
    single.device_compute_units = 82;
    single.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    single.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    single.batch_size = 1;
    single.max_seq_len = 4096;
    single.kv_precision = "fp16";

    auto single_plan = MemoryPlanner::plan(profile, {single});

    // TP-2 — KV heads should be divided across shards
    DevicePlanConfig shard0 = single;
    shard0.shard_index = 0;
    shard0.total_shards = 2;
    shard0.local_tp_backend = CollectiveBackendType::NCCL;

    DevicePlanConfig shard1 = single;
    shard1.device = DeviceId::cuda(1);
    shard1.shard_index = 1;
    shard1.total_shards = 2;
    shard1.local_tp_backend = CollectiveBackendType::NCCL;

    auto tp_plan = MemoryPlanner::plan(profile, {shard0, shard1});

    // Each shard should have less KV cache than the full model
    // (n_kv_heads=2, so TP-2 gives 1 head per shard)
    EXPECT_LT(tp_plan.devices[0].kv_cache_bytes(), single_plan.devices[0].kv_cache_bytes());
    EXPECT_LT(tp_plan.devices[1].kv_cache_bytes(), single_plan.devices[0].kv_cache_bytes());
}

/**
 * @brief Rank-local TP admission must consume the exact uneven GQA assignment.
 *
 * Qwen2 has fourteen query heads and two KV heads. TP4 therefore owns uneven
 * query ranges while replicating both KV heads on every participant. This is
 * the metadata-only regression for the production ROCm TP4 parity failure:
 * admission must neither reject 14/4 nor reconstruct one KV head per device.
 */
TEST(Test__MemoryPlanner, RankLocalTP4UsesExactUnevenQAndReplicatedKVGeometry)
{
    const auto profile = createTestProfile();
    const std::vector<DeviceId> devices{
        DeviceId::rocm(0),
        DeviceId::rocm(1),
        DeviceId::rocm(2),
        DeviceId::rocm(3),
    };
    const auto assignments = TensorParallelConfig::proportionalSplit(
        devices,
        std::vector<float>(devices.size(), 1.0f),
        profile.n_heads,
        profile.n_kv_heads,
        profile.d_ff,
        profile.vocab_size);

    std::vector<DevicePlanConfig> configs;
    configs.reserve(devices.size());
    for (std::size_t rank = 0; rank < devices.size(); ++rank)
    {
        DevicePlanConfig config;
        config.device = devices[rank];
        config.device_compute_units = 60;
        config.device_total_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
        config.device_free_bytes = config.device_total_bytes;
        config.shard_index = static_cast<int>(rank);
        config.total_shards = static_cast<int>(devices.size());
        config.local_tp_backend = CollectiveBackendType::RCCL;
        config.batch_size = 1;
        config.max_seq_len = 64;
        config.activation_seq_len = 16;
        config.kv_precision = "fp16";
        config.bindTensorParallelAssignment(
            assignments.forRank(static_cast<int>(rank)));
        configs.push_back(std::move(config));
    }

    const auto plan = MemoryPlanner::plan(profile, configs);
    ASSERT_EQ(plan.devices.size(), devices.size());
    for (std::size_t rank = 0; rank < devices.size(); ++rank)
    {
        EXPECT_EQ(assignments.forRank(static_cast<int>(rank)).kv_head_count, 2);
        EXPECT_GT(plan.devices[rank].kv_cache_bytes(), 0u);
        EXPECT_GT(plan.devices[rank].activation_bytes(), 0u);
        EXPECT_GT(plan.devices[rank].workspace_bytes(), 0u);
    }
    EXPECT_GT(plan.devices[0].activation_bytes(),
              plan.devices[3].activation_bytes())
        << "The two-query-head remainder shard must not be priced as a uniform three-head shard.";
}

TEST(Test__MemoryPlanner, PP2_SplitsLayersAcrossDevices)
{
    auto profile = createTestProfile(); // 24 layers

    // Stage 0: layers 0-11
    DevicePlanConfig stage0;
    stage0.device = DeviceId::cuda(0);
    stage0.device_compute_units = 82;
    stage0.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    stage0.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    stage0.batch_size = 1;
    stage0.max_seq_len = 4096;
    stage0.kv_precision = "fp16";
    stage0.first_layer = 0;
    stage0.last_layer = 11;

    // Stage 1: layers 12-23
    DevicePlanConfig stage1 = stage0;
    stage1.device = DeviceId::cuda(1);
    stage1.first_layer = 12;
    stage1.last_layer = 23;

    auto pp_plan = MemoryPlanner::plan(profile, {stage0, stage1});

    EXPECT_EQ(pp_plan.devices.size(), 2u);

    // Each PP stage owns only its layer interval and actual global endpoints.
    DevicePlanConfig full;
    full.device = DeviceId::cuda(0);
    full.device_compute_units = 82;
    full.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    full.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    full.batch_size = 1;
    full.max_seq_len = 4096;
    full.kv_precision = "fp16";

    auto full_plan = MemoryPlanner::plan(profile, {full});

    EXPECT_LT(pp_plan.devices[0].weight_bytes(), full_plan.devices[0].weight_bytes());
    EXPECT_LT(pp_plan.devices[1].weight_bytes(), full_plan.devices[0].weight_bytes());

    // Each stage should have half the KV cache layers (12 vs 24)
    EXPECT_LT(pp_plan.devices[0].kv_cache_bytes(), full_plan.devices[0].kv_cache_bytes());
    EXPECT_EQ(pp_plan.devices[0].kv_cache_bytes(), pp_plan.devices[1].kv_cache_bytes());
}

/** @brief PMA receives each PP stage's actual endpoint weights, not full-model globals. */
TEST(Test__MemoryPlanner, PipelineWeightBOMPartitionsEndpointsAndMirrors)
{
    const auto profile = createTestProfile();
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (int degree : {1, 2, 4, 8})
        {
            SCOPED_TRACE(std::to_string(static_cast<int>(backend)) + ":" + std::to_string(degree));
            DevicePlanConfig cfg;
            cfg.device = DeviceId(backend, 0);
            cfg.device_compute_units = 60;
            cfg.device_total_bytes = cfg.device_free_bytes = 64ull << 30;
            cfg.max_seq_len = 64;
            cfg.total_shards = degree;
            if (degree > 1)
                cfg.additional_weight_sets = {AdditionalPersistentWeightSet::MirroredDecodeEmbedding,
                                              AdditionalPersistentWeightSet::MirroredMTPTerminalHead};
            const auto full = MemoryPlanner::plan(profile, {cfg}).devices.front().total_weight_bytes();
            size_t stages = 0;
            for (int stage = 0; stage < 3; ++stage)
            {
                cfg.first_layer = stage * 8;
                cfg.last_layer = stage * 8 + 7;
                cfg.owns_embedding = stage == 0;
                stages += MemoryPlanner::plan(profile, {cfg}).devices.front().total_weight_bytes();
            }
            EXPECT_EQ(stages, full) << "A three-stage pipeline must not clone vocabulary weights or mirrors";
        }
}

/** @brief All-GDN PP followers price live and rollback metadata without dummy attention. */
TEST(Test__MemoryPlanner, RecurrentOnlyPipelineFollowerOwnsSequenceMetadata)
{
    const auto profile = createQwen36HybridMTPStateProfile();
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const std::string precision : {"fp32", "fp16", "bf16", "q8_1"})
        {
            DevicePlanConfig cfg;
            cfg.device = device;
            cfg.device_compute_units = 60;
            cfg.device_total_bytes = cfg.device_free_bytes = 64ull << 30;
            cfg.max_seq_len = 64;
            cfg.kv_precision = precision;
            cfg.first_layer = 0;
            cfg.last_layer = 1;
            cfg.owns_embedding = true;
            cfg.mtp_enabled = true;
            const auto plan = MemoryPlanner::plan(profile, {cfg}).devices.front();
            EXPECT_EQ(plan.kv_cache_bytes(), 2u * sizeof(int32_t));
            // Four live rollback slots plus one accepted-verifier publication.
            EXPECT_EQ(plan.sequence_metadata_bytes(), 5u * 2u * sizeof(int32_t));
            EXPECT_GT(plan.checkpoint_state_bytes(), 0u);
        }
}

/**
 * @brief Recurrent-only prefix archives retain their exact admitted physical owners.
 *
 * A PP head with no attention layers still exports a real recurrent snapshot.
 * Its GPU staging and hot slots contain that image alone; the RAM owner exists
 * on every backend. Neither MTP enablement nor KV precision may erase this BOM
 * or introduce a fictitious attention block.
 */
TEST(Test__MemoryPlanner, RecurrentOnlyPipelinePrefixAdmission)
{
    const auto profile = createQwen36HybridMTPStateProfile();
    const auto gdn = HybridGDNStateGeometry::resolve(profile.n_heads, 0,
        profile.n_heads, profile.gdn_group_count, profile.gdn_time_step_rank,
        profile.gdn_state_size, profile.gdn_inner_size, profile.gdn_conv_kernel_size);
    const auto recurrent_bytes = gdn.deviceSerializedPayloadBytes(2);
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const std::string precision : {"fp32", "fp16", "bf16", "q8_1"})
    for (const bool mtp_enabled : {false, true})
    for (const auto mode : {PrefixCacheStorageMode::Disabled,
                           PrefixCacheStorageMode::Ram, PrefixCacheStorageMode::Tiered})
    {
        SCOPED_TRACE(device.toString() + ":" + precision + ":mtp=" +
            std::to_string(mtp_enabled) + ":mode=" + std::to_string(static_cast<int>(mode)));
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.cpu_execution = device.is_cpu()
            ? test::kSyntheticCPUExecutionGeometry : CPUExecutionGeometry{};
        cfg.device_compute_units = 60;
        cfg.device_total_bytes = cfg.device_free_bytes = 64ull << 30;
        cfg.max_seq_len = 64;
        cfg.kv_precision = precision;
        cfg.first_layer = 0;
        cfg.last_layer = 1;
        cfg.owns_embedding = true;
        cfg.mtp_enabled = mtp_enabled;
        cfg.prefix_cache.enabled = true;
        cfg.prefix_cache.storage_mode = mode;
        cfg.prefix_cache.ram_budget_bytes = 64u << 20;
        cfg.prefix_cache.device_budget_bytes = 32u << 20;
        cfg.prefix_cache.disk_budget_bytes = 0;
        cfg.associated_host_memory = PhysicalMemoryResource{
            .world_rank = -1, .device = DeviceId::cpu(),
            .total_bytes = 64ull << 30, .admission_available_bytes = 64ull << 30};
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        const auto entry = std::find_if(plan.devices.begin(), plan.devices.end(),
            [&](const auto &candidate) { return candidate.device() == device; });
        ASSERT_NE(entry, plan.devices.end());
        const bool enabled = mode != PrefixCacheStorageMode::Disabled;
        EXPECT_EQ(entry->prefix_cache_staging_bytes(),
            enabled && device.is_gpu() ? recurrent_bytes : 0u);
        EXPECT_EQ(entry->prefix_cache_device_hot_bytes(),
            mode == PrefixCacheStorageMode::Tiered && device.is_gpu()
                ? prefixCacheWholeBlockReservationBytes(
                    cfg.prefix_cache.device_budget_bytes, recurrent_bytes) : 0u);
        const auto host = std::find_if(plan.devices.begin(), plan.devices.end(),
            [](const auto &candidate) { return candidate.device().is_cpu(); });
        if (enabled || device.is_cpu())
        {
            ASSERT_NE(host, plan.devices.end());
            EXPECT_EQ(host->bom().bytes(PhysicalMemoryOwner::PrefixHostTier),
                enabled ? cfg.prefix_cache.ram_budget_bytes : 0u);
        }
    }
}

/**
 * @brief PP followers retain main rollback state, never another shifted predictor.
 *
 * CPU and both GPU BOMs must use the same ownership rule. A follower still
 * verifies speculative rows in its own layers, so clearing MTP entirely would
 * under-admit rollback state even though it removes the excess shifted cache.
 */
TEST(Test__MemoryPlanner, PipelineFollowersRetainRollbackButNotPredictorCaches)
{
    auto profile = createTestProfile();
    profile.mtp_layer_count = 1;
    ++profile.n_layers;
    auto predictor = profile.tensors[3];
    predictor.name = "blk.24.attn_q.weight";
    predictor.layer_index = 24;
    profile.tensors.push_back(predictor);
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const std::string precision : {"fp32", "fp16"})
            for (const int degree : {1, 2})
                for (int stage = 0; stage < 3; ++stage)
                {
                    SCOPED_TRACE(device.toString() + ":" + precision + ":" +
                                 std::to_string(degree) + ":" + std::to_string(stage));
                    DevicePlanConfig cfg;
                    cfg.device = device;
                    cfg.cpu_execution = device.is_cpu()
                        ? test::kSyntheticCPUExecutionGeometry : CPUExecutionGeometry{};
                    cfg.device_compute_units = 60;
                    cfg.device_total_bytes = cfg.device_free_bytes = 64ull << 30;
                    cfg.max_seq_len = 64;
                    cfg.kv_precision = precision;
                    cfg.total_shards = degree;
                    cfg.first_layer = stage * 8;
                    cfg.last_layer = stage * 8 + 7;
                    cfg.owns_embedding = stage == 0;
                    const auto ordinary = MemoryPlanner::plan(profile, {cfg}).devices.front();
                    cfg.mtp_enabled = true;
                    const auto speculative = MemoryPlanner::plan(profile, {cfg}).devices.front();
                    if (stage == 2)
                        EXPECT_GT(speculative.kv_cache_bytes(), ordinary.kv_cache_bytes());
                    else
                        EXPECT_EQ(speculative.kv_cache_bytes(), ordinary.kv_cache_bytes())
                            << "A nonterminal stage owns no learned predictor";
                    EXPECT_GT(speculative.checkpoint_state_bytes(), ordinary.checkpoint_state_bytes())
                        << "Every main-model stage still owns rollback state";
                    if (device.is_gpu())
                        EXPECT_EQ(speculative.sequence_metadata_bytes(),
                                  static_cast<size_t>(4 * (8 + (stage == 2 ? 1 : 0)) + 8) *
                                      2 * sizeof(int32_t));
                }
}

/** @brief Only the pipeline tail owns learned predictor weights and its token lookup. */
TEST(Test__MemoryPlanner, PipelinePredictorWeightsBelongOnlyToTheTerminalStage)
{
    auto profile = createTestProfile();
    profile.mtp_layer_count = 1;
    ++profile.n_layers;
    auto predictor = profile.tensors[3];
    predictor.name = "blk.24.attn_q.weight";
    predictor.layer_index = 24;
    profile.tensors.push_back(predictor);
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        DevicePlanConfig cfg;
        cfg.device = DeviceId(backend, 0);
        cfg.device_compute_units = 60;
        cfg.device_total_bytes = cfg.device_free_bytes = 64ull << 30;
        cfg.max_seq_len = 64;
        cfg.total_shards = 2;
        cfg.mtp_enabled = true;
        cfg.additional_weight_sets = {AdditionalPersistentWeightSet::ReplicatedMTPSidecarDense};
        cfg.first_layer = 0;
        cfg.last_layer = 11;
        const auto entry = MemoryPlanner::plan(profile, {cfg}).devices.front();
        EXPECT_EQ(entry.additional_weight_bytes(), 0u);

        cfg.first_layer = 12;
        cfg.last_layer = 23;
        cfg.owns_embedding = false;
        const auto terminal = MemoryPlanner::plan(profile, {cfg}).devices.front();
        const auto primary = WeightMemoryEstimator::estimate(profile, cfg.device, 0, 2,
            12, 24, {}, {}, WeightComponentScope::EmbeddingAndTerminal);
        const auto replica = WeightMemoryEstimator::estimate(profile, cfg.device, 0, 1,
            24, 24, {}, {}, WeightComponentScope::LayersOnly);
        EXPECT_EQ(terminal.weight_bytes(), primary.device_bytes);
        EXPECT_EQ(terminal.total_weight_bytes(), primary.device_bytes + replica.device_bytes);
        EXPECT_EQ(terminal.additional_weight_bytes(), replica.device_bytes);
        EXPECT_GT(replica.device_bytes, 0u);
    }
}

TEST(Test__MemoryPlanner, MixedDevices_CUDAAndROCm)
{
    auto profile = createTestProfile();

    DevicePlanConfig cuda_cfg;
    cuda_cfg.device = DeviceId::cuda(0);
    cuda_cfg.device_compute_units = 82;
    cuda_cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024; // 24 GB (RTX 3090)
    cuda_cfg.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    cuda_cfg.batch_size = 1;
    cuda_cfg.max_seq_len = 4096;
    cuda_cfg.kv_precision = "fp16";
    cuda_cfg.shard_index = 0;
    cuda_cfg.total_shards = 2;
    cuda_cfg.local_tp_backend = CollectiveBackendType::HETEROGENEOUS;

    DevicePlanConfig rocm_cfg;
    rocm_cfg.device = DeviceId::rocm(0);
    rocm_cfg.device_compute_units = 60;
    rocm_cfg.device_total_bytes = 32ULL * 1024 * 1024 * 1024; // 32 GB (MI60)
    rocm_cfg.device_free_bytes = 31ULL * 1024 * 1024 * 1024;
    rocm_cfg.batch_size = 1;
    rocm_cfg.max_seq_len = 4096;
    rocm_cfg.kv_precision = "fp16";
    rocm_cfg.shard_index = 1;
    rocm_cfg.total_shards = 2;
    rocm_cfg.local_tp_backend = CollectiveBackendType::HETEROGENEOUS;

    auto plan = MemoryPlanner::plan(profile, {cuda_cfg, rocm_cfg});

    EXPECT_EQ(plan.devices.size(), 2u);
    EXPECT_TRUE(plan.fits());

    // Both devices should have weight estimates (GPU packing applies to both)
    EXPECT_GT(plan.devices[0].weight_bytes(), 0u);
    EXPECT_GT(plan.devices[1].weight_bytes(), 0u);

    // Weight bytes should be equal (same sharding, same packing formula)
    EXPECT_EQ(plan.devices[0].weight_bytes(), plan.devices[1].weight_bytes());

    // Render table should contain both device types
    auto table = plan.renderTable();
    EXPECT_NE(table.find("CUDA:0"), std::string::npos);
    EXPECT_NE(table.find("ROCm:0"), std::string::npos);
}

/**
 * @brief GPU upload rings aggregate against one shared rank-local RAM authority.
 *
 * Each accelerator owns an independent device staging allocation, but both
 * pinned rings consume the same CPU allocator. The topology plan must merge
 * those host charges before admission rather than allowing either logical GPU
 * view to fit against the complete host capacity on its own.
 */
TEST(Test__MemoryPlanner,
     MixedGPUWeightLoadStagingHasOneAggregatedHostAuthority)
{
    constexpr size_t MiB = 1024ULL * 1024ULL;
    constexpr size_t GiB = 1024ULL * MiB;
    const auto profile = createMoEOverlayProfile();
    const PhysicalMemoryResource host{
        .world_rank = 4,
        .device = DeviceId::cpu(),
        .total_bytes = 128ULL * GiB,
        .admission_available_bytes = 96ULL * GiB,
    };

    auto cuda = overlayDeviceConfig(DeviceId::cuda(0));
    cuda.world_rank = 4;
    cuda.associated_host_memory = host;
    cuda.weight_load_staging = {
        .device_bytes = 4ULL * MiB,
        .host_bytes = 6ULL * MiB,
    };

    auto rocm = overlayDeviceConfig(DeviceId::rocm(1));
    rocm.world_rank = 4;
    rocm.associated_host_memory = host;
    rocm.weight_load_staging = {
        .device_bytes = 8ULL * MiB,
        .host_bytes = 10ULL * MiB,
    };

    const auto plan = MemoryPlanner::plan(profile, {cuda, rocm});
    ASSERT_TRUE(plan.fits()) << plan.renderTable();
    ASSERT_EQ(plan.physicalPlan().resources().size(), 3u);

    const auto *cuda_bom = plan.physicalPlan().find(
        {4, DeviceId::cuda(0)});
    const auto *rocm_bom = plan.physicalPlan().find(
        {4, DeviceId::rocm(1)});
    const auto *host_bom = plan.physicalPlan().find(
        {4, DeviceId::cpu()});
    ASSERT_NE(cuda_bom, nullptr);
    ASSERT_NE(rocm_bom, nullptr);
    ASSERT_NE(host_bom, nullptr);
    EXPECT_EQ(
        cuda_bom->bytes(PhysicalMemoryOwner::WeightLoadStaging),
        4ULL * MiB);
    EXPECT_EQ(
        rocm_bom->bytes(PhysicalMemoryOwner::WeightLoadStaging),
        8ULL * MiB);
    EXPECT_EQ(
        host_bom->bytes(PhysicalMemoryOwner::WeightLoadStaging),
        16ULL * MiB);

    const auto admission = plan.admit();
    EXPECT_EQ(admission.plan().resources().size(), 3u);
}

TEST(Test__MemoryPlanner, CPUOnly_AttentionWorkspaceUsesPhysicalWorkerBOM)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cpu();
    cfg.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    cfg.device_total_bytes = 128ULL * 1024 * 1024 * 1024;
    cfg.device_free_bytes = 64ULL * 1024 * 1024 * 1024;
    cfg.device_compute_units = 28;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp32";

    auto plan = MemoryPlanner::plan(profile, {cfg});

    auto expected = attention_workspace::cpuParallelRequirements({
        .compact_query_rows =
            attention::kMaxGroupedVerifierAttentionRows,
        .local_query_heads = profile.n_heads,
        .head_dim = profile.head_dim,
        .worker_count = cfg.device_compute_units,
    });
    // Attention and the named FFN transform coexist in the graph workspace;
    // layer-serial transforms share one tile rather than one tile per layer.
    expected.merge(cpuSwiGLUWorkspaceRequirements(
        cfg.batch_size * cfg.max_seq_len, profile.d_ff));
    expected.merge(cpuProjectionQ8WorkspaceRequirements(
        cfg.batch_size * cfg.max_seq_len, profile.d_ff));
    // All ordinary projections contribute their common name maximum. Terminal
    // projection has one live sampling row, not the prefill family's row count.
    for (const auto &tensor : profile.tensors)
    {
        if (tensor.quant_type != "Q8_0" || tensor.name == "token_embd.weight") continue;
        expected.merge(CPUProjectionWorkspaceContract::sourceNative(tensor.quant_type, {
            .rows = tensor.name == "output.weight" ? 1 : cfg.batch_size * cfg.max_seq_len,
            .n = static_cast<int>(tensor.elements / tensor.K), .k = static_cast<int>(tensor.K),
            .workers = cfg.device_compute_units, .execution = cfg.cpu_execution}));
    }
    EXPECT_EQ(
        plan.devices[0].workspace_bytes(),
        expected.total_bytes_with_alignment());
    const auto *host_bom = plan.physicalPlan().find(
        {-1, DeviceId::cpu()});
    ASSERT_NE(host_bom, nullptr);
    EXPECT_EQ(
        host_bom->bytes(PhysicalMemoryOwner::ExecutionWorkspace),
        expected.total_bytes_with_alignment());
    EXPECT_TRUE(plan.fits());
}

TEST(Test__MemoryPlanner, RenderTable_ProducesOutput)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    cfg.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";

    auto plan = MemoryPlanner::plan(profile, {cfg});
    auto table = plan.renderTable();

    EXPECT_FALSE(table.empty());
    // Should contain device name (DeviceId::to_string() returns "CUDA:0")
    EXPECT_NE(table.find("CUDA:0"), std::string::npos);
    // Should contain column headers
    EXPECT_NE(table.find("Weights"), std::string::npos);
    EXPECT_NE(table.find("KV Cache"), std::string::npos);
}

TEST(Test__MemoryPlanner, Diagnostics_WarnsOnTightExactFit)
{
    auto profile = createTestProfile();

    // First, compute the plan with generous memory to learn the total
    DevicePlanConfig probe;
    probe.device = DeviceId::cuda(0);
    probe.device_compute_units = 82;
    probe.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    probe.device_free_bytes = 24ULL * 1024 * 1024 * 1024;
    probe.batch_size = 1;
    probe.max_seq_len = 256;
    probe.kv_precision = "fp16";
    auto probe_plan = MemoryPlanner::plan(profile, {probe});
    ASSERT_TRUE(probe_plan.fits());
    size_t total_needed = probe_plan.devices[0].total_bytes();

    // Leave 100 MiB unallocated (below the diagnostic threshold).
    DevicePlanConfig cfg = probe;
    cfg.device_free_bytes = total_needed + 100ULL * 1024 * 1024;

    auto plan = MemoryPlanner::plan(profile, {cfg});

    EXPECT_GT(plan.devices.size(), 0u);
    EXPECT_TRUE(plan.fits());
    // Remaining is only ~100 MB, which is < 256 MB threshold → tight headroom warning
    EXPECT_FALSE(plan.diagnostics.empty());
}

TEST(Test__MemoryPlanner, PP2_DoesNotFit_WithoutLayerSplit)
{
    // Regression test: verifies that PP must actually set per-device layer ranges.
    // A model that doesn't fit on one 24 GB GPU MUST fail if both devices get
    // all layers (the bug: global first_layer/last_layer assigned to all devices).
    auto profile = createTestProfile(); // 24 layers

    // Full model on a single 24 GB GPU that doesn't have enough free memory
    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_compute_units = 82;
    cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    cfg.device_free_bytes = 1ULL * 1024 * 1024 * 1024; // Only 1 GB free
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";
    cfg.first_layer = 0;
    cfg.last_layer = 23; // All layers

    auto full_plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_FALSE(full_plan.fits()) << "Precondition: model must NOT fit on 1 GB free";

    // BUG scenario: two devices but both get ALL layers (no PP layer split)
    DevicePlanConfig bad_stage0 = cfg;
    bad_stage0.first_layer = 0;
    bad_stage0.last_layer = 23; // All layers — WRONG for PP

    DevicePlanConfig bad_stage1 = cfg;
    bad_stage1.device = DeviceId::cuda(1);
    bad_stage1.first_layer = 0;
    bad_stage1.last_layer = 23; // All layers — WRONG for PP

    auto bad_plan = MemoryPlanner::plan(profile, {bad_stage0, bad_stage1});
    // Both devices still don't fit (each sees full weight)
    EXPECT_FALSE(bad_plan.devices[0].fits());
    EXPECT_FALSE(bad_plan.devices[1].fits());

    // CORRECT: PP2 with proper layer split — each device gets half
    DevicePlanConfig good_stage0 = cfg;
    good_stage0.first_layer = 0;
    good_stage0.last_layer = 11;

    DevicePlanConfig good_stage1 = cfg;
    good_stage1.device = DeviceId::cuda(1);
    good_stage1.first_layer = 12;
    good_stage1.last_layer = 23;

    auto good_plan = MemoryPlanner::plan(profile, {good_stage0, good_stage1});
    // Each device should have less weight than the full model
    EXPECT_LT(good_plan.devices[0].weight_bytes(), full_plan.devices[0].weight_bytes());
    EXPECT_LT(good_plan.devices[1].weight_bytes(), full_plan.devices[0].weight_bytes());
}

TEST(Test__MemoryPlanner, PP_LayerRange_ReducesWeightEstimate_Proportionally)
{
    // Validates that setting first_layer/last_layer to half the layers
    // reduces weight estimate to approximately half (plus non-layer weights).
    auto profile = createTestProfile(); // 24 layers

    // Full model
    DevicePlanConfig full;
    full.device = DeviceId::cuda(0);
    full.device_compute_units = 82;
    full.device_total_bytes = 48ULL * 1024 * 1024 * 1024;
    full.device_free_bytes = 48ULL * 1024 * 1024 * 1024;
    full.batch_size = 1;
    full.max_seq_len = 4096;
    full.kv_precision = "fp16";
    full.first_layer = 0;
    full.last_layer = 23;

    auto full_plan = MemoryPlanner::plan(profile, {full});
    size_t full_weight = full_plan.devices[0].weight_bytes();

    // Half layers (0-11)
    DevicePlanConfig half = full;
    half.first_layer = 0;
    half.last_layer = 11;

    auto half_plan = MemoryPlanner::plan(profile, {half});
    size_t half_weight = half_plan.devices[0].weight_bytes();

    // Half should have less weight than full
    EXPECT_LT(half_weight, full_weight);
    // Should be roughly 50-70% of full (has non-layer tensors like embed/output)
    double ratio = static_cast<double>(half_weight) / full_weight;
    EXPECT_GT(ratio, 0.40) << "Half layers should be at least 40% of full weight";
    EXPECT_LT(ratio, 0.75) << "Half layers should be less than 75% of full weight";
}

TEST(Test__MemoryPlanner, DeviceMemoryPlan_Summary)
{
    PhysicalMemoryBOMBuilder builder({
        .world_rank = -1,
        .device = DeviceId::cuda(0),
        .total_bytes = 1024ULL * 1024 * 1024 * 24,
        .admission_available_bytes = 1024ULL * 1024 * 1024 * 23,
    });
    builder
        .add(
            PhysicalMemoryOwner::PrimaryModelWeights,
            1024 * 1024 * 100)
        .add(PhysicalMemoryOwner::KVCache, 1024 * 1024 * 50)
        .add(PhysicalMemoryOwner::ActivationArena, 1024 * 1024 * 30)
        .add(PhysicalMemoryOwner::ExecutionWorkspace, 1024 * 1024 * 200);
    DeviceMemoryPlan plan(
        builder.build(), /*max_seq_len=*/0, /*activation_seq_len=*/0);

    auto summary = plan.summary();
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("CUDA:0"), std::string::npos);
    EXPECT_NE(summary.find("[OK]"), std::string::npos);
}

TEST(Test__MemoryPlanner, RoutedExpertParticipantOwnsNoDensePersistentState)
{
    const auto profile = createMoEOverlayProfile();

    auto continuation = overlayDeviceConfig(DeviceId::cuda(0));
    continuation.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {2, 2});

    auto participant = overlayDeviceConfig(DeviceId::rocm(0));
    participant.execution_role =
        DeviceExecutionMemoryRole::RoutedExpertParticipant;
    participant.weight_residency =
        DeviceWeightResidency::selectedRoutedExpertsOnly(
            profile.expert_count,
            {2, 2});

    const auto plan = MemoryPlanner::plan(
        profile, {continuation, participant});

    ASSERT_EQ(plan.devices.size(), 2u);
    const auto &continuation_plan = plan.devices[0];
    const auto &participant_plan = plan.devices[1];
    EXPECT_GT(continuation_plan.kv_cache_bytes(), 0u);
    EXPECT_GT(continuation_plan.activation_bytes(), 0u);
    EXPECT_GT(continuation_plan.workspace_bytes(), 0u);
    EXPECT_EQ(participant_plan.kv_cache_bytes(), 0u);
    EXPECT_EQ(participant_plan.live_recurrent_state_bytes(), 0u);
    EXPECT_EQ(participant_plan.checkpoint_state_bytes(), 0u);
    EXPECT_EQ(participant_plan.persistent_state_bytes(), 0u);
    EXPECT_GT(participant_plan.activation_bytes(), 0u);
    EXPECT_GT(participant_plan.workspace_bytes(), 0u);
    EXPECT_LT(participant_plan.weight_bytes(), continuation_plan.weight_bytes());
}

/**
 * @brief Two graph owners on one physical CPU resource need two scratch blocks.
 *
 * Serial prefill/decode graph families may alias within one owner, but local
 * GPU children can execute their CPU expert graphs concurrently. The PMA BOM
 * must reserve each owner's block before expert quota filling begins.
 */
TEST(Test__MemoryPlanner, RoutedExpertConcurrentWorkspaceOwnersAreAdditive)
{
    const auto profile = createMoEOverlayProfile();
    auto participant = overlayDeviceConfig(DeviceId::cpu());
    participant.execution_role =
        DeviceExecutionMemoryRole::RoutedExpertParticipant;
    participant.weight_residency =
        DeviceWeightResidency::selectedRoutedExpertsOnly(
            profile.expert_count, {2, 2});

    const auto one = MemoryPlanner::plan(profile, {participant});
    ASSERT_EQ(one.devices.size(), 1u);
    ASSERT_GT(one.devices.front().workspace_bytes(), 0u);

    participant.concurrent_workspace_owners = 2u;
    const auto two = MemoryPlanner::plan(profile, {participant});
    ASSERT_EQ(two.devices.size(), 1u);
    EXPECT_EQ(two.devices.front().workspace_bytes(),
              2u * one.devices.front().workspace_bytes());
    EXPECT_EQ(two.devices.front().activation_bytes(),
              one.devices.front().activation_bytes());
    EXPECT_EQ(two.devices.front().weight_bytes(),
              one.devices.front().weight_bytes());

    participant.concurrent_workspace_owners = 0u;
    EXPECT_THROW((void)MemoryPlanner::plan(profile, {participant}),
                 std::invalid_argument);
}

/**
 * @test CPU continuations admit the exact local sparse stage, not token-only scratch.
 *
 * A continuation still executes a compact expert packet: each token can route
 * to every top-k expert. Compare the physical BOM with the production stage's
 * metadata-only declaration, without allocating weights or multi-GiB scratch.
 * Explicit serial capacities are already flattened across requests; TP changes
 * expert ownership, never the shape of one whole expert's projection.
 */
TEST(Test__MemoryPlanner, CPUOverlayContinuationAdmitsCompactProjectionWorkspaceAllFormats)
{
    std::vector<std::pair<std::string, TensorType>> formats{
        {"F32", TensorType::FP32}, {"F16", TensorType::FP16}, {"BF16", TensorType::BF16}};
    for (const auto &format : test::quantizedVerifierFormats())
        formats.emplace_back(format.label, format.tensor_type);

    ModelMemoryProfile profile;
    profile.architecture = "qwen35moe";
    profile.n_layers = 1;
    profile.d_model = 2048;
    profile.d_ff = profile.expert_feed_forward_length = 512;
    profile.expert_count = 8;
    profile.expert_used_count = 8;
    profile.n_heads = 16;
    profile.n_kv_heads = 8;
    profile.head_dim = 128;
    profile.vocab_size = 1024;
    profile.max_seq_len = 8192;

    for (const int batch : {1, 2})
    for (const int declared_rows : {0, 31, 512, 2048, 4096})
    for (int shards = 1; shards <= 8; ++shards)
    {
        auto cfg = overlayDeviceConfig(DeviceId::cpu());
        cfg.device_total_bytes = cfg.device_free_bytes = 256ULL * 1024 * 1024 * 1024;
        cfg.device_compute_units = 28;
        cfg.batch_size = batch;
        cfg.max_seq_len = 8192;
        cfg.activation_seq_len = 2048;
        cfg.first_layer = cfg.last_layer = 0;
        cfg.total_shards = shards;
        cfg.weight_residency = DeviceWeightResidency::continuationWithSelectedRoutedExperts(8, {1});
        cfg.routed_expert_compact_buffer_lifetime = RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
        cfg.serial_routed_expert_participant_count = 1;
        cfg.serial_routed_expert_compact_rows = declared_rows;
        const int token_rows = declared_rows > 0 ? declared_rows : batch * cfg.activation_seq_len;
        declareCPUSerialFixture(cfg, profile);

        // Remove only the expert matrices to isolate their named workspace
        // contribution from the independent attention family in the same BOM.
        profile.tensors.clear();
        const auto base = MemoryPlanner::plan(profile, {cfg}).devices.front().workspace_bytes();
        for (const auto &[name, type] : formats)
        {
            SCOPED_TRACE(name + " batch=" + std::to_string(batch) +
                " declared_rows=" + std::to_string(declared_rows) + " tp=" + std::to_string(shards));
            profile.tensors = {
                {.name = "blk.0.ffn_gate_exps.weight", .quant_type = name,
                 .elements = 8u * 512 * 2048, .K = 2048, .layer_index = 0},
                {.name = "blk.0.ffn_up_exps.weight", .quant_type = name,
                 .elements = 8u * 512 * 2048, .K = 2048, .layer_index = 0},
                {.name = "blk.0.ffn_down_exps.weight", .quant_type = name,
                 .elements = 8u * 2048 * 512, .K = 512, .layer_index = 0}};
            MoELocalExpertStage::Params params;
            params.device_id = DeviceId::cpu();
            params.num_experts = profile.expert_count;
            params.top_k = profile.expert_used_count;
            params.d_model = profile.d_model;
            params.expert_intermediate = profile.expert_feed_forward_length;
            params.cpu_workspace_source = MoELocalExpertStage::CPUExpertWorkspaceSource{
                .gate = type, .up = type, .down = type,
                .workers = cfg.device_compute_units, .execution = cfg.cpu_execution};
            const MoELocalExpertStage stage(std::move(params));
            const auto required = stage.getWorkspaceRequirements(token_rows, 0, 0).total_bytes_with_alignment();
            const auto plan = MemoryPlanner::plan(profile, {cfg});
            EXPECT_EQ(plan.devices.front().workspace_bytes(), base + required);
            const auto *bom = plan.physicalPlan().find({-1, DeviceId::cpu()});
            ASSERT_NE(bom, nullptr);
            EXPECT_EQ(bom->bytes(PhysicalMemoryOwner::ExecutionWorkspace), base + required);

            // A follower consumes the same packet without the continuation's
            // attention scratch. In particular, batch=2 must not double an
            // explicitly flattened serial token capacity for a second time.
            auto follower = cfg;
            follower.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant;
            follower.weight_residency = DeviceWeightResidency::selectedRoutedExpertsOnly(8, {1});
            EXPECT_EQ(MemoryPlanner::plan(profile, {follower}).devices.front().workspace_bytes(),
                required + follower.cpu_serial_expert_storage->envelope->admitted()->workspaceBytes());
        }
    }
    auto invalid = overlayDeviceConfig(DeviceId::cpu());
    invalid.first_layer = invalid.last_layer = 0;
    invalid.weight_residency = DeviceWeightResidency::continuationWithSelectedRoutedExperts(8, {1});
    invalid.serial_routed_expert_compact_rows = -1;
    EXPECT_THROW(MemoryPlanner::plan(profile, {invalid}), std::invalid_argument);
    invalid.serial_routed_expert_compact_rows = std::numeric_limits<int>::max();
    EXPECT_THROW(MemoryPlanner::plan(profile, {invalid}), std::runtime_error);
}

/** @test Explicit compact capacity is already batched on every backend. */
TEST(Test__MemoryPlanner, OverlayCompactWorkspaceCapacityIsFlattenedOnEveryBackend)
{
    const auto profile = createMoEOverlayProfile();
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SCOPED_TRACE(device.toString());
        auto cfg = overlayDeviceConfig(device);
        cfg.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant;
        cfg.weight_residency = DeviceWeightResidency::selectedRoutedExpertsOnly(8, {2, 2});
        cfg.routed_expert_compact_buffer_lifetime = RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
        cfg.serial_routed_expert_participant_count = 1;
        cfg.serial_routed_expert_compact_rows = 31;
        declareCPUSerialFixture(cfg, profile);
        const auto one = MemoryPlanner::plan(profile, {cfg}).devices.front().workspace_bytes();
        ASSERT_GT(one, 0u);
        for (int batch : {2, 3, 8})
        {
            cfg.batch_size = batch;
            EXPECT_EQ(MemoryPlanner::plan(profile, {cfg}).devices.front().workspace_bytes(), one);
        }
    }
}

TEST(Test__MemoryPlanner, OverlayContinuationReservesItsOwnCompactRouteBuffers)
{
    const auto profile = createMoEOverlayProfile();

    auto dense_continuation = overlayDeviceConfig(DeviceId::cuda(0));
    const auto dense_plan = MemoryPlanner::plan(
        profile, {dense_continuation});
    ASSERT_EQ(dense_plan.devices.size(), 1u);

    auto overlay_continuation = dense_continuation;
    overlay_continuation.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {2, 2});
    const auto overlay_plan = MemoryPlanner::plan(
        profile, {overlay_continuation});
    ASSERT_EQ(overlay_plan.devices.size(), 1u);

    /*
     * MoELocalExpertStage owns hidden + output vectors and one routing-index
     * and routing-weight scalar per compact route.  Both selected layers keep
     * those buffers alive because a captured graph holds their addresses.
     */
    const size_t route_rows =
        static_cast<size_t>(overlay_continuation.batch_size) *
        static_cast<size_t>(overlay_continuation.activation_seq_len) *
        static_cast<size_t>(profile.expert_used_count);
    const size_t bytes_per_route =
        (2u * static_cast<size_t>(profile.d_model) + 2u) * sizeof(float);
    const size_t expected_compact_bytes =
        route_rows * bytes_per_route * static_cast<size_t>(profile.n_layers);

    EXPECT_EQ(
        overlay_plan.devices.front().activation_bytes(),
        dense_plan.devices.front().activation_bytes() + expected_compact_bytes)
        << "A continuation that owns hot routed experts must account for the "
           "same persistent compact tensors as an expert-only endpoint";
}

/**
 * @brief A serial graph family reserves one compact packet per participant, not per layer.
 *
 * This is the production ExpertOverlay contract used by Qwen3.5 MoE: graph
 * roles and transformer layers are explicitly ordered, while distinct logical
 * participants remain independently runnable and therefore retain separate
 * packet addresses.
 */
TEST(Test__MemoryPlanner,
     OverlaySerialParticipantFamilyReservesOneCompactRouteBufferPerParticipant)
{
    const auto profile = createMoEOverlayProfile();

    auto dense_continuation = overlayDeviceConfig(DeviceId::cuda(0));
    const auto dense_plan = MemoryPlanner::plan(
        profile, {dense_continuation});

    auto serial_overlay = dense_continuation;
    serial_overlay.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {2, 2});
    serial_overlay.routed_expert_compact_buffer_lifetime =
        RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
    serial_overlay.serial_routed_expert_participant_count = 2;

    const auto serial_plan = MemoryPlanner::plan(
        profile, {serial_overlay});
    ASSERT_EQ(serial_plan.devices.size(), 1u);

    const size_t route_rows =
        static_cast<size_t>(serial_overlay.batch_size) *
        static_cast<size_t>(serial_overlay.activation_seq_len) *
        static_cast<size_t>(profile.expert_used_count);
    const size_t bytes_per_route =
        (2u * static_cast<size_t>(profile.d_model) + 2u) * sizeof(float);
    const size_t expected_compact_bytes =
        route_rows * bytes_per_route *
        static_cast<size_t>(serial_overlay.serial_routed_expert_participant_count);

    EXPECT_EQ(
        serial_plan.devices.front().activation_bytes(),
        dense_plan.devices.front().activation_bytes() + expected_compact_bytes)
        << "The serial packet arena may alias layers and graph roles, but never "
           "two independently runnable overlay participants";
}

/**
 * @brief The planner prices every retained segment bucket, not the KV envelope.
 */
TEST(Test__MemoryPlanner,
     OverlaySerialSegmentFamiliesUseBoundedRowsAndExactPowerOfTwoBom)
{
    const auto profile = createMoEOverlayProfile();

    auto dense_continuation = overlayDeviceConfig(DeviceId::cuda(0));
    const auto dense_plan = MemoryPlanner::plan(
        profile, {dense_continuation});

    auto serial_overlay = dense_continuation;
    serial_overlay.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {2, 2});
    serial_overlay.routed_expert_compact_buffer_lifetime =
        RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
    serial_overlay.serial_routed_expert_participant_count = 2;
    /* Five flattened token rows at top-k two produce ten route rows. */
    serial_overlay.serial_routed_expert_compact_rows = 5;

    const auto serial_plan = MemoryPlanner::plan(
        profile, {serial_overlay});
    ASSERT_EQ(serial_plan.devices.size(), 1u);

    const size_t retained_route_rows = 1u + 2u + 4u + 8u + 10u;
    const size_t bytes_per_route =
        (2u * static_cast<size_t>(profile.d_model) + 2u) *
        sizeof(float);
    const size_t expected_compact_bytes =
        retained_route_rows * bytes_per_route *
        static_cast<size_t>(
            serial_overlay.serial_routed_expert_participant_count);

    EXPECT_EQ(
        serial_plan.devices.front().activation_bytes(),
        dense_plan.devices.front().activation_bytes() +
            expected_compact_bytes);
}

TEST(Test__MemoryPlanner,
     OverlaySerialParticipantFamilyRejectsMissingParticipantCount)
{
    const auto profile = createMoEOverlayProfile();
    auto serial_overlay = overlayDeviceConfig(DeviceId::cuda(0));
    serial_overlay.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {2, 2});
    serial_overlay.routed_expert_compact_buffer_lifetime =
        RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;

    EXPECT_THROW(
        (void)MemoryPlanner::plan(profile, {serial_overlay}),
        std::invalid_argument);
}

TEST(Test__MemoryPlanner, CpuExpertParticipantUsesTheSameResidentTicketBucket)
{
    const auto profile = createMoEOverlayProfile();

    auto continuation = overlayDeviceConfig(DeviceId::cuda(0));
    continuation.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            {4, 4});

    auto cpu_participant = overlayDeviceConfig(DeviceId::cpu());
    cpu_participant.execution_role =
        DeviceExecutionMemoryRole::RoutedExpertParticipant;
    cpu_participant.weight_residency =
        DeviceWeightResidency::selectedRoutedExpertsOnly(
            profile.expert_count,
            {2, 2});

    const auto selected =
        MemoryPlanner::planLargestFittingResidentGraphRows(
            profile,
            {continuation, cpu_participant},
            {8, 16, 32});

    ASSERT_TRUE(selected.fits()) << selected.memory_plan.renderTable();
    EXPECT_EQ(selected.resident_graph_rows, 32);
    ASSERT_EQ(selected.memory_plan.devices.size(), 2u);
    EXPECT_EQ(selected.memory_plan.devices[0].activation_seq_len(), 32);
    EXPECT_EQ(selected.memory_plan.devices[1].activation_seq_len(), 32)
        << "CPU cold endpoints consume the same segmented ticket geometry as captured GPU participants";
    ASSERT_EQ(selected.device_inputs.size(), 2u);
    EXPECT_EQ(selected.device_inputs[0].activation_seq_len, 32);
    EXPECT_EQ(selected.device_inputs[1].activation_seq_len, 32);
}

TEST(Test__MemoryPlanner, SelectedInputsPreserveCPUContinuationCapacityBesideGPUBuckets)
{
    const auto profile = createMoEOverlayProfile();
    for (auto gpu : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto accelerator = overlayDeviceConfig(gpu);
        auto cpu = overlayDeviceConfig(DeviceId::cpu());
        cpu.max_seq_len = 512;
        cpu.activation_seq_len = 512;
        const auto selected = MemoryPlanner::planLargestFittingResidentGraphRows(profile, {accelerator, cpu}, {16, 32});
        ASSERT_TRUE(selected.fits()) << selected.memory_plan.renderTable();
        ASSERT_EQ(selected.device_inputs.size(), 2u);
        EXPECT_EQ(selected.device_inputs[0].activation_seq_len, 32);
        EXPECT_EQ(selected.device_inputs[1].activation_seq_len, 512);
        EXPECT_EQ(selected.memory_plan.devices[1].activation_seq_len(), 512);
        const auto replay = MemoryPlanner::plan(profile, selected.device_inputs);
        EXPECT_FALSE(replay.admit().plan().requiredFootprintMismatch(selected.memory_plan.admit().plan()));
    }
}

/**
 * @brief Diagnostic graph storage is an exact accelerator BOM owner.
 *
 * This regression prevents parity capture from returning to the historical
 * pattern of allocating an unpriced VRAM arena after routed-expert admission.
 * CPU participants do not own device graph snapshots, while every GPU plan
 * must carry the caller-declared capacity byte-for-byte.
 */
TEST(Test__MemoryPlanner,
     GraphSnapshotCapacityIsChargedExactlyOnceToEachAccelerator)
{
    const auto profile = createTestProfile();

    DevicePlanConfig baseline;
    baseline.device = DeviceId::rocm(0);
    baseline.device_compute_units = 60;
    baseline.device_total_bytes = 32ULL * 1024 * 1024 * 1024;
    baseline.device_free_bytes = 31ULL * 1024 * 1024 * 1024;
    baseline.batch_size = 1;
    baseline.max_seq_len = 256;
    baseline.kv_precision = "fp16";

    constexpr std::size_t kSnapshotBytes = 417'234'944u;
    DevicePlanConfig diagnostic = baseline;
    diagnostic.graph_snapshot_memory.per_accelerator_bytes =
        kSnapshotBytes;

    const auto baseline_plan = MemoryPlanner::plan(profile, {baseline});
    const auto diagnostic_plan =
        MemoryPlanner::plan(profile, {diagnostic});
    ASSERT_EQ(baseline_plan.devices.size(), 1u);
    ASSERT_EQ(diagnostic_plan.devices.size(), 1u);
    EXPECT_EQ(
        diagnostic_plan.devices.front().graph_snapshot_bytes(),
        kSnapshotBytes);
    EXPECT_EQ(
        diagnostic_plan.devices.front().total_bytes(),
        baseline_plan.devices.front().total_bytes() + kSnapshotBytes);

    const auto *gpu_bom = diagnostic_plan.physicalPlan().find(
        {-1, DeviceId::rocm(0)});
    ASSERT_NE(gpu_bom, nullptr);
    EXPECT_EQ(
        gpu_bom->bytes(PhysicalMemoryOwner::GraphSnapshotArena),
        kSnapshotBytes);

    DevicePlanConfig cpu = diagnostic;
    cpu.device = DeviceId::cpu();
    cpu.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    const auto cpu_plan = MemoryPlanner::plan(profile, {cpu});
    ASSERT_EQ(cpu_plan.devices.size(), 1u);
    EXPECT_EQ(cpu_plan.devices.front().graph_snapshot_bytes(), 0u);
    const auto *cpu_bom = cpu_plan.physicalPlan().find(
        {-1, DeviceId::cpu()});
    ASSERT_NE(cpu_bom, nullptr);
    EXPECT_EQ(
        cpu_bom->bytes(PhysicalMemoryOwner::GraphSnapshotArena),
        0u);
}

/** @brief Full-cache diagnostics use context and TP geometry, not prompt rows. */
TEST(Test__MemoryPlanner, EffectiveKVSnapshotCapacityCoversEveryBackendFormatAndTPDegree)
{
    auto profile = createTestProfile();
    profile.n_kv_heads = 8;
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const std::string codec : {"fp16", "bf16", "fp32", "q8_1", "tq4", "tq"})
    for (int shards = 1; shards <= 8; ++shards)
    {
        SCOPED_TRACE(device.toString() + " " + codec + " TP=" + std::to_string(shards));
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.device_total_bytes = cfg.device_free_bytes = 32ull << 30;
        cfg.device_compute_units = 60;
        cfg.batch_size = 2;
        cfg.max_seq_len = 4096;
        cfg.activation_seq_len = 9;
        cfg.kv_precision = codec;
        cfg.total_shards = shards;
        // Uneven/replicated KV ownership is already resolved by the topology.
        cfg.local_kv_heads = (profile.n_kv_heads + shards - 1) / shards;
        cfg.graph_snapshot_memory = {
            .per_accelerator_bytes = 128,
            .effective_kv = GraphSnapshotMemoryCapacity::EffectiveKV{
                .layer = 7, .retained_arena_count = 3}};
        const auto plan = MemoryPlanner::plan(profile, {cfg});
        const std::size_t per_bank = 2u * 4096u * cfg.local_kv_heads * 64u * 2u * sizeof(float);
        const std::size_t metadata = 2u * 2u * sizeof(std::int32_t);
        EXPECT_EQ(plan.devices.front().graph_snapshot_bytes(), 128u + 3u * (per_bank + metadata));
        cfg.graph_snapshot_memory.effective_kv->layer.reset();
        const auto all = MemoryPlanner::plan(profile, {cfg});
        EXPECT_EQ(all.devices.front().graph_snapshot_bytes(), 128u + 3u * 24u * (per_bank + metadata));
    }
}

/** @brief Absent diagnostics, non-KV roles and recurrent layers add no banks. */
TEST(Test__MemoryPlanner, EffectiveKVSnapshotSelectionRespectsLayerOwnership)
{
    auto profile = createTestProfile();
    // Tensor metadata takes precedence over the model's periodic layer hint.
    profile.tensors.erase(std::remove_if(profile.tensors.begin(), profile.tensors.end(),
        [](const auto &tensor) { return tensor.layer_index == 5; }), profile.tensors.end());
    profile.full_attention_interval = 4;
    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_total_bytes = cfg.device_free_bytes = 32ull << 30;
    cfg.device_compute_units = 60;
    cfg.max_seq_len = 16;
    cfg.graph_snapshot_memory = {
        .per_accelerator_bytes = 128,
        .effective_kv = GraphSnapshotMemoryCapacity::EffectiveKV{
            .layer = 5, .retained_arena_count = 1}};
    // Query only the FA-containing PP interval, so an unrelated GDN layer's
    // recurrent dimensions are not part of this minimal fixture's admission.
    cfg.first_layer = 6;
    cfg.last_layer = 8;
    EXPECT_EQ(MemoryPlanner::plan(profile, {cfg}).devices.front().graph_snapshot_bytes(), 128u);
    cfg.graph_snapshot_memory.effective_kv->layer = 7;
    EXPECT_GT(MemoryPlanner::plan(profile, {cfg}).devices.front().graph_snapshot_bytes(), 128u);
    cfg.graph_snapshot_memory.effective_kv.reset();
    EXPECT_EQ(MemoryPlanner::plan(profile, {cfg}).devices.front().graph_snapshot_bytes(), 128u);
    cfg.device = DeviceId::cpu();
    cfg.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    EXPECT_EQ(MemoryPlanner::plan(profile, {cfg}).devices.front().graph_snapshot_bytes(), 0u);
}

/** @brief Missing inventories and wrapped byte counts cannot pass admission. */
TEST(Test__MemoryPlanner, EffectiveKVSnapshotRejectsIncompleteOrOverflowingInventory)
{
    const auto profile = createTestProfile();
    DevicePlanConfig cfg;
    cfg.device = DeviceId::rocm(0);
    cfg.max_seq_len = 4096;
    cfg.graph_snapshot_memory = {
        .per_accelerator_bytes = 128,
        .effective_kv = GraphSnapshotMemoryCapacity::EffectiveKV{}};
    EXPECT_THROW(MemoryPlanner::plan(profile, {cfg}), std::invalid_argument);
    cfg.graph_snapshot_memory.effective_kv->retained_arena_count = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(MemoryPlanner::plan(profile, {cfg}), std::runtime_error);
    cfg.graph_snapshot_memory.effective_kv->retained_arena_count = 1;
    cfg.graph_snapshot_memory.effective_kv->layer = profile.n_layers;
    EXPECT_THROW(MemoryPlanner::plan(profile, {cfg}), std::invalid_argument);
}

/**
 * @test Prefix admission retains the complete native-gather publication.
 *
 * The physical head is half-width on both backends. Runtime archives the
 * assembled FP32 vocabulary row, so its fixed prefix slot must use full width.
 * A distributed shard without this local publication keeps its local row.
 * Exact three-slot budgets expose the old rounding error before GPU startup.
 */
TEST(Test__MemoryPlanner, NativeVocabularyPrefixAdmissionUsesPublishedRows)
{
    auto profile = createQwen36HybridMTPStateProfile();
    profile.n_heads = 24;
    profile.n_kv_heads = 4;
    profile.head_dim = 256;
    profile.d_ff = 17408;
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto assignments = TensorParallelConfig::proportionalSplit(
            {device, DeviceId(device.type, 1)}, {1.0f, 1.0f},
            profile.n_heads, profile.n_kv_heads, profile.d_ff, profile.vocab_size);
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.device_total_bytes = cfg.device_free_bytes = 64ULL << 30;
        cfg.device_compute_units = 60;
        cfg.first_layer = 0;
        cfg.last_layer = 63;
        cfg.total_shards = 2;
        cfg.bindTensorParallelAssignment(assignments.forRank(0));
        cfg.batch_size = 1;
        cfg.max_seq_len = 8192;
        cfg.activation_seq_len = 512;
        cfg.kv_precision = "fp16";
        cfg.mtp_enabled = true;
        cfg.mtp_target_query_rows = 16;
        cfg.prefix_cache = PrefixCacheRuntimeConfig{};
        cfg.mtp_terminal_logits_layout =
            MTPTerminalLogitsLayout::VocabularyShardPerParticipant;
        cfg.associated_host_memory = PhysicalMemoryResource{
            .world_rank = -1, .device = DeviceId::cpu(),
            .total_bytes = 64ULL << 30, .admission_available_bytes = 64ULL << 30};
        const auto main_block = KVCacheMemoryEstimator::estimateGPULogicalBlock(
            KVCacheFamily::Hybrid, cfg.prefix_cache.block_size,
            2, profile.head_dim, cfg.kv_precision, device);
        const auto shifted_block = KVCacheMemoryEstimator::estimateGPULogicalBlock(
            KVCacheFamily::AttentionOnly, cfg.prefix_cache.block_size,
            2, profile.head_dim, cfg.kv_precision, device);
        const auto recurrent = HybridGDNStateGeometry::resolve(profile.n_heads,
            0, 12, profile.gdn_group_count, profile.gdn_time_step_rank,
            profile.gdn_state_size, profile.gdn_inner_size,
            profile.gdn_conv_kernel_size).deviceSerializedPayloadBytes(48);
        const size_t non_logit_payload = 16 * main_block.totalBytes() +
            shifted_block.totalBytes() + recurrent + profile.d_model * sizeof(float);
        for (const auto publication : {
                 MTPTerminalLogitsCollective::None,
                 MTPTerminalLogitsCollective::NativeLocalVocabularyAllGather})
        {
            SCOPED_TRACE(device.toString() + ":" + std::to_string(static_cast<int>(publication)));
            cfg.local_tp_backend = publication ==
                    MTPTerminalLogitsCollective::NativeLocalVocabularyAllGather
                ? std::optional(device.is_cuda() ? CollectiveBackendType::NCCL
                                                : CollectiveBackendType::RCCL)
                : std::nullopt;
            ASSERT_EQ(cfg.terminalLogitsCollective(true), publication);
            const size_t published_columns = publication ==
                    MTPTerminalLogitsCollective::NativeLocalVocabularyAllGather
                ? profile.vocab_size : assignments.forRank(0).vocab_count;
            const size_t archive_slot = non_logit_payload + published_columns * sizeof(float);
            cfg.prefix_cache.device_budget_bytes = 3 * archive_slot;
            const auto plan = MemoryPlanner::plan(profile, {cfg});
            const auto entry = std::find_if(plan.devices.begin(), plan.devices.end(),
                [&](const auto &candidate) { return candidate.device() == device; });
            ASSERT_NE(entry, plan.devices.end());
            EXPECT_EQ(entry->prefix_cache_device_hot_bytes(), 3 * archive_slot)
                << "The archived gathered surface must match the physical-memory certificate.";
        }
    }
}

/**
 * @test The real schema and configured codec reach the estimator before admission.
 *
 * Existing dense/MoE fixtures exercise the public planner without models, runtime
 * discovery or allocations. The independent geometry names native FP16 causal
 * operands explicitly; a default FP32/General record must produce a different
 * M4096 workspace, preventing an omitted producer field from passing unnoticed.
 * Sharded KV and full replicated KV also require different native head mappings.
 */
TEST(Test__MemoryPlanner, FP16PrefillSchemaAndCodecReachWorkspaceAdmission)
{
    /** Keep fixture capture settings automatic and restore every authored override. */
    struct AutomaticPolicy
    {
        AttentionConfig original = mutableDebugEnv().attention;
        /** Select ordinary production geometry for this device-free admission proof. */
        AutomaticPolicy()
        {
            auto &policy = mutableDebugEnv().attention;
            policy.cuda_fa2_q_warp_groups = policy.cuda_fa2_hd256_pv_warps = policy.cuda_fa2_tile_kv = 0;
        }
        /** Restore the immutable process snapshot after the fixture. */
        ~AutomaticPolicy() { mutableDebugEnv().attention = original; }
    } automatic;
    /** A schema fixture whose declared visibility must exclude the native full-context body. */
    class WindowAttentionSchema final : public Qwen35SchemaFactory
    {
    public:
        /** @return The same real schema with every attention reader explicitly windowed. */
        GraphSchema createSchema() const override
        {
            auto schema = Qwen35SchemaFactory::createSchema();
            const auto window = [](LayerTemplate &layer) {
                for (auto &stage : layer.attention_stages)
                    if (stage.type == StageType::AttentionCompute) stage.window_size = 96;
            };
            window(schema.layer_template);
            for (auto &[name, layer] : schema.named_templates) window(layer);
            return schema;
        }
        /** @return Fixture-local registry identity, never a production model claim. */
        std::string architectureName() const override { return "unit_attention_admission_window"; }
    };
    SchemaFactoryRegistry::registerFactory("unit_attention_admission_window",
        [] { return std::make_unique<WindowAttentionSchema>(); });

    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const bool moe : {false, true})
    for (const bool replicated : {false, true})
    for (const bool windowed_schema : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << device.toString() << " MoE=" << moe
            << " replicated=" << replicated << " windowed_schema=" << windowed_schema);
        auto profile = moe ? createMoEOverlayProfile() : createTestProfile();
        profile.architecture = windowed_schema ? "unit_attention_admission_window" : (moe ? "qwen35moe" : "qwen35");
        profile.n_layers = 2;
        profile.n_heads = moe ? 16 : 24;
        profile.n_kv_heads = moe ? 2 : 4;
        profile.head_dim = 256;
        profile.max_seq_len = 8193;
        // The fixture prices metadata declarations, not prepared projection weights.
        profile.tensors.clear();
        profile.total_native_bytes = 0;
        auto config = overlayDeviceConfig(device);
        config.owns_embedding = false;
        config.activation_seq_len = 4096;
        config.max_seq_len = profile.max_seq_len;
        config.total_shards = 2;
        config.shard_index = 1;
        config.local_kv_heads = replicated ? profile.n_kv_heads : profile.n_kv_heads / 2;
        const WorkspaceMemoryGeometry native{
            .device = device, .device_compute_units = config.device_compute_units,
            .batch_size = 1, .resident_graph_rows = 4096, .max_context_rows = profile.max_seq_len,
            .shard_index = 1, .local_d_ff = profile.d_ff / 2,
            .local_query_head_start = profile.n_heads / 2, .local_query_heads = profile.n_heads / 2,
            .local_kv_head_start = config.local_kv_heads, .local_kv_heads = config.local_kv_heads,
            .first_layer = 0, .last_layer = 1, .total_shards = 2,
            .attention_kv_format = TensorType::FP16,
            .attention_visibility = windowed_schema ? attention::AttentionVisibilityPolicy::General
                : attention::AttentionVisibilityPolicy::CausalFullContext};
        const auto planned = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(planned.devices.size(), 1u);
        const size_t expected = WorkspaceMemoryEstimator::estimate(profile, native);
        EXPECT_EQ(planned.devices.front().workspace_bytes(), expected);
        auto missing_format = native;
        missing_format.attention_kv_format = TensorType::FP32;
        auto missing_visibility = native;
        missing_visibility.attention_visibility = attention::AttentionVisibilityPolicy::General;
        if (!windowed_schema)
        {
            EXPECT_NE(expected, WorkspaceMemoryEstimator::estimate(profile, missing_format));
            EXPECT_NE(expected, WorkspaceMemoryEstimator::estimate(profile, missing_visibility));
        }
        else
        {
            auto invented_causal = native;
            invented_causal.attention_visibility = attention::AttentionVisibilityPolicy::CausalFullContext;
            EXPECT_NE(expected, WorkspaceMemoryEstimator::estimate(profile, invented_causal));
        }
    }
}

/** @test FP16 codec spelling is strict before it can select native prefill admission. */
TEST(Test__MemoryPlanner, AttentionOperandCodecClassificationIsStrict)
{
    for (const auto *name : {"FP16", "fp16", "fP16"})
        EXPECT_EQ(KVCacheMemoryEstimator::attentionExecutionFormat(name), TensorType::FP16) << name;
    EXPECT_THROW(KVCacheMemoryEstimator::attentionExecutionFormat("fp16-extra"), std::invalid_argument);
    EXPECT_THROW(KVCacheMemoryEstimator::attentionExecutionFormat(""), std::invalid_argument);
}

/** @test Hosted storage shares one exact host owner while every actual GPU binds the sealed tier. */
TEST(Test__MemoryPlanner, NgramStorageHasOneExactHostOwnerAcrossGpuParticipants)
{
    // The shared config fixture intentionally has no learned FA/GDN matrices.
    // Full activation admission is source-inventory driven, so supply this
    // synthetic model's complete projection geometry before comparing the
    // frozen base plus concrete lookup delta with the full resolved plan.
    auto directory=test::qwen4MetadataFixture(false);
    const auto projection=[&](const char *name,std::uint64_t rows,std::uint64_t columns) {
        directory.tensors.push_back({.name=name,.dimensions={rows,columns},.type=GGUFTensorType::F16});
    };
    projection("blk.3.attn_q.weight",6144,2560);
    projection("blk.3.attn_gate.weight",6144,2560);
    projection("blk.0.attn_qkv.weight",10240,2560);
    projection("blk.0.ssm_out.weight",2560,6144);
    projection("blk.0.ssm_alpha.weight",48,2560);
    projection("blk.0.ssm_beta.weight",48,2560);
    const auto profile = ModelMemoryProfile::fromGGUF(directory);
    DevicePlanConfig first;
    first.world_rank = 0; first.device = DeviceId::cuda(0); first.device_compute_units = 82;
    first.device_total_bytes = 64ull << 30; first.device_free_bytes = 60ull << 30;
    first.max_seq_len = 64;
    first.associated_host_memory = PhysicalMemoryResource{
        .world_rank = 0, .device = DeviceId{DeviceType::CPU,1}, .total_bytes = 128ull << 30,
        .admission_available_bytes = 120ull << 30};
    auto second = first; second.device = DeviceId::rocm(0);
    const std::vector<std::vector<DevicePlanConfig>> groups{{first,second}};
    const auto baseline = MemoryPlanner::plan(profile, groups.front());
    const auto input = MemoryPlanner::immutableRowTablePlacementInput(profile,groups,*first.associated_host_memory);
    const auto modes = MemoryPlanner::immutableRowTableModeContributions(profile,groups,{});
    for (const auto requested : {ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped})
    {
        const auto selected=ImmutableRowTablePlacementPlanner::select({requested,65536,3},input,baseline.physicalPlan(),modes);
        first.ngram_table_placement=selected.placement;second.ngram_table_placement=selected.placement;
        const auto plan=MemoryPlanner::plan(profile,{first,second});
        const auto *host=plan.physicalPlan().find({0,DeviceId{DeviceType::CPU,1}});
        ASSERT_NE(host,nullptr);
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::ImmutableTableStorage),
            ImmutableRowTable::physicalBytesForNativeExtent(profile.flash_next->ple_table_native_bytes,
                selected.placement->storagePolicy()));
        const auto selected_mismatch=selected.physical_plan.requiredFootprintMismatch(plan.physicalPlan());
        const auto resolved_mismatch=plan.physicalPlan().requiredFootprintMismatch(selected.physical_plan);
        EXPECT_FALSE(selected_mismatch.has_value())<<selected_mismatch.value_or("");
        EXPECT_FALSE(resolved_mismatch.has_value())<<resolved_mismatch.value_or("");
        for (const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
        {
            const auto *actual=plan.physicalPlan().find({0,device});
            const auto *prior=baseline.physicalPlan().find({0,device});
            ASSERT_NE(actual,nullptr);ASSERT_NE(prior,nullptr);
            EXPECT_EQ(actual->bytes(PhysicalMemoryOwner::ImmutableTableStorage),0u);
            EXPECT_EQ(actual->bytes(PhysicalMemoryOwner::PrimaryModelWeights),prior->bytes(PhysicalMemoryOwner::PrimaryModelWeights));
        }
    }
}


/** @test Final table admission derives one frozen host and rejects missing or conflicting owners. */
TEST(Test__MemoryPlanner, NgramFinalHostComesFromActualConsumerInputs)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto &profile = source.metadata().memoryProfile();
    const ImmutableRowTablePolicy policy{ImmutableRowStoragePreference::RAM, 65536, 3};
    DevicePlanConfig first;
    first.ngram_execution_policy = policy;
    first.world_rank = 2;
    first.device = DeviceId::cuda(0);
    first.device_compute_units = 64;
    first.max_seq_len = 64;
    first.device_total_bytes = first.device_free_bytes = 64ull << 30;
    first.associated_host_memory = PhysicalMemoryResource{
        2, DeviceId::cpu(), 128ull << 30, 120ull << 30};
    auto second = first;
    second.device = DeviceId::rocm(1);
    auto expert = first;
    expert.device = DeviceId{DeviceType::CPU, 7};
    expert.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant;
    expert.associated_host_memory.reset();
    std::vector<std::vector<DevicePlanConfig>> groups{{first, second, expert}};
    const auto request = MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy());
    ASSERT_TRUE(request);
    EXPECT_EQ(request->input.host().world_rank, 2);
    EXPECT_EQ(request->input.host().device, DeviceId::cpu());
    EXPECT_EQ(request->input.consumers().size(), 2u);
    EXPECT_TRUE(request->input.hostPlacement().usesTaskPolicy());
    const auto bound = MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},
        ImmutableRowTableHostPlacement::fromObservedNUMANode(3));
    ASSERT_TRUE(bound);
    EXPECT_EQ(bound->input.host().device, DeviceId::cpu());
    EXPECT_EQ(bound->input.hostPlacement(), ImmutableRowTableHostPlacement::onNode(3));

    groups[0][1].associated_host_memory.reset();
    EXPECT_THROW((void)MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy()),
        std::invalid_argument);
    groups[0][1] = second;
    groups[0][1].associated_host_memory->device = DeviceId{DeviceType::CPU, 3};
    EXPECT_THROW((void)MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy()),
        std::invalid_argument);
    groups[0][1] = second;
    groups[0][1].associated_host_memory->world_rank = 9;
    EXPECT_THROW((void)MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy()),
        std::invalid_argument);
    groups[0][1] = second;
    groups[0][1].associated_host_memory->total_bytes += 4096;
    EXPECT_THROW((void)MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy()),
        std::invalid_argument);

    // A CPU continuation already is its host owner; an expert-only CPU does
    // not select storage for a PLE consumer and cannot supply a missing host.
    first.device = DeviceId::cpu();
    first.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    first.device_compute_units = 8;
    first.associated_host_memory.reset();
    groups = {{first, expert}};
    const auto cpu = MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy());
    ASSERT_TRUE(cpu);
    EXPECT_EQ(cpu->input.host().device, first.device);
    EXPECT_FALSE(cpu->input.permitsVRAM());
    groups = {{expert}};
    EXPECT_FALSE(MemoryPlanner::immutableRowTableAdmissionRequest(profile, policy, groups, {},ImmutableRowTableHostPlacement::taskPolicy()));
}

/** @test Each consumer authenticates its own original upload tuple, including unequal mixed GPUs. */
TEST(Test__MemoryPlanner, NgramVRAMEligibilityUsesEachOriginalUploadGeometry)
{
    const auto profile=ModelMemoryProfile::fromGGUF(test::qwen4MetadataFixture(false));
    DevicePlanConfig first;
    first.world_rank=0;first.device=DeviceId::cuda(0);first.max_seq_len=64;first.device_total_bytes=64ull<<30;
    first.device_free_bytes=60ull<<30;first.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),128ull<<30,120ull<<30};
    const auto row=static_cast<std::size_t>(profile.flash_next->ple_table_row_bytes);
    const auto wide=resolveGPUWeightLoadMemoryGeometry(row+4096,{.staging_stream_count=1,.staging_budget_bytes=4096});
    first.ordinary_weight_load_geometry=wide;
    first.weight_load_staging={wide.staging_bytes,wide.host_staging_bytes};
    auto second=first;second.device=DeviceId::rocm(1);
    const auto other=resolveGPUWeightLoadMemoryGeometry(row+8192,{.staging_stream_count=2,.staging_budget_bytes=8192});
    second.ordinary_weight_load_geometry=other;
    second.weight_load_staging={other.staging_bytes,other.host_staging_bytes};
    auto expert_only=first;
    expert_only.device=DeviceId{DeviceType::CPU,2};
    expert_only.execution_role=DeviceExecutionMemoryRole::RoutedExpertParticipant;
    expert_only.weight_load_staging={};expert_only.ordinary_weight_load_geometry.reset();
    std::vector<std::vector<DevicePlanConfig>> groups{{first,second,expert_only}};
    const auto placement_input=MemoryPlanner::immutableRowTablePlacementInput(profile,groups,*first.associated_host_memory);
    EXPECT_TRUE(placement_input.permitsVRAM());
    EXPECT_EQ(placement_input.consumers().size(),2u);
    const auto bindings=MemoryPlanner::immutableRowTableUploadBindings(profile,groups);
    ASSERT_EQ(bindings.size(),2u);
    EXPECT_TRUE(MemoryPlanner::immutableRowTableModeContributions(profile,groups,bindings).vram);
    EXPECT_FALSE(MemoryPlanner::immutableRowTableModeContributions(profile,groups,{}).vram);
    ASSERT_GT(row,1u);
    const auto narrow=resolveGPUWeightLoadMemoryGeometry(row-1,{.staging_stream_count=1,.staging_budget_bytes=0});
    groups.front()[1].ordinary_weight_load_geometry=narrow;
    groups.front()[1].weight_load_staging={narrow.staging_bytes,narrow.host_staging_bytes};
    EXPECT_FALSE(MemoryPlanner::immutableRowTableModeContributions(profile,groups,
        MemoryPlanner::immutableRowTableUploadBindings(profile,groups)).vram);
    auto foreign=bindings;foreign.front().consumer.device=DeviceId::cuda(7);
    EXPECT_THROW(MemoryPlanner::immutableRowTableModeContributions(profile,groups,foreign),std::invalid_argument);
}

/** @test Native and hosted PP participants price one sampler-free parent per admitted query program. */
TEST(Test__MemoryPlanner, NativePipelineForwardOnlyParentHasNamedPhysicalBOM)
{
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const auto domain : {DeviceGenerationExecutionDomain::NativeDomain, DeviceGenerationExecutionDomain::HostedBoundary})
    for (const size_t programs : {1u, 2u})
    {
        auto profile = createTestProfile();
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.device_compute_units = 82;
        cfg.device_total_bytes = cfg.device_free_bytes = 64ULL * GiB;
        cfg.max_seq_len = 4096;
        cfg.activation_seq_len = 64;
        cfg.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory(
            {64},
            {},
            resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(declaredDeviceGenerationExecutionPolicy(
                device.type, domain))),
            MTPGenerationParentRole::Controller,
            domain, programs);
        const auto ordinary = MemoryPlanner::plan(profile, {cfg});
        cfg.pipeline_forward_only_graph_owner = PipelineForwardOnlyGraphOwner::Participant;
        const auto pipeline = MemoryPlanner::plan(profile, {cfg});
        ASSERT_EQ(ordinary.devices.size(), 1u);
        ASSERT_EQ(pipeline.devices.size(), 1u);
        EXPECT_EQ(pipeline.devices.front().captured_graph_bytes() - ordinary.devices.front().captured_graph_bytes(),
            estimateCapturedGraphExecutableBytes(device, CapturedGraphExecutableInventory{
                .model_graph_identity_count = 0u,
                .model_graph_topology_variant_count = 0u,
                .auxiliary_executable_count = OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(programs, PipelineForwardOnlyGraphOwner::Participant)}));
    }
}

/** @test Native and captured-domain PP declarations own parents independently of transport. */
TEST(Test__MemoryPlanner, ForwardOnlyBOMFollowsOriginalNativePipelineProjection)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int topology : {0, 1, 2}) // singleton homogeneous, nested TP, mixed vendor
    {
        RankExecutionPlan parent;
        parent.rank = 0;
        const auto first = GlobalDeviceAddress::fromLocalDeviceId(DeviceId(type, 0));
        const auto second_type = topology == 2 ? (type == DeviceType::CUDA ? DeviceType::ROCm : DeviceType::CUDA) : type;
        const auto second = GlobalDeviceAddress::fromLocalDeviceId(DeviceId(second_type, topology == 2 ? 0 : 1));
        parent.local_pp_devices = {first, second};
        parent.local_pp_layer_boundaries = {0, 2, 4};
        if (topology == 1)
            parent.local_pp_stage_tp_info = {{.devices = {first,
                GlobalDeviceAddress::fromLocalDeviceId(DeviceId(type, 2))}},
                {.devices = {second, GlobalDeviceAddress::fromLocalDeviceId(DeviceId(type, 3))}}};
        DevicePlanConfig cfg;
        cfg.world_rank = 0;
        cfg.device = first.toLocalDeviceId();
        cfg.first_layer = 0;
        cfg.last_layer = 1;
        cfg.execution_role = DeviceExecutionMemoryRole::ContinuationGraph;
        cfg.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
            .device = DeviceId::cpu(), .total_bytes = 4096, .admission_available_bytes = 4096};
        PipelineStageTransferMemory::forStage(parent, 0).bind(cfg);
        EXPECT_EQ(cfg.pipeline_forward_only_graph_owner, PipelineForwardOnlyGraphOwner::Participant);
        EXPECT_THROW(PipelineStageTransferMemory::forStage(parent, 0).bind(cfg), std::invalid_argument);
        EXPECT_EQ(cfg.local_pipeline_backend.has_value(), topology == 0);
        if (topology == 0)
        {
            EXPECT_EQ(*cfg.local_pipeline_backend, type == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        }
    }
}

/** @test Frozen domains assign exactly one parent/program to every actual GPU, including TP siblings. */
TEST(Test__MemoryPlanner, CapturedDomainForwardOnlyOwnersIncludeEveryPhysicalParticipant)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
    for (int members = 1; members <= 8; ++members)
    {
        RankExecutionPlan parent;
        parent.rank = 0;
        const auto other = type == DeviceType::CUDA ? DeviceType::ROCm : DeviceType::CUDA;
        const auto first = GlobalDeviceAddress::fromLocalDeviceId(DeviceId(type, 0));
        const auto tail = GlobalDeviceAddress::fromLocalDeviceId(DeviceId(other, 0));
        parent.local_pp_devices = {first, tail};
        parent.local_pp_layer_boundaries = {0, 2, 4};
        parent.local_pp_stage_tp_info.resize(2);
        for (int member = 0; member < members; ++member)
        {
            parent.local_pp_stage_tp_info[0].devices.push_back(
                GlobalDeviceAddress::fromLocalDeviceId(DeviceId(type, member)));
            parent.local_pp_stage_tp_info[1].devices.push_back(
                GlobalDeviceAddress::fromLocalDeviceId(DeviceId(other, member)));
        }
        for (size_t stage = 0; stage != 2; ++stage)
        {
            const auto transfer = PipelineStageTransferMemory::forStage(parent, stage);
            for (const auto &member : parent.local_pp_stage_tp_info[stage].devices)
            {
                DevicePlanConfig cfg;
                cfg.world_rank = 0;
                cfg.device = member.toLocalDeviceId();
                cfg.first_layer = static_cast<int>(stage) * 2;
                cfg.last_layer = cfg.first_layer + 1;
                cfg.execution_role = DeviceExecutionMemoryRole::ContinuationGraph;
                cfg.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                    .device = DeviceId::cpu(), .total_bytes = 4096, .admission_available_bytes = 4096};
                transfer.bind(cfg);
                EXPECT_FALSE(cfg.local_pipeline_backend);
                EXPECT_EQ(cfg.pipeline_forward_only_graph_owner, PipelineForwardOnlyGraphOwner::Participant);
                for (const size_t programs : {1u, 2u})
                    EXPECT_EQ(OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(
                        programs, cfg.pipeline_forward_only_graph_owner), programs);
                EXPECT_THROW(transfer.bind(cfg), std::invalid_argument);
            }
        }
    }
    EXPECT_EQ(OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(
        2u, PipelineForwardOnlyGraphOwner::None), 0u);
    EXPECT_THROW(OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(
        1u, static_cast<PipelineForwardOnlyGraphOwner>(255)), std::invalid_argument);
    DevicePlanConfig invalid;
    invalid.device = DeviceId::cpu();
    invalid.pipeline_forward_only_graph_owner = PipelineForwardOnlyGraphOwner::Participant;
    EXPECT_THROW(MemoryPlanner::plan(createTestProfile(), {invalid}), std::invalid_argument);
    invalid.pipeline_forward_only_graph_owner = static_cast<PipelineForwardOnlyGraphOwner>(255);
    EXPECT_THROW(MemoryPlanner::plan(createTestProfile(), {invalid}), std::invalid_argument);
}

/**
 * @test Original source geometry and actual stage declarations own handoff bytes.
 *
 * Sparse parser-native source metadata is never loaded as model weights. The
 * existing stage query supplies the descriptors; only collective membership is
 * mocked. Global layer subranges, scalar families, non-native contexts and
 * already-full primary state preserve their actual stage-selection boundaries.
 */
TEST(Test__MemoryPlanner, ReplicatedDecodeIncludesActualGDNHandoffWorkspace)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto &profile = source.metadata().memoryProfile();
    const GDNLinkedLiveStateGeometry geometry{
        .global_key_heads = profile.gdn_group_count,
        .global_value_heads = profile.gdn_time_step_rank,
        .key_width = profile.gdn_state_size,
        .value_width = profile.gdn_state_size,
        .conv_history_length = profile.gdn_conv_kernel_size - 1};
    for (const DeviceId first : {DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int degree : {2, 4, 8})
    {
        const auto backend = first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        std::vector<GlobalDeviceAddress> addresses;
        std::vector<DeviceId> devices;
        for (int rank = 0; rank < degree; ++rank)
        {
            devices.emplace_back(first.type, rank);
            addresses.push_back(first.is_cuda() ? GlobalDeviceAddress::cuda(rank) : GlobalDeviceAddress::rocm(rank));
        }
        test::MockLocalTPContext collective;
        collective.setDevices(addresses);
        collective.setBackend(backend);
        const auto assignment = TensorParallelConfig::proportionalSplit(devices,
            std::vector<float>(static_cast<size_t>(degree), 1.f), profile.n_heads,
            profile.n_kv_heads, profile.d_ff, profile.vocab_size);
        for (const auto [first_layer, last_layer] : {std::pair{0, 47}, std::pair{12, 27}})
        {
            WorkspaceRequirements actual_stages;
            std::set<int> layers;
            for (const auto &tensor : profile.tensors)
                if (tensor.name.ends_with(".ssm_out.weight") &&
                    tensor.layer_index >= first_layer && tensor.layer_index <= last_layer)
                    layers.insert(tensor.layer_index);
            ASSERT_EQ(layers.size(), first_layer == 0 ? 36u : 12u);
            for (const int layer : layers)
            {
                GDNLiveStateAllGatherStage::Params params;
                params.device_id = first;
                params.tp_ctx = &collective;
                params.layer_idx = layer;
                params.tp_device_idx = 0;
                params.geometry = geometry;
                GDNLiveStateAllGatherStage stage(params);
                const auto one = stage.getWorkspaceRequirements(9);
                ASSERT_EQ(one.buffers.size(), 6u);
                for (const auto &descriptor : one.buffers)
                {
                    EXPECT_TRUE(descriptor.name.ends_with("_layer" + std::to_string(layer)));
                    EXPECT_EQ(descriptor.alignment, 256u);
                    EXPECT_TRUE(descriptor.required);
                    EXPECT_EQ(descriptor.regime, WorkspaceExecutionRegime::Any);
                    EXPECT_EQ(descriptor.content_lifetime, WorkspaceContentLifetime::ParticipantExecution);
                }
                actual_stages.merge(one);
            }
            EXPECT_EQ(actual_stages.buffers.size(), layers.size() * 6u);
            const auto handoff_bytes = actual_stages.total_bytes_with_alignment();
            if (degree == 4 && first_layer == 0)
            {
                EXPECT_EQ(handoff_bytes, 264757248u);
            }
            for (const auto [batch, rows] : {std::pair{1, 1}, std::pair{1, 9},
                                           std::pair{1, 512}, std::pair{2, 1}})
            {
                auto cfg = overlayDeviceConfig(first);
                cfg.world_rank = -1;
                cfg.device_total_bytes = cfg.device_free_bytes = 1ull << 40;
                cfg.total_shards = degree;
                cfg.local_tp_backend = backend;
                cfg.bindTensorParallelAssignment(assignment.forRank(0));
                cfg.first_layer = first_layer;
                cfg.last_layer = last_layer;
                cfg.max_seq_len = 4096;
                cfg.activation_seq_len = rows;
                cfg.batch_size = batch;
                cfg.weight_residency = DeviceWeightResidency::continuationWithSelectedRoutedExperts(
                    profile.expert_count, std::vector<int>(static_cast<size_t>(profile.n_layers), 0));
                cfg.ngram_execution_policy = ImmutableRowTablePolicy{ImmutableRowStoragePreference::RAM, 65536, 1024};
                cfg.ngram_host_placement = ImmutableRowTableHostPlacement::fromObservedNUMANode(0);
                cfg.associated_host_memory = PhysicalMemoryResource{
                    .world_rank = -1, .device = DeviceId::cpu(),
                    .total_bytes = 1ull << 40, .admission_available_bytes = 1ull << 40};
                const auto find = [&](const MemoryPlan &plan) -> const DeviceMemoryPlan * {
                    for (const auto &device : plan.devices)
                        if (device.device() == first) return &device;
                    return nullptr;
                };
                const auto primary = MemoryPlanner::plan(profile, {cfg});
                cfg.additional_weight_sets = {AdditionalPersistentWeightSet::ReplicatedDenseDecode};
                const auto phase_split = MemoryPlanner::plan(profile, {cfg});
                const auto *before = find(primary);
                const auto *after = find(phase_split);
                ASSERT_NE(before, nullptr);
                ASSERT_NE(after, nullptr);
                if (batch * rows > 1)
                {
                    EXPECT_GE(after->workspace_bytes(), before->workspace_bytes() + handoff_bytes)
                        << "participant=" << first.toString() << " batch=" << batch << " rows=" << rows << " degree=" << degree;
                }
                auto without_native = cfg;
                without_native.local_tp_backend.reset();
                const auto nonlocal = MemoryPlanner::plan(profile, {without_native});
                const auto *without = find(nonlocal);
                ASSERT_NE(without, nullptr);
                EXPECT_EQ(after->workspace_bytes() - without->workspace_bytes(), batch * rows > 1 ? handoff_bytes : 0u);

                if (first_layer == 0)
                {
                    auto implicit_end = cfg;
                    implicit_end.last_layer = -1;
                    const auto default_scope = MemoryPlanner::plan(profile, {implicit_end});
                    ASSERT_NE(find(default_scope), nullptr);
                    EXPECT_EQ(find(default_scope)->workspace_bytes(), after->workspace_bytes());
                }

                // The same declared maximum layout is not a linked-local
                // producer when its original primary assignment is already full.
                auto full = assignment.forRank(0);
                full.head_start = 0; full.head_count = profile.n_heads;
                full.kv_head_start = 0; full.kv_head_count = profile.n_kv_heads;
                full.d_ff_start = 0; full.d_ff_count = profile.d_ff;
                full.vocab_start = 0; full.vocab_count = profile.vocab_size;
                auto full_primary = cfg;
                full_primary.bindTensorParallelAssignment(full);
                const auto full_native = MemoryPlanner::plan(profile, {full_primary});
                full_primary.local_tp_backend.reset();
                const auto full_nonlocal = MemoryPlanner::plan(profile, {full_primary});
                ASSERT_NE(find(full_native), nullptr);
                ASSERT_NE(find(full_nonlocal), nullptr);
                EXPECT_EQ(find(full_native)->workspace_bytes(), find(full_nonlocal)->workspace_bytes());
            }
        }
    }

    // A scalar CPU primary cannot request a TP replica sidecar. Preserve this
    // original admission guard instead of fabricating native CPU membership.
    auto cpu = overlayDeviceConfig(DeviceId::cpu());
    cpu.world_rank = -1;
    cpu.device_total_bytes = cpu.device_free_bytes = 1ull << 40;
    cpu.total_shards = 1;
    cpu.first_layer = 0;
    cpu.last_layer = profile.mainLayerCount() - 1;
    cpu.activation_seq_len = 9;
    cpu.additional_weight_sets = {AdditionalPersistentWeightSet::ReplicatedDenseDecode};
    EXPECT_THROW((void)MemoryPlanner::plan(profile, {cpu}), std::invalid_argument);
}

/**
 * @test Ordinary native terminal publication owns full logits without an MTP family.
 *
 * Keep the planner's historical target-row default deliberately untouched:
 * workspace propagation must mask it to zero while mtp_enabled is false. An
 * ordinary singleton gather needs no transpose scratch, predictor weights or
 * recurrent checkpoints. The no-native comparison differs only in publication.
 */
TEST(Test__MemoryPlanner, OrdinaryNativeVocabularyHasNoRetainedMTPOrSingletonScratch)
{
    const auto profile = createTestProfile();
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int degree : {2, 4, 8})
    {
        SCOPED_TRACE(DeviceId(backend, 0).toString() + "/" + std::to_string(degree));
        DevicePlanConfig cfg;
        cfg.device = DeviceId(backend, 0);
        cfg.device_total_bytes = cfg.device_free_bytes = 64ULL << 30;
        cfg.device_compute_units = 60;
        cfg.total_shards = degree;
        cfg.first_layer = 0;
        cfg.last_layer = profile.n_layers - 1;
        cfg.batch_size = 1;
        cfg.max_seq_len = cfg.activation_seq_len = 64;
        cfg.mtp_enabled = false;
        cfg.mtp_terminal_logits_layout = MTPTerminalLogitsLayout::VocabularyShardPerParticipant;
        cfg.local_tp_backend = backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        ASSERT_EQ(cfg.terminalLogitsCollective(true), MTPTerminalLogitsCollective::NativeLocalVocabularyAllGather);
        EXPECT_EQ(cfg.terminalLogitsCollective(false), MTPTerminalLogitsCollective::None);
        MTPRuntimeConfig no_mtp;
        no_mtp.enabled = false;
        no_mtp.graph_capacity_draft_tokens = 0;
        no_mtp.terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded;
        EXPECT_FALSE(retainsMTPGraphCapacity(no_mtp));
        cfg.additional_weight_sets = resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallel, degree, no_mtp);
        EXPECT_TRUE(cfg.additional_weight_sets.empty());
        const auto native = MemoryPlanner::plan(profile, {cfg});
        cfg.local_tp_backend.reset(); // Explicit global/non-local shard owns no native publication.
        EXPECT_EQ(cfg.terminalLogitsCollective(true), MTPTerminalLogitsCollective::None);
        const auto without_native = MemoryPlanner::plan(profile, {cfg});
        ASSERT_EQ(native.devices.size(), 1u);
        ASSERT_EQ(without_native.devices.size(), 1u);
        EXPECT_EQ(native.devices.front().workspace_bytes(), without_native.devices.front().workspace_bytes());
        EXPECT_EQ(native.devices.front().additional_weight_bytes(), 0u);
        EXPECT_EQ(native.devices.front().checkpoint_state_bytes(), 0u);
    }
}

/** @test Ordinary native scratch follows declared output rows, never prompt/MTP placeholders. */
TEST(Test__MemoryPlanner, OrdinaryProjectedRowsPriceExactNativeBank)
{
    const auto profile = createTestProfile();
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int degree : {2, 4, 8})
    {
        DevicePlanConfig cfg;
        cfg.device = DeviceId(backend, 0);
        cfg.device_total_bytes = cfg.device_free_bytes = 64ULL << 30;
        cfg.device_compute_units = 60;
        cfg.total_shards = degree;
        cfg.last_layer = profile.n_layers - 1;
        cfg.max_seq_len = 4096;
        cfg.activation_seq_len = 512;
        cfg.batch_size = 1;
        cfg.mtp_enabled = false; // Keep the inactive target-row placeholder unchanged.
        cfg.mtp_terminal_logits_layout = MTPTerminalLogitsLayout::VocabularyShardPerParticipant;
        cfg.local_tp_backend = backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        const auto singleton = MemoryPlanner::plan(profile, {cfg});
        for (const int rows : {2, 16, 512})
        {
            cfg.ordinary_terminal_projection = OrdinaryTerminalProjectionCapacity::projectedRows(rows);
            const auto diagnostic = MemoryPlanner::plan(profile, {cfg});
            const auto bank = VocabularyGatherWorkspaceContract::requirements(rows, profile.vocab_size);
            EXPECT_EQ(diagnostic.devices.front().workspace_bytes() - singleton.devices.front().workspace_bytes(),
                      bank.total_bytes_with_alignment());
            EXPECT_EQ(diagnostic.devices.front().additional_weight_bytes(), 0u);
            EXPECT_EQ(diagnostic.devices.front().checkpoint_state_bytes(), 0u);
        }
        cfg.ordinary_terminal_projection = OrdinaryTerminalProjectionCapacity::projectedRows(513);
        EXPECT_THROW(MemoryPlanner::plan(profile, {cfg}), std::invalid_argument);
        cfg.ordinary_terminal_projection = OrdinaryTerminalProjectionCapacity::projectedRows(16);
        cfg.local_tp_backend.reset(); // Global/non-native ownership contributes no transpose bank.
        const auto global = MemoryPlanner::plan(profile, {cfg});
        cfg.ordinary_terminal_projection = {};
        const auto global_single = MemoryPlanner::plan(profile, {cfg});
        EXPECT_EQ(global.devices.front().workspace_bytes(), global_single.devices.front().workspace_bytes());
        cfg.local_tp_backend = backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        cfg.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant;
        EXPECT_EQ(cfg.terminalLogitsCollective(true), MTPTerminalLogitsCollective::None);
    }
    DevicePlanConfig cpu;
    cpu.device = DeviceId::cpu();
    cpu.cpu_execution = test::kSyntheticCPUExecutionGeometry;
    cpu.device_compute_units = 8;
    cpu.device_total_bytes = cpu.device_free_bytes = 64ULL << 30;
    cpu.max_seq_len = cpu.activation_seq_len = 512;
    const auto singleton = MemoryPlanner::plan(profile, {cpu});
    cpu.ordinary_terminal_projection = OrdinaryTerminalProjectionCapacity::projectedRows(16);
    const auto diagnostic = MemoryPlanner::plan(profile, {cpu});
    EXPECT_EQ(cpu.terminalLogitsCollective(true), MTPTerminalLogitsCollective::None);
    EXPECT_EQ(diagnostic.devices.front().workspace_bytes(), singleton.devices.front().workspace_bytes());
}


/** @test Extra main-query programs add only their real native owners, never duplicated sidecars/helpers. */
TEST(Test__MemoryPlanner, SparseQueryProgramInventoryPricesOnlyAdditionalMainOwners)
{
    for(bool retains:{false,true}){
        MTPRuntimeConfig policy;
        policy.enabled=false;
        policy.graph_capacity_draft_tokens=retains?15:0;
        const auto one=resolveCapturedServingGraphMemoryInventory(
            {32,64},
            policy,
            MTPVerifierGraphFamily::BoundedBuckets,
            MTPGenerationParentRole::Controller,
            DeviceGenerationExecutionDomain::NativeDomain);
        const auto two=resolveCapturedServingGraphMemoryInventory(
            {32,64},
            policy,
            MTPVerifierGraphFamily::BoundedBuckets,
            MTPGenerationParentRole::Controller,
            DeviceGenerationExecutionDomain::NativeDomain,
            2u);
        EXPECT_EQ(one.sparse_query_program_count,1u);
        EXPECT_EQ(two.sparse_query_program_count,2u);
        EXPECT_EQ(two.prefill_bucket_rows,one.prefill_bucket_rows);
        EXPECT_EQ(two.fixed_executable_count,2u*one.fixed_executable_count);
        EXPECT_EQ(two.mtp_graph_owners.sidecarGraphSlots(),one.mtp_graph_owners.sidecarGraphSlots());
        EXPECT_EQ(two.mtp_graph_owners.boundedHelperExecutableSlotCount(),one.mtp_graph_owners.boundedHelperExecutableSlotCount());
        EXPECT_EQ(two.mtp_graph_owners.controllerGraphSlots(),one.mtp_graph_owners.controllerGraphSlots()+(retains?2u:0u));
        EXPECT_EQ(one.mtp_graph_owners.speculativePublicationGraphSlots(), retains ? 1u : 0u);
        EXPECT_EQ(two.mtp_graph_owners.speculativePublicationGraphSlots(), retains ? 2u : 0u);
        EXPECT_EQ(two.mtp_graph_owners.generalAuxiliaryExecutableSlotCount(),
            one.mtp_graph_owners.generalAuxiliaryExecutableSlotCount() + (retains ? 2u : 0u));
        EXPECT_EQ(OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(1u, PipelineForwardOnlyGraphOwner::Participant),1u);
        EXPECT_EQ(OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(2u, PipelineForwardOnlyGraphOwner::Participant),2u);
        for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
            EXPECT_EQ(OrdinaryGenerationGraphPlan::additionalExecutableCount(
                device,
                retains,
                declaredDeviceGenerationExecutionPolicy(device.type,DeviceGenerationExecutionDomain::NativeDomain),
                MTPGenerationParentRole::Controller,
                2u),
                2u*OrdinaryGenerationGraphPlan::additionalExecutableCount(
                    device,
                    retains,
                    declaredDeviceGenerationExecutionPolicy(device.type,DeviceGenerationExecutionDomain::NativeDomain),
                    MTPGenerationParentRole::Controller));
    }
    EXPECT_THROW((void)resolveCapturedServingGraphMemoryInventory(
        {32},
        MTPRuntimeConfig{},
        MTPVerifierGraphFamily::BoundedBuckets,
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::NativeDomain,
        0u),std::invalid_argument);
    EXPECT_THROW((void)resolveCapturedServingGraphMemoryInventory(
        {32},
        MTPRuntimeConfig{},
        MTPVerifierGraphFamily::BoundedBuckets,
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::NativeDomain,
        3u),std::invalid_argument);
    EXPECT_THROW((void)MTPGraphOwnerPlan(MTPRuntimeConfig{},MTPVerifierGraphFamily::BoundedBuckets,MTPGenerationParentRole::Controller,3u),std::invalid_argument);
    EXPECT_THROW((void)OrdinaryGenerationGraphPlan::additionalExecutableCount(
        DeviceId::rocm(0),
        false,
        DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions,
        MTPGenerationParentRole::Controller,
        3u),std::invalid_argument);
    EXPECT_THROW((void)OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(3u, PipelineForwardOnlyGraphOwner::Participant),std::invalid_argument);
}

/** @test Dedicated scalar diagnostic arenas multiply; proven wide/condition lanes and Dense0 stay shared. */
TEST(Test__MemoryPlanner, SparseQueryProgramSnapshotOwnersMatchPhysicalSharing)
{
    const auto one=ServingSnapshotArenaInventory::ordinary(true,true,7u,1u);
    const auto two=ServingSnapshotArenaInventory::ordinary(true,true,7u,1u,2u);
    EXPECT_EQ(one.scalarMainOwners(),2u);
    EXPECT_EQ(two.scalarMainOwners(),4u);
    EXPECT_EQ(two.wideMainOwners(),one.wideMainOwners());
    EXPECT_EQ(two.conditionMainOwners(),one.conditionMainOwners());
    EXPECT_EQ(two.predictorOwners(),one.predictorOwners());
    const MainSnapshotPhasePayload payload{.wide_main_bytes=1024u,.scalar_main_bytes=128u,.separate_reference_bytes=17u};
    EXPECT_EQ(payload.bytesFor(two)-payload.bytesFor(one),2u*128u);
    EXPECT_THROW((void)ServingSnapshotArenaInventory::ordinary(true,false,0u,0u,0u),std::invalid_argument);
    EXPECT_THROW((void)ServingSnapshotArenaInventory::ordinary(true,false,0u,0u,3u),std::invalid_argument);
}

/** @test Complete-main sparse topology determines immutable programs, independent of stage scope. */
TEST(Test__MemoryPlanner, SparseQueryProfileProjectionExcludesPredictorOnlyLayers)
{
    ModelMemoryProfile profile;
    profile.n_layers = 4;
    profile.mtp_layer_count = 1;
    EXPECT_EQ(profile.sparseQueryProgramCount(), 1u);
    profile.flash_next.emplace();
    profile.flash_next->sparse_decoder_layers = {4};
    EXPECT_EQ(profile.sparseQueryProgramCount(), 1u); // Predictor-only sparse geometry is not a main program.
    profile.flash_next->sparse_decoder_layers = {3, 4};
    EXPECT_EQ(profile.sparseQueryProgramCount(), 2u);
    profile.n_layers = -1;
    EXPECT_THROW(profile.sparseQueryProgramCount(), std::invalid_argument);
}

/** @test Two decode programs never create a second RankingPossible-only prefill family. */
TEST(Test__MemoryPlanner, SparseQueryProgramBOMKeepsPrefillSingleAndPricesHostedCudaTail)
{
    const auto profile = createTestProfile();
    DevicePlanConfig config;
    config.device = DeviceId::cuda(0);
    config.device_total_bytes = config.device_free_bytes = 64ull << 30;
    config.device_compute_units = 60;
    config.max_seq_len = 4096;
    config.activation_seq_len = 64;
    MTPRuntimeConfig retained;
    retained.graph_capacity_draft_tokens = 15;
    for (const auto role : {MTPGenerationParentRole::Controller, MTPGenerationParentRole::PipelineFollower})
    {
        config.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory(
            {32, 64},
            retained,
            MTPVerifierGraphFamily::ExactRows,
            role,
            DeviceGenerationExecutionDomain::HostedBoundary,
            2u);
        const auto plan = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(plan.devices.size(), 1u);
        const auto &serving = config.captured_serving_graphs;
        const auto ordinary = OrdinaryGenerationGraphPlan::additionalExecutableCount(config.device, true, DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions, role, 2u);
        const CapturedGraphExecutableInventory expected{
            .model_graph_identity_count = 2u + serving.fixed_executable_count,
            .model_graph_topology_variant_count = 1u,
            .auxiliary_executable_count = serving.mtp_graph_owners.generalAuxiliaryExecutableSlotCount() + ordinary,
            .bounded_helper_executable_count = serving.mtp_graph_owners.boundedHelperExecutableSlotCount()};
        EXPECT_EQ(plan.devices.front().captured_graph_bytes(), estimateCapturedGraphExecutableBytes(config.device, expected));
        auto wrong_prefill = expected;
        wrong_prefill.model_graph_identity_count += 2u; // Blanket program multiplication invents two prefills.
        EXPECT_NE(plan.devices.front().captured_graph_bytes(), estimateCapturedGraphExecutableBytes(config.device, wrong_prefill));
    }
}

/** @test A per-GPU BOM rejects missing or contradictory execution declarations before pricing. */
TEST(Test__MemoryPlanner, CapturedServingFamilyMustMatchDeclaredGpuDomain)
{
    const auto profile = createTestProfile();
    DevicePlanConfig config;
    config.device = DeviceId::cuda(0);
    config.device_total_bytes = config.device_free_bytes = 64ull << 30;
    config.device_compute_units = 60;
    config.max_seq_len = 4096;
    config.activation_seq_len = 64;
    MTPRuntimeConfig retained;
    retained.graph_capacity_draft_tokens = 15;
    config.captured_serving_graphs = resolveCapturedServingGraphMemoryInventory(
        {32},
        retained,
        MTPVerifierGraphFamily::ExactRows,
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::NativeDomain);
    EXPECT_THROW(MemoryPlanner::plan(profile,{config}), std::invalid_argument); // CUDA native requires bounded.
    config.device = DeviceId::rocm(0);
    EXPECT_NO_THROW(MemoryPlanner::plan(profile,{config})); // ROCm native domain declares hosted exact.
    config.device = DeviceId::cuda(0);
    config.captured_serving_graphs.generation_domain = DeviceGenerationExecutionDomain::HostedBoundary;
    EXPECT_NO_THROW(MemoryPlanner::plan(profile,{config}));
    config.captured_serving_graphs.verifier_graph_family = MTPVerifierGraphFamily::BoundedBuckets;
    EXPECT_THROW(MemoryPlanner::plan(profile,{config}), std::invalid_argument);
    config.captured_serving_graphs.generation_domain.reset();
    EXPECT_THROW(MemoryPlanner::plan(profile,{config}), std::invalid_argument);
}

/**
 * @test One flat scalar history owner is admitted for each MTP controller.
 * @details Depth, request capacity, query programs and initial request enablement
 * do not multiply this owner. Its one bounded-helper charge reaches the actual
 * backend byte estimator; followers and absent retained capacity own none.
 */
TEST(Test__MemoryPlanner, MTPDirectTokenHistoryOwnerIsSingleBoundedControllerHelper)
{
    for (int requests = 1; requests <= 8; ++requests)
    for (int depth : {1, 3, 15})
    for (std::size_t programs : {1u, 2u})
    for (bool enabled : {false, true})
    for (auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Observe, MTPDepthPolicyMode::Dynamic})
    for (auto family : {MTPVerifierGraphFamily::BoundedBuckets, MTPVerifierGraphFamily::ExactRows})
    for (auto role : {MTPGenerationParentRole::Controller, MTPGenerationParentRole::PipelineFollower})
    {
        MTPRuntimeConfig policy;
        policy.enabled = enabled;
        policy.draft_tokens = 1;
        policy.graph_capacity_draft_tokens = depth;
        policy.max_request_batch = requests;
        policy.depth_policy.mode = mode;
        const MTPGraphOwnerPlan owners(policy, family, role, programs);
        const std::size_t direct = role == MTPGenerationParentRole::Controller ? 1u : 0u;
        EXPECT_EQ(owners.directTokenHistoryGraphSlots(), direct);
        EXPECT_EQ(owners.controllerGraphSlots(), owners.samplerHelperGraphSlots() +
            owners.generationParentGraphSlots() + owners.speculativePublicationGraphSlots() + direct);
        const auto bounded_without_direct = owners.boundedHelperExecutableSlotCount() - direct;
        const auto total_without_direct = owners.auxiliaryExecutableSlotCount() - direct;
        EXPECT_EQ(owners.generalAuxiliaryExecutableSlotCount() + bounded_without_direct,
                  total_without_direct);
        for (auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            CapturedGraphExecutableInventory inventory{
                .model_graph_identity_count = 1u,
                .model_graph_topology_variant_count = 1u,
                .auxiliary_executable_count = owners.generalAuxiliaryExecutableSlotCount(),
                .bounded_helper_executable_count = bounded_without_direct,
            };
            const auto without_direct = estimateCapturedGraphExecutableBytes(device, inventory);
            inventory.bounded_helper_executable_count += direct;
            EXPECT_EQ(estimateCapturedGraphExecutableBytes(device, inventory), without_direct + direct *
                GPUGraphMemoryContract::reservationBytesPerExecutable(device, GPUGraphExecutableClass::BoundedFlatHelper));
        }
    }
    for (auto role : {MTPGenerationParentRole::Controller, MTPGenerationParentRole::PipelineFollower})
    {
        const MTPGraphOwnerPlan absent(MTPRuntimeConfig{}, MTPVerifierGraphFamily::ExactRows, role);
        EXPECT_EQ(absent.directTokenHistoryGraphSlots(), 0u);
        EXPECT_EQ(absent.auxiliaryExecutableSlotCount(), 0u);
    }
}

/** @test Every independently retained controller is explicit in the serial workspace BOM. */
TEST(Test__MemoryPlanner, CPUDeclaredSerialControllersRequireExactWorkspaceConcurrency)
{
    const auto profile = createMoEOverlayProfile();
    auto config = overlayDeviceConfig(DeviceId::cpu());
    config.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant;
    // This profile owns both routed layers; concurrency must not be tested
    // through an incomplete per-layer residency declaration.
    config.weight_residency = DeviceWeightResidency::selectedRoutedExpertsOnly(
        profile.expert_count,std::vector<int>(static_cast<size_t>(profile.n_layers),2));
    config.routed_expert_compact_buffer_lifetime = RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
    config.serial_routed_expert_participant_count = 2;
    config.serial_routed_expert_compact_rows = 31;
    declareCPUSerialFixture(config,profile);
    const auto single = MemoryPlanner::plan(profile,{config}).devices.front().workspace_bytes();
    auto declared = std::make_shared<CPUSerialExpertStorageAdmission>(*config.cpu_serial_expert_storage);
    declared->controller_devices.push_back(DeviceId{DeviceType::CPU,1});
    std::sort(declared->controller_devices.begin(),declared->controller_devices.end());
    config.cpu_serial_expert_storage = declared;
    EXPECT_THROW((void)MemoryPlanner::plan(profile,{config}),std::invalid_argument);
    config.concurrent_workspace_owners = declared->controller_devices.size();
    EXPECT_EQ(MemoryPlanner::plan(profile,{config}).devices.front().workspace_bytes(),single*2);
}

namespace
{
/**
 * @brief Reproduce the existing named-workspace source projection for this
 * tiny CPU fixture with balanced TP; the grouped physical roots are intentionally
 * priced independently through their shared production declaration.
 */
size_t ordinaryCPUNamedWorkspace(const ModelMemoryProfile &profile,const DevicePlanConfig &cfg)
{
    WorkspaceMemoryGeometry geometry{
        .device=cfg.device,.device_compute_units=cfg.device_compute_units,
        .cpu_execution=cfg.cpu_execution,.batch_size=cfg.batch_size,
        .ordinary_terminal_projection=cfg.ordinary_terminal_projection,
        .resident_graph_rows=cfg.activation_seq_len,.max_context_rows=cfg.max_seq_len,
        .owns_embedding=cfg.execution_role==DeviceExecutionMemoryRole::ContinuationGraph && cfg.owns_embedding,
        .shard_index=cfg.shard_index,.local_d_ff=std::max(1,profile.d_ff/cfg.total_shards),
        .local_query_head_start=cfg.shard_index*std::max(1,profile.n_heads/cfg.total_shards),
        .local_query_heads=std::max(1,profile.n_heads/cfg.total_shards),
        .local_kv_head_start=cfg.shard_index*std::max(1,profile.n_kv_heads/cfg.total_shards),
        .local_kv_heads=std::max(1,profile.n_kv_heads/cfg.total_shards),
        .first_layer=cfg.first_layer,.last_layer=cfg.last_layer,
        .total_shards=cfg.total_shards,
        .apportioned_routed_experts=cfg.weight_residency.selectsRoutedExperts(),
        .runtime_device_policy_available=cfg.runtime_device_policy_available,
        .mtp_target_query_rows=cfg.mtp_enabled?cfg.mtp_target_query_rows:0,
        .generation_request_capacity=cfg.generation_request_capacity,
    };
    if(cfg.weight_residency.selectsRoutedExperts() ||
        cfg.execution_role==DeviceExecutionMemoryRole::RoutedExpertParticipant)
        geometry.compact_routed_expert_token_rows=cfg.serial_routed_expert_compact_rows>0?
            cfg.serial_routed_expert_compact_rows:cfg.batch_size*cfg.activation_seq_len;
    if(cfg.execution_role==DeviceExecutionMemoryRole::ContinuationGraph)
        geometry.attention_kv_format=KVCacheMemoryEstimator::attentionExecutionFormat(cfg.kv_precision);
    return cfg.execution_role==DeviceExecutionMemoryRole::ContinuationGraph?
        WorkspaceMemoryEstimator::estimate(profile,geometry):
        WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(profile,geometry);
}
}
/** @test One canonical eight-root bundle covers ordinary and retained CPU rows, once per concurrent owner. */
TEST(Test__MemoryPlanner, OrdinaryCPUGroupedRootsUseExactDeclarationAndOwnerMultiplicity)
{
    const auto profile=createMoEOverlayProfile(); // F=64 exceeds D=32.
    for(const auto device:{DeviceId::cpu(),DeviceId{DeviceType::CPU,1}})
    for(const bool retained:{false,true})
    for(const int degree:{1,2})
    for(const size_t owners:{1u,3u})
    {
        auto cfg=overlayDeviceConfig(device);cfg.batch_size=2;cfg.activation_seq_len=3;
        cfg.mtp_enabled=retained;cfg.mtp_target_query_rows=17;cfg.total_shards=degree;
        cfg.concurrent_workspace_owners=owners;
        const size_t rows=retained?17u:6u; // Target rows are already flattened.
        const auto expected=CPUGroupedMoEStorageLayout::declare(rows,profile.d_model,
            profile.expert_feed_forward_length,profile.expert_count,profile.expert_used_count,device);
        const auto planned=MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes();
        EXPECT_EQ(planned,(ordinaryCPUNamedWorkspace(profile,cfg)+expected.physicalBytes())*owners)
            << "device="<<device.to_string()<<" retained="<<retained<<" degree="<<degree<<" owners="<<owners;
    }
}
/** @test Retained local endpoints retain original compact rows and model top-k, without flattening K twice. */
TEST(Test__MemoryPlanner, RetainedNonserialCPUGroupedRootsPreserveCompactRowsAndModelTopK)
{
    const auto profile=createMoEOverlayProfile();auto cfg=overlayDeviceConfig(DeviceId{DeviceType::CPU,1});
    cfg.execution_role=DeviceExecutionMemoryRole::RoutedExpertParticipant;
    cfg.weight_residency=DeviceWeightResidency::selectedRoutedExpertsOnly(profile.expert_count,{2,2});
    cfg.batch_size=2;cfg.activation_seq_len=3;cfg.serial_routed_expert_compact_rows=11;
    cfg.mtp_enabled=true;cfg.mtp_target_query_rows=7;
    const auto grouped=CPUGroupedMoEStorageLayout::declare(11,profile.d_model,
        profile.expert_feed_forward_length,profile.expert_count,profile.expert_used_count,cfg.device);
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg)+grouped.physicalBytes());
}
/** @test A PP interval without a routed graph stage must not acquire grouped roots from global model metadata. */
TEST(Test__MemoryPlanner, CPUGroupedRootsRespectActualRoutedLayerInventory)
{
    auto profile=createMoEOverlayProfile();
    std::erase_if(profile.tensors,[](const auto &tensor){return tensor.layer_index==1 &&
        (tensor.name.ends_with(".ffn_gate_exps.weight") || tensor.name.ends_with(".ffn_up_exps.weight") ||
         tensor.name.ends_with(".ffn_down_exps.weight"));});
    for(const int layer:{0,1})
    {
        auto cfg=overlayDeviceConfig(DeviceId::cpu());cfg.first_layer=cfg.last_layer=layer;
        const auto grouped=CPUGroupedMoEStorageLayout::declare(size_t(cfg.activation_seq_len),profile.d_model,
            profile.expert_feed_forward_length,profile.expert_count,profile.expert_used_count,cfg.device);
        EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
            ordinaryCPUNamedWorkspace(profile,cfg)+(layer==0?grouped.physicalBytes():0u))<<"PP layer="<<layer;
    }
}
/** @test An existing immutable serial envelope owns the bundle; ordinary admission must not add another copy. */
TEST(Test__MemoryPlanner, DeclaredCPUSerialBundleHasNoOrdinaryDuplicate)
{
    const auto profile=createMoEOverlayProfile();auto cfg=overlayDeviceConfig(DeviceId::cpu());
    cfg.execution_role=DeviceExecutionMemoryRole::RoutedExpertParticipant;
    cfg.weight_residency=DeviceWeightResidency::selectedRoutedExpertsOnly(profile.expert_count,{2,2});
    cfg.routed_expert_compact_buffer_lifetime=RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
    cfg.serial_routed_expert_participant_count=2;cfg.serial_routed_expert_compact_rows=11;
    declareCPUSerialFixture(cfg,profile);
    const auto grouped=cfg.cpu_serial_expert_storage->envelope->admitted()->workspaceBytes();
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg)+grouped*2u);
    // A real retained endpoint remains present at an initially empty epoch;
    // topology identities, not current ownership, retain the serial bundle.
    cfg.weight_residency=DeviceWeightResidency::selectedRoutedExpertsOnly(profile.expert_count,{0,0});
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg)+grouped*2u);
}

/** @test Empty selected portable endpoints and dense-only serial continuations own no grouped stage. */
TEST(Test__MemoryPlanner, CPUGroupedRootsFollowEndpointTopologyForSelectedZeroOwners)
{
    const auto profile=createMoEOverlayProfile();
    for(const auto lifetime:{RoutedExpertCompactBufferLifetime::PerLayerGraphOwned,
                             RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant})
    {
        auto cfg=overlayDeviceConfig(DeviceId{DeviceType::CPU,1});
        cfg.weight_residency=DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,{0,0});
        cfg.routed_expert_compact_buffer_lifetime=lifetime;
        cfg.serial_routed_expert_participant_count=0;
        ASSERT_EQ(cfg.cpu_serial_expert_storage,nullptr);
        EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
            ordinaryCPUNamedWorkspace(profile,cfg));
    }
    // Per-layer endpoint masks are layer-local: positive ownership outside
    // this PP interval cannot author an otherwise absent local stage.
    auto cfg=overlayDeviceConfig(DeviceId::cpu());cfg.first_layer=cfg.last_layer=1;
    cfg.weight_residency=DeviceWeightResidency::continuationWithSelectedRoutedExperts(
        profile.expert_count,{2,0});
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg));
}

/** @test Dormant predictor expert parents are absent; a retained terminal-owned predictor uses the same bundle. */
TEST(Test__MemoryPlanner, CPUGroupedRootsRespectRetainedPredictorOwnership)
{
    auto profile=createMoEOverlayProfile();profile.mtp_layer_count=1;
    // Ordinary Qwen metadata includes its predictor in n_layers. The only
    // routed parent is the trailing predictor; the main PP interval is dense.
    std::erase_if(profile.tensors,[](const auto &tensor){return tensor.layer_index==0 &&
        (tensor.name.ends_with(".ffn_gate_exps.weight") || tensor.name.ends_with(".ffn_up_exps.weight") ||
         tensor.name.ends_with(".ffn_down_exps.weight"));});
    for(const bool retained:{false,true})
    {
        auto cfg=overlayDeviceConfig(DeviceId::cpu());cfg.last_layer=0;
        cfg.mtp_enabled=retained;cfg.mtp_target_query_rows=37;
        const auto grouped=CPUGroupedMoEStorageLayout::declare(37,profile.d_model,
            profile.expert_feed_forward_length,profile.expert_count,profile.expert_used_count,cfg.device);
        EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
            ordinaryCPUNamedWorkspace(profile,cfg)+(retained?grouped.physicalBytes():0u));
    }
}

namespace
{
/** @brief Add canonical source-owned router matrices absent from the legacy expert-only metadata fixture. */
ModelMemoryProfile cpuRouterPlanningProfile()
{
    auto profile=createMoEOverlayProfile();
    for(int layer=0;layer<profile.n_layers;++layer)
    {
        TensorSizeInfo tensor; tensor.name="blk."+std::to_string(layer)+".ffn_gate_inp.weight";
        tensor.layer_index=layer;tensor.K=profile.d_model;
        tensor.elements=size_t(profile.expert_count)*size_t(profile.d_model);
        tensor.quant_type="F32";tensor.native_bytes=tensor.elements*sizeof(float);
        profile.total_native_bytes+=tensor.native_bytes;profile.tensors.push_back(std::move(tensor));
    }
    return profile;
}
}
/** @test Router-only continuations own precisely one Q8 publication root even at zero selected experts. */
TEST(Test__MemoryPlanner, CPUOnlyRouterPublicationUsesCanonicalPhysicalRoot)
{
    const auto profile=cpuRouterPlanningProfile();
    for(const auto device:{DeviceId::cpu(),DeviceId{DeviceType::CPU,1}})
    for(const bool retained:{false,true})
    for(const size_t owners:{1u,2u})
    {
        auto cfg=overlayDeviceConfig(device);cfg.batch_size=2;cfg.activation_seq_len=3;
        cfg.mtp_enabled=retained;cfg.mtp_target_query_rows=17;cfg.concurrent_workspace_owners=owners;
        cfg.weight_residency=DeviceWeightResidency::continuationWithSelectedRoutedExperts(profile.expert_count,{0,0});
        const auto router=CPURouterQ8StorageLayout::declare(retained?17:6,profile.d_model,device);
        EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
            (ordinaryCPUNamedWorkspace(profile,cfg)+router.physicalBytes())*owners);
    }
}
/** @test Same-manager grouped storage supplies its router root exactly once. */
TEST(Test__MemoryPlanner, CPUOrdinaryRouterCoalescesWithItsGroupedExpertBundle)
{
    const auto profile=cpuRouterPlanningProfile();auto cfg=overlayDeviceConfig(DeviceId::cpu());
    const auto grouped=CPUGroupedMoEStorageLayout::declare(size_t(cfg.activation_seq_len),profile.d_model,
        profile.expert_feed_forward_length,profile.expert_count,profile.expert_used_count,cfg.device);
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg)+grouped.physicalBytes());
}
/** @test A transported serial endpoint's kernel cannot provide a separate continuation router's backing. */
TEST(Test__MemoryPlanner, CPUSerialEndpointDoesNotAliasContinuationRouterPublication)
{
    const auto profile=cpuRouterPlanningProfile();auto cfg=overlayDeviceConfig(DeviceId{DeviceType::CPU,1});
    cfg.weight_residency=DeviceWeightResidency::continuationWithSelectedRoutedExperts(profile.expert_count,{0,0});
    cfg.routed_expert_compact_buffer_lifetime=RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant;
    cfg.serial_routed_expert_participant_count=2;cfg.serial_routed_expert_compact_rows=11;
    declareCPUSerialFixture(cfg,profile);
    const auto endpoint=cfg.cpu_serial_expert_storage->envelope->admitted()->workspaceBytes();
    const auto router=CPURouterQ8StorageLayout::declare(size_t(cfg.activation_seq_len),profile.d_model,cfg.device);
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg)+endpoint*2u+router.physicalBytes());
    cfg.execution_role=DeviceExecutionMemoryRole::RoutedExpertParticipant;
    cfg.weight_residency=DeviceWeightResidency::selectedRoutedExpertsOnly(profile.expert_count,{0,0});
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),
        ordinaryCPUNamedWorkspace(profile,cfg)+endpoint*2u);
}
/** @test A router outside the current PP interval cannot author a local root. */
TEST(Test__MemoryPlanner, CPURouterPublicationRespectsActualGraphScope)
{
    auto profile=cpuRouterPlanningProfile();
    std::erase_if(profile.tensors,[](const auto &tensor){return tensor.layer_index==1 &&
        tensor.name.ends_with(".ffn_gate_inp.weight");});
    auto cfg=overlayDeviceConfig(DeviceId::cpu());cfg.first_layer=cfg.last_layer=1;
    cfg.weight_residency=DeviceWeightResidency::continuationWithSelectedRoutedExperts(profile.expert_count,{0,0});
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().workspace_bytes(),ordinaryCPUNamedWorkspace(profile,cfg));
}

/** @test Per-layer CPU packet BOM uses exactly the common four native roots for each layer. */
TEST(Test__MemoryPlanner, CpuPerLayerPacketUsesExactIndependentRootDeclarations)
{
    const auto profile=createMoEOverlayProfile();
    for(const auto device:{DeviceId::cpu(),DeviceId(DeviceType::CPU,1)})
    for(const int batch:{1,2})
    {
        auto config=overlayDeviceConfig(device);
        config.execution_role=DeviceExecutionMemoryRole::RoutedExpertParticipant;
        config.batch_size=batch;config.activation_seq_len=3;
        config.weight_residency=DeviceWeightResidency::selectedRoutedExpertsOnly(profile.expert_count,
            std::vector<int>(static_cast<size_t>(profile.n_layers),2));
        const auto layout=CPUCompactPacketStorageLayout::declare(size_t(batch)*3,
            profile.d_model,profile.expert_used_count,device);
        auto plan=MemoryPlanner::plan(profile,{config});
        ASSERT_EQ(plan.devices.size(),1u);
        EXPECT_EQ(plan.devices.front().activation_bytes(),layout.physicalBytes()*profile.n_layers);
        config.weight_residency=DeviceWeightResidency::selectedRoutedExpertsOnly(profile.expert_count,{2,0});
        plan=MemoryPlanner::plan(profile,{config});
        EXPECT_EQ(plan.devices.front().activation_bytes(),layout.physicalBytes());
    }
}

/** @brief Followers retain native main/publication parents without acquiring terminal helper owners. */
TEST(Test__MemoryPlanner, MTPFollowerExecutableOwnershipPreservesRegistryGeometry)
{
    for (const int depth : {1, 3, 15})
    for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Dynamic})
    for (const auto family : {MTPVerifierGraphFamily::BoundedBuckets, MTPVerifierGraphFamily::ExactRows})
    for (std::size_t programs = 1; programs <= kSparseQueryExecutionProgramCount; ++programs)
    {
        MTPRuntimeConfig config;
        config.enabled = true;
        config.draft_tokens = depth;
        config.graph_capacity_draft_tokens = depth;
        config.max_request_batch = 2;
        config.depth_policy.mode = mode;
        config.depth_policy.max_depth = depth;
        const MTPGraphOwnerPlan controller(config, family, MTPGenerationParentRole::Controller, programs);
        const MTPGraphOwnerPlan follower(config, family, MTPGenerationParentRole::PipelineFollower, programs);
        EXPECT_EQ(follower.parentRole(), MTPGenerationParentRole::PipelineFollower);
        EXPECT_EQ(follower.verifierPreparationRegistryCapacity(), controller.verifierPreparationRegistryCapacity());
        EXPECT_EQ(follower.draftPublicationRegistryCapacity(), controller.draftPublicationRegistryCapacity());
        EXPECT_GT(follower.verifierPreparationRegistryCapacity(), 0u);
        EXPECT_EQ(follower.sidecarGraphSlots(), 0u);
        EXPECT_EQ(follower.terminalHiddenGraphSlots(), 0u);
        EXPECT_EQ(follower.draftPublicationGraphSlots(), 0u);
        EXPECT_EQ(follower.verifierPreparationGraphSlots(), 0u);
        EXPECT_EQ(follower.samplerHelperGraphSlots(), 0u);
        EXPECT_EQ(follower.boundedHelperExecutableSlotCount(), 0u);
        EXPECT_EQ(follower.auxiliaryExecutableSlotCount(), follower.generationParentGraphSlots() +
            follower.speculativePublicationGraphSlots());
        EXPECT_EQ(follower.generalAuxiliaryExecutableSlotCount(), follower.auxiliaryExecutableSlotCount());
        EXPECT_GT(follower.generationParentGraphSlots(), 0u);
        EXPECT_EQ(follower.speculativePublicationGraphSlots(), controller.speculativePublicationGraphSlots());
    }
}
