/**
 * @file Test__MemoryPlanner.cpp
 * @brief Unit coverage for metadata-only device memory admission.
 *
 * These tests pin the boundary between a declarative execution placement and
 * the bytes that its real graph allocates.  In particular, mixed ExpertOverlay
 * continuation endpoints must reserve both dense graph state and their own
 * stable sparse-route buffers before captured graph construction begins.
 */

#include <gtest/gtest.h>
#include "planning/MemoryPlanner.h"
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/ActivationBufferSizing.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "backends/DeviceId.h"

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

        addTensor("token_embd.weight", 896, 151936, "Q8_0");
        addTensor("output.weight", 896, 151936, "Q8_0");
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
        p.architecture = "qwen3.6";
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
        profile.architecture = "qwen3.6";
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
        profile.architecture = "qwen3.5moe";
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

} // anonymous namespace

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
    EXPECT_GT(plan.devices[0].weight_bytes, 0u);
    EXPECT_GT(plan.devices[0].kv_cache_bytes, 0u);
    EXPECT_GT(plan.devices[0].activation_bytes, 0u);
    EXPECT_GT(plan.devices[0].workspace_bytes, 0u);
    EXPECT_TRUE(plan.devices[0].fits());
    EXPECT_TRUE(plan.fits());
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
    EXPECT_LT(device_plan.activation_bytes, 2ULL * GiB)
        << "normal prefill must not reserve full-context all-position logits";
    EXPECT_GT(device_plan.workspace_bytes, 768ULL * 1024ULL * 1024ULL)
        << "large grouped-prefill accumulators must exceed the minimum floor";
    EXPECT_LT(
        device_plan.workspace_bytes,
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

    const auto plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan.devices.size(), 1u);
    const auto &device = plan.devices.front();

    constexpr size_t expected_main_gdn_payload =
        156893184ULL;
    constexpr size_t expected_live_gdn_arena =
        313786368ULL;
    constexpr size_t expected_terminal_hidden =
        5120ULL * sizeof(float);
    const size_t expected_kv = KVCacheMemoryEstimator::estimate(
        /*n_layers=*/17,
        /*batch_size=*/1,
        /*max_seq_len=*/4096,
        /*n_kv_heads=*/8,
        /*head_dim=*/128,
        "fp16",
        DeviceId::cuda(0));

    EXPECT_EQ(device.kv_cache_bytes, expected_kv)
        << "Only 16 main FA layers plus one shifted MTP FA layer own KV.";
    EXPECT_EQ(
        device.live_recurrent_state_bytes,
        expected_live_gdn_arena);
    EXPECT_EQ(
        device.checkpoint_state_bytes,
        4ULL *
            (expected_main_gdn_payload +
             expected_terminal_hidden))
        << "The shifted FA cache must not reserve a recurrent payload slot.";
    EXPECT_GT(
        device.persistent_state_bytes,
        device.live_recurrent_state_bytes +
            device.checkpoint_state_bytes)
        << "Device sequence metadata must remain explicitly accounted.";

    const auto logical_prefix =
        KVCacheMemoryEstimator::estimateGPULogicalBlock(
            cfg.prefix_cache.block_size,
            profile.n_kv_heads,
            profile.head_dim,
            cfg.kv_precision,
            cfg.device);
    EXPECT_EQ(
        device.prefix_cache_staging_bytes,
        expected_main_gdn_payload +
            2ULL * logical_prefix.totalBytes())
        << "Main KV, hybrid state, and shifted-MTP KV each own one archive slot.";
    EXPECT_EQ(
        device.prefix_cache_device_hot_bytes,
        cfg.prefix_cache.device_budget_bytes)
        << "The default tier must retain its configured bounded hot capacity.";

    DevicePlanConfig rocm_cfg = cfg;
    rocm_cfg.device = DeviceId::rocm(0);
    const auto rocm_plan = MemoryPlanner::plan(profile, {rocm_cfg});
    ASSERT_EQ(rocm_plan.devices.size(), 1u);
    EXPECT_EQ(
        rocm_plan.devices.front().prefix_cache_staging_bytes,
        device.prefix_cache_staging_bytes);
    EXPECT_EQ(
        rocm_plan.devices.front().prefix_cache_device_hot_bytes,
        device.prefix_cache_device_hot_bytes);
}

TEST(Test__MemoryPlanner, TerminalParticipantOwnsTrailingMTPWeights)
{
    ModelMemoryProfile profile;
    profile.architecture = "qwen3.6";
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
        with_mtp.devices.front().weight_bytes -
            without_mtp.devices.front().weight_bytes,
        32ULL * 32ULL * sizeof(float))
        << "The terminal main-graph participant must reserve the trailing "
           "predictor block loaded by the MTP sidecar.";
}

TEST(Test__MemoryPlanner, GPUActivationBufferSizing_UsesLargestPrefillBucket)
{
    EXPECT_EQ(resolveActivationBufferSeqLen(131072, DeviceId::cuda(0)), 4096);
    EXPECT_EQ(resolveActivationBufferSeqLen(2048, DeviceId::cuda(0)), 2048);
    EXPECT_EQ(resolveActivationBufferSeqLen(131072, DeviceId::rocm(0)), 4096);
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

    EXPECT_GT(long_device.kv_cache_bytes, short_device.kv_cache_bytes)
        << "KV cache must still reserve the requested long context";
    EXPECT_EQ(long_device.activation_bytes, short_device.activation_bytes)
        << "Activation arena must be sized to graph chunk capacity, not context capacity";
    EXPECT_GT(long_device.workspace_bytes, short_device.workspace_bytes)
        << "FA2 context-summary capacity follows the full KV horizon even "
           "when ordinary activation rows remain bucketed";
    EXPECT_EQ(long_device.max_seq_len, 16384);
    EXPECT_EQ(long_device.activation_seq_len, 4096);
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
    EXPECT_EQ(selected.memory_plan.devices.front().activation_seq_len, 2048);
    EXPECT_EQ(selected.memory_plan.devices.front().max_seq_len, 16384)
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
    };

    const auto plan_64 = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan_64.devices.size(), 1u);
    EXPECT_EQ(
        plan_64.devices.front().captured_graph_bytes,
        estimateCapturedGraphExecutableBytes(
            profile.n_layers,
            /*two prefill + decode + prefix bridge=*/4u));

    cfg.activation_seq_len = 128;
    const auto plan_128 = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan_128.devices.size(), 1u);
    EXPECT_EQ(
        plan_128.devices.front().captured_graph_bytes,
        estimateCapturedGraphExecutableBytes(
            profile.n_layers,
            /*three prefill + decode + prefix bridge=*/5u));
    EXPECT_GT(
        plan_128.devices.front().captured_graph_bytes,
        plan_64.devices.front().captured_graph_bytes);

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
        selected.memory_plan.devices.front().captured_graph_bytes,
        plan_64.devices.front().captured_graph_bytes);
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
    EXPECT_EQ(selected.memory_plan.devices[0].activation_seq_len, 16);
    EXPECT_EQ(selected.memory_plan.devices[1].activation_seq_len, 16)
        << "CUDA and ROCm participants must receive the same retained shape";
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
        config.device_total_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
        config.device_free_bytes = config.device_total_bytes;
        config.batch_size = 1;
        config.max_seq_len = 4096;
        config.kv_precision = "fp16";

        const auto complete = MemoryPlanner::plan(profile, {config});
        ASSERT_EQ(complete.devices.size(), 1u);
        const auto &complete_device = complete.devices.front();
        ASSERT_GT(complete_device.weight_bytes, 0u);
        ASSERT_EQ(complete_device.retained_weight_bytes, 0u);

        const size_t non_weight_bytes =
            complete_device.total_bytes() - complete_device.weight_bytes;
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
        EXPECT_EQ(retained_device.weight_bytes, complete_device.weight_bytes)
            << "The final capacity BOM must retain the complete model footprint";
        EXPECT_EQ(
            retained_device.retained_weight_bytes,
            retained_device.weight_bytes);
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
    ASSERT_GT(fresh_device.workspace_bytes, 0u);
    ASSERT_EQ(fresh_device.retained_workspace_bytes, 0u);

    config.retained_workspace_bytes = fresh_device.workspace_bytes;
    const auto reused = MemoryPlanner::plan(profile, {config});
    ASSERT_EQ(reused.devices.size(), 1u);
    const DeviceMemoryPlan &reused_device = reused.devices.front();

    EXPECT_EQ(reused_device.total_bytes(), fresh_device.total_bytes())
        << "Retained workspace remains part of the final device BOM";
    EXPECT_EQ(
        reused_device.retained_workspace_bytes,
        fresh_device.workspace_bytes);
    EXPECT_EQ(
        fresh_device.incremental_bytes() - reused_device.incremental_bytes(),
        fresh_device.workspace_bytes)
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

    DevicePlanConfig shard1 = single;
    shard1.device = DeviceId::cuda(1);
    shard1.shard_index = 1;
    shard1.total_shards = 2;

    auto tp_plan = MemoryPlanner::plan(profile, {shard0, shard1});

    EXPECT_EQ(tp_plan.devices.size(), 2u);
    // Each shard should have less weight than single
    EXPECT_LT(tp_plan.devices[0].weight_bytes, single_plan.devices[0].weight_bytes);
    EXPECT_LT(tp_plan.devices[1].weight_bytes, single_plan.devices[0].weight_bytes);
}

TEST(Test__MemoryPlanner,
     PhaseSplitDensePolicyPricesConcurrentReplicatedDecodeWeights)
{
    const auto profile = createMoEOverlayProfile();
    auto config = overlayDeviceConfig(DeviceId::cuda(0));
    config.shard_index = 0;
    config.total_shards = 2;
    const std::vector<int> no_routed_experts(
        static_cast<std::size_t>(profile.n_layers), 0);
    config.weight_residency =
        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
            profile.expert_count,
            no_routed_experts);
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
            config.total_shards);

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

    EXPECT_EQ(planned.devices.front().weight_bytes, primary.device_bytes);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes,
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
    config.first_layer = 0;
    config.last_layer = profile.n_layers - 1;
    config.batch_size = 1;
    config.max_seq_len = 4096;
    config.mtp_enabled = true;
    config.mtp_terminal_logits_layout =
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant;
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding,
            config.total_shards,
            MTPTerminalHeadPolicy::MirroredFullVocabulary);

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
    EXPECT_EQ(planned.devices.front().weight_bytes, primary.device_bytes);
    EXPECT_EQ(
        planned.devices.front().additional_weight_bytes,
        mirrored.prepared_embedding_bytes + mirrored.lm_head_bytes);
    EXPECT_EQ(
        planned.devices.front().collective_bytes,
        CollectiveMemoryEstimator::localTP(
            config.max_seq_len,
            profile.d_model)
            .perDeviceBytes());
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
    config.first_layer = 0;
    config.last_layer = profile.n_layers - 1;
    config.batch_size = 1;
    config.max_seq_len = 64;
    config.mtp_enabled = false;
    config.additional_weight_sets =
        resolveAdditionalPersistentWeightSets(
            DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding,
            config.total_shards,
            MTPTerminalHeadPolicy::MirroredFullVocabulary);

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
        planned.devices.front().additional_weight_bytes,
        mirrored.prepared_embedding_bytes + mirrored.lm_head_bytes)
        << "Serial decode and grouped MTP must retain the same mirrored terminal surface.";
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

    DevicePlanConfig shard1 = single;
    shard1.device = DeviceId::cuda(1);
    shard1.shard_index = 1;
    shard1.total_shards = 2;

    auto tp_plan = MemoryPlanner::plan(profile, {shard0, shard1});

    // Each shard should have less KV cache than the full model
    // (n_kv_heads=2, so TP-2 gives 1 head per shard)
    EXPECT_LT(tp_plan.devices[0].kv_cache_bytes, single_plan.devices[0].kv_cache_bytes);
    EXPECT_LT(tp_plan.devices[1].kv_cache_bytes, single_plan.devices[0].kv_cache_bytes);
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
        EXPECT_GT(plan.devices[rank].kv_cache_bytes, 0u);
        EXPECT_GT(plan.devices[rank].activation_bytes, 0u);
        EXPECT_GT(plan.devices[rank].workspace_bytes, 0u);
    }
    EXPECT_GT(plan.devices[0].activation_bytes,
              plan.devices[3].activation_bytes)
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

    // Each PP stage should have roughly half the weights of the full model
    // (plus replicated embedding/lm_head/norms, so > 50% each)
    DevicePlanConfig full;
    full.device = DeviceId::cuda(0);
    full.device_compute_units = 82;
    full.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    full.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    full.batch_size = 1;
    full.max_seq_len = 4096;
    full.kv_precision = "fp16";

    auto full_plan = MemoryPlanner::plan(profile, {full});

    EXPECT_LT(pp_plan.devices[0].weight_bytes, full_plan.devices[0].weight_bytes);
    EXPECT_LT(pp_plan.devices[1].weight_bytes, full_plan.devices[0].weight_bytes);

    // Each stage should have half the KV cache layers (12 vs 24)
    EXPECT_LT(pp_plan.devices[0].kv_cache_bytes, full_plan.devices[0].kv_cache_bytes);
    EXPECT_EQ(pp_plan.devices[0].kv_cache_bytes, pp_plan.devices[1].kv_cache_bytes);
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

    auto plan = MemoryPlanner::plan(profile, {cuda_cfg, rocm_cfg});

    EXPECT_EQ(plan.devices.size(), 2u);
    EXPECT_TRUE(plan.fits());

    // Both devices should have weight estimates (GPU packing applies to both)
    EXPECT_GT(plan.devices[0].weight_bytes, 0u);
    EXPECT_GT(plan.devices[1].weight_bytes, 0u);

    // Weight bytes should be equal (same sharding, same packing formula)
    EXPECT_EQ(plan.devices[0].weight_bytes, plan.devices[1].weight_bytes);

    // Render table should contain both device types
    auto table = plan.renderTable();
    EXPECT_NE(table.find("CUDA:0"), std::string::npos);
    EXPECT_NE(table.find("ROCm:0"), std::string::npos);
}

TEST(Test__MemoryPlanner, CPUOnly_ZeroWorkspace)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cpu();
    cfg.device_total_bytes = 128ULL * 1024 * 1024 * 1024;
    cfg.device_free_bytes = 64ULL * 1024 * 1024 * 1024;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp32";

    auto plan = MemoryPlanner::plan(profile, {cfg});

    EXPECT_EQ(plan.devices[0].workspace_bytes, 0u);
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
    EXPECT_LT(good_plan.devices[0].weight_bytes, full_plan.devices[0].weight_bytes);
    EXPECT_LT(good_plan.devices[1].weight_bytes, full_plan.devices[0].weight_bytes);
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
    size_t full_weight = full_plan.devices[0].weight_bytes;

    // Half layers (0-11)
    DevicePlanConfig half = full;
    half.first_layer = 0;
    half.last_layer = 11;

    auto half_plan = MemoryPlanner::plan(profile, {half});
    size_t half_weight = half_plan.devices[0].weight_bytes;

    // Half should have less weight than full
    EXPECT_LT(half_weight, full_weight);
    // Should be roughly 50-70% of full (has non-layer tensors like embed/output)
    double ratio = static_cast<double>(half_weight) / full_weight;
    EXPECT_GT(ratio, 0.40) << "Half layers should be at least 40% of full weight";
    EXPECT_LT(ratio, 0.75) << "Half layers should be less than 75% of full weight";
}

TEST(Test__MemoryPlanner, DeviceMemoryPlan_Summary)
{
    DeviceMemoryPlan plan;
    plan.device = DeviceId::cuda(0);
    plan.weight_bytes = 1024 * 1024 * 100; // 100 MB
    plan.kv_cache_bytes = 1024 * 1024 * 50;
    plan.activation_bytes = 1024 * 1024 * 30;
    plan.workspace_bytes = 1024 * 1024 * 200;
    plan.device_total_bytes = 1024ULL * 1024 * 1024 * 24;
    plan.device_free_bytes = 1024ULL * 1024 * 1024 * 23;

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
    EXPECT_GT(continuation_plan.kv_cache_bytes, 0u);
    EXPECT_GT(continuation_plan.activation_bytes, 0u);
    EXPECT_GT(continuation_plan.workspace_bytes, 0u);
    EXPECT_EQ(participant_plan.kv_cache_bytes, 0u);
    EXPECT_EQ(participant_plan.live_recurrent_state_bytes, 0u);
    EXPECT_EQ(participant_plan.checkpoint_state_bytes, 0u);
    EXPECT_EQ(participant_plan.persistent_state_bytes, 0u);
    EXPECT_GT(participant_plan.activation_bytes, 0u);
    EXPECT_GT(participant_plan.workspace_bytes, 0u);
    EXPECT_LT(participant_plan.weight_bytes, continuation_plan.weight_bytes);
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
        overlay_plan.devices.front().activation_bytes,
        dense_plan.devices.front().activation_bytes + expected_compact_bytes)
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
        serial_plan.devices.front().activation_bytes,
        dense_plan.devices.front().activation_bytes + expected_compact_bytes)
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
        serial_plan.devices.front().activation_bytes,
        dense_plan.devices.front().activation_bytes +
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
    EXPECT_EQ(selected.memory_plan.devices[0].activation_seq_len, 32);
    EXPECT_EQ(selected.memory_plan.devices[1].activation_seq_len, 32)
        << "CPU cold endpoints consume the same segmented ticket geometry as captured GPU participants";
}
