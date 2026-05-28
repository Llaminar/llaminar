#include <gtest/gtest.h>
#include "planning/MemoryPlanner.h"
#include "planning/MemoryPlan.h"
#include "planning/ModelMemoryProfile.h"
#include "backends/DeviceId.h"

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

    auto addTensor = [&](const std::string& name, size_t rows, size_t cols,
                         const std::string& quant, int layer = -1) {
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

} // anonymous namespace

TEST(Test__MemoryPlanner, SingleGPU_Fits)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024;  // 24 GB
    cfg.device_free_bytes = 23ULL * 1024 * 1024 * 1024;   // 23 GB free
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

TEST(Test__MemoryPlanner, SingleGPU_DoesNotFit)
{
    auto profile = createTestProfile();

    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_total_bytes = 1ULL * 1024 * 1024 * 1024;  // 1 GB
    cfg.device_free_bytes = 512ULL * 1024 * 1024;         // 512 MB free
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";

    auto plan = MemoryPlanner::plan(profile, {cfg});

    EXPECT_FALSE(plan.fits());
    EXPECT_GT(plan.devices[0].deficit(), 0u);
    EXPECT_FALSE(plan.diagnostics.empty());
}

TEST(Test__MemoryPlanner, TP2_ReducesPerDeviceWeight)
{
    auto profile = createTestProfile();

    // Single device
    DevicePlanConfig single;
    single.device = DeviceId::cuda(0);
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

TEST(Test__MemoryPlanner, TP2_ReducesKVCachePerDevice)
{
    auto profile = createTestProfile();

    // Single device — all KV heads
    DevicePlanConfig single;
    single.device = DeviceId::cuda(0);
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

TEST(Test__MemoryPlanner, PP2_SplitsLayersAcrossDevices)
{
    auto profile = createTestProfile();  // 24 layers

    // Stage 0: layers 0-11
    DevicePlanConfig stage0;
    stage0.device = DeviceId::cuda(0);
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
    cuda_cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024;   // 24 GB (RTX 3090)
    cuda_cfg.device_free_bytes = 23ULL * 1024 * 1024 * 1024;
    cuda_cfg.batch_size = 1;
    cuda_cfg.max_seq_len = 4096;
    cuda_cfg.kv_precision = "fp16";
    cuda_cfg.shard_index = 0;
    cuda_cfg.total_shards = 2;

    DevicePlanConfig rocm_cfg;
    rocm_cfg.device = DeviceId::rocm(0);
    rocm_cfg.device_total_bytes = 32ULL * 1024 * 1024 * 1024;   // 32 GB (MI60)
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

TEST(Test__MemoryPlanner, Diagnostics_WarnsOnTightHeadroom)
{
    auto profile = createTestProfile();

    // First, compute the plan with generous memory to learn the total
    DevicePlanConfig probe;
    probe.device = DeviceId::cuda(0);
    probe.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    probe.device_free_bytes = 24ULL * 1024 * 1024 * 1024;
    probe.batch_size = 1;
    probe.max_seq_len = 256;
    probe.kv_precision = "fp16";
    probe.headroom_bytes = 128ULL * 1024 * 1024;

    auto probe_plan = MemoryPlanner::plan(profile, {probe});
    ASSERT_TRUE(probe_plan.fits());
    size_t total_needed = probe_plan.devices[0].total_bytes();

    // Now set free bytes to total + headroom + 100 MB (remaining < 256 MB → diagnostic)
    DevicePlanConfig cfg = probe;
    cfg.device_free_bytes = total_needed + cfg.headroom_bytes + 100ULL * 1024 * 1024;

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
    auto profile = createTestProfile();  // 24 layers

    // Full model on a single 24 GB GPU that doesn't have enough free memory
    DevicePlanConfig cfg;
    cfg.device = DeviceId::cuda(0);
    cfg.device_total_bytes = 24ULL * 1024 * 1024 * 1024;
    cfg.device_free_bytes = 1ULL * 1024 * 1024 * 1024;  // Only 1 GB free
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.kv_precision = "fp16";
    cfg.first_layer = 0;
    cfg.last_layer = 23;  // All layers

    auto full_plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_FALSE(full_plan.fits()) << "Precondition: model must NOT fit on 1 GB free";

    // BUG scenario: two devices but both get ALL layers (no PP layer split)
    DevicePlanConfig bad_stage0 = cfg;
    bad_stage0.first_layer = 0;
    bad_stage0.last_layer = 23;  // All layers — WRONG for PP

    DevicePlanConfig bad_stage1 = cfg;
    bad_stage1.device = DeviceId::cuda(1);
    bad_stage1.first_layer = 0;
    bad_stage1.last_layer = 23;  // All layers — WRONG for PP

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
    auto profile = createTestProfile();  // 24 layers

    // Full model
    DevicePlanConfig full;
    full.device = DeviceId::cuda(0);
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
    plan.weight_bytes = 1024 * 1024 * 100;  // 100 MB
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
