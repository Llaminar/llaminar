/**
 * @file Test__Qwen4ExpGDNQKVWorkspace.cpp
 * @brief Device-free original-mixer/source selection and canonical QKV residual-bank admission.
 *
 * Existing GGUF metadata parsing and the actual model planning profile own all
 * source facts. Tests allocate no model weights, discover no accelerator, and
 * contribute named requirements to the sole PMA capacity checker.
 */
#include "../../utils/Qwen4ExpHCWorkspaceFixture.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "models/qwen4exp/Qwen4ExpProjectionWorkspace.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "planning/MemoryPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <tuple>

namespace
{
using namespace llaminar2;
constexpr auto single = QuantizedActivationOperandPrecision::SingleQ8;
constexpr auto two = QuantizedActivationOperandPrecision::TwoTermQ8;
/** @return Existing source-owned QFN profile with a small explicit native source inventory. */
ModelMemoryProfile source()
{
    auto profile = ModelMemoryProfile::fromGGUF(test::qwen4MetadataFixture(false));
    profile.vocab_size = 32;
    profile.tensors.clear();
    for (const auto &[name, n, k, layer] : {
        std::tuple{"blk.0.attn_qkv.weight", 10240, 2560, 0},
        std::tuple{"blk.0.attn_gate.weight", 6144, 2560, 0},
        std::tuple{"blk.0.hc_attn_up.weight", 10240, 320, 0},
        std::tuple{"blk.1.ssm_out.weight", 2560, 6144, 1},
        std::tuple{"blk.3.attn_output.weight", 2560, 6144, 3}})
        profile.tensors.push_back({.name=name, .native_bytes=static_cast<size_t>(n)*k/32*34,
            .quant_type="Q8_0", .elements=static_cast<size_t>(n)*k, .K=static_cast<size_t>(k), .layer_index=layer});
    for (const auto name : {"blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"})
        profile.tensors.push_back({.name=name, .native_bytes=size_t{48}*2560*sizeof(float),
            .quant_type="F32", .elements=size_t{48}*2560, .K=2560, .layer_index=0});
    return profile;
}
/** @return Exact existing backend correction names absent from native projection scratch. */
WorkspaceRequirements residual(DeviceId device, int rows, int columns)
{
    if (device.is_gpu()) return gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 10240, columns);
    WorkspaceRequirements requirements;
    for (const auto &buffer : cpuQ8ResidualCorrectionWorkspaceRequirements(rows, 10240, columns).buffers)
        if (buffer.name == kCPUProjectionResidualQ8) requirements.buffers.push_back(buffer);
    return requirements;
}
/** @return Complete metadata-only participant geometry using the canonical CPU worker declaration. */
WorkspaceMemoryGeometry geometry(DeviceId device, int rows)
{
    auto value = test::qfnHCWorkspaceGeometry(device, rows);
    value.local_d_ff = 640;
    value.local_query_heads = 24;
    value.local_kv_heads = 2;
    value.local_vocab = 32;
    value.first_layer = 0;
    value.last_layer = 47;
    return value;
}
}

/** @test Graph and planner use their actual original source kinds, excluding FA/predictor and malformed identities. */
TEST(Qwen4ExpGDNQKVWorkspace, OriginalMixerAndRoleOwnSelection)
{
    const Qwen4ExpConfig config(test::qwen4MetadataFixture(false));
    const auto profile = source();
    size_t selected = 0;
    for (uint32_t layer = 0; layer < config.mainLayers(); ++layer)
    {
        const std::string name = "blk." + std::to_string(layer) + ".attn_qkv.weight";
        const auto expected = config.mixers()[layer] == Qwen4ExpMixer::GatedDeltaNet ? two : single;
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnQKVPrecision(two, true, WeightRole::FusedQKV, name, &config), expected);
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnQKVProfilePrecision(two, true, WeightRole::FusedQKV, name, &profile), expected);
        selected += expected == two;
    }
    EXPECT_EQ(selected, 36u);
    for (const auto name : {"blk.48.attn_qkv.weight", "blk.49.attn_qkv.weight", "blk.03.attn_qkv.weight",
        "blk.-1.attn_qkv.weight", "blk.+0.attn_qkv.weight", "blk.x.attn_qkv.weight", "blk.0.attn_qkv.bias",
        "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_out.weight"})
    {
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnQKVPrecision(two, true, WeightRole::FusedQKV, name, &config), single);
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnQKVProfilePrecision(two, true, WeightRole::FusedQKV, name, &profile), single);
    }
    EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnQKVPrecision(two, false, WeightRole::FusedQKV,
        "blk.0.attn_qkv.weight", nullptr), single);
    EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnQKVPrecision(two, true, WeightRole::GDNProjection,
        "blk.0.attn_qkv.weight", &config), single);
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::gdnQKVPrecision(two, true, WeightRole::FusedQKV,
        "blk.0.attn_qkv.weight", nullptr), std::invalid_argument);
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::gdnQKVProfilePrecision(two, true, WeightRole::FusedQKV,
        "blk.0.attn_qkv.weight", nullptr), std::invalid_argument);
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::gdnQKVPrecision(static_cast<QuantizedActivationOperandPrecision>(255),
        false, WeightRole::Other, {}, nullptr), std::invalid_argument);
    EXPECT_EQ(DevicePlanConfig{}.qfn_gdn_qkv_activation, single);
    EXPECT_EQ(WorkspaceMemoryGeometry{}.qfn_gdn_qkv_activation, single);
}

/** @test Only the unchanged complete original native QKV matrix and one request may contribute banks. */
TEST(Qwen4ExpGDNQKVWorkspace, FullSourceFormatAndRequestAreStrict)
{
    const Qwen4ExpConfig config(test::qwen4MetadataFixture(false));
    const auto profile = source();
    const WeightSliceSpec full{.source_rows=10240, .source_cols=2560, .row_count=10240, .col_count=2560};
    for (int mutation = 0; mutation < 9; ++mutation)
    {
        auto slice = full;
        int requests = 1;
        if (mutation == 0) slice.row_start = 1;
        if (mutation == 1) slice.col_start = 1;
        if (mutation == 2) slice.row_count /= 2;
        if (mutation == 3) slice.col_count /= 2;
        if (mutation == 4) slice.source_rows /= 2;
        if (mutation == 5) slice.source_cols /= 2;
        if (mutation == 6) requests = 2;
        if (mutation == 7) slice.expert_count = 1;
        if (mutation == 8) slice.expert_ids = {0};
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::validateGDNQKVInvocation(two, true, WeightRole::FusedQKV,
            "blk.0.attn_qkv.weight", &config, requests, slice), std::invalid_argument);
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::validateGDNQKVProfileInvocation(two, true, WeightRole::FusedQKV,
            "blk.0.attn_qkv.weight", &profile, requests, slice), std::invalid_argument);
    }
    EXPECT_NO_THROW(Qwen4ExpProjectionWorkspace::validateGDNQKVInvocation(two, true, WeightRole::FusedQKV,
        "blk.0.attn_qkv.weight", &config, 1, full));
    for (const auto format : {"F32", "F16", "BF16", "Q4_K", "Q8_K"})
    {
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::requirements(DeviceId::cpu(), single, single, true,
            WeightRole::FusedQKV, "blk.0.attn_qkv.weight", format, 1, 10240, 2560,
            single, 48, two, &profile), std::invalid_argument);
    }
}

/** @test All sixteen choices merge the same source-owned named bank union, never one allocation per layer. */
TEST(Qwen4ExpGDNQKVWorkspace, SixteenSelectionsUseOneCanonicalNamedUnion)
{
    const auto profile = source();
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 15, 512})
    {
        auto plan = geometry(device, rows);
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, plan);
        for (const auto hc : {single, two})
        for (const auto gdn : {single, two})
        for (const auto fa : {single, two})
        for (const auto qkv : {single, two})
        {
            SCOPED_TRACE(::testing::Message() << device.toString() << " M=" << rows << " qkv=" << static_cast<int>(qkv));
            plan.qfn_hc_up_activation = hc;
            plan.qfn_gdn_output_activation = gdn;
            plan.qfn_fa_output_activation = fa;
            plan.qfn_gdn_qkv_activation = qkv;
            WorkspaceRequirements expected;
            if (hc == two) expected.merge(residual(device, rows, 320));
            if (gdn == two || fa == two) expected.merge(residual(device, rows, 6144));
            if (qkv == two) expected.merge(residual(device, rows, 2560));
            EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, plan), ordinary + expected.total_bytes_with_alignment());
        }
    }
}

/** @test Ownership scope and existing prepared representation are checked before the PMA receives capacity. */
TEST(Qwen4ExpGDNQKVWorkspace, OwnedIntervalsRepresentationAndExactFitRemainCanonical)
{
    auto profile = source();
    auto plan = geometry(DeviceId::cpu(), 15);
    plan.first_layer = plan.last_layer = 3;
    const auto absent = WorkspaceMemoryEstimator::estimate(profile, plan);
    plan.qfn_gdn_qkv_activation = two;
    EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, plan), absent);
    plan.first_layer = plan.last_layer = 0;
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        GDNProjectionArithmetic previous;
        ~Restore() { setup.gdn_projection_arithmetic = previous; }
    } restore{setup, setup.gdn_projection_arithmetic};
    for (const auto converted : {GDNProjectionArithmetic::FP16, GDNProjectionArithmetic::FP32})
    {
        setup.gdn_projection_arithmetic = converted;
        EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile, plan), std::invalid_argument);
    }
    setup.gdn_projection_arithmetic = GDNProjectionArithmetic::FP16MoEAlphaBeta;
    const auto bytes = WorkspaceMemoryEstimator::estimate(profile, plan);
    PhysicalMemoryPlanBuilder fit;
    fit.add({.world_rank=0, .device=DeviceId::cpu(), .total_bytes=bytes, .admission_available_bytes=bytes},
        PhysicalMemoryOwner::ExecutionWorkspace, bytes);
    const auto admitted_plan = fit.build();
    EXPECT_TRUE(admitted_plan.fits());
    EXPECT_NO_THROW((void)PhysicalMemoryPlanAdmissionCertificate(admitted_plan));
    ASSERT_GT(bytes, 0u);
    PhysicalMemoryPlanBuilder short_plan;
    short_plan.add({.world_rank=0, .device=DeviceId::cpu(), .total_bytes=bytes-1, .admission_available_bytes=bytes-1},
        PhysicalMemoryOwner::ExecutionWorkspace, bytes);
    EXPECT_FALSE(short_plan.build().fits());
    EXPECT_THROW((void)PhysicalMemoryPlanAdmissionCertificate(short_plan.build()), PhysicalMemoryCapacityExhausted);
    profile.tensors[0].quant_type = "F32";
    EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile, plan), std::invalid_argument);
}

/** @test Floating GDN setup rejects both incompatible correction roles before admitting their banks. */
TEST(Qwen4ExpGDNQKVWorkspace, FloatingGDNRejectsOutputCorrectionBeforeAdmission)
{
    const auto profile = source();
    struct Restore
    {
        GDNProjectionArithmetic previous = debugEnv().gemm.gdn_projection_arithmetic;
        MoERouterArithmetic router = debugEnv().gemm.moe_router_arithmetic;
        ~Restore()
        {
            mutableDebugEnv().gemm.gdn_projection_arithmetic = previous;
            mutableDebugEnv().gemm.moe_router_arithmetic = router;
        }
    } restore;
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const auto arithmetic : {GDNProjectionArithmetic::FP16MoEAlphaBeta, GDNProjectionArithmetic::FP16,
                                     GDNProjectionArithmetic::FP32})
            for (const auto router : {MoERouterArithmetic::Native, MoERouterArithmetic::FP16})
            {
                mutableDebugEnv().gemm.gdn_projection_arithmetic = arithmetic;
                mutableDebugEnv().gemm.moe_router_arithmetic = router;
                auto plan = geometry(device, 9);
                EXPECT_NO_THROW((void)WorkspaceMemoryEstimator::estimate(profile, plan));
                plan.qfn_gdn_output_activation = two;
                if (arithmetic == GDNProjectionArithmetic::FP16MoEAlphaBeta)
                {
                    EXPECT_NO_THROW((void)WorkspaceMemoryEstimator::estimate(profile, plan));
                }
                else
                {
                    try
                    {
                        (void)WorkspaceMemoryEstimator::estimate(profile, plan);
                        FAIL() << "Converted GDN output was admitted as native Q8 correction";
                    }
                    catch (const std::invalid_argument &error)
                    {
                        const std::string message = error.what();
                        EXPECT_NE(message.find("tensor=blk.1.ssm_out.weight"), std::string::npos);
                        EXPECT_NE(message.find(std::string("requested_gdn_arithmetic=") +
                            std::string(gdnProjectionArithmeticName(arithmetic))), std::string::npos);
                    }
                }
            }
}
