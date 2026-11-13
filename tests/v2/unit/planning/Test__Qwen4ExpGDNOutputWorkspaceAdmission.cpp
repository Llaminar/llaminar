/**
 * @file Test__Qwen4ExpGDNOutputWorkspaceAdmission.cpp
 * @brief Complete estimator coverage for exact GDN correction banks and ownership.
 *
 * Original source metadata contains K320 HC-up and K6144 GDN output weights.
 * Serial per-layer calls reuse one participant workspace. This suite verifies
 * both the metadata-only CUDA residual addition and complete CPU/ROCm unions,
 * without preparing weights or discovering a backend.
 */
#include "../../utils/Qwen4ExpHCWorkspaceFixture.h"
#include "models/qwen4exp/Qwen4ExpProjectionWorkspace.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace
{
constexpr auto single = QuantizedActivationOperandPrecision::SingleQ8;
constexpr auto two = QuantizedActivationOperandPrecision::TwoTermQ8;
/** @return Source geometry for actual output/HC projection widths and a predictor without GDN. */
ModelMemoryProfile source()
{
    auto profile = test::qfnHCWorkspaceProfile();
    profile.n_layers = 2;
    profile.mtp_layer_count = 1;
    profile.flash_next_predictor = Qwen4ExpPredictorMetadata{
        2, Qwen4ExpPredictorAttention::Dense, Qwen4ExpPredictorHiddenNorm::PerBranch,
        Qwen4ExpPredictorSource::SharedGGUF};
    profile.d_model = 2560;
    profile.d_ff = 640;
    profile.n_heads = 24;
    profile.n_kv_heads = 2;
    profile.head_dim = 256;
    profile.gdn_group_count = 16;
    profile.gdn_time_step_rank = 48;
    profile.gdn_state_size = 128;
    profile.gdn_inner_size = 6144;
    profile.gdn_conv_kernel_size = 4;
    profile.flash_next->residual_width = 10240;
    profile.flash_next->terminal_head_width = 2560;
    profile.flash_next->residual_gate_rank = 320;
    profile.tensors.clear();
    profile.tensors.push_back({.name = "blk.0.hc_attn_up.weight", .native_bytes = 10240ull * 320 / 32 * 34,
        .quant_type = "Q8_0", .elements = 10240ull * 320, .K = 320, .layer_index = 0});
    profile.tensors.push_back({.name = "blk.1.ssm_out.weight", .native_bytes = 2560ull * 6144 / 32 * 34,
        .quant_type = "Q8_0", .elements = 2560ull * 6144, .K = 6144, .layer_index = 1});
    auto predictor = profile.tensors.front();
    predictor.name = "blk.2.nextn.hc_head_up.weight";
    predictor.layer_index = 2;
    profile.tensors.push_back(predictor);
    return profile;
}
/** @return Admitted rows and source-local dense geometry with no runtime queries. */
WorkspaceMemoryGeometry geometry(DeviceId device, int rows)
{
    auto value = test::qfnHCWorkspaceGeometry(device, rows);
    value.local_d_ff = 640;
    value.local_query_heads = 24;
    value.local_kv_heads = 2;
    value.last_layer = 2;
    value.mtp_target_query_rows = 15;
    value.mtp_dense_layout = MTPShiftedKVHeadLayout::FullModelPerParticipant;
    return value;
}
/** @return Only newly required banks beyond the normal single-term stage. */
WorkspaceRequirements residual(DeviceId device, int rows, int k)
{
    if (device.is_gpu()) return gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 2560, k);
    WorkspaceRequirements result;
    for (const auto &buffer : cpuQ8ResidualCorrectionWorkspaceRequirements(rows, 2560, k).buffers)
        if (buffer.name == kCPUProjectionResidualQ8) result.buffers.push_back(buffer);
    return result;
}
}
/** @test Independent selected sources merge exactly once at the maximum live operation's admitted K. */
TEST(Qwen4ExpGDNOutputWorkspaceAdmission, CompleteEstimatorMergesActualSourceBanks)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 15, 512, 1024, 2048})
    for (const bool runtime_available : {false, true})
    {
        // A device-free CUDA test must never impersonate a live context. ROCm's
        // codebook-independent factory supports both flags without any query.
        if (device.is_cuda() && runtime_available) continue;
        SCOPED_TRACE(::testing::Message() << device.toString() << " M=" << rows << " runtime=" << runtime_available);
        auto profile = source();
        auto plan = geometry(device, rows);
        plan.runtime_device_policy_available = runtime_available;
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, plan);
        plan.qfn_gdn_output_activation = two;
        const auto gdn = WorkspaceMemoryEstimator::estimate(profile, plan);
        EXPECT_EQ(gdn, ordinary + residual(device, rows, 6144).total_bytes_with_alignment());
        plan.qfn_hc_up_activation = two;
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, plan), gdn);
        plan.qfn_gdn_output_activation = single;
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, plan),
            ordinary + residual(device, std::max(rows, 15), 320).total_bytes_with_alignment());
        plan.qfn_hc_up_activation = single;
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, plan), ordinary);
    }
}
/** @test FA/HC-only and predictor-only PP intervals gain no phantom output correction banks. */
TEST(Qwen4ExpGDNOutputWorkspaceAdmission, OwnedLayerIntervalControlsContribution)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int layer : {0, 1, 2})
    {
        auto profile = source();
        auto plan = geometry(device, 512);
        plan.first_layer = plan.last_layer = layer;
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, plan);
        plan.qfn_gdn_output_activation = two;
        const auto selected = WorkspaceMemoryEstimator::estimate(profile, plan);
        EXPECT_EQ(selected, ordinary + (layer == 1 ? residual(device, 512, 6144).total_bytes_with_alignment() : 0u));
        // Source validation is ownership-specific: a foreign format outside the
        // exact local interval is not consumed by this participant.
        profile.tensors[1].quant_type = "F16";
        if (layer == 1) EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile, plan), std::invalid_argument);
        else EXPECT_NO_THROW(WorkspaceMemoryEstimator::estimate(profile, plan));
    }
}
/** @test Reject request holes and genuinely partitioned source extents, while ordinary defaults remain unchanged. */
TEST(Qwen4ExpGDNOutputWorkspaceAdmission, RequestGeometryAndDenseSlicesFailBeforeAdmission)
{
    auto profile = source();
    auto plan = geometry(DeviceId::rocm(0), 15);
    plan.qfn_gdn_output_activation = two;
    plan.batch_size = 2;
    EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile, plan), std::invalid_argument);
    plan.batch_size = 1;
    plan.total_shards = 2;
    plan.local_query_heads = 12;
    plan.local_kv_heads = 1;
    EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile, plan), std::invalid_argument);
    plan.qfn_gdn_output_activation = single;
    EXPECT_NO_THROW(WorkspaceMemoryEstimator::estimate(profile, plan));
}
