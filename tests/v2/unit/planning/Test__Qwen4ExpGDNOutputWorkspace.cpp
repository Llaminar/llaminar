/**
 * @file Test__Qwen4ExpGDNOutputWorkspace.cpp
 * @brief Exact role, invocation and shared-bank tests for optional GDN correction.
 *
 * These source-only tests execute no backend. They exercise the same canonical
 * factories used by setup and GEMMStage, including the real K6144 output role
 * and its mutually exclusive reuse with K320 HC-up invocations.
 */
#include "models/qwen4exp/Qwen4ExpProjectionWorkspace.h"
#include "planning/MemoryPlanner.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include <gtest/gtest.h>
#include <map>
using namespace llaminar2;
namespace
{
constexpr auto single = QuantizedActivationOperandPrecision::SingleQ8;
constexpr auto two = QuantizedActivationOperandPrecision::TwoTermQ8;
/** @return Canonical named capacities, independent of ordering in the union. */
std::map<std::string, std::size_t> capacities(const WorkspaceRequirements &value)
{
    std::map<std::string, std::size_t> result;
    for (const auto &buffer : value.buffers) result.emplace(buffer.name, buffer.size_bytes);
    return result;
}
/** @return A role-specific contribution without source conversion or runtime queries. */
WorkspaceRequirements output(DeviceId device, int rows, QuantizedActivationOperandPrecision hc = single, QuantizedActivationOperandPrecision gdn = two)
{
    return Qwen4ExpProjectionWorkspace::requirements(device, hc, gdn, true,
        WeightRole::GDNProjection, "blk.0.ssm_out.weight", "Q8_0", rows, 2560, 6144);
}
}
/** @test The separate policy selects original output weights, never QKV/Z/alpha/beta/HC/router or foreign models. */
TEST(Qwen4ExpGDNOutputWorkspace, ExactOriginalSourceRoleAndDefaultIsolation)
{
    for (const auto name : {"blk.0.ssm_out.weight", "blk.47.ssm_out.weight"})
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnOutputPrecision(two, true, WeightRole::GDNProjection, name), two);
    for (const auto name : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight", "blk.0.ffn_gate_inp.weight", "blk.0.hc_attn_up.weight", "output.weight"})
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnOutputPrecision(two, true, WeightRole::GDNProjection, name), single);
    EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnOutputPrecision(two, false, WeightRole::GDNProjection, "blk.0.ssm_out.weight"), single);
    EXPECT_EQ(Qwen4ExpProjectionWorkspace::gdnOutputPrecision(two, true, WeightRole::Other, "blk.0.ssm_out.weight"), single);
    EXPECT_TRUE(output(DeviceId::cpu(), 512, single, single).buffers.empty());
    EXPECT_TRUE(Qwen4ExpProjectionWorkspace::requirements(DeviceId::cuda(0), single, two, false,
        WeightRole::GDNProjection, "blk.0.ssm_out.weight", "F16", 512, 2560, 6144).buffers.empty());
    EXPECT_EQ(DevicePlanConfig{}.qfn_gdn_output_activation, single);
    EXPECT_EQ(WorkspaceMemoryGeometry{}.qfn_gdn_output_activation, single);
}
/** @test Every admitted bucket uses actual K6144 native banks; no output-sized or per-layer reserve exists. */
TEST(Qwen4ExpGDNOutputWorkspace, ExactNativeBanksAndHCMaximumReuse)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 2, 15, 16, 31, 127, 512, 1024, 2048, 8192})
    {
        SCOPED_TRACE(::testing::Message() << device.toString() << " M=" << rows);
        auto gdn = output(device, rows);
        const auto expected = device.is_cpu() ? cpuQ8ResidualCorrectionWorkspaceRequirements(rows, 2560, 6144)
            : gpuQ8ResidualCorrectionWorkspaceRequirements(rows, 2560, 6144);
        EXPECT_EQ(capacities(gdn), capacities(expected));
        const auto hc = Qwen4ExpProjectionWorkspace::requirements(device, two, two, true,
            WeightRole::HyperConnectionProjection, "blk.0.hc_attn_up.weight", "Q8_0", rows, 10240, 320);
        EXPECT_GT(gdn.total_bytes_with_alignment(), hc.total_bytes_with_alignment());
        gdn.merge(hc); gdn.merge(output(device, rows));
        EXPECT_EQ(capacities(gdn), capacities(expected));
        EXPECT_EQ(gdn.total_bytes_with_alignment(), expected.total_bytes_with_alignment());
        if (device.is_gpu())
        {
            auto residual = gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 10240, 320);
            residual.merge(gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 2560, 6144));
            EXPECT_EQ(capacities(residual), capacities(gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 2560, 6144)));
            gdn.merge(residual);
            EXPECT_EQ(gdn.total_bytes_with_alignment(), expected.total_bytes_with_alignment());
        }
    }
}
/** @test Floating/unknown sources and unsupported physical/count geometry fail before capture. */
TEST(Qwen4ExpGDNOutputWorkspace, PreciseSourceAndInvocationRejection)
{
    for (const auto format : {"F32", "FP32", "F16", "FP16", "BF16", "UNKNOWN", ""})
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::requirements(DeviceId::cpu(), single, two, true,
            WeightRole::GDNProjection, "blk.0.ssm_out.weight", format, 15, 2560, 6144), std::invalid_argument);
    for (const int rows : {0, -1}) EXPECT_THROW(output(DeviceId::cpu(), rows), std::invalid_argument);
    EXPECT_THROW(output(DeviceId::invalid(), 15), std::invalid_argument);
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::requirements(DeviceId::cpu(), single, two, true,
        WeightRole::GDNProjection, "blk.0.ssm_out.weight", "Q8_0", 15, 2560, 6143), std::invalid_argument);
    const WeightSliceSpec complete{.source_rows = 2560, .source_cols = 6144, .row_count = 2560, .col_count = 6144};
    for (const int requests : {0, 2, 15})
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::validateGDNOutputInvocation(two, true,
            WeightRole::GDNProjection, "blk.0.ssm_out.weight", requests, complete), std::invalid_argument);
    for (int mutation = 0; mutation < 5; ++mutation)
    {
        auto partial = complete;
        if (mutation == 0) partial.row_count /= 2;
        if (mutation == 1) partial.col_count /= 2;
        if (mutation == 2) partial.row_start = 1;
        if (mutation == 3) partial.col_start = 1;
        if (mutation == 4) partial.source_rows = 0;
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::validateGDNOutputInvocation(two, true,
            WeightRole::GDNProjection, "blk.0.ssm_out.weight", 1, partial), std::invalid_argument);
    }
    EXPECT_NO_THROW(Qwen4ExpProjectionWorkspace::validateGDNOutputInvocation(two, true,
        WeightRole::GDNProjection, "blk.0.ssm_out.weight", 1, complete));
    EXPECT_NO_THROW(Qwen4ExpProjectionWorkspace::validateGDNOutputInvocation(single, true,
        WeightRole::GDNProjection, "blk.0.ssm_out.weight", 15, complete));
    EXPECT_NO_THROW(Qwen4ExpProjectionWorkspace::validateGDNOutputInvocation(two, true,
        WeightRole::HyperConnectionProjection, "blk.48.nextn.hc_head_up.weight", 15, complete));
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::selected(single,
        static_cast<QuantizedActivationOperandPrecision>(255), true, WeightRole::Other, {}), std::invalid_argument);
}

/** @test Every canonical native source owns the exact original output banks on all backends. */
TEST(Qwen4ExpGDNOutputWorkspace, AllNativeSourceFormatsUseCanonicalCorrectionBanks)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 15, 512})
    for (const auto &source : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(::testing::Message() << source.quant_type << " " << device.toString() << " M=" << rows);
        const auto actual = Qwen4ExpProjectionWorkspace::requirements(device, single, two, true,
            WeightRole::GDNProjection, "blk.0.ssm_out.weight", source.quant_type, rows, 2560, 6144);
        const auto expected = device.is_cpu() ? cpuQ8ResidualCorrectionWorkspaceRequirements(rows, 2560, 6144)
            : gpuQ8ResidualCorrectionWorkspaceRequirements(rows, 2560, 6144);
        EXPECT_EQ(actual.buffers.size(), expected.buffers.size());
        EXPECT_EQ(capacities(actual), capacities(expected));
        EXPECT_EQ(actual.total_bytes_with_alignment(), expected.total_bytes_with_alignment());
        EXPECT_TRUE(Qwen4ExpProjectionWorkspace::requirements(device, single, single, true,
            WeightRole::GDNProjection, "blk.0.ssm_out.weight", source.quant_type, rows, 2560, 6144).buffers.empty());
    }
}
