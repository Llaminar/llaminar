/**
 * @file Test__Qwen4ExpHCUpWorkspaceAdmission.cpp
 * @brief Real estimator coverage for native HC correction banks and ownership.
 *
 * These device-free tests compare the full production BOM against the actual
 * stage factory. CUDA exercises remote metadata planning; its runtime policy
 * is covered separately by an explicit native CUDA integration test.
 */
#include "../../utils/Qwen4ExpHCWorkspaceFixture.h"
#include "models/qwen4exp/Qwen4ExpHCUpWorkspace.h"
#include <gtest/gtest.h>
#include <array>

using namespace llaminar2;

/** @test Every backend admits precisely one new residual bank family across serial layers. */
TEST(Qwen4ExpHCUpWorkspaceAdmission, ExactIncrementAndSerialNameReuse)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 15, 512, 1024, 2048})
    {
        SCOPED_TRACE(::testing::Message() << device.toString() << " rows=" << rows);
        auto profile = test::qfnHCWorkspaceProfile();
        auto geometry = test::qfnHCWorkspaceGeometry(device, rows);
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, geometry);
        geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
        const auto corrected = WorkspaceMemoryEstimator::estimate(profile, geometry);
        WorkspaceRequirements residual;
        if (device.is_cpu())
        {
            const auto complete = cpuQ8ResidualCorrectionWorkspaceRequirements(rows, 512, 32);
            for (const auto& buffer : complete.buffers)
                if (buffer.name == kCPUProjectionResidualQ8)
                    residual.buffers.push_back(buffer);
        }
        else
            residual = gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 512, 32);
        EXPECT_EQ(corrected, ordinary + residual.total_bytes_with_alignment());
        auto second = profile.tensors.front();
        second.name = "blk.1.hc_ffn_up.weight";
        second.layer_index = 1;
        profile.tensors.push_back(second);
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), corrected);
        geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), ordinary);
    }
}

/**
 * @test ROCm uses one exact named-bank factory independently of live-policy availability.
 *
 * ROCm's ordinary workspace descriptor is codebook independent and device free.
 * Unlike CUDA, setting runtime_device_policy_available does not select another
 * query or residual reserve. Both flags must admit precisely the same union.
 */
TEST(Qwen4ExpHCUpWorkspaceAdmission, ROCmRuntimeFlagDoesNotAddASecondResidualReserve)
{
    for (const int rows : {1, 15, 512, 1024, 2048})
    {
        SCOPED_TRACE(rows);
        std::array<std::size_t, 2> ordinary{};
        std::array<std::size_t, 2> corrected{};
        for (std::size_t policy = 0; policy < ordinary.size(); ++policy)
        {
            SCOPED_TRACE(policy);
            auto profile = test::qfnHCWorkspaceProfile();
            auto geometry = test::qfnHCWorkspaceGeometry(DeviceId::rocm(0), rows);
            geometry.runtime_device_policy_available = policy != 0;
            ordinary[policy] = WorkspaceMemoryEstimator::estimate(profile, geometry);
            geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
            corrected[policy] = WorkspaceMemoryEstimator::estimate(profile, geometry);
            const auto residual = gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 512, 32);
            EXPECT_EQ(corrected[policy], ordinary[policy] + residual.total_bytes_with_alignment());

            auto second = profile.tensors.front();
            second.name = "blk.1.hc_ffn_up.weight";
            second.layer_index = 1;
            profile.tensors.push_back(second);
            EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), corrected[policy]);
            geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
            EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), ordinary[policy]);
        }
        EXPECT_EQ(ordinary[0], ordinary[1]);
        EXPECT_EQ(corrected[0], corrected[1]);
    }
}

/** @test Source validation rejects selected foreign formats but respects participant ownership. */
TEST(Qwen4ExpHCUpWorkspaceAdmission, NativeSourceValidationPrecedesAdmission)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        auto profile = test::qfnHCWorkspaceProfile();
        auto geometry = test::qfnHCWorkspaceGeometry(device, 512);
        geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
        profile.tensors.front().quant_type = "F16";
        EXPECT_THROW(WorkspaceMemoryEstimator::estimate(profile, geometry), std::invalid_argument);
        geometry.first_layer = 1;
        EXPECT_NO_THROW(WorkspaceMemoryEstimator::estimate(profile, geometry));
    }
}

#ifndef HAVE_CUDA
/** @test Metadata-only CPU builds cannot lose residual banks through an unavailable CUDA query. */
TEST(Qwen4ExpHCUpWorkspaceAdmission, BuildWithoutCUDAStillPricesRemoteCorrection)
{
    auto profile = test::qfnHCWorkspaceProfile();
    auto geometry = test::qfnHCWorkspaceGeometry(DeviceId::cuda(0), 512);
    geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    const auto remote = WorkspaceMemoryEstimator::estimate(profile, geometry);
    geometry.runtime_device_policy_available = true;
    EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), remote);
}
#endif
