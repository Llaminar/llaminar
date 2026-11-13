/**
 * @file Test__CUDAHCUpWorkspaceAdmission.cpp
 * @brief Local CUDA dispatch merges HC residual banks once into its real BOM.
 *
 * This explicit device integration complements metadata-only Unit coverage.
 * It asks the actual CUDA launch policy for ordinary projection requirements;
 * no model file, allocation reserve or synthetic dispatch is substituted.
 */
#include "../../utils/Qwen4ExpHCWorkspaceFixture.h"
#include "backends/BackendManager.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include <gtest/gtest.h>

using namespace llaminar2;

/** @test Native dispatch includes the exact residual increment without charging primary banks twice. */
TEST(CUDAHCUpWorkspaceAdmission, RuntimePolicyMergesResidualBanksOnce)
{
    ASSERT_NE(getCUDABackend(), nullptr);
    for (const int rows : {1, 15, 512, 1024, 2048})
    {
        SCOPED_TRACE(rows);
        auto profile = test::qfnHCWorkspaceProfile();
        auto geometry = test::qfnHCWorkspaceGeometry(DeviceId::cuda(0), rows);
        geometry.runtime_device_policy_available = true;
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, geometry);
        geometry.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
        const auto corrected = WorkspaceMemoryEstimator::estimate(profile, geometry);
        const auto residual = gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(rows, 512, 32);
        EXPECT_EQ(corrected, ordinary + residual.total_bytes_with_alignment());
        auto second = profile.tensors.front();
        second.name = "blk.1.hc_ffn_up.weight";
        second.layer_index = 1;
        profile.tensors.push_back(second);
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry), corrected);
    }
}
