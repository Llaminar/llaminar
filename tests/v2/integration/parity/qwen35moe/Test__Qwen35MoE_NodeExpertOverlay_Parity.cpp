/**
 * @file Test__Qwen35MoE_NodeExpertOverlay_Parity.cpp
 * @brief 35B model registration for the generated node ExpertOverlay matrix.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "node_overlay/NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

INSTANTIATE_TEST_SUITE_P(
    Qwen35_35B_ExpertOverlay,
    Qwen35MoENodeExpertOverlayParityTest,
    ::testing::ValuesIn(qwen35GraphNativeParityCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    { return info.param.testName(); });

}
