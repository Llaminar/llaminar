/**
 * @file Test__Qwen35MoE_SingleDevice_Parity.cpp
 * @brief Single-device Qwen3.5 MoE parity tests (CPU, CUDA, ROCm)
 *
 * Tests that single-device Qwen3.5 MoE inference produces results matching
 * PyTorch reference outputs. Validates:
 *   - GDN (Gated Delta Network) layer integration (same as dense Qwen3.5)
 *   - Full Attention layer integration (same as dense Qwen3.5)
 *   - MoE Router: softmax top-k expert selection (256 experts, top-8)
 *   - MoE Expert FFN: per-expert SwiGLU + weighted combine
 *   - Shared Expert: always-active dense SwiGLU + sigmoid gate
 *   - MoE Combined Output: routed + gated shared expert
 *   - Residual connections across the MoE FFN block
 *
 * Configurations:
 *   - CPU: Full-precision baseline with FP16 KV cache
 *   - CUDA: Single NVIDIA GPU with the smaller Q3_K_S proving model
 *   - ROCm: Single AMD GPU
 *
 * Model: Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf at /opt/llaminar-models/
 *   - Q4_K/Q5_K/Q6_K quantization (expect wider tolerances than Q8_0)
 *   - 40 layers, 256 experts (top-8), 2048 hidden dim, 512 expert intermediate
 *
 * NOTE: Decode attention drops (0.04-0.12 cosine/layer) are expected due to
 * Llaminar's FP16 KV cache vs PyTorch's FP32 DynamicCache. MoE expert drops
 * (0.01-0.035/layer) are from block-wise GEMM accumulation order differences.
 * End-to-end token predictions match (100% top-1 across 5 decode steps).
 *
 * @author David Sanftenberg
 * @date 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <unistd.h>
#include "Qwen35MoEModelParityDefinitions.h"
#include "Qwen35MoEParityTestBase.h"
#include "collective/BackendRouter.h"
#include "backends/GPUDeviceContextPool.h"

#include <array>
#include <iterator>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen35moe;

/** @return Canonically expanded CPU, CUDA, and ROCm real-weight cases. */
static const std::vector<ModelParityCase> &qwen35MoESingleDeviceCases()
{
    static const auto cases = []
    {
        const std::array definitions = {
            qwen35MoEParityDefinition(
                qwen35MoE35BQ4KXLParityModel(),
                ModelParityTopologyDefinition{
                    .test_id = "CPU0",
                    .kind = ModelParityTopologyKind::SingleDevice,
                    .participants = {{GlobalDeviceAddress::cpu(), 0}},
                },
                qwen35MoESingleDeviceThresholds()),
            qwen35MoEParityDefinition(
                qwen35MoE35BQ4KXLParityModel(),
                ModelParityTopologyDefinition{
                    .test_id = "ROCm0",
                    .kind = ModelParityTopologyKind::SingleDevice,
                    .participants = {{GlobalDeviceAddress::rocm(0), 0}},
                },
                qwen35MoESingleDeviceThresholds()),
            qwen35MoEParityDefinition(
                qwen35MoE35BQ3KSParityModel(),
                ModelParityTopologyDefinition{
                    .test_id = "CUDA0",
                    .kind = ModelParityTopologyKind::SingleDevice,
                    .participants = {{GlobalDeviceAddress::cuda(0), 0}},
                },
                qwen35MoESingleDeviceThresholds()),
        };
        std::vector<ModelParityCase> expanded;
        for (const auto &definition : definitions)
        {
            auto definition_cases =
                expandModelParityDefinition(definition);
            expanded.insert(
                expanded.end(),
                std::make_move_iterator(definition_cases.begin()),
                std::make_move_iterator(definition_cases.end()));
        }
        return expanded;
    }();
    return cases;
}

// =============================================================================
// Parameterized Test Fixture
// =============================================================================

class Qwen35MoESingleDeviceParityTest
    : public Qwen35MoEConfigDrivenParityTest<Qwen35MoESingleDeviceParityTest>,
      public ModelParityCaseParameter
{};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(Qwen35MoESingleDeviceParityTest, ProductionParity)
{
    runProductionParityCampaign();
}

// =============================================================================
// Test Instantiation
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    Qwen35MoE,
    Qwen35MoESingleDeviceParityTest,
    ::testing::ValuesIn(qwen35MoESingleDeviceCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info)
    {
        return info.param.testName();
    });

// =============================================================================
// Custom Main with MPI Initialization
// =============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // CRITICAL: Shutdown GlobalBackendRouter before MPI_Finalize to ensure
    // NCCLCoordinator cleanup happens while CUDA runtime is still active.
    GlobalBackendRouter::shutdown();
    GPUDeviceContextPool::instance().shutdown();

    MPI_Finalize();

    // Skip static destructors — avoid CUDA/ROCm atexit races.
    std::cout.flush();
    std::cerr.flush();
    _exit(result);
}
