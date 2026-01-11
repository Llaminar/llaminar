/**
 * @file Test__Qwen2_CPU_vs_PyTorch.cpp
 * @brief Integration: Qwen2 CPU Pipeline vs PyTorch Reference
 *
 * CPU uses per-block Q8_1 quantization with asymmetric correction (sum_qs),
 * which provides better accuracy than CUDA's per-row symmetric INT8.
 *
 * Thresholds: 0.999 avg cosine for first 4 layers, KL < 0.15
 */

#include <gtest/gtest.h>
#include "Qwen2ParityTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================

class Test__Qwen2_CPU_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    BackendThresholds getBackendThresholds() override
    {
        // CPU uses per-block Q8_1 quantization with asymmetric correction,
        // which provides better accuracy than CUDA's per-row symmetric INT8.
        return {
            .cosine_threshold = 0.999f,
            .decode_cosine_threshold = 0.99f, // Tighter threshold for CPU
            .early_layers_count = 4,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.15f};
    }

    DeviceId getDevice() override { return DeviceId::cpu(); }
    std::string getBackendName() override { return "CPU"; }
};

INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_CPU_vs_PyTorch);
