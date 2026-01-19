/**
 * @file Test__Qwen2_ROCm_vs_PyTorch.cpp
 * @brief Integration: Qwen2 ROCm Pipeline vs PyTorch Reference
 *
 * ROCm uses similar quantization strategy to CUDA (per-row symmetric INT8).
 * Thresholds are set similarly to CUDA tests.
 *
 * Thresholds: 0.99 avg cosine for first 6 layers, KL < 0.15
 */

#include <gtest/gtest.h>
#include "Qwen2ParityTestBase.h"

#include "backends/ComputeBackend.h"

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================

class Test__Qwen2_ROCm_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    DeviceId gpu_device_{};  // Intentionally invalid - must be set in setupDeviceSpecific()

    BackendThresholds getBackendThresholds() override
    {
        // ROCm uses similar quantization strategy to CUDA.
        // Relax thresholds accordingly.
        return {
            .cosine_threshold = 0.99f,
            .decode_cosine_threshold = 0.985f, // Relaxed for INT8 accumulation error
            .early_layers_count = 6,
            .min_early_layers_passed = 6,
            .kl_threshold = 0.15f};
    }

    void setupDeviceSpecific() override
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "Built without ROCm support";
#else
        // Clear any lingering HIP errors
        (void)hipGetLastError();
        (void)hipDeviceSynchronize();

        auto &dm = DeviceManager::instance();
        dm.initialize(-1);

        int gpu_idx = dm.find_device(ComputeBackendType::GPU_ROCM);
        if (gpu_idx < 0)
        {
            GTEST_SKIP() << "No ROCm device found";
        }

        gpu_device_ = DeviceId::rocm(gpu_idx - 1);
        LOG_INFO("[Qwen2 ROCm Parity] Using ROCm device " << (gpu_idx - 1));
#endif
    }

    DeviceId getDevice() override
    {
        // Fail loudly if setupDeviceSpecific() wasn't called or failed to set device
        EXPECT_TRUE(gpu_device_.is_rocm())
            << "ROCm parity test requires ROCm device but got: " << gpu_device_.to_string();
        return gpu_device_;
    }
    std::string getBackendName() override { return "ROCm"; }
};

INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_ROCm_vs_PyTorch);
