/**
 * @file Test__Qwen2_CUDA_vs_PyTorch.cpp
 * @brief Integration: Qwen2 CUDA Pipeline vs PyTorch Reference
 *
 * CUDA uses per-row symmetric INT8 quantization (CUTLASS), which accumulates
 * error faster than CPU's per-block Q8_1. Relaxed thresholds account for this.
 *
 * Thresholds: 0.99 avg cosine for first 6 layers, KL < 0.15
 */

#include <gtest/gtest.h>
#include "Qwen2ParityTestBase.h"

#include "backends/ComputeBackend.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen2;

// =============================================================================

class Test__Qwen2_CUDA_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    DeviceId gpu_device_ = DeviceId::cpu();

    BackendThresholds getBackendThresholds() override
    {
        // CUDA uses per-row symmetric INT8 quantization which accumulates more
        // error than CPU's per-block Q8_1. Relax thresholds accordingly.
        return {
            .cosine_threshold = 0.99f,
            .decode_cosine_threshold = 0.985f, // Relaxed for INT8 accumulation error
            .early_layers_count = 6,
            .min_early_layers_passed = 6,
            .kl_threshold = 0.15f};
    }

    void setupDeviceSpecific() override
    {
#ifndef HAVE_CUDA
        GTEST_SKIP() << "Built without CUDA support";
#else
        // Clear any lingering CUDA errors
        cudaGetLastError();
        cudaDeviceSynchronize();

        auto &dm = DeviceManager::instance();
        dm.initialize(-1);

        if (!dm.has_gpu())
        {
            GTEST_SKIP() << "No CUDA GPU available";
        }

        int gpu_idx = dm.find_device(ComputeBackendType::GPU_CUDA);
        if (gpu_idx < 0)
        {
            GTEST_SKIP() << "No CUDA device found";
        }

        gpu_device_ = DeviceId::cuda(gpu_idx - 1);
        LOG_INFO("[Qwen2 CUDA Parity] Using CUDA device " << (gpu_idx - 1));
#endif
    }

    DeviceId getDevice() override { return gpu_device_; }
    std::string getBackendName() override { return "CUDA"; }
};

INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_CUDA_vs_PyTorch);
