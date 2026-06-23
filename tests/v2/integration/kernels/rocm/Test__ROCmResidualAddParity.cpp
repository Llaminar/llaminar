/**
 * @file Test__ROCmResidualAddParity.cpp
 * @brief Parity tests for ROCm ResidualAdd kernel vs CPU reference
 *
 * **Purpose**: Validate that ROCm ResidualAdd kernels produce numerically equivalent
 * results to CPU kernels with high cosine similarity (>= 0.99999).
 *
 * **Tests**:
 * - ROCmResidualAddKernelT vs CPUResidualAddKernelT (FP32, BF16, FP16)
 * - Small (4×64) and large (32×896) tensor sizes
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.99999 (very high correlation - elementwise add is simple)
 * - No NaN/Inf in outputs
 * - Relative L2 error < 0.1% for FP32, < 1% for BF16/FP16
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/ops/ROCmResidualAddKernelT.h"
#include "kernels/cpu/ops/CPUResidualAddKernelT.h"
#endif

#include "../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    bool hasROCm()
    {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
#else
        return false;
#endif
    }

#define SKIP_IF_NO_ROCM()                                           \
    do                                                              \
    {                                                               \
        if (!hasROCm())                                             \
        {                                                           \
            GTEST_SKIP() << "No ROCm GPU available, skipping test"; \
        }                                                           \
    } while (0)

    // ============================================================================
    // Similarity Utilities
    // ============================================================================

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
            return 0.0;
        return dot / denom;
    }

    double relativeL2Error(const float *actual, const float *expected, size_t count)
    {
        double diff_norm = 0.0, expected_norm = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = actual[i] - expected[i];
            diff_norm += diff * diff;
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (expected_norm < 1e-12)
            return 0.0;
        return std::sqrt(diff_norm / expected_norm);
    }

    bool hasNanOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
                return true;
        }
        return false;
    }

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class ROCmResidualAddParityTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            SKIP_IF_NO_ROCM();
        }

#ifdef HAVE_ROCM
        // Helper to allocate GPU memory and copy data
        float *allocGpuFloat(size_t count)
        {
            float *d_ptr = nullptr;
            (void)hipMalloc(&d_ptr, count * sizeof(float));
            return d_ptr;
        }

        uint16_t *allocGpuU16(size_t count)
        {
            uint16_t *d_ptr = nullptr;
            (void)hipMalloc(&d_ptr, count * sizeof(uint16_t));
            return d_ptr;
        }

        void copyToGpu(float *d_dst, const float *h_src, size_t count)
        {
            (void)hipMemcpy(d_dst, h_src, count * sizeof(float), hipMemcpyHostToDevice);
        }

        void copyToGpu(uint16_t *d_dst, const uint16_t *h_src, size_t count)
        {
            (void)hipMemcpy(d_dst, h_src, count * sizeof(uint16_t), hipMemcpyHostToDevice);
        }

        void copyFromGpu(float *h_dst, const float *d_src, size_t count)
        {
            (void)hipMemcpy(h_dst, d_src, count * sizeof(float), hipMemcpyDeviceToHost);
        }

        void copyFromGpu(uint16_t *h_dst, const uint16_t *d_src, size_t count)
        {
            (void)hipMemcpy(h_dst, d_src, count * sizeof(uint16_t), hipMemcpyDeviceToHost);
        }

        void freeGpu(void *d_ptr)
        {
            if (d_ptr)
                (void)hipFree(d_ptr);
        }
#endif
    };

    // ============================================================================
    // FP32 Tests
    // ============================================================================

#ifdef HAVE_ROCM

    TEST_F(ROCmResidualAddParityTest, FP32_SmallTensor)
    {
        const int rows = 4;
        const int cols = 64;
        const size_t num_elements = rows * cols;

        // Create random test data
        auto input = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto residual = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output = TestTensorFactory::createFP32({rows, cols});
        auto rocm_output = TestTensorFactory::createFP32({rows, cols});

        // CPU reference
        llaminar2::CPUResidualAddKernelT<ActivationPrecision::FP32> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply(
            input->data(), residual->data(), cpu_output->mutable_data(),
            num_elements, nullptr, -1));

        // ROCm kernel
        float *d_input = allocGpuFloat(num_elements);
        float *d_residual = allocGpuFloat(num_elements);
        float *d_output = allocGpuFloat(num_elements);

        copyToGpu(d_input, input->data(), num_elements);
        copyToGpu(d_residual, residual->data(), num_elements);

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP32> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply(d_input, d_residual, d_output, num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        copyFromGpu(rocm_output->mutable_data(), d_output, num_elements);

        // Validate
        EXPECT_FALSE(hasNanOrInf(rocm_output->data(), num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(rocm_output->data(), cpu_output->data(), num_elements);
        double rel_error = relativeL2Error(rocm_output->data(), cpu_output->data(), num_elements);

        std::cout << "[FP32_SmallTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.99999) << "Cosine similarity too low";
        EXPECT_LE(rel_error, 0.001) << "Relative L2 error too high";

        freeGpu(d_input);
        freeGpu(d_residual);
        freeGpu(d_output);
    }

    TEST_F(ROCmResidualAddParityTest, FP32_LargeTensor)
    {
        const int rows = 32;
        const int cols = 896;
        const size_t num_elements = rows * cols;

        auto input = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto residual = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output = TestTensorFactory::createFP32({rows, cols});
        auto rocm_output = TestTensorFactory::createFP32({rows, cols});

        // CPU reference
        llaminar2::CPUResidualAddKernelT<ActivationPrecision::FP32> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply(
            input->data(), residual->data(), cpu_output->mutable_data(),
            num_elements, nullptr, -1));

        // ROCm kernel
        float *d_input = allocGpuFloat(num_elements);
        float *d_residual = allocGpuFloat(num_elements);
        float *d_output = allocGpuFloat(num_elements);

        copyToGpu(d_input, input->data(), num_elements);
        copyToGpu(d_residual, residual->data(), num_elements);

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP32> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply(d_input, d_residual, d_output, num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        copyFromGpu(rocm_output->mutable_data(), d_output, num_elements);

        EXPECT_FALSE(hasNanOrInf(rocm_output->data(), num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(rocm_output->data(), cpu_output->data(), num_elements);
        double rel_error = relativeL2Error(rocm_output->data(), cpu_output->data(), num_elements);

        std::cout << "[FP32_LargeTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.99999) << "Cosine similarity too low";
        EXPECT_LE(rel_error, 0.001) << "Relative L2 error too high";

        freeGpu(d_input);
        freeGpu(d_residual);
        freeGpu(d_output);
    }

    // ============================================================================
    // BF16 Tests
    // ============================================================================

    TEST_F(ROCmResidualAddParityTest, BF16_SmallTensor)
    {
        const int rows = 4;
        const int cols = 64;
        const size_t num_elements = rows * cols;

        // Create BF16 test data
        auto input_bf16 = TestTensorFactory::createBF16Random({rows, cols}, -1.0f, 1.0f);
        auto residual_bf16 = TestTensorFactory::createBF16Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output_bf16 = TestTensorFactory::createBF16({rows, cols});
        auto rocm_output_bf16 = TestTensorFactory::createBF16({rows, cols});

        // CPU reference
        llaminar2::CPUResidualAddKernelT<ActivationPrecision::BF16> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply_bf16(
            input_bf16->typed_data(), residual_bf16->typed_data(), cpu_output_bf16->mutable_typed_data(),
            num_elements, nullptr, -1));

        // ROCm kernel
        uint16_t *d_input = allocGpuU16(num_elements);
        uint16_t *d_residual = allocGpuU16(num_elements);
        uint16_t *d_output = allocGpuU16(num_elements);

        copyToGpu(d_input, input_bf16->typed_data(), num_elements);
        copyToGpu(d_residual, residual_bf16->typed_data(), num_elements);

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::BF16> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_bf16(d_input, d_residual, d_output, num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        copyFromGpu(rocm_output_bf16->mutable_typed_data(), d_output, num_elements);

        // Dequantize for comparison
        std::vector<float> cpu_fp32(num_elements);
        std::vector<float> rocm_fp32(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
        {
            cpu_fp32[i] = cpu_output_bf16->fp32_data()[i];
            rocm_fp32[i] = rocm_output_bf16->fp32_data()[i];
        }

        EXPECT_FALSE(hasNanOrInf(rocm_fp32.data(), num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(rocm_fp32.data(), cpu_fp32.data(), num_elements);
        double rel_error = relativeL2Error(rocm_fp32.data(), cpu_fp32.data(), num_elements);

        std::cout << "[BF16_SmallTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.9999) << "Cosine similarity too low";
        EXPECT_LE(rel_error, 0.01) << "Relative L2 error too high";

        freeGpu(d_input);
        freeGpu(d_residual);
        freeGpu(d_output);
    }

    TEST_F(ROCmResidualAddParityTest, BF16_LargeTensor)
    {
        const int rows = 32;
        const int cols = 896;
        const size_t num_elements = rows * cols;

        auto input_bf16 = TestTensorFactory::createBF16Random({rows, cols}, -1.0f, 1.0f);
        auto residual_bf16 = TestTensorFactory::createBF16Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output_bf16 = TestTensorFactory::createBF16({rows, cols});
        auto rocm_output_bf16 = TestTensorFactory::createBF16({rows, cols});

        llaminar2::CPUResidualAddKernelT<ActivationPrecision::BF16> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply_bf16(
            input_bf16->typed_data(), residual_bf16->typed_data(), cpu_output_bf16->mutable_typed_data(),
            num_elements, nullptr, -1));

        uint16_t *d_input = allocGpuU16(num_elements);
        uint16_t *d_residual = allocGpuU16(num_elements);
        uint16_t *d_output = allocGpuU16(num_elements);

        copyToGpu(d_input, input_bf16->typed_data(), num_elements);
        copyToGpu(d_residual, residual_bf16->typed_data(), num_elements);

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::BF16> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_bf16(d_input, d_residual, d_output, num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        copyFromGpu(rocm_output_bf16->mutable_typed_data(), d_output, num_elements);

        std::vector<float> cpu_fp32(num_elements);
        std::vector<float> rocm_fp32(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
        {
            cpu_fp32[i] = cpu_output_bf16->fp32_data()[i];
            rocm_fp32[i] = rocm_output_bf16->fp32_data()[i];
        }

        EXPECT_FALSE(hasNanOrInf(rocm_fp32.data(), num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(rocm_fp32.data(), cpu_fp32.data(), num_elements);
        double rel_error = relativeL2Error(rocm_fp32.data(), cpu_fp32.data(), num_elements);

        std::cout << "[BF16_LargeTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.9999) << "Cosine similarity too low";
        EXPECT_LE(rel_error, 0.01) << "Relative L2 error too high";

        freeGpu(d_input);
        freeGpu(d_residual);
        freeGpu(d_output);
    }

    // ============================================================================
    // FP16 Tests
    // ============================================================================

    TEST_F(ROCmResidualAddParityTest, FP16_SmallTensor)
    {
        const int rows = 4;
        const int cols = 64;
        const size_t num_elements = rows * cols;

        auto input_fp16 = TestTensorFactory::createFP16Random({rows, cols}, -1.0f, 1.0f);
        auto residual_fp16 = TestTensorFactory::createFP16Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output_fp16 = TestTensorFactory::createFP16({rows, cols});
        auto rocm_output_fp16 = TestTensorFactory::createFP16({rows, cols});

        llaminar2::CPUResidualAddKernelT<ActivationPrecision::FP16> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply_fp16(
            input_fp16->typed_data(), residual_fp16->typed_data(), cpu_output_fp16->mutable_typed_data(),
            num_elements, nullptr, -1));

        uint16_t *d_input = allocGpuU16(num_elements);
        uint16_t *d_residual = allocGpuU16(num_elements);
        uint16_t *d_output = allocGpuU16(num_elements);

        copyToGpu(d_input, input_fp16->typed_data(), num_elements);
        copyToGpu(d_residual, residual_fp16->typed_data(), num_elements);

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP16> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_fp16(d_input, d_residual, d_output, num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        copyFromGpu(rocm_output_fp16->mutable_typed_data(), d_output, num_elements);

        std::vector<float> cpu_fp32(num_elements);
        std::vector<float> rocm_fp32(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
        {
            cpu_fp32[i] = cpu_output_fp16->fp32_data()[i];
            rocm_fp32[i] = rocm_output_fp16->fp32_data()[i];
        }

        EXPECT_FALSE(hasNanOrInf(rocm_fp32.data(), num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(rocm_fp32.data(), cpu_fp32.data(), num_elements);
        double rel_error = relativeL2Error(rocm_fp32.data(), cpu_fp32.data(), num_elements);

        std::cout << "[FP16_SmallTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.9999) << "Cosine similarity too low";
        EXPECT_LE(rel_error, 0.01) << "Relative L2 error too high";

        freeGpu(d_input);
        freeGpu(d_residual);
        freeGpu(d_output);
    }

    TEST_F(ROCmResidualAddParityTest, FP16_LargeTensor)
    {
        const int rows = 32;
        const int cols = 896;
        const size_t num_elements = rows * cols;

        auto input_fp16 = TestTensorFactory::createFP16Random({rows, cols}, -1.0f, 1.0f);
        auto residual_fp16 = TestTensorFactory::createFP16Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output_fp16 = TestTensorFactory::createFP16({rows, cols});
        auto rocm_output_fp16 = TestTensorFactory::createFP16({rows, cols});

        llaminar2::CPUResidualAddKernelT<ActivationPrecision::FP16> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply_fp16(
            input_fp16->typed_data(), residual_fp16->typed_data(), cpu_output_fp16->mutable_typed_data(),
            num_elements, nullptr, -1));

        uint16_t *d_input = allocGpuU16(num_elements);
        uint16_t *d_residual = allocGpuU16(num_elements);
        uint16_t *d_output = allocGpuU16(num_elements);

        copyToGpu(d_input, input_fp16->typed_data(), num_elements);
        copyToGpu(d_residual, residual_fp16->typed_data(), num_elements);

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP16> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_fp16(d_input, d_residual, d_output, num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        copyFromGpu(rocm_output_fp16->mutable_typed_data(), d_output, num_elements);

        std::vector<float> cpu_fp32(num_elements);
        std::vector<float> rocm_fp32(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
        {
            cpu_fp32[i] = cpu_output_fp16->fp32_data()[i];
            rocm_fp32[i] = rocm_output_fp16->fp32_data()[i];
        }

        EXPECT_FALSE(hasNanOrInf(rocm_fp32.data(), num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(rocm_fp32.data(), cpu_fp32.data(), num_elements);
        double rel_error = relativeL2Error(rocm_fp32.data(), cpu_fp32.data(), num_elements);

        std::cout << "[FP16_LargeTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.9999) << "Cosine similarity too low";
        EXPECT_LE(rel_error, 0.01) << "Relative L2 error too high";

        freeGpu(d_input);
        freeGpu(d_residual);
        freeGpu(d_output);
    }

    // ============================================================================
    // apply_tensor() API Tests
    // ============================================================================
    // These tests exercise the TensorBase* API (apply_tensor) which is what the
    // actual pipeline uses. This is critical because the raw pointer apply() API
    // tests might pass while apply_tensor() has bugs (e.g., using data() instead
    // of active_data_ptr() for GPU tensors).

    TEST_F(ROCmResidualAddParityTest, FP32_ApplyTensor_SmallTensor)
    {
        const int rows = 4;
        const int cols = 64;
        const size_t num_elements = rows * cols;

        // Create tensors on CPU
        auto input = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto residual = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output = TestTensorFactory::createFP32({rows, cols});
        auto rocm_output = TestTensorFactory::createFP32({rows, cols});

        // CPU reference using raw apply()
        llaminar2::CPUResidualAddKernelT<ActivationPrecision::FP32> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply(
            input->data(), residual->data(), cpu_output->mutable_data(),
            num_elements, nullptr, -1));

        // Upload tensors to GPU using ensureOnDevice (as the pipeline does)
        DeviceId rocm_device = DeviceId::rocm(0);
        ASSERT_TRUE(input->ensureOnDevice(rocm_device));
        ASSERT_TRUE(residual->ensureOnDevice(rocm_device));
        ASSERT_TRUE(rocm_output->ensureOnDevice(rocm_device));

        // ROCm kernel using apply_tensor() API (what the pipeline actually uses)
        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP32> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_tensor(
            input.get(), residual.get(), rocm_output.get(),
            num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();

        // Mark output dirty and sync back to host
        rocm_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *result = rocm_output->data(); // This syncs GPU→host

        EXPECT_FALSE(hasNanOrInf(result, num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(result, cpu_output->data(), num_elements);
        double rel_error = relativeL2Error(result, cpu_output->data(), num_elements);

        std::cout << "[FP32_ApplyTensor_SmallTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.99999) << "Cosine similarity too low - apply_tensor() may have pointer issues";
        EXPECT_LE(rel_error, 0.001) << "Relative L2 error too high";
    }

    TEST_F(ROCmResidualAddParityTest, FP32_ApplyTensor_LargeTensor)
    {
        const int rows = 32;
        const int cols = 896;
        const size_t num_elements = rows * cols;

        auto input = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto residual = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f);
        auto cpu_output = TestTensorFactory::createFP32({rows, cols});
        auto rocm_output = TestTensorFactory::createFP32({rows, cols});

        llaminar2::CPUResidualAddKernelT<ActivationPrecision::FP32> cpu_kernel;
        ASSERT_TRUE(cpu_kernel.apply(
            input->data(), residual->data(), cpu_output->mutable_data(),
            num_elements, nullptr, -1));

        DeviceId rocm_device = DeviceId::rocm(0);
        ASSERT_TRUE(input->ensureOnDevice(rocm_device));
        ASSERT_TRUE(residual->ensureOnDevice(rocm_device));
        ASSERT_TRUE(rocm_output->ensureOnDevice(rocm_device));

        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP32> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_tensor(
            input.get(), residual.get(), rocm_output.get(),
            num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        rocm_output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *result = rocm_output->data();

        EXPECT_FALSE(hasNanOrInf(result, num_elements)) << "ROCm output contains NaN/Inf";

        double cosine = cosineSimilarity(result, cpu_output->data(), num_elements);
        double rel_error = relativeL2Error(result, cpu_output->data(), num_elements);

        std::cout << "[FP32_ApplyTensor_LargeTensor] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.99999) << "Cosine similarity too low - apply_tensor() may have pointer issues";
        EXPECT_LE(rel_error, 0.001) << "Relative L2 error too high";
    }

    TEST_F(ROCmResidualAddParityTest, FP32_ApplyTensor_InPlaceResidual_QwenShape)
    {
        const int rows = 64;
        const int cols = 896;
        const size_t num_elements = static_cast<size_t>(rows) * cols;

        auto input = TestTensorFactory::createFP32Random({rows, cols}, -4.0f, 4.0f);
        auto residual = TestTensorFactory::createFP32Random({rows, cols}, -4.0f, 4.0f);

        std::vector<float> expected(num_elements);
        const float *input_host = input->data();
        const float *residual_host = residual->data();
        for (size_t i = 0; i < num_elements; ++i)
        {
            expected[i] = input_host[i] + residual_host[i];
        }

        DeviceId rocm_device = DeviceId::rocm(0);
        ASSERT_TRUE(input->ensureOnDevice(rocm_device));
        ASSERT_TRUE(residual->ensureOnDevice(rocm_device));

        // Mirror the pipeline's in-place residual pattern: output aliases residual/hidden.
        llaminar2::rocm::ROCmResidualAddKernelT<ActivationPrecision::FP32> rocm_kernel;
        ASSERT_TRUE(rocm_kernel.apply_tensor(
            input.get(), residual.get(), residual.get(),
            num_elements, nullptr, 0));

        (void)hipDeviceSynchronize();
        residual->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        const float *result = residual->data();
        ASSERT_NE(result, nullptr);

        EXPECT_FALSE(hasNanOrInf(result, num_elements)) << "ROCm in-place residual output contains NaN/Inf";

        double cosine = cosineSimilarity(result, expected.data(), num_elements);
        double rel_error = relativeL2Error(result, expected.data(), num_elements);

        std::cout << "[FP32_ApplyTensor_InPlaceResidual_QwenShape] Cosine similarity: "
                  << std::fixed << std::setprecision(6) << cosine
                  << ", Relative L2 error: " << std::scientific << rel_error << std::endl;

        EXPECT_GE(cosine, 0.99999) << "Cosine similarity too low for in-place residual alias pattern";
        EXPECT_LE(rel_error, 0.001) << "Relative L2 error too high for in-place residual alias pattern";
    }

#endif // HAVE_ROCM

} // anonymous namespace
