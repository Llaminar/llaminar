/**
 * @file Test__CUDAVectorAddKernels.cpp
 * @brief Integration tests for CUDA vector addition kernels against CPU reference
 *
 * Tests FP32, FP16, BF16, INT8, INT32 in-place reduction kernels for accuracy.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <limits>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#include "kernels/cuda/ops/CUDAVectorAddKernels.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace test
    {

        class CUDAVectorAddKernelsTest : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
#ifdef HAVE_CUDA
                int deviceCount = 0;
                cudaGetDeviceCount(&deviceCount);
                if (deviceCount == 0)
                {
                    GTEST_SKIP() << "No CUDA devices available";
                }
                cudaSetDevice(0);
#else
                GTEST_SKIP() << "CUDA not enabled";
#endif
            }

            void TearDown() override
            {
#ifdef HAVE_CUDA
                cudaDeviceSynchronize();
#endif
            }

            // Helper to generate random floats
            std::vector<float> generateRandomFloats(size_t count, float min_val = -10.0f, float max_val = 10.0f)
            {
                std::vector<float> data(count);
                std::mt19937 gen(42); // Fixed seed for reproducibility
                std::uniform_real_distribution<float> dist(min_val, max_val);
                for (size_t i = 0; i < count; ++i)
                {
                    data[i] = dist(gen);
                }
                return data;
            }

            // Helper to generate random int8 values
            std::vector<int8_t> generateRandomInt8(size_t count)
            {
                std::vector<int8_t> data(count);
                std::mt19937 gen(42);
                // Use smaller range to avoid saturation in most cases
                std::uniform_int_distribution<int> dist(-50, 50);
                for (size_t i = 0; i < count; ++i)
                {
                    data[i] = static_cast<int8_t>(dist(gen));
                }
                return data;
            }

            // Helper to generate random int32 values
            std::vector<int32_t> generateRandomInt32(size_t count)
            {
                std::vector<int32_t> data(count);
                std::mt19937 gen(42);
                std::uniform_int_distribution<int32_t> dist(-100000, 100000);
                for (size_t i = 0; i < count; ++i)
                {
                    data[i] = dist(gen);
                }
                return data;
            }
        };

        // =============================================================================
        // FP32 Tests
        // =============================================================================

        TEST_F(CUDAVectorAddKernelsTest, FP32_InplaceAdd_Small)
        {
#ifdef HAVE_CUDA
            const size_t count = 128;

            auto output_host = generateRandomFloats(count);
            auto input_host = generateRandomFloats(count);

            // CPU reference
            std::vector<float> expected = output_host;
            for (size_t i = 0; i < count; ++i)
            {
                expected[i] += input_host[i];
            }

            // GPU execution
            float *d_output;
            float *d_input;
            cudaMalloc(&d_output, count * sizeof(float));
            cudaMalloc(&d_input, count * sizeof(float));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(float), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_f32(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<float> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost);

            // Verify
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at index " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, FP32_InplaceAdd_Large_Vectorized)
        {
#ifdef HAVE_CUDA
            const size_t count = 1024 * 1024; // 1M elements, will use vectorized path

            auto output_host = generateRandomFloats(count);
            auto input_host = generateRandomFloats(count);

            // CPU reference
            std::vector<float> expected = output_host;
            for (size_t i = 0; i < count; ++i)
            {
                expected[i] += input_host[i];
            }

            // GPU execution
            float *d_output;
            float *d_input;
            cudaMalloc(&d_output, count * sizeof(float));
            cudaMalloc(&d_input, count * sizeof(float));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(float), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_f32(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<float> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost);

            // Sample verification (check every 1000th element to speed up)
            size_t errors = 0;
            for (size_t i = 0; i < count; i += 1000)
            {
                if (std::abs(result[i] - expected[i]) > 1e-5f)
                {
                    errors++;
                    if (errors <= 5)
                    {
                        LOG_ERROR("FP32 mismatch at " << i << ": got " << result[i] << ", expected " << expected[i]);
                    }
                }
            }
            EXPECT_EQ(errors, 0) << "Found " << errors << " mismatches in sampled elements";

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        // =============================================================================
        // FP16 Tests
        // =============================================================================

        TEST_F(CUDAVectorAddKernelsTest, FP16_InplaceAdd_Small)
        {
#ifdef HAVE_CUDA
            const size_t count = 128;

            auto output_f32 = generateRandomFloats(count, -5.0f, 5.0f); // Smaller range for FP16
            auto input_f32 = generateRandomFloats(count, -5.0f, 5.0f);

            // Convert to FP16
            std::vector<__half> output_host(count);
            std::vector<__half> input_host(count);
            for (size_t i = 0; i < count; ++i)
            {
                output_host[i] = __float2half(output_f32[i]);
                input_host[i] = __float2half(input_f32[i]);
            }

            // CPU reference (compute in float, then convert expected)
            std::vector<float> expected_f32(count);
            for (size_t i = 0; i < count; ++i)
            {
                expected_f32[i] = __half2float(output_host[i]) + __half2float(input_host[i]);
            }

            // GPU execution
            __half *d_output;
            __half *d_input;
            cudaMalloc(&d_output, count * sizeof(__half));
            cudaMalloc(&d_input, count * sizeof(__half));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(__half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(__half), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_f16(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<__half> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(__half), cudaMemcpyDeviceToHost);

            // Verify with FP16 tolerance
            for (size_t i = 0; i < count; ++i)
            {
                float result_f32 = __half2float(result[i]);
                EXPECT_NEAR(result_f32, expected_f32[i], 0.01f) << "FP16 mismatch at index " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, FP16_InplaceAdd_Large_Vectorized)
        {
#ifdef HAVE_CUDA
            const size_t count = 512 * 1024; // 512K elements, will use half2 vectorized path

            auto output_f32 = generateRandomFloats(count, -5.0f, 5.0f);
            auto input_f32 = generateRandomFloats(count, -5.0f, 5.0f);

            std::vector<__half> output_host(count);
            std::vector<__half> input_host(count);
            std::vector<float> expected_f32(count);

            for (size_t i = 0; i < count; ++i)
            {
                output_host[i] = __float2half(output_f32[i]);
                input_host[i] = __float2half(input_f32[i]);
                expected_f32[i] = __half2float(output_host[i]) + __half2float(input_host[i]);
            }

            __half *d_output;
            __half *d_input;
            cudaMalloc(&d_output, count * sizeof(__half));
            cudaMalloc(&d_input, count * sizeof(__half));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(__half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(__half), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_f16(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<__half> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(__half), cudaMemcpyDeviceToHost);

            // Sample verification
            size_t errors = 0;
            for (size_t i = 0; i < count; i += 1000)
            {
                float result_f32 = __half2float(result[i]);
                if (std::abs(result_f32 - expected_f32[i]) > 0.01f)
                {
                    errors++;
                    if (errors <= 5)
                    {
                        LOG_ERROR("FP16 mismatch at " << i << ": got " << result_f32 << ", expected " << expected_f32[i]);
                    }
                }
            }
            EXPECT_EQ(errors, 0) << "Found " << errors << " mismatches in sampled FP16 elements";

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, FP16_InplaceAdd_OddCount)
        {
#ifdef HAVE_CUDA
            // Test non-vectorizable count (odd number)
            const size_t count = 1001;

            auto output_f32 = generateRandomFloats(count, -5.0f, 5.0f);
            auto input_f32 = generateRandomFloats(count, -5.0f, 5.0f);

            std::vector<__half> output_host(count);
            std::vector<__half> input_host(count);
            std::vector<float> expected_f32(count);

            for (size_t i = 0; i < count; ++i)
            {
                output_host[i] = __float2half(output_f32[i]);
                input_host[i] = __float2half(input_f32[i]);
                expected_f32[i] = __half2float(output_host[i]) + __half2float(input_host[i]);
            }

            __half *d_output;
            __half *d_input;
            cudaMalloc(&d_output, count * sizeof(__half));
            cudaMalloc(&d_input, count * sizeof(__half));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(__half), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(__half), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_f16(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<__half> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(__half), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < count; ++i)
            {
                float result_f32 = __half2float(result[i]);
                EXPECT_NEAR(result_f32, expected_f32[i], 0.01f) << "FP16 odd-count mismatch at " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        // =============================================================================
        // BF16 Tests
        // =============================================================================

        TEST_F(CUDAVectorAddKernelsTest, BF16_InplaceAdd_Small)
        {
#ifdef HAVE_CUDA
            const size_t count = 128;

            auto output_f32 = generateRandomFloats(count, -10.0f, 10.0f);
            auto input_f32 = generateRandomFloats(count, -10.0f, 10.0f);

            // Convert to BF16
            std::vector<__nv_bfloat16> output_host(count);
            std::vector<__nv_bfloat16> input_host(count);
            for (size_t i = 0; i < count; ++i)
            {
                output_host[i] = __float2bfloat16(output_f32[i]);
                input_host[i] = __float2bfloat16(input_f32[i]);
            }

            // CPU reference
            std::vector<float> expected_f32(count);
            for (size_t i = 0; i < count; ++i)
            {
                expected_f32[i] = __bfloat162float(output_host[i]) + __bfloat162float(input_host[i]);
            }

            // GPU execution
            __nv_bfloat16 *d_output;
            __nv_bfloat16 *d_input;
            cudaMalloc(&d_output, count * sizeof(__nv_bfloat16));
            cudaMalloc(&d_input, count * sizeof(__nv_bfloat16));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_bf16(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<__nv_bfloat16> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

            // BF16 has less precision than FP16, use larger tolerance
            for (size_t i = 0; i < count; ++i)
            {
                float result_f32 = __bfloat162float(result[i]);
                EXPECT_NEAR(result_f32, expected_f32[i], 0.1f) << "BF16 mismatch at index " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, BF16_InplaceAdd_Large_Vectorized)
        {
#ifdef HAVE_CUDA
            const size_t count = 512 * 1024;

            auto output_f32 = generateRandomFloats(count, -10.0f, 10.0f);
            auto input_f32 = generateRandomFloats(count, -10.0f, 10.0f);

            std::vector<__nv_bfloat16> output_host(count);
            std::vector<__nv_bfloat16> input_host(count);
            std::vector<float> expected_f32(count);

            for (size_t i = 0; i < count; ++i)
            {
                output_host[i] = __float2bfloat16(output_f32[i]);
                input_host[i] = __float2bfloat16(input_f32[i]);
                expected_f32[i] = __bfloat162float(output_host[i]) + __bfloat162float(input_host[i]);
            }

            __nv_bfloat16 *d_output;
            __nv_bfloat16 *d_input;
            cudaMalloc(&d_output, count * sizeof(__nv_bfloat16));
            cudaMalloc(&d_input, count * sizeof(__nv_bfloat16));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_bf16(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<__nv_bfloat16> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

            size_t errors = 0;
            for (size_t i = 0; i < count; i += 1000)
            {
                float result_f32 = __bfloat162float(result[i]);
                if (std::abs(result_f32 - expected_f32[i]) > 0.1f)
                {
                    errors++;
                    if (errors <= 5)
                    {
                        LOG_ERROR("BF16 mismatch at " << i << ": got " << result_f32 << ", expected " << expected_f32[i]);
                    }
                }
            }
            EXPECT_EQ(errors, 0) << "Found " << errors << " mismatches in sampled BF16 elements";

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        // =============================================================================
        // INT8 Tests
        // =============================================================================

        TEST_F(CUDAVectorAddKernelsTest, INT8_InplaceAdd_Small)
        {
#ifdef HAVE_CUDA
            const size_t count = 128;

            auto output_host = generateRandomInt8(count);
            auto input_host = generateRandomInt8(count);

            // CPU reference with saturation
            std::vector<int8_t> expected(count);
            for (size_t i = 0; i < count; ++i)
            {
                int32_t sum = static_cast<int32_t>(output_host[i]) + static_cast<int32_t>(input_host[i]);
                expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, sum)));
            }

            // GPU execution
            int8_t *d_output;
            int8_t *d_input;
            cudaMalloc(&d_output, count * sizeof(int8_t));
            cudaMalloc(&d_input, count * sizeof(int8_t));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_i8(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<int8_t> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(int8_t), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(result[i], expected[i]) << "INT8 mismatch at index " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, INT8_InplaceAdd_Saturation)
        {
#ifdef HAVE_CUDA
            // Test saturation behavior at boundaries
            const size_t count = 8;

            std::vector<int8_t> output_host = {100, -100, 127, -128, 50, -50, 0, 1};
            std::vector<int8_t> input_host = {50, -50, 1, -1, 100, -100, 127, -128};

            // CPU reference with saturation
            std::vector<int8_t> expected(count);
            for (size_t i = 0; i < count; ++i)
            {
                int32_t sum = static_cast<int32_t>(output_host[i]) + static_cast<int32_t>(input_host[i]);
                expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, sum)));
            }

            // Expected: {127 (saturated), -128 (saturated), 127 (saturated), -128 (saturated), 127 (saturated), -128 (saturated), 127, -127}

            int8_t *d_output;
            int8_t *d_input;
            cudaMalloc(&d_output, count * sizeof(int8_t));
            cudaMalloc(&d_input, count * sizeof(int8_t));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_i8(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<int8_t> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(int8_t), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(result[i], expected[i]) << "INT8 saturation mismatch at index " << i
                                                  << " (out=" << (int)output_host[i] << " + in=" << (int)input_host[i] << ")";
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, INT8_InplaceAdd_Large_Vectorized)
        {
#ifdef HAVE_CUDA
            const size_t count = 1024 * 1024; // 1M elements, will use char4 vectorized path

            auto output_host = generateRandomInt8(count);
            auto input_host = generateRandomInt8(count);

            std::vector<int8_t> expected(count);
            for (size_t i = 0; i < count; ++i)
            {
                int32_t sum = static_cast<int32_t>(output_host[i]) + static_cast<int32_t>(input_host[i]);
                expected[i] = static_cast<int8_t>(std::max(-128, std::min(127, sum)));
            }

            int8_t *d_output;
            int8_t *d_input;
            cudaMalloc(&d_output, count * sizeof(int8_t));
            cudaMalloc(&d_input, count * sizeof(int8_t));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_i8(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<int8_t> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(int8_t), cudaMemcpyDeviceToHost);

            size_t errors = 0;
            for (size_t i = 0; i < count; ++i)
            {
                if (result[i] != expected[i])
                {
                    errors++;
                    if (errors <= 5)
                    {
                        LOG_ERROR("INT8 mismatch at " << i << ": got " << (int)result[i] << ", expected " << (int)expected[i]);
                    }
                }
            }
            EXPECT_EQ(errors, 0) << "Found " << errors << " mismatches in INT8 elements";

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        // =============================================================================
        // INT32 Tests
        // =============================================================================

        TEST_F(CUDAVectorAddKernelsTest, INT32_InplaceAdd_Small)
        {
#ifdef HAVE_CUDA
            const size_t count = 128;

            auto output_host = generateRandomInt32(count);
            auto input_host = generateRandomInt32(count);

            // CPU reference
            std::vector<int32_t> expected(count);
            for (size_t i = 0; i < count; ++i)
            {
                expected[i] = output_host[i] + input_host[i];
            }

            // GPU execution
            int32_t *d_output;
            int32_t *d_input;
            cudaMalloc(&d_output, count * sizeof(int32_t));
            cudaMalloc(&d_input, count * sizeof(int32_t));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_i32(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<int32_t> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(int32_t), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(result[i], expected[i]) << "INT32 mismatch at index " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, INT32_InplaceAdd_Large_Vectorized)
        {
#ifdef HAVE_CUDA
            const size_t count = 1024 * 1024; // 1M elements, will use int4 vectorized path

            auto output_host = generateRandomInt32(count);
            auto input_host = generateRandomInt32(count);

            std::vector<int32_t> expected(count);
            for (size_t i = 0; i < count; ++i)
            {
                expected[i] = output_host[i] + input_host[i];
            }

            int32_t *d_output;
            int32_t *d_input;
            cudaMalloc(&d_output, count * sizeof(int32_t));
            cudaMalloc(&d_input, count * sizeof(int32_t));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_i32(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<int32_t> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(int32_t), cudaMemcpyDeviceToHost);

            size_t errors = 0;
            for (size_t i = 0; i < count; ++i)
            {
                if (result[i] != expected[i])
                {
                    errors++;
                    if (errors <= 5)
                    {
                        LOG_ERROR("INT32 mismatch at " << i << ": got " << result[i] << ", expected " << expected[i]);
                    }
                }
            }
            EXPECT_EQ(errors, 0) << "Found " << errors << " mismatches in INT32 elements";

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, INT32_InplaceAdd_NonVectorizable)
        {
#ifdef HAVE_CUDA
            // Test count not divisible by 4 (uses scalar path)
            const size_t count = 1025;

            auto output_host = generateRandomInt32(count);
            auto input_host = generateRandomInt32(count);

            std::vector<int32_t> expected(count);
            for (size_t i = 0; i < count; ++i)
            {
                expected[i] = output_host[i] + input_host[i];
            }

            int32_t *d_output;
            int32_t *d_input;
            cudaMalloc(&d_output, count * sizeof(int32_t));
            cudaMalloc(&d_input, count * sizeof(int32_t));
            cudaMemcpy(d_output, output_host.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input, input_host.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_i32(d_output, d_input, count));
            cudaDeviceSynchronize();

            std::vector<int32_t> result(count);
            cudaMemcpy(result.data(), d_output, count * sizeof(int32_t), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(result[i], expected[i]) << "INT32 non-vec mismatch at index " << i;
            }

            cudaFree(d_output);
            cudaFree(d_input);
#endif
        }

        // =============================================================================
        // Edge Cases
        // =============================================================================

        TEST_F(CUDAVectorAddKernelsTest, EmptyBuffer)
        {
#ifdef HAVE_CUDA
            // All kernels should handle count=0 gracefully
            EXPECT_TRUE(cuda::launchVectorAddInplace_f32(nullptr, nullptr, 0));
            EXPECT_TRUE(cuda::launchVectorAddInplace_f16(nullptr, nullptr, 0));
            EXPECT_TRUE(cuda::launchVectorAddInplace_bf16(nullptr, nullptr, 0));
            EXPECT_TRUE(cuda::launchVectorAddInplace_i8(nullptr, nullptr, 0));
            EXPECT_TRUE(cuda::launchVectorAddInplace_i32(nullptr, nullptr, 0));
#endif
        }

        TEST_F(CUDAVectorAddKernelsTest, SingleElement)
        {
#ifdef HAVE_CUDA
            float f32_out = 1.5f, f32_in = 2.5f;
            float *d_f32_out;
            float *d_f32_in;
            cudaMalloc(&d_f32_out, sizeof(float));
            cudaMalloc(&d_f32_in, sizeof(float));
            cudaMemcpy(d_f32_out, &f32_out, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_f32_in, &f32_in, sizeof(float), cudaMemcpyHostToDevice);

            ASSERT_TRUE(cuda::launchVectorAddInplace_f32(d_f32_out, d_f32_in, 1));
            cudaDeviceSynchronize();

            float result;
            cudaMemcpy(&result, d_f32_out, sizeof(float), cudaMemcpyDeviceToHost);
            EXPECT_NEAR(result, 4.0f, 1e-6f);

            cudaFree(d_f32_out);
            cudaFree(d_f32_in);
#endif
        }

    } // namespace test
} // namespace llaminar2
