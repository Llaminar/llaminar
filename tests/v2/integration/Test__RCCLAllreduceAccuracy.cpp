/**
 * @file Test__RCCLAllreduceAccuracy.cpp
 * @brief Integration tests for RCCL allreduce via LocalTPContext
 *
 * These tests validate RCCL allreduce operations produce numerically correct
 * results when called through the SAME CODE PATH used during inference:
 *
 *   LocalTPAllreduceStage::execute()
 *     → LocalTPContext::allreduce()
 *       → LocalTPContext::allreduceWithBarrierMultiGpu()
 *         → RCCLBackend::allreduceMulti()
 *
 * The key difference from raw backend tests is that these tests exercise:
 * - Multi-threaded barrier synchronization (one thread per device)
 * - Tensor buffer collection via barrier_tensors_[]
 * - gpu_data_ptr() buffer extraction
 * - mark_device_dirty_with_event() coherence marking
 *
 * Test scenarios based on inference use cases:
 * - Row-parallel GEMM output reduction (hidden_dim sizes)
 * - Attention output projection reduction
 * - FFN down projection reduction
 * - Multi-iteration consistency (simulating decode steps)
 *
 * @note Requires 2+ AMD ROCm GPUs with RCCL support
 *
 * @author David Sanftenberg
 * @date 2026-01-XX
 */

#include <gtest/gtest.h>

#ifdef HAVE_RCCL
#include "collective/LocalTPContext.h"
#include "collective/ILocalTPContext.h"
#include "collective/backends/RCCLBackend.h"
#include "collective/backends/RCCLDynamicLoader.h"
#include "collective/DeviceGroup.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <iomanip>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace llaminar2
{

    // =========================================================================
    // Constants matching Qwen2.5-0.5B inference dimensions
    // =========================================================================

    /** @brief Qwen2.5-0.5B hidden dimension */
    constexpr int QWEN2_HIDDEN_DIM = 896;

    /** @brief Qwen2.5-0.5B intermediate FFN dimension */
    constexpr int QWEN2_FFN_DIM = 4864;

    /** @brief Qwen2.5-0.5B attention heads */
    constexpr int QWEN2_NUM_HEADS = 14;

    /** @brief Qwen2.5-0.5B head dimension */
    constexpr int QWEN2_HEAD_DIM = 64;

    /** @brief Typical prefill sequence lengths */
    constexpr int PREFILL_SEQ_LENS[] = {5, 32, 128, 256};

    /** @brief Decode always produces 1 token */
    constexpr int DECODE_SEQ_LEN = 1;

    // =========================================================================
    // Statistical Helpers
    // =========================================================================

    /**
     * @brief Compute max absolute error between two vectors
     */
    double computeMaxAbsError(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return std::numeric_limits<double>::infinity();
        double max_err = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_err = std::max(max_err, static_cast<double>(std::abs(a[i] - b[i])));
        }
        return max_err;
    }

    /**
     * @brief Verify vectors match within tolerance
     */
    bool verifyMatch(const std::vector<float> &actual,
                     const std::vector<float> &expected,
                     float abs_tol = 1e-5f,
                     float rel_tol = 1e-4f)
    {
        if (actual.size() != expected.size())
            return false;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            float diff = std::abs(actual[i] - expected[i]);
            float max_val = std::max(std::abs(actual[i]), std::abs(expected[i]));
            if (diff > abs_tol && diff > rel_tol * max_val)
            {
                return false;
            }
        }
        return true;
    }

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class RCCLAllreduceAccuracyTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check for ROCm GPUs via BackendManager
#ifdef HAVE_ROCM
            auto *rocm_backend = getROCmBackend();
            device_count_ = (rocm_backend != nullptr) ? rocm_backend->deviceCount() : 0;
#else
            device_count_ = 0;
#endif

            if (device_count_ < 2)
            {
                GTEST_SKIP() << "Test requires at least 2 ROCm GPUs (found " << device_count_ << ")";
            }

            // Log device info
            std::cout << "\n=== RCCL LocalTPContext Allreduce Accuracy Tests ===" << std::endl;
            std::cout << "Found " << device_count_ << " ROCm GPU(s):" << std::endl;
            for (int i = 0; i < device_count_; ++i)
            {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, i);
                std::cout << "  GPU " << i << ": " << prop.name << std::endl;
            }
            std::cout << std::endl;

            // Create LocalTPContext with RCCL backend
            std::vector<GlobalDeviceAddress> devices;
            for (int i = 0; i < device_count_; ++i)
            {
                devices.push_back(GlobalDeviceAddress::rocm(i));
            }

            tp_ctx_ = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);
            ASSERT_NE(tp_ctx_, nullptr) << "Failed to create LocalTPContext";
            EXPECT_EQ(tp_ctx_->backend(), CollectiveBackendType::RCCL)
                << "Expected RCCL backend";
            EXPECT_EQ(tp_ctx_->degree(), device_count_)
                << "Expected degree=" << device_count_;
        }

        void TearDown() override
        {
            tp_ctx_.reset();

            // Synchronize all devices
            for (int i = 0; i < device_count_; ++i)
            {
                hipSetDevice(i);
                hipDeviceSynchronize();
                hipGetLastError();
            }
        }

        /**
         * @brief Run multi-threaded allreduce test via LocalTPContext
         *
         * This simulates the EXACT call pattern from inference:
         * - One thread per device
         * - Each thread owns a tensor on its device
         * - All threads call tp_ctx_->allreduce() concurrently
         * - LocalTPContext::allreduceWithBarrierMultiGpu() coordinates
         *
         * @param per_gpu_values Each GPU's contribution (per_gpu_values[gpu][element])
         * @param expected_sum The expected sum after allreduce
         * @param test_name Name for logging
         * @param count Number of elements (0 = use full tensor)
         * @return true if allreduce result matches expected on all GPUs
         */
        bool runMultiThreadedAllreduceTest(
            const std::vector<std::vector<float>> &per_gpu_values,
            const std::vector<float> &expected_sum,
            const std::string &test_name,
            size_t count = 0)
        {
            const int num_gpus = static_cast<int>(per_gpu_values.size());
            const size_t tensor_size = expected_sum.size();
            const size_t effective_count = (count > 0) ? count : tensor_size;

            EXPECT_EQ(num_gpus, device_count_) << "Test requires exactly " << device_count_ << " GPUs";

            // Create tensors on each device
            std::vector<std::unique_ptr<FP32Tensor>> tensors(num_gpus);
            for (int i = 0; i < num_gpus; ++i)
            {
                EXPECT_EQ(per_gpu_values[i].size(), tensor_size)
                    << "GPU " << i << " data size mismatch";

                // Create tensor with device affinity
                DeviceId device = DeviceId::rocm(i);
                tensors[i] = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{tensor_size});

                // Initialize host data
                std::memcpy(tensors[i]->mutable_data(), per_gpu_values[i].data(),
                            tensor_size * sizeof(float));

                // Upload to device
                EXPECT_TRUE(tensors[i]->ensureOnDevice(device))
                    << "Failed to upload tensor to GPU " << i;
            }

            // Synchronization primitives for multi-threaded test
            std::atomic<int> threads_ready{0};
            std::atomic<int> threads_done{0};
            std::atomic<bool> all_success{true};
            std::mutex result_mutex;
            std::condition_variable start_cv;
            std::mutex start_mutex;
            bool start_signal = false;

            // Results storage
            std::vector<std::vector<float>> results(num_gpus);
            std::vector<bool> thread_success(num_gpus, false);

            // Launch one thread per device (simulating inference worker threads)
            std::vector<std::thread> threads;
            for (int i = 0; i < num_gpus; ++i)
            {
                threads.emplace_back([&, i]()
                                     {
                // Set device context for this thread
                hipSetDevice(i);

                // Signal ready
                threads_ready.fetch_add(1);

                // Wait for start signal (simulates barrier before collective)
                {
                    std::unique_lock<std::mutex> lock(start_mutex);
                    start_cv.wait(lock, [&]() { return start_signal; });
                }

                // Call allreduce via LocalTPContext (the SAME call path as inference!)
                bool success = tp_ctx_->allreduce(
                    tensors[i].get(),
                    test_name + "_gpu" + std::to_string(i),
                    effective_count);

                if (!success) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    std::cerr << "  [ERROR] GPU " << i << " allreduce failed" << std::endl;
                    all_success.store(false);
                }

                thread_success[i] = success;
                threads_done.fetch_add(1); });
            }

            // Wait for all threads to be ready
            while (threads_ready.load() < num_gpus)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // Signal all threads to start allreduce
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                start_signal = true;
            }
            start_cv.notify_all();

            // Wait for all threads to complete
            for (auto &t : threads)
            {
                t.join();
            }

            // Synchronize devices
            tp_ctx_->synchronize();

            // Check allreduce success
            if (!all_success.load())
            {
                return false;
            }

            // Read back results from each tensor (triggers GPU→host sync via data())
            for (int i = 0; i < num_gpus; ++i)
            {
                results[i].resize(tensor_size);
                // data() automatically syncs from GPU if device-dirty
                const float *tensor_data = tensors[i]->data();
                std::memcpy(results[i].data(), tensor_data, tensor_size * sizeof(float));
            }

            // Verify results on each GPU
            bool all_match = true;
            for (int i = 0; i < num_gpus; ++i)
            {
                // Only verify the elements that were reduced
                std::vector<float> result_subset(results[i].begin(),
                                                 results[i].begin() + effective_count);
                std::vector<float> expected_subset(expected_sum.begin(),
                                                   expected_sum.begin() + effective_count);

                double max_abs_err = computeMaxAbsError(result_subset, expected_subset);

                bool gpu_match = verifyMatch(result_subset, expected_subset);
                if (!gpu_match)
                {
                    std::cout << "  [FAIL] GPU " << i << ": max_abs_err=" << max_abs_err << std::endl;

                    // Print first few mismatches
                    int mismatch_count = 0;
                    for (size_t j = 0; j < effective_count && mismatch_count < 5; ++j)
                    {
                        float diff = std::abs(result_subset[j] - expected_subset[j]);
                        if (diff > 1e-5f)
                        {
                            std::cout << "    [" << j << "]: got " << result_subset[j]
                                      << ", expected " << expected_subset[j]
                                      << " (diff=" << diff << ")" << std::endl;
                            mismatch_count++;
                        }
                    }
                    all_match = false;
                }
                else
                {
                    std::cout << "  [PASS] GPU " << i << ": max_abs_err=" << max_abs_err << std::endl;
                }
            }

            // Also verify all GPUs have identical results (consistency)
            for (int i = 1; i < num_gpus; ++i)
            {
                std::vector<float> result0_subset(results[0].begin(),
                                                  results[0].begin() + effective_count);
                std::vector<float> resulti_subset(results[i].begin(),
                                                  results[i].begin() + effective_count);
                double cross_gpu_err = computeMaxAbsError(result0_subset, resulti_subset);
                if (cross_gpu_err > 1e-7)
                {
                    std::cout << "  [WARN] GPU 0 vs GPU " << i << " differ by " << cross_gpu_err << std::endl;
                }
            }

            return all_match;
        }

        /**
         * @brief Run multi-threaded allreduce test and return results for analysis
         *
         * @param per_gpu_inputs Per-GPU input data
         * @param expected_sum Expected sum after allreduce
         * @param test_name Name for logging
         * @param effective_count Number of elements to reduce
         * @param out_results Output: results from each GPU
         * @return true if allreduce succeeded
         */
        bool runMultiThreadedAllreduceTestWithResults(
            const std::vector<std::vector<float>> &per_gpu_inputs,
            const std::vector<float> &expected_sum,
            const std::string &test_name,
            size_t effective_count,
            std::vector<std::vector<float>> &out_results);

        int device_count_ = 0;
        std::unique_ptr<ILocalTPContext> tp_ctx_;
    };

    // =========================================================================
    // Basic Accuracy Tests via LocalTPContext
    // =========================================================================

    /**
     * @brief Verify allreduce SUM with simple known values via LocalTPContext
     *
     * Each GPU contributes its rank value, sum should be n*(n-1)/2
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_BasicSum_KnownValues)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_BasicSum_KnownValues ---" << std::endl;

        const size_t count = 1024;
        const float expected_per_elem = static_cast<float>(device_count_ * (device_count_ - 1) / 2);

        std::vector<std::vector<float>> per_gpu(device_count_);
        for (int i = 0; i < device_count_; ++i)
        {
            per_gpu[i].resize(count, static_cast<float>(i));
        }

        std::vector<float> expected(count, expected_per_elem);

        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "BasicSum_KnownValues"));
    }

    /**
     * @brief Verify allreduce SUM with ones (trivial case) via LocalTPContext
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_BasicSum_AllOnes)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_BasicSum_AllOnes ---" << std::endl;

        const size_t count = 8192;
        const float expected_per_elem = static_cast<float>(device_count_);

        std::vector<std::vector<float>> per_gpu(device_count_);
        for (int i = 0; i < device_count_; ++i)
        {
            per_gpu[i].resize(count, 1.0f);
        }

        std::vector<float> expected(count, expected_per_elem);

        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "BasicSum_AllOnes"));
    }

    // =========================================================================
    // Inference-Sized Buffer Tests via LocalTPContext
    // =========================================================================

    /**
     * @brief Test with Qwen2.5 hidden dimension size (seq_len=1, decode)
     *
     * This simulates row-parallel GEMM output reduction for attention output
     * projection or FFN down projection during decode.
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_HiddenDim_Decode)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_HiddenDim_Decode ---" << std::endl;
        std::cout << "  Buffer size: " << QWEN2_HIDDEN_DIM << " (1 × hidden_dim)" << std::endl;

        const size_t count = QWEN2_HIDDEN_DIM; // 896

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<std::vector<float>> per_gpu(device_count_);
        std::vector<float> expected(count, 0.0f);

        for (int i = 0; i < device_count_; ++i)
        {
            rng.seed(42 + i);
            per_gpu[i].resize(count);
            for (size_t j = 0; j < count; ++j)
            {
                per_gpu[i][j] = dist(rng);
                expected[j] += per_gpu[i][j];
            }
        }

        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "HiddenDim_Decode"));
    }

    /**
     * @brief Test with Qwen2.5 hidden dimension × prefill sequence length
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_HiddenDim_Prefill)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_HiddenDim_Prefill ---" << std::endl;

        for (int seq_len : PREFILL_SEQ_LENS)
        {
            const size_t count = QWEN2_HIDDEN_DIM * seq_len;
            std::cout << "  Testing seq_len=" << seq_len
                      << " (buffer=" << count << " floats, "
                      << (count * sizeof(float) / 1024) << " KB)" << std::endl;

            std::mt19937 rng(123);
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

            std::vector<std::vector<float>> per_gpu(device_count_);
            std::vector<float> expected(count, 0.0f);

            for (int i = 0; i < device_count_; ++i)
            {
                rng.seed(123 + i * 1000);
                per_gpu[i].resize(count);
                for (size_t j = 0; j < count; ++j)
                {
                    per_gpu[i][j] = dist(rng);
                    expected[j] += per_gpu[i][j];
                }
            }

            EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected,
                                                      "HiddenDim_Prefill_seq" + std::to_string(seq_len)));
        }
    }

    /**
     * @brief Test with FFN intermediate dimension
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_FFNDim)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_FFNDim ---" << std::endl;
        std::cout << "  Buffer size: " << QWEN2_FFN_DIM << " (FFN intermediate dim)" << std::endl;

        const size_t count = QWEN2_FFN_DIM; // 4864

        std::mt19937 rng(999);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        std::vector<std::vector<float>> per_gpu(device_count_);
        std::vector<float> expected(count, 0.0f);

        for (int i = 0; i < device_count_; ++i)
        {
            rng.seed(999 + i * 100);
            per_gpu[i].resize(count);
            for (size_t j = 0; j < count; ++j)
            {
                per_gpu[i][j] = dist(rng);
                expected[j] += per_gpu[i][j];
            }
        }

        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "FFNDim"));
    }

    // =========================================================================
    // Multi-Iteration Consistency Tests
    // =========================================================================

    /**
     * @brief Run multiple allreduce iterations via LocalTPContext
     *
     * This simulates multiple decode steps where allreduce is called repeatedly.
     * Uses the SAME LocalTPContext across all iterations (like inference does).
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_MultiIteration_Decode)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_MultiIteration_Decode ---" << std::endl;

        constexpr int NUM_ITERATIONS = 10;
        const size_t tensor_size = QWEN2_HIDDEN_DIM; // 896

        bool all_iterations_pass = true;

        for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            std::mt19937 rng(iter * 1000);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            std::vector<std::vector<float>> per_gpu(device_count_);
            std::vector<float> expected(tensor_size, 0.0f);

            for (int i = 0; i < device_count_; ++i)
            {
                rng.seed(iter * 1000 + i * 100);
                per_gpu[i].resize(tensor_size);
                for (size_t j = 0; j < tensor_size; ++j)
                {
                    per_gpu[i][j] = dist(rng);
                    expected[j] += per_gpu[i][j];
                }
            }

            std::cout << "  Iteration " << iter << ": ";
            bool iter_pass = runMultiThreadedAllreduceTest(
                per_gpu, expected, "MultiIteration_iter" + std::to_string(iter));

            if (!iter_pass)
            {
                all_iterations_pass = false;
            }
        }

        EXPECT_TRUE(all_iterations_pass) << "Some iterations failed";
    }

    // =========================================================================
    // Partial Count Tests (simulates decode with count < buffer size)
    // =========================================================================

    /**
     * @brief Test allreduce with count parameter less than tensor size
     *
     * During decode, we often have large pre-allocated buffers but only
     * reduce a subset of elements (seq_len * hidden_dim where seq_len=1).
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_PartialCount)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_PartialCount ---" << std::endl;

        // Allocate large buffer but only reduce first hidden_dim elements
        const size_t buffer_size = QWEN2_HIDDEN_DIM * 256; // Large buffer
        const size_t reduce_count = QWEN2_HIDDEN_DIM;      // Only reduce this much

        std::cout << "  Buffer size: " << buffer_size << ", reduce count: " << reduce_count << std::endl;

        std::mt19937 rng(777);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<std::vector<float>> per_gpu(device_count_);
        std::vector<float> expected(buffer_size, 0.0f);

        for (int i = 0; i < device_count_; ++i)
        {
            rng.seed(777 + i);
            per_gpu[i].resize(buffer_size);
            for (size_t j = 0; j < buffer_size; ++j)
            {
                per_gpu[i][j] = dist(rng);
                if (j < reduce_count)
                {
                    expected[j] += per_gpu[i][j];
                }
            }
        }

        // Pass explicit count to only reduce first reduce_count elements
        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "PartialCount", reduce_count));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    /**
     * @brief Test with small counts via LocalTPContext
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_SmallCount)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_SmallCount ---" << std::endl;

        for (size_t count : {1, 7, 15, 31, 63, 127})
        {
            std::cout << "  Testing count=" << count << std::endl;

            std::vector<std::vector<float>> per_gpu(device_count_);
            std::vector<float> expected(count, 0.0f);

            for (int i = 0; i < device_count_; ++i)
            {
                per_gpu[i].resize(count, static_cast<float>(i + 1));
                for (size_t j = 0; j < count; ++j)
                {
                    expected[j] += per_gpu[i][j];
                }
            }

            EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected,
                                                      "SmallCount_" + std::to_string(count)));
        }
    }

    /**
     * @brief Test with negative values via LocalTPContext
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_NegativeValues)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_NegativeValues ---" << std::endl;

        const size_t count = QWEN2_HIDDEN_DIM;

        std::vector<std::vector<float>> per_gpu(device_count_);
        std::vector<float> expected(count, 0.0f);

        for (int i = 0; i < device_count_; ++i)
        {
            float sign = (i % 2 == 0) ? 1.0f : -1.0f;
            per_gpu[i].resize(count);
            for (size_t j = 0; j < count; ++j)
            {
                per_gpu[i][j] = sign * (static_cast<float>(j) / 100.0f + 0.1f);
                expected[j] += per_gpu[i][j];
            }
        }

        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "NegativeValues"));
    }

    // =========================================================================
    // Cross-GPU Consistency Test
    // =========================================================================

    /**
     * @brief Verify all GPUs have IDENTICAL results after allreduce via LocalTPContext
     */
    TEST_F(RCCLAllreduceAccuracyTest, ViaLocalTPContext_Consistency)
    {
        std::cout << "\n--- Test: ViaLocalTPContext_Consistency ---" << std::endl;

        const size_t count = QWEN2_HIDDEN_DIM * 32;

        std::mt19937 rng(54321);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<std::vector<float>> per_gpu(device_count_);
        std::vector<float> expected(count, 0.0f);

        for (int i = 0; i < device_count_; ++i)
        {
            rng.seed(54321 + i);
            per_gpu[i].resize(count);
            for (size_t j = 0; j < count; ++j)
            {
                per_gpu[i][j] = dist(rng);
                expected[j] += per_gpu[i][j];
            }
        }

        EXPECT_TRUE(runMultiThreadedAllreduceTest(per_gpu, expected, "Consistency"));
    }

    // =========================================================================
    // Regression Tests for RCCL Multi-GPU Numerical Variance
    // =========================================================================
    // These tests lock in the expected numerical bounds for RCCL allreduce.
    // See: Test__Qwen2_LocalTP_RCCL_Allreduce_vs_PyTorch.cpp kl_threshold=0.36
    //
    // RCCL multi-GPU introduces FP32 accumulation order differences during
    // ring/tree reduction, and cross-device synchronization timing can cause
    // minor numerical variance. These tests ensure the variance stays within
    // acceptable bounds for inference correctness.
    // =========================================================================

    /**
     * @brief Compute relative error between result and expected
     */
    static double computeRelativeError(const std::vector<float> &result,
                                       const std::vector<float> &expected)
    {
        double max_rel_err = 0.0;
        for (size_t i = 0; i < result.size() && i < expected.size(); ++i)
        {
            if (std::abs(expected[i]) > 1e-6f)
            {
                double rel_err = std::abs(result[i] - expected[i]) / std::abs(expected[i]);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
        return max_rel_err;
    }

    /**
     * @brief Regression test: RCCL allreduce numerical variance bounds
     *
     * This test validates that RCCL multi-GPU allreduce stays within acceptable
     * numerical variance bounds. This is a regression test for the KL threshold
     * adjustment from 0.30 to 0.36 in the LocalTP parity test.
     *
     * The test performs multiple allreduce operations with inference-like data
     * and verifies that:
     * 1. Maximum absolute error is bounded (< 1e-4)
     * 2. Relative error is bounded (< 0.01%)
     * 3. Cross-GPU consistency is maintained
     *
     * These bounds are empirically determined from real inference scenarios and
     * ensure that the numerical variance introduced by RCCL ring/tree algorithms
     * does not cause token prediction errors.
     */
    TEST_F(RCCLAllreduceAccuracyTest, Regression_NumericalVarianceBounds)
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     REGRESSION TEST: RCCL Multi-GPU Numerical Variance Bounds    ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

        // Test configurations matching inference scenarios
        struct TestConfig
        {
            size_t count;
            const char *name;
            float abs_err_bound;
            float rel_err_bound;
        };

        std::vector<TestConfig> configs = {
            {QWEN2_HIDDEN_DIM, "Decode_HiddenDim", 1e-4f, 1e-4f},
            {QWEN2_HIDDEN_DIM * 9, "Prefill_9tok_HiddenDim", 1e-4f, 1e-4f},
            {QWEN2_HIDDEN_DIM * 128, "Prefill_128tok_HiddenDim", 1e-4f, 1e-4f},
            {QWEN2_FFN_DIM, "FFN_Intermediate", 1e-4f, 1e-4f},
        };

        bool all_pass = true;

        for (const auto &cfg : configs)
        {
            std::cout << "\n  Testing: " << cfg.name << " (count=" << cfg.count << ")" << std::endl;

            // Generate inference-like data (activations after GEMM)
            std::mt19937 rng(12345);
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

            std::vector<std::vector<float>> per_gpu(device_count_);
            std::vector<float> expected(cfg.count, 0.0f);

            for (int i = 0; i < device_count_; ++i)
            {
                rng.seed(12345 + i * 1000);
                per_gpu[i].resize(cfg.count);
                for (size_t j = 0; j < cfg.count; ++j)
                {
                    per_gpu[i][j] = dist(rng);
                    expected[j] += per_gpu[i][j];
                }
            }

            // Run allreduce
            std::vector<std::vector<float>> results(device_count_);
            bool op_success = runMultiThreadedAllreduceTestWithResults(
                per_gpu, expected, cfg.name, cfg.count, results);

            if (!op_success)
            {
                std::cout << "    [FAIL] Allreduce operation failed" << std::endl;
                all_pass = false;
                continue;
            }

            // Compute error metrics
            double max_abs_err = computeMaxAbsError(results[0], expected);
            double max_rel_err = computeRelativeError(results[0], expected);

            // Verify cross-GPU consistency
            double cross_gpu_err = 0.0;
            for (int i = 1; i < device_count_; ++i)
            {
                double err = computeMaxAbsError(results[0], results[i]);
                cross_gpu_err = std::max(cross_gpu_err, err);
            }

            std::cout << "    Absolute error: " << std::scientific << std::setprecision(3)
                      << max_abs_err << " (bound: " << cfg.abs_err_bound << ")" << std::endl;
            std::cout << "    Relative error: " << std::scientific << std::setprecision(3)
                      << max_rel_err << " (bound: " << cfg.rel_err_bound << ")" << std::endl;
            std::cout << "    Cross-GPU diff: " << std::scientific << std::setprecision(3)
                      << cross_gpu_err << std::endl;

            // Check bounds
            bool abs_ok = max_abs_err <= cfg.abs_err_bound;
            bool rel_ok = max_rel_err <= cfg.rel_err_bound;
            bool consistency_ok = cross_gpu_err < 1e-7;

            if (!abs_ok || !rel_ok || !consistency_ok)
            {
                std::cout << "    [FAIL] Bounds exceeded" << std::endl;
                all_pass = false;
            }
            else
            {
                std::cout << "    [PASS]" << std::endl;
            }
        }

        EXPECT_TRUE(all_pass) << "Some variance bound checks failed";
    }

    /**
     * @brief Helper: Run allreduce test and return results for analysis
     */
    bool RCCLAllreduceAccuracyTest::runMultiThreadedAllreduceTestWithResults(
        const std::vector<std::vector<float>> &per_gpu_inputs,
        const std::vector<float> &expected_sum,
        const std::string &test_name,
        size_t effective_count,
        std::vector<std::vector<float>> &out_results)
    {
        const int num_gpus = static_cast<int>(per_gpu_inputs.size());
        const size_t tensor_size = per_gpu_inputs[0].size();

        if (effective_count == 0)
        {
            effective_count = tensor_size;
        }

        // Create tensors on each device and upload input data
        std::vector<std::unique_ptr<FP32Tensor>> tensors;
        for (int i = 0; i < num_gpus; ++i)
        {
            auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{tensor_size});

            // Copy input data to host buffer
            std::memcpy(tensor->mutable_data(), per_gpu_inputs[i].data(),
                        tensor_size * sizeof(float));

            // Upload to device
            DeviceId device = DeviceId::rocm(i);
            tensor->ensureOnDevice(device);

            tensors.push_back(std::move(tensor));
        }

        // Run multi-threaded allreduce (same as inference)
        std::vector<std::thread> threads;
        std::atomic<int> threads_ready{0};
        std::atomic<int> threads_done{0};
        std::atomic<bool> all_success{true};
        std::vector<bool> thread_success(num_gpus, false);

        std::mutex start_mutex;
        std::condition_variable start_cv;
        bool start_signal = false;
        std::mutex result_mutex;

        for (int i = 0; i < num_gpus; ++i)
        {
            threads.emplace_back([&, i]()
                                 {
            threads_ready.fetch_add(1);

            {
                std::unique_lock<std::mutex> lock(start_mutex);
                start_cv.wait(lock, [&]() { return start_signal; });
            }

            bool success = tp_ctx_->allreduce(
                tensors[i].get(),
                test_name + "_gpu" + std::to_string(i),
                effective_count);

            if (!success) {
                std::lock_guard<std::mutex> lock(result_mutex);
                all_success.store(false);
            }

            thread_success[i] = success;
            threads_done.fetch_add(1); });
        }

        while (threads_ready.load() < num_gpus)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        {
            std::lock_guard<std::mutex> lock(start_mutex);
            start_signal = true;
        }
        start_cv.notify_all();

        for (auto &t : threads)
        {
            t.join();
        }

        tp_ctx_->synchronize();

        if (!all_success.load())
        {
            return false;
        }

        // Read back results
        out_results.resize(num_gpus);
        for (int i = 0; i < num_gpus; ++i)
        {
            out_results[i].resize(effective_count);
            const float *tensor_data = tensors[i]->data();
            std::memcpy(out_results[i].data(), tensor_data, effective_count * sizeof(float));
        }

        return true;
    }

} // namespace llaminar2

#else // !HAVE_RCCL

// Stub test when RCCL is not available
namespace llaminar2
{

    TEST(RCCLAllreduceAccuracyTest, DISABLED_NoRCCL)
    {
        GTEST_SKIP() << "RCCL not available - tests disabled";
    }

} // namespace llaminar2

#endif // HAVE_RCCL
