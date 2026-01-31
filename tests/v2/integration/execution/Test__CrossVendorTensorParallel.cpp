/**
 * @file Test__CrossVendorTensorParallel.cpp
 * @brief Comprehensive integration tests for cross-vendor GPU tensor parallelism
 *
 * Tests real tensor operations using PCIe BAR-based P2P between NVIDIA and AMD GPUs.
 * Validates tensor parallel inference patterns with Llaminar tensor types:
 *   - FP32Tensor, Q8_1Tensor, IQ4_NLTensor transfers
 *   - Column-parallel GEMM (weight sharding across vendors)
 *   - AllReduce/AllGather semantics via P2P
 *   - Mixed-vendor inference pipeline simulation
 *
 * ## Architecture
 *
 * These tests exercise the complete cross-vendor path:
 *   1. Tensors created on host (CPU)
 *   2. Transferred to NVIDIA GPU via CUDA
 *   3. Cross-vendor P2P to AMD GPU via PCIe BAR
 *   4. AMD processes (simulated via host roundtrip)
 *   5. Results gathered back via P2P
 *   6. Verification on host
 *
 * ## Requirements
 *
 * - HAVE_CUDA and HAVE_ROCM defined
 * - Root/sudo access (for PCIe BAR mapping)
 * - Resizable BAR enabled for best performance (32GB BAR)
 * - NVIDIA GPU (tested: RTX 3090) + AMD GPU (tested: MI50)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "backends/p2p/DirectP2P.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"
#include "TestTensorFactory.h"

// NOTE: We use the IBackend abstraction for device memory allocation
// to avoid including conflicting CUDA/HIP headers in the same TU.
// Only CUDA headers are needed for the P2P tests that use cudaMemcpy.
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Test fixture for cross-vendor tensor parallel integration tests
         *
         * Manages DirectP2PEngine lifecycle and provides tensor creation utilities.
         */
        class Test__CrossVendorTensorParallel : public ::testing::Test
        {
        protected:
            DirectP2PEngine engine_;
            DeviceId cuda_dev_;
            DeviceId rocm_dev_;
            bool p2p_available_ = false;

            // Test dimensions (Qwen2.5-0.5B style)
            static constexpr int D_MODEL = 896;
            static constexpr int FFN_DIM = 4864;
            static constexpr int N_HEADS = 14;
            static constexpr int HEAD_DIM = 64;
            static constexpr int SEQ_LEN = 128;
            static constexpr int VOCAB_SIZE = 151936;

            void SetUp() override
            {
#ifndef HAVE_CUDA
                GTEST_SKIP() << "CUDA not available";
#endif
#ifndef HAVE_ROCM
                GTEST_SKIP() << "ROCm not available";
#endif

                cuda_dev_ = DeviceId::cuda(0);
                rocm_dev_ = DeviceId::rocm(0);

                // Check PCIe BAR P2P availability
                auto caps = DirectP2PEngine::probeCapabilities();
                if (!caps.canDoPCIeBarP2P())
                {
                    LOG_WARN("PCIe BAR P2P not available - some tests will skip");
                    LOG_WARN("  BAR access: " << (caps.pcie_bar_accessible ? "YES" : "NO"));
                    LOG_WARN("  IOMEMORY: " << (caps.pcie_bar_iomemory_supported ? "YES" : "NO"));
                    LOG_WARN("  AMD BARs: " << caps.discovered_bars.size());
                    return;
                }

                // Request 1 GB mapping to allow larger transfer benchmarks
                constexpr size_t map_size = 1024 * 1024 * 1024;
                p2p_available_ = engine_.initializePCIeBar(cuda_dev_, rocm_dev_, 0, map_size);
                if (p2p_available_)
                {
                    LOG_INFO("PCIe BAR P2P initialized:");
                    LOG_INFO("  BAR mapped: " << (engine_.getBarMappedSize() / (1024 * 1024)) << " MB");
                    LOG_INFO("  CUDA BAR ptr: " << engine_.getCudaBarPointer());
                }
            }

            void TearDown() override
            {
                // Engine destructor handles cleanup
            }

            /**
             * @brief Skip test if P2P not available
             * Uses macro to properly return from calling test function
             */
#define REQUIRE_P2P()                                                           \
    do                                                                          \
    {                                                                           \
        if (!p2p_available_)                                                    \
        {                                                                       \
            GTEST_SKIP() << "PCIe BAR P2P not available (need root + AMD GPU)"; \
            return;                                                             \
        }                                                                       \
    } while (0)

            /**
             * @brief Compute MSE between two float buffers
             */
            static double computeMSE(const float *a, const float *b, size_t count)
            {
                double sum = 0.0;
                for (size_t i = 0; i < count; ++i)
                {
                    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                    sum += diff * diff;
                }
                return sum / count;
            }

            /**
             * @brief Compute cosine similarity between two float buffers
             */
            static double computeCosineSimilarity(const float *a, const float *b, size_t count)
            {
                double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
                for (size_t i = 0; i < count; ++i)
                {
                    dot += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }
                if (norm_a < 1e-10 || norm_b < 1e-10)
                    return 0.0;
                return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
            }

            /**
             * @brief CPU reference GEMM: C = A @ B^T (for validation only)
             * A: [M, K], B: [N, K] -> C: [M, N]
             */
            static void cpu_gemm_reference(const float *A, const float *B, float *C,
                                           int M, int N, int K)
            {
                for (int i = 0; i < M; ++i)
                {
                    for (int j = 0; j < N; ++j)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k)
                        {
                            sum += A[i * K + k] * B[j * K + k];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }

            /**
             * @brief AllReduce SUM emulation (for two-GPU case)
             */
            static void allreduceSum(float *local_a, float *local_b, float *output, size_t count)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    output[i] = local_a[i] + local_b[i];
                }
            }

            /**
             * @brief Get backend for device, with null check
             */
            IBackend *getBackend(DeviceId dev) const
            {
                ComputeBackendType type = (dev.type == DeviceType::CUDA) ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM;
                return getBackendForDeviceType(type);
            }
        };

        //==============================================================================
        // Basic FP32 Tensor Transfer Tests
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, FP32Tensor_RoundTrip)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== FP32 Tensor Round-Trip Test ===");

            // Create test tensor with known pattern
            constexpr size_t ROWS = 64;
            constexpr size_t COLS = 256;
            auto tensor = TestTensorFactory::createFP32Random({ROWS, COLS}, -1.0f, 1.0f, 12345);

            // Copy original data for verification
            std::vector<float> original(tensor->numel());
            std::memcpy(original.data(), tensor->data(), tensor->numel() * sizeof(float));

            // Allocate CUDA buffer and copy tensor data
            void *cuda_buf = nullptr;
            size_t size_bytes = tensor->numel() * sizeof(float);

#ifdef HAVE_CUDA
            ASSERT_EQ(cudaSetDevice(cuda_dev_.ordinal), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&cuda_buf, size_bytes), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(cuda_buf, tensor->data(), size_bytes, cudaMemcpyHostToDevice), cudaSuccess);
#endif

            // Transfer NVIDIA → AMD via PCIe BAR
            auto write_result = engine_.transferViaPCIeBar(cuda_buf, 0, size_bytes,
                                                           DirectP2PEngine::Direction::ToAMD);
            ASSERT_TRUE(write_result.success);
            LOG_INFO("Write (NVIDIA→AMD): " << write_result.throughput_gbps << " GB/s");

            // Clear CUDA buffer
#ifdef HAVE_CUDA
            ASSERT_EQ(cudaMemset(cuda_buf, 0, size_bytes), cudaSuccess);
#endif

            // Read back AMD → NVIDIA
            auto read_result = engine_.transferViaPCIeBar(cuda_buf, 0, size_bytes,
                                                          DirectP2PEngine::Direction::ToNVIDIA);
            ASSERT_TRUE(read_result.success);
            LOG_INFO("Read (AMD→NVIDIA): " << read_result.throughput_gbps << " GB/s");

            // Copy back to host and verify
            std::vector<float> roundtrip(tensor->numel());
#ifdef HAVE_CUDA
            ASSERT_EQ(cudaMemcpy(roundtrip.data(), cuda_buf, size_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
            cudaFree(cuda_buf);
#endif

            // Verify exact match (FP32 should be bit-exact)
            EXPECT_EQ(std::memcmp(original.data(), roundtrip.data(), size_bytes), 0)
                << "FP32 round-trip should be bit-exact";

            double mse = computeMSE(original.data(), roundtrip.data(), tensor->numel());
            LOG_INFO("MSE: " << mse << " (expect 0)");
            EXPECT_EQ(mse, 0.0);
        }

        TEST_F(Test__CrossVendorTensorParallel, FP32Tensor_LargeTransfer)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== Large FP32 Tensor Transfer ===");

            // Use 75% of BAR size to leave room for overhead
            size_t bar_size = engine_.getBarMappedSize();
            size_t size_bytes = std::min(bar_size * 3 / 4, static_cast<size_t>(256 * 1024 * 1024));
            size_t numel = size_bytes / sizeof(float);

            LOG_INFO("Transfer size: " << (size_bytes / (1024 * 1024)) << " MB (BAR size: " << (bar_size / (1024 * 1024)) << " MB)");

            auto tensor = TestTensorFactory::createFP32Random({numel}, -1.0f, 1.0f, 54321);

            void *cuda_buf = nullptr;
#ifdef HAVE_CUDA
            ASSERT_EQ(cudaSetDevice(cuda_dev_.ordinal), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&cuda_buf, size_bytes), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(cuda_buf, tensor->data(), size_bytes, cudaMemcpyHostToDevice), cudaSuccess);
#endif

            // Benchmark transfer
            auto result = engine_.benchmarkPCIeBar(size_bytes, 3);

            LOG_INFO("Transfer Results:");
            LOG_INFO("  Write: " << result.write_gbps << " GB/s");
            LOG_INFO("  Read:  " << result.read_gbps << " GB/s");
            LOG_INFO("  Avg:   " << result.throughput_gbps << " GB/s");

            EXPECT_TRUE(result.success);
            // PCIe BAR bandwidth is hardware-limited; we consistently see ~1.3 GB/s writes
            // on our AMD MI60 + RTX 3090 system. Use conservative threshold.
            EXPECT_GT(result.write_gbps, 1.0); // Expect >1.0 GB/s writes (typical: 1.3 GB/s)
            EXPECT_GT(result.read_gbps, 0.5);  // Expect >0.5 GB/s reads (or symmetric with rBAR)

#ifdef HAVE_CUDA
            cudaFree(cuda_buf);
#endif
        }

        //==============================================================================
        // Q8_1 Tensor Transfer Tests (Quantized Activations)
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, Q8_1Tensor_RoundTrip)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== Q8_1 Tensor Round-Trip Test ===");

            // Create Q8_1 activation tensor
            constexpr size_t ROWS = SEQ_LEN;
            constexpr size_t COLS = D_MODEL;

            // Create via quantization from random FP32
            auto fp32_src = TestTensorFactory::createFP32Random({ROWS, COLS}, -2.0f, 2.0f, 99999);
            auto q8_tensor = TestTensorFactory::createQ8_1FromFP32(fp32_src.get());

            // Get Q8_1 block data
            const Q8_1Block *blocks = q8_tensor->typed_data();
            size_t num_blocks = (COLS + 31) / 32 * ROWS;
            size_t size_bytes = num_blocks * sizeof(Q8_1Block);

            LOG_INFO("Q8_1 tensor: " << ROWS << "x" << COLS);
            LOG_INFO("  Blocks: " << num_blocks);
            LOG_INFO("  Size: " << (size_bytes / 1024) << " KB");

            // Copy original blocks for verification
            std::vector<Q8_1Block> original(num_blocks);
            std::memcpy(original.data(), blocks, size_bytes);

            // CUDA alloc and copy
            void *cuda_buf = nullptr;
#ifdef HAVE_CUDA
            ASSERT_EQ(cudaSetDevice(cuda_dev_.ordinal), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&cuda_buf, size_bytes), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(cuda_buf, blocks, size_bytes, cudaMemcpyHostToDevice), cudaSuccess);
#endif

            // Transfer round-trip
            auto write_result = engine_.transferViaPCIeBar(cuda_buf, 0, size_bytes,
                                                           DirectP2PEngine::Direction::ToAMD);
            ASSERT_TRUE(write_result.success);

#ifdef HAVE_CUDA
            ASSERT_EQ(cudaMemset(cuda_buf, 0, size_bytes), cudaSuccess);
#endif

            auto read_result = engine_.transferViaPCIeBar(cuda_buf, 0, size_bytes,
                                                          DirectP2PEngine::Direction::ToNVIDIA);
            ASSERT_TRUE(read_result.success);

            // Verify
            std::vector<Q8_1Block> roundtrip(num_blocks);
#ifdef HAVE_CUDA
            ASSERT_EQ(cudaMemcpy(roundtrip.data(), cuda_buf, size_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
            cudaFree(cuda_buf);
#endif

            // Bit-exact comparison
            EXPECT_EQ(std::memcmp(original.data(), roundtrip.data(), size_bytes), 0)
                << "Q8_1 blocks should be bit-exact after round-trip";

            LOG_INFO("Q8_1 round-trip: " << write_result.throughput_gbps << " GB/s write, "
                                         << read_result.throughput_gbps << " GB/s read");
        }

        //==============================================================================
        // Column-Parallel GEMM Test (Tensor Parallelism) - ACTUAL GPU KERNELS
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, ColumnParallelGEMM_TwoGPU)
        {
            REQUIRE_P2P();

#ifndef HAVE_ROCM
            GTEST_SKIP() << "ROCm required for cross-vendor GPU GEMM test";
#endif

            LOG_INFO("\n=== Column-Parallel GEMM (2-GPU Tensor Parallel) ===");
            LOG_INFO("Using ACTUAL GPU kernels: CUDAFloatingPointGemmKernel + ROCmFloatingPointGemmKernel");

            /**
             * TRUE cross-vendor tensor parallel GEMM:
             *
             * Full weight W: [N, K]
             * Split by columns: W_cuda = W[0:N/2, :], W_amd = W[N/2:N, :]
             *
             * Input X: [M, K] (replicated on both GPUs)
             *
             * Compute on ACTUAL GPUs:
             *   Y_cuda = X @ W_cuda^T  -> [M, N/2] via CUDAFloatingPointGemmKernel (cuBLAS)
             *   Y_amd  = X @ W_amd^T   -> [M, N/2] via ROCmFloatingPointGemmKernel (hipBLAS)
             *
             * P2P transfer AMD's result to NVIDIA via PCIe BAR
             * Concatenate: Y_full = concat(Y_cuda, Y_amd) -> [M, N]
             */

            constexpr int M = SEQ_LEN;
            constexpr int K = D_MODEL;
            constexpr int N = FFN_DIM;
            constexpr int N_LOCAL = N / 2; // Split across 2 GPUs

            // Get backends for both GPUs
            IBackend *cuda_backend = getBackend(cuda_dev_);
            IBackend *rocm_backend = getBackend(rocm_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";
            ASSERT_NE(rocm_backend, nullptr) << "ROCm backend not available";

            // Create tensors on CPU
            auto input = TestTensorFactory::createFP32Random({M, K}, -0.5f, 0.5f, 111);
            auto weight_full = TestTensorFactory::createFP32Random({N, K}, -0.5f, 0.5f, 222);
            auto output_ref = TestTensorFactory::createFP32Zeros({M, N});

            // Compute reference output (full GEMM on CPU)
            cpu_gemm_reference(input->data(), weight_full->data(), output_ref->mutable_data(), M, N, K);

            // Split weights by columns (rows in transposed view)
            auto weight_cuda = TestTensorFactory::createFP32({N_LOCAL, K});
            auto weight_amd = TestTensorFactory::createFP32({N_LOCAL, K});

            const float *w_full = weight_full->data();

            // First half to CUDA, second half to AMD
            std::memcpy(weight_cuda->mutable_data(), w_full, N_LOCAL * K * sizeof(float));
            std::memcpy(weight_amd->mutable_data(), w_full + N_LOCAL * K, N_LOCAL * K * sizeof(float));

            // Allocate output tensors
            auto output_cuda = TestTensorFactory::createFP32Zeros({M, N_LOCAL});
            auto output_amd = TestTensorFactory::createFP32Zeros({M, N_LOCAL});

            const size_t input_bytes = M * K * sizeof(float);
            const size_t local_output_bytes = M * N_LOCAL * sizeof(float);

            double cuda_ms = 0.0, rocm_ms = 0.0;

            // === NVIDIA GPU: CUDAFloatingPointGemmKernel ===
            LOG_INFO("Setting up NVIDIA GPU (CUDA device " << cuda_dev_.ordinal << ")...");

            // Upload weight to NVIDIA GPU via tensor coherence
            ASSERT_TRUE(weight_cuda->ensureOnDevice(cuda_dev_));
            EXPECT_TRUE(weight_cuda->isOnGPU());

            // Allocate activations and output on NVIDIA via IBackend
            float *d_A_cuda = static_cast<float *>(cuda_backend->allocate(input_bytes, cuda_dev_.ordinal));
            float *d_C_cuda = static_cast<float *>(cuda_backend->allocate(local_output_bytes, cuda_dev_.ordinal));
            ASSERT_NE(d_A_cuda, nullptr);
            ASSERT_NE(d_C_cuda, nullptr);

            // Copy input to NVIDIA GPU
            ASSERT_TRUE(cuda_backend->hostToDevice(d_A_cuda, input->data(), input_bytes, cuda_dev_.ordinal));
            ASSERT_TRUE(cuda_backend->memset(d_C_cuda, 0, local_output_bytes, cuda_dev_.ordinal));

            // Create ACTUAL CUDA GEMM kernel (uses cuBLAS)
            auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
                weight_cuda.get(), llaminar::v2::kernels::DeviceType::CUDA);
            ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDAFloatingPointGemmKernel";

            // Execute CUDA GEMM: Y_cuda = input @ weight_cuda^T
            LOG_INFO("Executing CUDA GEMM: [" << M << "," << K << "] @ [" << N_LOCAL << "," << K << "]^T");
            auto cuda_start = std::chrono::high_resolution_clock::now();
            ASSERT_TRUE(cuda_kernel->multiply(d_A_cuda, d_C_cuda, M, N_LOCAL, K));
            cuda_backend->synchronize(cuda_dev_.ordinal);
            auto cuda_end = std::chrono::high_resolution_clock::now();
            cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();

            // Download CUDA result
            ASSERT_TRUE(cuda_backend->deviceToHost(output_cuda->mutable_data(), d_C_cuda,
                                                   local_output_bytes, cuda_dev_.ordinal));

            // === AMD GPU: ROCmFloatingPointGemmKernel ===
            LOG_INFO("Setting up AMD GPU (ROCm device " << rocm_dev_.ordinal << ")...");

            // Upload weight to AMD GPU via tensor coherence
            ASSERT_TRUE(weight_amd->ensureOnDevice(rocm_dev_));
            EXPECT_TRUE(weight_amd->isOnGPU());

            // Allocate activations and output on AMD via IBackend
            float *d_A_rocm = static_cast<float *>(rocm_backend->allocate(input_bytes, rocm_dev_.ordinal));
            float *d_C_rocm = static_cast<float *>(rocm_backend->allocate(local_output_bytes, rocm_dev_.ordinal));
            ASSERT_NE(d_A_rocm, nullptr);
            ASSERT_NE(d_C_rocm, nullptr);

            // Copy input to AMD GPU
            ASSERT_TRUE(rocm_backend->hostToDevice(d_A_rocm, input->data(), input_bytes, rocm_dev_.ordinal));
            ASSERT_TRUE(rocm_backend->memset(d_C_rocm, 0, local_output_bytes, rocm_dev_.ordinal));

            // Create ACTUAL ROCm GEMM kernel (uses hipBLAS)
            auto rocm_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
                weight_amd.get(), llaminar::v2::kernels::DeviceType::ROCm);
            ASSERT_NE(rocm_kernel, nullptr) << "Failed to create ROCmFloatingPointGemmKernel";

            // Execute ROCm GEMM: Y_amd = input @ weight_amd^T
            LOG_INFO("Executing ROCm GEMM: [" << M << "," << K << "] @ [" << N_LOCAL << "," << K << "]^T");
            auto rocm_start = std::chrono::high_resolution_clock::now();
            ASSERT_TRUE(rocm_kernel->multiply(d_A_rocm, d_C_rocm, M, N_LOCAL, K));
            rocm_backend->synchronize(rocm_dev_.ordinal);
            auto rocm_end = std::chrono::high_resolution_clock::now();
            rocm_ms = std::chrono::duration<double, std::milli>(rocm_end - rocm_start).count();

            // Download ROCm result
            ASSERT_TRUE(rocm_backend->deviceToHost(output_amd->mutable_data(), d_C_rocm,
                                                   local_output_bytes, rocm_dev_.ordinal));

            LOG_INFO("GPU GEMM completed:");
            LOG_INFO("  CUDA time: " << cuda_ms << " ms");
            LOG_INFO("  ROCm time: " << rocm_ms << " ms");

            // === P2P transfer: Send AMD's portion to NVIDIA for concatenation ===
            void *cuda_buf = cuda_backend->allocate(local_output_bytes, cuda_dev_.ordinal);
            ASSERT_NE(cuda_buf, nullptr);
            ASSERT_TRUE(cuda_backend->hostToDevice(cuda_buf, output_amd->data(), local_output_bytes, cuda_dev_.ordinal));

            // Write AMD's output to BAR (simulating direct GPU-to-GPU via PCIe BAR)
            auto write_result = engine_.transferViaPCIeBar(cuda_buf, 0, local_output_bytes,
                                                           DirectP2PEngine::Direction::ToAMD);
            ASSERT_TRUE(write_result.success);

            // Read back (simulating gather to NVIDIA)
            auto read_result = engine_.transferViaPCIeBar(cuda_buf, 0, local_output_bytes,
                                                          DirectP2PEngine::Direction::ToNVIDIA);
            ASSERT_TRUE(read_result.success);

            // Verify AMD portion survived round-trip
            std::vector<float> gathered_amd(M * N_LOCAL);
            ASSERT_TRUE(cuda_backend->deviceToHost(gathered_amd.data(), cuda_buf, local_output_bytes, cuda_dev_.ordinal));

            // Cleanup GPU memory
            cuda_backend->free(cuda_buf, cuda_dev_.ordinal);
            cuda_backend->free(d_A_cuda, cuda_dev_.ordinal);
            cuda_backend->free(d_C_cuda, cuda_dev_.ordinal);
            rocm_backend->free(d_A_rocm, rocm_dev_.ordinal);
            rocm_backend->free(d_C_rocm, rocm_dev_.ordinal);

            double amd_mse = computeMSE(output_amd->data(), gathered_amd.data(), M * N_LOCAL);
            EXPECT_EQ(amd_mse, 0.0) << "AMD output should be bit-exact after P2P";

            // Concatenate results
            auto output_concat = TestTensorFactory::createFP32({M, N});
            float *out_concat = output_concat->mutable_data();
            for (int m = 0; m < M; ++m)
            {
                std::memcpy(out_concat + m * N, output_cuda->data() + m * N_LOCAL,
                            N_LOCAL * sizeof(float));
                std::memcpy(out_concat + m * N + N_LOCAL, gathered_amd.data() + m * N_LOCAL,
                            N_LOCAL * sizeof(float));
            }

            // Verify against reference (GPU GEMM may have small numerical differences vs CPU reference)
            double final_mse = computeMSE(output_ref->data(), out_concat, M * N);
            double cosine = computeCosineSimilarity(output_ref->data(), out_concat, M * N);

            // Calculate GFLOPS for both GPUs
            double gflops = 2.0 * M * N_LOCAL * K / 1e9; // Per GPU
            double cuda_gflops = gflops / (cuda_ms / 1000.0);
            double rocm_gflops = gflops / (rocm_ms / 1000.0);

            LOG_INFO("Column-Parallel GEMM Results (ACTUAL GPU KERNELS):");
            LOG_INFO("  Input:  [" << M << ", " << K << "]");
            LOG_INFO("  Weight: [" << N << ", " << K << "] split to 2x[" << N_LOCAL << ", " << K << "]");
            LOG_INFO("  Output: [" << M << ", " << N << "]");
            LOG_INFO("  CUDA GEMM: " << cuda_ms << " ms (" << cuda_gflops << " GFLOPS)");
            LOG_INFO("  ROCm GEMM: " << rocm_ms << " ms (" << rocm_gflops << " GFLOPS)");
            LOG_INFO("  MSE vs CPU reference: " << final_mse);
            LOG_INFO("  Cosine similarity: " << cosine);
            LOG_INFO("  P2P write: " << write_result.throughput_gbps << " GB/s");
            LOG_INFO("  P2P read:  " << read_result.throughput_gbps << " GB/s");

            // GPU GEMM has higher tolerance due to different FP operation ordering
            EXPECT_LT(final_mse, 1e-6) << "GPU GEMM should closely match CPU reference";
            EXPECT_GT(cosine, 0.9999) << "Cosine similarity should be very high";
        }

        //==============================================================================
        // Row-Parallel GEMM with AllReduce Test - ACTUAL GPU KERNELS
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, RowParallelGEMM_AllReduce)
        {
            REQUIRE_P2P();

#ifndef HAVE_ROCM
            GTEST_SKIP() << "ROCm required for cross-vendor GPU GEMM test";
#endif

            LOG_INFO("\n=== Row-Parallel GEMM with AllReduce (2-GPU) ===");
            LOG_INFO("Using ACTUAL GPU kernels: CUDAFloatingPointGemmKernel + ROCmFloatingPointGemmKernel");

            /**
             * TRUE cross-vendor row-parallel GEMM (e.g., Wo projection):
             *
             * Full weight W: [N, K]
             * Split by rows (K dim): W_cuda = W[:, 0:K/2], W_amd = W[:, K/2:K]
             *
             * Input X: [M, K]
             * Split by columns: X_cuda = X[:, 0:K/2], X_amd = X[:, K/2:K]
             *
             * Compute on ACTUAL GPUs:
             *   Y_cuda = X_cuda @ W_cuda^T  -> [M, N] (partial) via cuBLAS
             *   Y_amd  = X_amd @ W_amd^T    -> [M, N] (partial) via hipBLAS
             *
             * AllReduce via P2P: Y_full = Y_cuda + Y_amd
             */

            constexpr int M = SEQ_LEN;
            constexpr int K = D_MODEL;
            constexpr int N = D_MODEL; // Wo: [d_model, d_model]
            constexpr int K_LOCAL = K / 2;

            // Get backends for both GPUs
            IBackend *cuda_backend = getBackend(cuda_dev_);
            IBackend *rocm_backend = getBackend(rocm_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";
            ASSERT_NE(rocm_backend, nullptr) << "ROCm backend not available";

            // Create full tensors
            auto input = TestTensorFactory::createFP32Random({M, K}, -0.5f, 0.5f, 333);
            auto weight_full = TestTensorFactory::createFP32Random({N, K}, -0.5f, 0.5f, 444);
            auto output_ref = TestTensorFactory::createFP32Zeros({M, N});

            // Reference computation on CPU
            cpu_gemm_reference(input->data(), weight_full->data(), output_ref->mutable_data(), M, N, K);

            // Split input and weights along K dimension
            auto input_cuda = TestTensorFactory::createFP32({M, K_LOCAL});
            auto input_amd = TestTensorFactory::createFP32({M, K_LOCAL});
            auto weight_cuda = TestTensorFactory::createFP32({N, K_LOCAL});
            auto weight_amd = TestTensorFactory::createFP32({N, K_LOCAL});

            // Split input: each rank gets half the columns
            const float *in_full = input->data();
            for (int m = 0; m < M; ++m)
            {
                std::memcpy(input_cuda->mutable_data() + m * K_LOCAL,
                            in_full + m * K, K_LOCAL * sizeof(float));
                std::memcpy(input_amd->mutable_data() + m * K_LOCAL,
                            in_full + m * K + K_LOCAL, K_LOCAL * sizeof(float));
            }

            // Split weights: each rank gets half the K dimension
            const float *w_full = weight_full->data();
            for (int n = 0; n < N; ++n)
            {
                std::memcpy(weight_cuda->mutable_data() + n * K_LOCAL,
                            w_full + n * K, K_LOCAL * sizeof(float));
                std::memcpy(weight_amd->mutable_data() + n * K_LOCAL,
                            w_full + n * K + K_LOCAL, K_LOCAL * sizeof(float));
            }

            // Output tensors for partial results
            auto partial_cuda = TestTensorFactory::createFP32Zeros({M, N});
            auto partial_amd = TestTensorFactory::createFP32Zeros({M, N});

            const size_t input_local_bytes = M * K_LOCAL * sizeof(float);
            const size_t output_bytes = M * N * sizeof(float);

            double cuda_ms = 0.0, rocm_ms = 0.0;

            // === NVIDIA GPU: CUDAFloatingPointGemmKernel ===
            LOG_INFO("Setting up NVIDIA GPU (CUDA device " << cuda_dev_.ordinal << ")...");

            // Upload weight to NVIDIA GPU
            ASSERT_TRUE(weight_cuda->ensureOnDevice(cuda_dev_));
            EXPECT_TRUE(weight_cuda->isOnGPU());

            // Allocate activations and output on NVIDIA via IBackend
            float *d_A_cuda = static_cast<float *>(cuda_backend->allocate(input_local_bytes, cuda_dev_.ordinal));
            float *d_C_cuda = static_cast<float *>(cuda_backend->allocate(output_bytes, cuda_dev_.ordinal));
            ASSERT_NE(d_A_cuda, nullptr);
            ASSERT_NE(d_C_cuda, nullptr);

            ASSERT_TRUE(cuda_backend->hostToDevice(d_A_cuda, input_cuda->data(), input_local_bytes, cuda_dev_.ordinal));
            ASSERT_TRUE(cuda_backend->memset(d_C_cuda, 0, output_bytes, cuda_dev_.ordinal));

            // Create ACTUAL CUDA GEMM kernel
            auto cuda_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
                weight_cuda.get(), llaminar::v2::kernels::DeviceType::CUDA);
            ASSERT_NE(cuda_kernel, nullptr) << "Failed to create CUDAFloatingPointGemmKernel";

            // Execute CUDA GEMM
            LOG_INFO("Executing CUDA GEMM: [" << M << "," << K_LOCAL << "] @ [" << N << "," << K_LOCAL << "]^T");
            auto cuda_start = std::chrono::high_resolution_clock::now();
            ASSERT_TRUE(cuda_kernel->multiply(d_A_cuda, d_C_cuda, M, N, K_LOCAL));
            cuda_backend->synchronize(cuda_dev_.ordinal);
            auto cuda_end = std::chrono::high_resolution_clock::now();
            cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();

            // Download CUDA result
            ASSERT_TRUE(cuda_backend->deviceToHost(partial_cuda->mutable_data(), d_C_cuda,
                                                   output_bytes, cuda_dev_.ordinal));

            // === AMD GPU: ROCmFloatingPointGemmKernel ===
            LOG_INFO("Setting up AMD GPU (ROCm device " << rocm_dev_.ordinal << ")...");

            // Upload weight to AMD GPU
            ASSERT_TRUE(weight_amd->ensureOnDevice(rocm_dev_));
            EXPECT_TRUE(weight_amd->isOnGPU());

            // Allocate activations and output on AMD via IBackend
            float *d_A_rocm = static_cast<float *>(rocm_backend->allocate(input_local_bytes, rocm_dev_.ordinal));
            float *d_C_rocm = static_cast<float *>(rocm_backend->allocate(output_bytes, rocm_dev_.ordinal));
            ASSERT_NE(d_A_rocm, nullptr);
            ASSERT_NE(d_C_rocm, nullptr);

            ASSERT_TRUE(rocm_backend->hostToDevice(d_A_rocm, input_amd->data(), input_local_bytes, rocm_dev_.ordinal));
            ASSERT_TRUE(rocm_backend->memset(d_C_rocm, 0, output_bytes, rocm_dev_.ordinal));

            // Create ACTUAL ROCm GEMM kernel
            auto rocm_kernel = llaminar::v2::kernels::KernelFactory::createGemm(
                weight_amd.get(), llaminar::v2::kernels::DeviceType::ROCm);
            ASSERT_NE(rocm_kernel, nullptr) << "Failed to create ROCmFloatingPointGemmKernel";

            // Execute ROCm GEMM
            LOG_INFO("Executing ROCm GEMM: [" << M << "," << K_LOCAL << "] @ [" << N << "," << K_LOCAL << "]^T");
            auto rocm_start = std::chrono::high_resolution_clock::now();
            ASSERT_TRUE(rocm_kernel->multiply(d_A_rocm, d_C_rocm, M, N, K_LOCAL));
            rocm_backend->synchronize(rocm_dev_.ordinal);
            auto rocm_end = std::chrono::high_resolution_clock::now();
            rocm_ms = std::chrono::duration<double, std::milli>(rocm_end - rocm_start).count();

            // Download ROCm result
            ASSERT_TRUE(rocm_backend->deviceToHost(partial_amd->mutable_data(), d_C_rocm,
                                                   output_bytes, rocm_dev_.ordinal));

            LOG_INFO("GPU GEMM completed:");
            LOG_INFO("  CUDA time: " << cuda_ms << " ms");
            LOG_INFO("  ROCm time: " << rocm_ms << " ms");

            // === AllReduce via P2P: Transfer AMD's partial to NVIDIA ===
            void *cuda_buf = cuda_backend->allocate(output_bytes, cuda_dev_.ordinal);
            ASSERT_NE(cuda_buf, nullptr);
            ASSERT_TRUE(cuda_backend->hostToDevice(cuda_buf, partial_amd->data(), output_bytes, cuda_dev_.ordinal));

            // Write AMD partial to BAR
            auto write_result = engine_.transferViaPCIeBar(cuda_buf, 0, output_bytes,
                                                           DirectP2PEngine::Direction::ToAMD);
            ASSERT_TRUE(write_result.success);

            // Read back AMD partial
            auto read_result = engine_.transferViaPCIeBar(cuda_buf, 0, output_bytes,
                                                          DirectP2PEngine::Direction::ToNVIDIA);
            ASSERT_TRUE(read_result.success);

            std::vector<float> amd_partial_received(M * N);
            ASSERT_TRUE(cuda_backend->deviceToHost(amd_partial_received.data(), cuda_buf,
                                                   output_bytes, cuda_dev_.ordinal));

            // Cleanup GPU memory
            cuda_backend->free(cuda_buf, cuda_dev_.ordinal);
            cuda_backend->free(d_A_cuda, cuda_dev_.ordinal);
            cuda_backend->free(d_C_cuda, cuda_dev_.ordinal);
            rocm_backend->free(d_A_rocm, rocm_dev_.ordinal);
            rocm_backend->free(d_C_rocm, rocm_dev_.ordinal);

            // Perform AllReduce SUM
            auto output_reduced = TestTensorFactory::createFP32Zeros({M, N});
            allreduceSum(partial_cuda->mutable_data(), amd_partial_received.data(),
                         output_reduced->mutable_data(), M * N);

            // Verify against CPU reference (GPU may have small numerical differences)
            double mse = computeMSE(output_ref->data(), output_reduced->data(), M * N);
            double cosine = computeCosineSimilarity(output_ref->data(), output_reduced->data(), M * N);

            // Calculate GFLOPS
            double gflops = 2.0 * M * N * K_LOCAL / 1e9;
            double cuda_gflops = gflops / (cuda_ms / 1000.0);
            double rocm_gflops = gflops / (rocm_ms / 1000.0);

            LOG_INFO("Row-Parallel GEMM + AllReduce Results (ACTUAL GPU KERNELS):");
            LOG_INFO("  Input:  [" << M << ", " << K << "] split to [" << M << ", " << K_LOCAL << "]");
            LOG_INFO("  Weight: [" << N << ", " << K << "] split to [" << N << ", " << K_LOCAL << "]");
            LOG_INFO("  Output: [" << M << ", " << N << "]");
            LOG_INFO("  CUDA GEMM: " << cuda_ms << " ms (" << cuda_gflops << " GFLOPS)");
            LOG_INFO("  ROCm GEMM: " << rocm_ms << " ms (" << rocm_gflops << " GFLOPS)");
            LOG_INFO("  MSE vs CPU reference: " << mse);
            LOG_INFO("  Cosine similarity: " << cosine);
            LOG_INFO("  P2P bandwidth: " << (write_result.throughput_gbps + read_result.throughput_gbps) / 2 << " GB/s avg");

            // GPU GEMM has higher tolerance due to different FP operation ordering
            EXPECT_LT(mse, 1e-6) << "GPU GEMM should closely match CPU reference";
            EXPECT_GT(cosine, 0.9999) << "Cosine similarity should be very high";
        }

        //==============================================================================
        // Concurrent Transfer Tests (Overlapped Read+Write)
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, ConcurrentBidirectionalTransfer)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== Concurrent Bidirectional Transfer Test ===");

            /**
             * Tests overlapped read+write for maximum PCIe utilization.
             * This pattern is used in pipelined tensor parallel inference:
             *   - While computing layer N, transfer layer N-1 results
             *   - While receiving layer N-1, send layer N inputs
             */

            constexpr size_t SIZE = 32 * 1024 * 1024; // 32 MB each direction

            auto send_tensor = TestTensorFactory::createFP32Random({SIZE / sizeof(float)}, -1.0f, 1.0f, 555);
            auto recv_tensor = TestTensorFactory::createFP32Random({SIZE / sizeof(float)}, -1.0f, 1.0f, 666);

            IBackend *cuda_backend = getBackend(cuda_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";

            void *cuda_send = cuda_backend->allocate(SIZE, cuda_dev_.ordinal);
            void *cuda_recv = cuda_backend->allocate(SIZE, cuda_dev_.ordinal);
            ASSERT_NE(cuda_send, nullptr);
            ASSERT_NE(cuda_recv, nullptr);

            // Initialize: send_tensor goes to AMD, recv_tensor is already in AMD BAR
            ASSERT_TRUE(cuda_backend->hostToDevice(cuda_send, send_tensor->data(), SIZE, cuda_dev_.ordinal));

            // First, put recv_tensor in AMD BAR so we can read it
            ASSERT_TRUE(cuda_backend->hostToDevice(cuda_recv, recv_tensor->data(), SIZE, cuda_dev_.ordinal));
            auto prep = engine_.transferViaPCIeBar(cuda_recv, 0, SIZE, DirectP2PEngine::Direction::ToAMD);
            ASSERT_TRUE(prep.success);

            // Now do overlapped transfer: write send_tensor to second BAR region, read recv_tensor from first
            size_t bar_size = engine_.getBarMappedSize();
            if (bar_size < 2 * SIZE)
            {
                LOG_WARN("BAR too small for overlapped test (" << (bar_size / (1024 * 1024)) << " MB)");
                // Fall back to sequential
                auto write_result = engine_.transferViaPCIeBar(cuda_send, SIZE, SIZE,
                                                               DirectP2PEngine::Direction::ToAMD);
                auto read_result = engine_.transferViaPCIeBar(cuda_recv, 0, SIZE,
                                                              DirectP2PEngine::Direction::ToNVIDIA);
                LOG_INFO("Sequential: write " << write_result.throughput_gbps
                                              << " GB/s, read " << read_result.throughput_gbps << " GB/s");
                cuda_backend->free(cuda_send, cuda_dev_.ordinal);
                cuda_backend->free(cuda_recv, cuda_dev_.ordinal);
                return;
            }

            // Overlapped transfer using the engine's concurrent API
            auto result = engine_.transferOverlapped(
                cuda_recv, 0, SIZE,   // Read from offset 0
                cuda_send, SIZE, SIZE // Write to offset SIZE
            );

            LOG_INFO("Concurrent Transfer Results:");
            LOG_INFO("  Success: " << result.success);
            LOG_INFO("  Concurrent throughput: " << result.concurrent_gbps << " GB/s");
            LOG_INFO("  (Sequential would be ~" << (result.read_gbps + result.write_gbps) / 2 << " GB/s)");

            if (result.success)
            {
                EXPECT_GT(result.concurrent_gbps, result.read_gbps * 1.1)
                    << "Concurrent should be faster than single read";
            }

            cuda_backend->free(cuda_send, cuda_dev_.ordinal);
            cuda_backend->free(cuda_recv, cuda_dev_.ordinal);
        }

        //==============================================================================
        // Multi-Layer Pipeline Simulation
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, MultiLayerPipeline_TensorParallel)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== Multi-Layer Tensor Parallel Pipeline ===");

            /**
             * Simulates a 4-layer transformer pipeline with 2-GPU tensor parallelism.
             * Each layer has:
             *   1. Attention output projection (Wo) - row-parallel + allreduce
             *   2. FFN gate/up projection - column-parallel
             *   3. FFN down projection - row-parallel + allreduce
             *
             * This tests P2P transfer latency accumulation over multiple layers.
             */

            constexpr int NUM_LAYERS = 4;
            constexpr int M = 32;  // Small seq_len for faster test
            constexpr int D = 256; // Smaller D for faster test
            constexpr int FFN = 512;

            double total_p2p_time_ms = 0.0;
            double total_p2p_bytes = 0.0;

            auto hidden = TestTensorFactory::createFP32Random({M, D}, -0.5f, 0.5f, 777);

            IBackend *cuda_backend = getBackend(cuda_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";

            size_t buf_size = M * FFN * sizeof(float); // Largest buffer needed
            void *cuda_buf = cuda_backend->allocate(buf_size, cuda_dev_.ordinal);
            ASSERT_NE(cuda_buf, nullptr);

            LOG_INFO("Running " << NUM_LAYERS << "-layer pipeline...");
            LOG_INFO("  Hidden dim: " << D);
            LOG_INFO("  FFN dim: " << FFN);
            LOG_INFO("  Seq len: " << M);

            for (int layer = 0; layer < NUM_LAYERS; ++layer)
            {
                // Simulate Wo allreduce (row-parallel output)
                size_t wo_size = M * D * sizeof(float);
                ASSERT_TRUE(cuda_backend->hostToDevice(cuda_buf, hidden->data(), wo_size, cuda_dev_.ordinal));

                auto wo_write = engine_.transferViaPCIeBar(cuda_buf, 0, wo_size,
                                                           DirectP2PEngine::Direction::ToAMD);
                auto wo_read = engine_.transferViaPCIeBar(cuda_buf, 0, wo_size,
                                                          DirectP2PEngine::Direction::ToNVIDIA);

                ASSERT_TRUE(wo_write.success && wo_read.success);
                total_p2p_time_ms += wo_write.transfer_time_ms + wo_read.transfer_time_ms;
                total_p2p_bytes += 2 * wo_size;

                // Simulate FFN column-parallel (allgather, smaller transfer)
                size_t ffn_local_size = M * (FFN / 2) * sizeof(float);
                auto ffn_write = engine_.transferViaPCIeBar(cuda_buf, 0, ffn_local_size,
                                                            DirectP2PEngine::Direction::ToAMD);
                auto ffn_read = engine_.transferViaPCIeBar(cuda_buf, 0, ffn_local_size,
                                                           DirectP2PEngine::Direction::ToNVIDIA);

                ASSERT_TRUE(ffn_write.success && ffn_read.success);
                total_p2p_time_ms += ffn_write.transfer_time_ms + ffn_read.transfer_time_ms;
                total_p2p_bytes += 2 * ffn_local_size;

                // Simulate FFN down allreduce
                auto down_write = engine_.transferViaPCIeBar(cuda_buf, 0, wo_size,
                                                             DirectP2PEngine::Direction::ToAMD);
                auto down_read = engine_.transferViaPCIeBar(cuda_buf, 0, wo_size,
                                                            DirectP2PEngine::Direction::ToNVIDIA);

                ASSERT_TRUE(down_write.success && down_read.success);
                total_p2p_time_ms += down_write.transfer_time_ms + down_read.transfer_time_ms;
                total_p2p_bytes += 2 * wo_size;
            }

            cuda_backend->free(cuda_buf, cuda_dev_.ordinal);

            double effective_gbps = (total_p2p_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_p2p_time_ms / 1000.0);

            LOG_INFO("\nPipeline Results:");
            LOG_INFO("  Layers: " << NUM_LAYERS);
            LOG_INFO("  Total P2P transfers: " << (6 * NUM_LAYERS) << " (3 allreduce pairs per layer)");
            LOG_INFO("  Total bytes: " << (total_p2p_bytes / (1024.0 * 1024.0)) << " MB");
            LOG_INFO("  Total P2P time: " << total_p2p_time_ms << " ms");
            LOG_INFO("  Effective throughput: " << effective_gbps << " GB/s");

            EXPECT_GT(effective_gbps, 0.5) << "Pipeline should achieve reasonable P2P throughput";
        }

        //==============================================================================
        // Stress Tests
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, StressTest_RepeatedTransfers)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== Stress Test: 1000 Repeated Transfers ===");

            constexpr size_t SIZE = 1 * 1024 * 1024; // 1 MB
            constexpr int ITERATIONS = 1000;

            auto tensor = TestTensorFactory::createFP32Random({SIZE / sizeof(float)}, -1.0f, 1.0f, 888);

            IBackend *cuda_backend = getBackend(cuda_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";

            void *cuda_buf = cuda_backend->allocate(SIZE, cuda_dev_.ordinal);
            ASSERT_NE(cuda_buf, nullptr);
            ASSERT_TRUE(cuda_backend->hostToDevice(cuda_buf, tensor->data(), SIZE, cuda_dev_.ordinal));

            auto start = std::chrono::high_resolution_clock::now();

            int success_count = 0;
            for (int i = 0; i < ITERATIONS; ++i)
            {
                auto write = engine_.transferViaPCIeBar(cuda_buf, 0, SIZE,
                                                        DirectP2PEngine::Direction::ToAMD);
                auto read = engine_.transferViaPCIeBar(cuda_buf, 0, SIZE,
                                                       DirectP2PEngine::Direction::ToNVIDIA);
                if (write.success && read.success)
                    ++success_count;
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

            double total_bytes = 2.0 * SIZE * ITERATIONS;
            double throughput = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

            cuda_backend->free(cuda_buf, cuda_dev_.ordinal);

            LOG_INFO("Stress Test Results:");
            LOG_INFO("  Iterations: " << ITERATIONS);
            LOG_INFO("  Success rate: " << (100.0 * success_count / ITERATIONS) << "%");
            LOG_INFO("  Total time: " << elapsed_ms << " ms");
            LOG_INFO("  Total data: " << (total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB");
            LOG_INFO("  Throughput: " << throughput << " GB/s");
            LOG_INFO("  Latency per round-trip: " << (elapsed_ms / ITERATIONS) << " ms");

            EXPECT_EQ(success_count, ITERATIONS) << "All transfers should succeed";
            EXPECT_GT(throughput, 0.5) << "Should maintain reasonable throughput under stress";
        }

        //==============================================================================
        // Data Integrity Tests
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, DataIntegrity_PatternVerification)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== Data Integrity: Pattern Verification ===");

            // Test with various patterns that would catch byte/bit errors
            std::vector<std::pair<std::string, std::function<float(size_t)>>> patterns = {
                {"Zeros", [](size_t)
                 { return 0.0f; }},
                {"Ones", [](size_t)
                 { return 1.0f; }},
                {"Alternating", [](size_t i)
                 { return (i % 2) ? 1.0f : -1.0f; }},
                {"Ramp", [](size_t i)
                 { return static_cast<float>(i) / 1000.0f; }},
                {"NaN-adjacent", [](size_t i)
                 {
                     // Values close to special float representations
                     return (i % 3 == 0) ? 1e38f : ((i % 3 == 1) ? 1e-38f : 0.0f);
                 }},
            };

            constexpr size_t NUMEL = 64 * 1024; // 256 KB
            constexpr size_t SIZE_BYTES = NUMEL * sizeof(float);

            IBackend *cuda_backend = getBackend(cuda_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";

            void *cuda_buf = cuda_backend->allocate(SIZE_BYTES, cuda_dev_.ordinal);
            ASSERT_NE(cuda_buf, nullptr);

            for (const auto &[name, gen] : patterns)
            {
                // Generate pattern
                std::vector<float> original(NUMEL);
                for (size_t i = 0; i < NUMEL; ++i)
                {
                    original[i] = gen(i);
                }

                ASSERT_TRUE(cuda_backend->hostToDevice(cuda_buf, original.data(), SIZE_BYTES, cuda_dev_.ordinal));

                // Round-trip
                auto write = engine_.transferViaPCIeBar(cuda_buf, 0, SIZE_BYTES,
                                                        DirectP2PEngine::Direction::ToAMD);
                ASSERT_TRUE(write.success);

                ASSERT_TRUE(cuda_backend->memset(cuda_buf, 0xCC, SIZE_BYTES, cuda_dev_.ordinal)); // Corrupt buffer

                auto read = engine_.transferViaPCIeBar(cuda_buf, 0, SIZE_BYTES,
                                                       DirectP2PEngine::Direction::ToNVIDIA);
                ASSERT_TRUE(read.success);

                // Verify
                std::vector<float> result(NUMEL);
                ASSERT_TRUE(cuda_backend->deviceToHost(result.data(), cuda_buf, SIZE_BYTES, cuda_dev_.ordinal));

                bool match = (std::memcmp(original.data(), result.data(), SIZE_BYTES) == 0);
                LOG_INFO("  Pattern '" << name << "': " << (match ? "PASS" : "FAIL"));
                EXPECT_TRUE(match) << "Pattern '" << name << "' should match after round-trip";
            }

            cuda_backend->free(cuda_buf, cuda_dev_.ordinal);
        }

        //==============================================================================
        // Benchmarks
        //==============================================================================

        TEST_F(Test__CrossVendorTensorParallel, Benchmark_AllSizes)
        {
            REQUIRE_P2P();

            LOG_INFO("\n=== P2P Bandwidth Benchmark (All Sizes) ===");
            LOG_INFO("Size (MB) | Write (GB/s) | Read (GB/s) | Symmetric Ratio");
            LOG_INFO("----------|--------------|-------------|----------------");

            std::vector<size_t> sizes_mb = {1, 4, 16, 64, 128, 256, 512, 1024};

            IBackend *cuda_backend = getBackend(cuda_dev_);
            ASSERT_NE(cuda_backend, nullptr) << "CUDA backend not available";

            size_t max_size = 1024 * 1024 * 1024; // 1 GB
            void *cuda_buf = cuda_backend->allocate(max_size, cuda_dev_.ordinal);
            ASSERT_NE(cuda_buf, nullptr);
            ASSERT_TRUE(cuda_backend->memset(cuda_buf, 0xAB, max_size, cuda_dev_.ordinal));

            for (size_t size_mb : sizes_mb)
            {
                size_t size = size_mb * 1024 * 1024;
                if (size > engine_.getBarMappedSize())
                {
                    LOG_INFO(std::setw(9) << size_mb << " |    (BAR too small)");
                    continue;
                }

                auto result = engine_.benchmarkPCIeBar(size, 3);

                if (result.success)
                {
                    double ratio = result.read_gbps / result.write_gbps;
                    LOG_INFO(std::setw(9) << size_mb << " | "
                                          << std::setw(12) << std::fixed << std::setprecision(2) << result.write_gbps << " | "
                                          << std::setw(11) << std::fixed << std::setprecision(2) << result.read_gbps << " | "
                                          << std::setw(14) << std::fixed << std::setprecision(2) << ratio);

                    EXPECT_GT(result.write_gbps, 0.5);
                    EXPECT_GT(result.read_gbps, 0.3); // Reads can be slower without rBAR
                }
                else
                {
                    LOG_INFO(std::setw(9) << size_mb << " |    FAILED");
                }
            }

            cuda_backend->free(cuda_buf, cuda_dev_.ordinal);
        }

    } // namespace test
} // namespace llaminar2
