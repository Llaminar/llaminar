/**
 * @file Test__Phase7_CUTLASS_Simple.cpp
 * @brief Simplified Phase 7 test without IQ4_NL complexity
 * 
 * Tests CUTLASS int8 GEMM path directly without quantization/dequantization
 * to verify the core GEMM functionality works.
 * 
 * @author David Sanftenberg
 * @date 2025-01-10
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

// CUTLASS int8 GEMM type
using CutlassGemm = cutlass::gemm::device::Gemm<
    int8_t,                                      // ElementA
    cutlass::layout::RowMajor,                   // LayoutA
    int8_t,                                      // ElementB
    cutlass::layout::RowMajor,                   // LayoutB
    int32_t,                                     // ElementOutput
    cutlass::layout::RowMajor,                   // LayoutC
    int32_t,                                     // ElementAccumulator
    cutlass::arch::OpClassSimt,                  // OpClass
    cutlass::arch::Sm61,                         // ArchTag
    cutlass::gemm::GemmShape<256, 128, 64>,      // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,        // WarpShape
    cutlass::gemm::GemmShape<1, 1, 4>,           // InstructionShape (DP4A)
    cutlass::epilogue::thread::LinearCombination<
        int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
>;

/**
 * @brief CPU reference int8 GEMM
 */
void cpu_gemm_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Test 1: Identity matrices (should give identity result)
 */
TEST(Phase7CUTLASSSimple, IdentityMatrices) {
    constexpr int M = 64, N = 64, K = 64;
    
    // Create identity matrices
    std::vector<int8_t> A(M * K, 0);
    std::vector<int8_t> B(K * N, 0);
    
    for (int i = 0; i < std::min(M, K); ++i) {
        A[i * K + i] = 1;
    }
    for (int i = 0; i < std::min(K, N); ++i) {
        B[i * N + i] = 1;
    }
    
    // CPU reference
    std::vector<int32_t> C_cpu(M * N);
    cpu_gemm_int8(A.data(), B.data(), C_cpu.data(), M, N, K);
    
    // GPU CUTLASS
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));
    
    cudaMemcpy(d_A, A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {1, 0}
    );
    
    cutlass::Status status = gemm_op(args);
    ASSERT_EQ(status, cutlass::Status::kSuccess) << "CUTLASS GEMM failed";
    
    std::vector<int32_t> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Compare
    int correct = 0;
    int total = M * N;
    for (int i = 0; i < total; ++i) {
        if (C_gpu[i] == C_cpu[i]) {
            correct++;
        }
    }
    
    float accuracy = (float)correct / total;
    std::cout << "Identity test:\n";
    std::cout << "  Correct: " << correct << "/" << total << " (" << (accuracy * 100.0f) << "%)\n";
    std::cout << "  CPU C[0,0]: " << C_cpu[0] << "\n";
    std::cout << "  GPU C[0,0]: " << C_gpu[0] << "\n";
    std::cout << "  CPU C[1,1]: " << C_cpu[1 * N + 1] << "\n";
    std::cout << "  GPU C[1,1]: " << C_gpu[1 * N + 1] << "\n";
    
    EXPECT_GT(accuracy, 0.99f) << "Identity pattern not preserved";
}

/**
 * @brief Test 2: Small random matrices
 */
TEST(Phase7CUTLASSSimple, SmallRandomMatrices) {
    constexpr int M = 64, N = 64, K = 64;
    
    // Random data
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10, 10);
    
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    
    for (auto& val : A) val = (int8_t)dist(gen);
    for (auto& val : B) val = (int8_t)dist(gen);
    
    // CPU reference
    std::vector<int32_t> C_cpu(M * N);
    cpu_gemm_int8(A.data(), B.data(), C_cpu.data(), M, N, K);
    
    // GPU CUTLASS
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));
    
    cudaMemcpy(d_A, A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {1, 0}
    );
    
    cutlass::Status status = gemm_op(args);
    ASSERT_EQ(status, cutlass::Status::kSuccess) << "CUTLASS GEMM failed";
    
    std::vector<int32_t> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Compare
    int correct = 0;
    int32_t max_diff = 0;
    for (int i = 0; i < M * N; ++i) {
        if (C_gpu[i] == C_cpu[i]) {
            correct++;
        }
        max_diff = std::max(max_diff, std::abs(C_gpu[i] - C_cpu[i]));
    }
    
    float accuracy = (float)correct / (M * N);
    std::cout << "Random test:\n";
    std::cout << "  Exact matches: " << correct << "/" << (M * N) << " (" << (accuracy * 100.0f) << "%)\n";
    std::cout << "  Max difference: " << max_diff << "\n";
    std::cout << "  CPU C[0,0]: " << C_cpu[0] << "\n";
    std::cout << "  GPU C[0,0]: " << C_gpu[0] << "\n";
    
    EXPECT_GT(accuracy, 0.99f) << "Too many mismatches";
    EXPECT_EQ(max_diff, 0) << "GEMM results should be bit-exact for int8×int8→int32";
}
