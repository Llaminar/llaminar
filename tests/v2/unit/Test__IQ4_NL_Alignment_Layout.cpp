/**
 * @file Test__IQ4_NL_Alignment_Layout.cpp
 * @brief Diagnostic tests for IQ4_NL tensor alignment and weight layout
 *
 * Investigates two potential performance regression causes:
 * 1. SIMD alignment issues (AlignedVector usage)
 * 2. Weight layout assumptions (transpose changes)
 *
 * @author David Sanftenberg
 * @date November 9, 2025
 */

#include <gtest/gtest.h>
#include "../../../src/v2/tensors/IQ4_NLTensor.h"
#include "../../../src/v2/loaders/ModelLoader.h"
#include "../../../src/v2/utils/MPIContext.h"
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace llaminar2;

class IQ4_NL_Alignment_Layout : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize MPI context (required for ModelLoader)
        mpi_ctx_ = std::make_shared<MPIContext>();

        // Load model
        loader_ = std::make_unique<ModelLoader>("models/qwen2.5-0.5b-instruct-iq4_nl.gguf", mpi_ctx_);
        ASSERT_TRUE(loader_->load()) << "Failed to load model";
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<ModelLoader> loader_;
};

/**
 * @brief Test 1: Check if IQ4_NL tensors use AlignedVector
 */
TEST_F(IQ4_NL_Alignment_Layout, CheckAlignedVectorUsage)
{
    // Load a weight tensor
    auto weight = loader_->loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(weight, nullptr);

    auto iq4_nl = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);
    ASSERT_NE(iq4_nl, nullptr) << "Weight is not IQ4_NL format";

    // Check raw data pointer alignment (use raw_blocks() for typed access)
    const uint8_t *raw_ptr = iq4_nl->raw_blocks();
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);

    std::cout << "\n=== IQ4_NL Tensor Alignment ===" << std::endl;
    std::cout << "Raw data pointer: 0x" << std::hex << addr << std::dec << std::endl;
    std::cout << "64-byte aligned: " << (addr % 64 == 0 ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "32-byte aligned: " << (addr % 32 == 0 ? "✅ YES" : "❌ NO") << std::endl;
    std::cout << "16-byte aligned: " << (addr % 16 == 0 ? "✅ YES" : "❌ NO") << std::endl;

    // SIMD operations typically require 16-byte (SSE), 32-byte (AVX2), or 64-byte (AVX512) alignment
    // IQ4_NL uses std::vector<uint8_t> which typically has 16-byte alignment
    // For optimal SIMD performance, we'd want AlignedVector with 64-byte alignment

    // Decode to FP32 and check alignment
    auto shape = iq4_nl->shape();
    size_t rows = shape[0];
    size_t cols = shape[1];

    std::vector<float> decoded(rows * cols);
    iq4_nl->decode_to_fp32(decoded.data(), rows, cols);

    uintptr_t decoded_addr = reinterpret_cast<uintptr_t>(decoded.data());
    std::cout << "\nDecoded FP32 pointer: 0x" << std::hex << decoded_addr << std::dec << std::endl;
    std::cout << "64-byte aligned: " << (decoded_addr % 64 == 0 ? "✅ YES" : "❌ NO") << std::endl;

    // Check if the issue is that IQ4_NL uses std::vector instead of AlignedVector
    std::cout << "\n⚠️  IQ4_NL currently uses std::vector<uint8_t> for raw_data_" << std::endl;
    std::cout << "    Should it use AlignedVector for better SIMD performance?" << std::endl;
}

/**
 * @brief Test 2: Verify weight layout assumptions
 */
TEST_F(IQ4_NL_Alignment_Layout, CheckWeightLayout)
{
    // Load Q-projection weight (should be [d_model, d_model] = [896, 896])
    auto q_weight = loader_->loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(q_weight, nullptr);

    auto iq4_nl = std::dynamic_pointer_cast<IQ4_NLTensor>(q_weight);
    ASSERT_NE(iq4_nl, nullptr);

    auto shape = iq4_nl->shape();

    std::cout << "\n=== IQ4_NL Weight Layout ===" << std::endl;
    std::cout << "blk.0.attn_q.weight shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
    std::cout << "Expected: [896, 896] (out_features, in_features)" << std::endl;

    // Check if shape matches expected layout
    bool is_correct_layout = (shape[0] == 896 && shape[1] == 896);
    std::cout << "Layout correct: " << (is_correct_layout ? "✅ YES" : "❌ NO") << std::endl;

    // Check logical_k (should be same as shape[1] for non-transposed)
    size_t logical_k = iq4_nl->logical_k();
    std::cout << "logical_k(): " << logical_k << std::endl;
    std::cout << "Matches shape[1]: " << (logical_k == shape[1] ? "✅ YES" : "❌ NO") << std::endl;

    // Test decode_block_at to verify layout
    // For weight [n=896, k=896], blocks should be organized as [n_blocks_per_row, k_blocks]
    // Each row has k/32 = 28 blocks (896/32 = 28)

    size_t k_blocks_per_row = (shape[1] + 31) / 32;
    std::cout << "\nBlock organization:" << std::endl;
    std::cout << "K blocks per row: " << k_blocks_per_row << std::endl;
    std::cout << "Expected: 28 (896/32)" << std::endl;

    // Decode first block of first row and last row to verify layout
    alignas(64) float first_row_block0[32];
    alignas(64) float last_row_block0[32];

    iq4_nl->decode_block_at(0, 0, first_row_block0);
    iq4_nl->decode_block_at(shape[0] - 1, 0, last_row_block0);

    std::cout << "\nFirst block decoded successfully" << std::endl;
    std::cout << "First row, block 0, first 4 values: ";
    for (int i = 0; i < 4; i++)
    {
        std::cout << std::setprecision(4) << first_row_block0[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Last row, block 0, first 4 values: ";
    for (int i = 0; i < 4; i++)
    {
        std::cout << std::setprecision(4) << last_row_block0[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Test 3: Check GEMM kernel alignment during computation
 */
TEST_F(IQ4_NL_Alignment_Layout, CheckGEMMAlignment)
{
    auto q_weight = loader_->loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(q_weight, nullptr);

    auto iq4_nl = std::dynamic_pointer_cast<IQ4_NLTensor>(q_weight);
    ASSERT_NE(iq4_nl, nullptr);

    auto shape = iq4_nl->shape();
    int n = shape[0]; // 896
    int k = shape[1]; // 896
    int m = 128;      // Batch size

    // Create test activation tensor
    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    // Get GEMM kernel
    auto gemm = iq4_nl->createGemm();
    ASSERT_NE(gemm, nullptr);

    std::cout << "\n=== GEMM Kernel Alignment ===" << std::endl;
    std::cout << "Input A [" << m << ", " << k << "] at 0x" << std::hex
              << reinterpret_cast<uintptr_t>(A.data()) << std::dec << std::endl;
    std::cout << "Output C [" << m << ", " << n << "] at 0x" << std::hex
              << reinterpret_cast<uintptr_t>(C.data()) << std::dec << std::endl;

    // Check alignment of vectors
    std::cout << "\nStd::vector alignment (typically 16-byte on x86-64):" << std::endl;
    std::cout << "A: " << (reinterpret_cast<uintptr_t>(A.data()) % 64 == 0 ? "64-byte ✅" : reinterpret_cast<uintptr_t>(A.data()) % 32 == 0 ? "32-byte ⚠️"
                                                                                        : reinterpret_cast<uintptr_t>(A.data()) % 16 == 0   ? "16-byte ⚠️"
                                                                                                                                            : "unaligned ❌")
              << std::endl;
    std::cout << "C: " << (reinterpret_cast<uintptr_t>(C.data()) % 64 == 0 ? "64-byte ✅" : reinterpret_cast<uintptr_t>(C.data()) % 32 == 0 ? "32-byte ⚠️"
                                                                                        : reinterpret_cast<uintptr_t>(C.data()) % 16 == 0   ? "16-byte ⚠️"
                                                                                                                                            : "unaligned ❌")
              << std::endl;

    // Weight-owned GEMM path removed: IQ4_NL now participates in GEMM only via
    // activation-tensor-centric kernels. This test remains as an alignment and
    // layout diagnostic without invoking the legacy gemm->multiply API.
}

/**
 * @brief Test 4: Compare transpose_B=true vs transpose_B=false
 */
TEST_F(IQ4_NL_Alignment_Layout, CheckTransposeFlag)
{
    auto q_weight = loader_->loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    ASSERT_NE(q_weight, nullptr);

    auto iq4_nl = std::dynamic_pointer_cast<IQ4_NLTensor>(q_weight);
    ASSERT_NE(iq4_nl, nullptr);

    auto shape = iq4_nl->shape();
    int n = shape[0]; // 896
    int k = shape[1]; // 896
    int m = 128;

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> C_transpose(m * n, 0.0f);
    std::vector<float> C_no_transpose(m * n, 0.0f);

    auto gemm = iq4_nl->createGemm();
    ASSERT_NE(gemm, nullptr);

    std::cout << "\n=== Transpose Flag Test ===" << std::endl;

    // Weight-owned GEMM path removed: transpose_B correctness is now validated
    // through activation-driven GEMM tests rather than directly calling
    // gemm->multiply on the IQ4_NL weight tensor.
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
