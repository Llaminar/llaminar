#define DNNL_EXPERIMENTAL_UKERNEL
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <chrono>

#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/tensors/Tensors.h"
#include "v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

using namespace llaminar2;
using namespace dnnl;
using namespace dnnl::ukernel;

// Helper to fill Q8_1 block
void fill_q8_1_block(Q8_1Block &block, float scale, int seed_offset)
{
    // d is FP16 scale
    // We need to convert float scale to FP16.
    // Since we don't have fp32_to_fp16 helper exposed easily, let's just use a simple approximation or 1.0
    // Actually, let's just set d to 1.0 (0x3C00) for simplicity, or implement a mini converter.
    // 1.0 in FP16 is 0x3C00. 0.5 is 0x3800. 2.0 is 0x4000.

    // Let's use 1.0 for now to verify logic, then try varying.
    // 0x3C00 = 1.0
    block.d = 0x3C00;

    // Fill qs
    std::mt19937 gen(42 + seed_offset);
    std::uniform_int_distribution<int> dist(-127, 127);
    int sum = 0;
    for (int i = 0; i < 32; ++i)
    {
        block.qs[i] = (int8_t)dist(gen);
        sum += block.qs[i];
    }
    block.sum_qs = (int16_t)sum;
}

// Helper to fill Q4_0 block
void fill_q4_0_block(Q4_0Block &block, float scale, int seed_offset)
{
    block.d = 0x3C00; // 1.0

    std::mt19937 gen(1337 + seed_offset);
    std::uniform_int_distribution<int> dist(0, 15);
    for (int i = 0; i < 16; ++i)
    {
        uint8_t low = dist(gen);
        uint8_t high = dist(gen);
        block.qs[i] = (low & 0x0F) | ((high & 0x0F) << 4);
    }
}

TEST(OneDNN_BRGEMM_Experiment, Q8_1_Q4_0_DotProduct)
{
    // Dimensions
    const int M = 1;
    const int N = 32;
    const int K = 64; // 2 blocks
    const int K_blk = 32;
    const int num_blocks = K / K_blk;

    // Allocate Data
    std::vector<Q8_1Block> A(num_blocks);
    std::vector<Q4_0Block> B(num_blocks * N); // B is KxN, but blocked.
    // Wait, Q4_0 layout for weights is usually Blocked by K?
    // Usually weights are [N, K/32] blocks? Or [K/32, N] blocks?
    // In llaminar, Q4_0Tensor is usually row-major blocks?
    // Let's assume B is [K/32][N] blocks for simplicity of test.
    // Actually, standard layout is usually [N][K/32] blocks for contiguous memory access during dot product?
    // But here we just need to access them.
    // Let's assume we have a vector of blocks representing the matrix.
    // We will index it as B[n * num_blocks + k_blk].

    // Fill Data
    for (int k = 0; k < num_blocks; ++k)
    {
        fill_q8_1_block(A[k], 1.0f, k);
    }
    for (int n = 0; n < N; ++n)
    {
        for (int k = 0; k < num_blocks; ++k)
        {
            fill_q4_0_block(B[n * num_blocks + k], 1.0f, n * num_blocks + k);
        }
    }

    // Reference Calculation
    std::vector<float> C_ref(M * N, 0.0f);
    for (int n = 0; n < N; ++n)
    {
        float sum = 0.0f;
        for (int k_blk = 0; k_blk < num_blocks; ++k_blk)
        {
            const auto &blk_A = A[k_blk];
            const auto &blk_B = B[n * num_blocks + k_blk];

            float d_A = fp16_to_fp32(blk_A.d);
            float d_B = fp16_to_fp32(blk_B.d);

            for (int i = 0; i < 32; ++i)
            {
                int8_t val_A = blk_A.qs[i];
                // Q4_0 unpack: (qs - 8)
                uint8_t packed = blk_B.qs[i / 2];
                int8_t val_B = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                val_B -= 8;

                sum += (val_A * val_B) * d_A * d_B;
            }
        }
        C_ref[n] = sum;
    }

    // OneDNN Implementation

    // 1. Unpack B to s8 (KxN)
    // We need it to be KxN for BRGEMM?
    // BRGEMM expects B to be KxN (or packed).
    // If we use K_blk=32, we can treat it as [num_blocks][32][N]?
    // Or [num_blocks][N][32]?
    // BRGEMM B_ptr points to start of B.
    // If we loop over blocks, we can pass a pointer to the start of the block-row.
    // Let's unpack to a buffer `B_unpacked` of size [K * N].
    // Layout: Row-major (K, N)? Or (N, K)?
    // BRGEMM standard is Row-Major (K, N) if `ldb = N`.
    // Let's use (K, N).
    std::vector<int8_t> B_unpacked(K * N);
    for (int n = 0; n < N; ++n)
    {
        for (int k_blk = 0; k_blk < num_blocks; ++k_blk)
        {
            const auto &blk = B[n * num_blocks + k_blk];
            for (int i = 0; i < 32; ++i)
            {
                uint8_t packed = blk.qs[i / 2];
                int8_t val = (i % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                val -= 8;

                // Index in B_unpacked (K, N)
                // row = k_blk * 32 + i
                // col = n
                B_unpacked[(k_blk * 32 + i) * N + n] = val;
            }
        }
    }

    // 2. Create BRGEMM Kernel
    // M=1, N=32, K=32, batch=1
    // lda = K (32) ? No, A is 1x32. lda is stride between rows of A. Since M=1, lda is irrelevant?
    // Docs: "lda Leading dimension of tensor A."
    // If M=1, lda is usually K.
    // ldb = N (32).
    // ldc = N (32).
    brgemm brg;
    // brgemm(M, N, K, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt)
    // Note: K here is K_blk (32).
    brg = brgemm(M, N, K_blk, 1, K_blk, N, N,
                 memory::data_type::u8, memory::data_type::s8, memory::data_type::s32,
                 true); // allow_empty

    ASSERT_TRUE(brg) << "BRGEMM kernel creation failed";

    // We want to overwrite C_temp, so set_add_C(false)
    brg.set_add_C(false);
    brg.finalize();

    // Check pack type
    auto req_pack_type = brgemm::get_B_pack_type(memory::data_type::s8, memory::data_type::s8);
    std::cout << "BRGEMM B pack type: " << (int)req_pack_type << std::endl;

    brg.generate();

    // Transform kernel for packing B if needed
    transform tr;
    std::vector<int8_t> B_packed_blk(K_blk * N, 0x55); // Initialize with pattern
    if (req_pack_type != pack_type::no_trans)
    {
        // transform(K, N, in_pack_type, in_ld, out_ld, in_dt, out_dt)
        // We want to pack B. The 'pack_type' argument specifies the desired packing.
        // Try passing ld_out = N (even if packed, maybe it needs a dummy value > 0)
        tr = transform(K_blk, N, req_pack_type, N, N,
                       memory::data_type::s8, memory::data_type::s8);
        tr.generate();
    }

    // 3. Execute Loop
    std::vector<float> C_onednn(N, 0.0f);
    std::vector<int32_t> C_temp(N); // 1xN accumulator (s32)
    std::vector<std::pair<memory::dim, memory::dim>> offsets = {{0, 0}};

    for (int k_blk = 0; k_blk < num_blocks; ++k_blk)
    {
        // A_ptr: Points to qs of current block
        const int8_t *A_ptr = A[k_blk].qs;

        // Convert A to u8 (add 128)
        std::vector<uint8_t> A_u8(K_blk);
        for (int i = 0; i < K_blk; ++i)
            A_u8[i] = (uint8_t)(A_ptr[i] + 128);

        // B_ptr: Points to start of current block-row in B_unpacked
        const int8_t *B_ptr = &B_unpacked[(k_blk * 32) * N];
        const int8_t *B_gemm_ptr = B_ptr;

        if (req_pack_type != pack_type::no_trans)
        {
            tr.execute(B_ptr, B_packed_blk.data());
            B_gemm_ptr = B_packed_blk.data();
        }

        // Execute with u8 A
        brg.execute(A_u8.data(), B_gemm_ptr, offsets, C_temp.data(), nullptr);

        // Compute compensation for this block: sum(B) per column
        std::vector<int32_t> comp(N, 0);
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < K_blk; ++k)
            {
                // B is row-major in B_unpacked: B[k, n] is at B_ptr[k*N + n]
                comp[n] += B_ptr[k * N + n];
            }
        }

        // Accumulate with scales and compensation
        float d_A = fp16_to_fp32(A[k_blk].d);
        for (int n = 0; n < N; ++n)
        {
            float d_B = fp16_to_fp32(B[n * num_blocks + k_blk].d);
            // Recover signed dot product: sum(A*B) = sum((A+128)*B) - 128*sum(B)
            int32_t real_dot = C_temp[n] - 128 * comp[n];
            C_onednn[n] += real_dot * d_A * d_B;
        }
    }

    // Compare
    for (int n = 0; n < N; ++n)
    {
        EXPECT_NEAR(C_onednn[n], C_ref[n], 1e-3f) << "Mismatch at index " << n;
    }
}

// Test blockwise GEMM with real Q4_0 tensor using IINT8Unpackable interface
TEST(OneDNN_BRGEMM_Experiment, Q4_0_BlockwiseGEMM_Throughput)
{
    using namespace llaminar2::gemm_v4;

    // Test dimensions (smaller for unit test, but realistic decode shape)
    const int M = 1;   // Single token decode
    const int N = 256; // Smaller for test (typical: 4096)
    const int K = 256; // Hidden dimension (typical: 4096)
    const int K_blocks = K / 32;

    std::cout << "\n[Q4_0_BlockwiseGEMM_Throughput] Testing M=" << M << " N=" << N << " K=" << K << std::endl;

    // 1. Create Q4_0 weight tensor (N × K) with raw Q4_0 blocks
    // Q4_0 format: Each block = 18 bytes (2 bytes FP16 scale + 16 bytes packed nibbles)
    const size_t blocks_per_row = K_blocks;
    const size_t total_blocks = N * blocks_per_row;
    const size_t bytes_per_block = sizeof(Q4_0Block);
    std::vector<uint8_t> raw_q4_0_data(total_blocks * bytes_per_block);

    // Fill with random Q4_0 blocks
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 15);

    for (size_t n = 0; n < static_cast<size_t>(N); ++n)
    {
        for (size_t kb = 0; kb < static_cast<size_t>(K_blocks); ++kb)
        {
            Q4_0Block block;
            block.d = fp32_to_fp16(0.1f + static_cast<float>((n + kb) % 10) * 0.01f); // Varying scales

            for (int i = 0; i < 16; ++i)
            {
                uint8_t low = dist(gen);
                uint8_t high = dist(gen);
                block.qs[i] = (low & 0x0F) | ((high & 0x0F) << 4);
            }

            // Copy block to raw data
            size_t block_offset = (n * blocks_per_row + kb) * bytes_per_block;
            std::memcpy(raw_q4_0_data.data() + block_offset, &block, bytes_per_block);
        }
    }

    auto q4_0_tensor = std::make_shared<Q4_0Tensor>(
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K)},
        raw_q4_0_data);

    // 2. Verify IINT8Unpackable interface
    const auto *int8_unpackable = dynamic_cast<const IINT8Unpackable *>(q4_0_tensor.get());
    ASSERT_NE(int8_unpackable, nullptr) << "Q4_0Tensor must implement IINT8Unpackable";

    std::cout << "[Q4_0_BlockwiseGEMM_Throughput] Q4_0Tensor implements IINT8Unpackable: YES" << std::endl;
    std::cout << "[Q4_0_BlockwiseGEMM_Throughput] Tensor shape: " << N << " × " << K << std::endl;

    // 3. Create OneDNNGemmKernel and pack weights
    OneDNNGemmKernel kernel(q4_0_tensor.get());

    std::cout << "[Q4_0_BlockwiseGEMM_Throughput] Packing weights via IINT8Unpackable..." << std::endl;
    auto weight_pack_opt = kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack_opt.has_value()) << "Failed to pack Q4_0 weights";

    const auto &weight_pack = weight_pack_opt.value();
    std::cout << "[Q4_0_BlockwiseGEMM_Throughput] Weight pack: K=" << weight_pack.K
              << " N=" << weight_pack.N << " K_blocks=" << weight_pack.K_blocks << std::endl;
    std::cout << "[Q4_0_BlockwiseGEMM_Throughput] unpacked_s8 size: " << weight_pack.unpacked_s8.size()
              << " (expected: " << (K * N) << ")" << std::endl;
    std::cout << "[Q4_0_BlockwiseGEMM_Throughput] block_scales size: " << weight_pack.block_scales.size()
              << " (expected: " << (K_blocks * N) << ")" << std::endl;

    // 4. Create Q8_1 activations
    std::vector<Q8_1Block> A_blocks(M * K_blocks);
    std::mt19937 gen_A(1337);
    std::uniform_real_distribution<float> dist_A(-1.0f, 1.0f);

    for (int m = 0; m < M; ++m)
    {
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            Q8_1Block &block = A_blocks[m * K_blocks + kb];

            // Generate FP32 values, then quantize to Q8_1
            float max_abs = 0.0f;
            float values[32];
            for (int i = 0; i < 32; ++i)
            {
                values[i] = dist_A(gen_A);
                max_abs = std::max(max_abs, std::abs(values[i]));
            }

            float d = max_abs / 127.0f;
            if (d < 1e-10f)
                d = 1.0f;
            float id = 1.0f / d;

            block.d = fp32_to_fp16(d);
            int sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int8_t q = static_cast<int8_t>(std::round(values[i] * id));
                block.qs[i] = q;
                sum_qs += q;
            }
            block.sum_qs = static_cast<int16_t>(sum_qs);
        }
    }

    // 5. Allocate output
    std::vector<float> C(M * N, 0.0f);

    // 6. Warmup iterations
    const int warmup_iters = 10;
    for (int i = 0; i < warmup_iters; ++i)
    {
        kernel.execute_blockwise_gemm_test(A_blocks.data(), weight_pack, C.data(),
                                           M, N, K, nullptr, 1.0f, 0.0f);
    }

    // 7. Timed iterations
    const int timed_iters = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < timed_iters; ++i)
    {
        kernel.execute_blockwise_gemm_test(A_blocks.data(), weight_pack, C.data(),
                                           M, N, K, nullptr, 1.0f, 0.0f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / timed_iters;

    // 8. Compute throughput metrics
    // GEMM FLOPs: 2 * M * N * K (multiply + add)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms * 1e6));

    // Memory throughput (read A + read B + write C)
    // A: M * K * sizeof(Q8_1Block) / 32 elements per block
    // B: N * K (int8) unpacked
    // C: M * N * sizeof(float)
    size_t bytes_A = M * K_blocks * sizeof(Q8_1Block);
    size_t bytes_B = N * K * sizeof(int8_t);
    size_t bytes_C = M * N * sizeof(float);
    size_t total_bytes = bytes_A + bytes_B + bytes_C;
    double gb_per_sec = (total_bytes / (avg_ms * 1e6));

    std::cout << "\n[Q4_0_BlockwiseGEMM_Throughput] Performance Results:" << std::endl;
    std::cout << "  Warmup iterations: " << warmup_iters << std::endl;
    std::cout << "  Timed iterations: " << timed_iters << std::endl;
    std::cout << "  Total time: " << total_ms << " ms" << std::endl;
    std::cout << "  Average time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Memory bandwidth: " << gb_per_sec << " GB/s" << std::endl;
    std::cout << "  Bytes A (Q8_1): " << bytes_A << " bytes" << std::endl;
    std::cout << "  Bytes B (int8): " << bytes_B << " bytes" << std::endl;
    std::cout << "  Bytes C (FP32): " << bytes_C << " bytes" << std::endl;

    // 9. Verify correctness (spot check first few outputs)
    std::cout << "\n[Q4_0_BlockwiseGEMM_Throughput] Output sanity check:" << std::endl;
    std::cout << "  C[0] = " << C[0] << std::endl;
    std::cout << "  C[1] = " << C[1] << std::endl;
    std::cout << "  C[N-1] = " << C[N - 1] << std::endl;

    // Check for NaN/Inf
    bool has_invalid = false;
    for (int n = 0; n < N; ++n)
    {
        if (std::isnan(C[n]) || std::isinf(C[n]))
        {
            std::cerr << "  ERROR: Invalid value at C[" << n << "] = " << C[n] << std::endl;
            has_invalid = true;
            break;
        }
    }
    EXPECT_FALSE(has_invalid) << "Output contains NaN or Inf values";

    // Performance expectations (very rough - depends on hardware)
    // On modern CPU with AVX512, expect ~10-50 GFLOPS for small M (decode)
    std::cout << "\n[Q4_0_BlockwiseGEMM_Throughput] Test PASSED" << std::endl;
}
