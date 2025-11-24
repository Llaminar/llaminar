#define DNNL_EXPERIMENTAL_UKERNEL
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <numeric>
#include <cstring>

#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/tensors/Tensors.h"
#include "v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

using namespace dnnl;
using namespace dnnl::ukernel;
using namespace llaminar2;

class OneDNN_Perf : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure OpenMP is active
        if (omp_get_max_threads() < 2)
        {
            std::cout << "WARNING: OpenMP threads < 2. Performance may be poor." << std::endl;
        }
    }

    // Helper to fill random data
    void fill_random(std::vector<int8_t> &data)
    {
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto &x : data)
            x = (int8_t)dist(gen);
    }
};

TEST_F(OneDNN_Perf, Compare_CustomBRGEMM_vs_FlatInt8)
{
    // Configuration
    const int M = 1;    // Single token
    const int N = 4096; // Output dim
    const int K = 4096; // Input dim
    const int K_blk = 32;
    const int num_blocks = K / K_blk;
    const int iters = 100;
    const int warmup = 10;

    std::cout << "Benchmarking M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // ========================================================================
    // 1. Custom BRGEMM Setup (Q8_1 x Q4_0 simulation)
    // ========================================================================

    // Data Allocation
    std::vector<Q8_1Block> A_blocks(num_blocks);
    // B is logically [N][num_blocks] of Q4_0 blocks, but we unpack to flat s8 for BRGEMM
    std::vector<int8_t> B_unpacked(K * N);

    // Fill B with random data
    fill_random(B_unpacked);

    // Pre-compute Compensation Vector (Sum of B columns)
    // This is done ONCE at weight loading time in a real engine
    std::vector<int32_t> comp_vec(N, 0);
    for (int n = 0; n < N; ++n)
    {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k)
        {
            sum += B_unpacked[k * N + n]; // Assuming (K, N) layout for BRGEMM
        }
        comp_vec[n] = sum;
    }

    // Setup BRGEMM Kernel
    // M=1, N=N_blk_gemm, K=K_blk (32)
    // We iterate over K blocks.
    // lda = K_blk (since we copy A to a small buffer)
    // ldb = N
    // ldc = N
    const int N_blk_gemm = 64;
    brgemm brg = brgemm(M, N_blk_gemm, K_blk, 1, K_blk, N, N,
                        memory::data_type::u8, memory::data_type::s8, memory::data_type::s32,
                        true); // allow_empty
    brg.set_add_C(true);       // Accumulate into C
    brg.finalize();
    brg.generate();

    // Packing B (if needed)
    auto req_pack_type = brgemm::get_B_pack_type(memory::data_type::s8, memory::data_type::s8);
    std::cout << "BRGEMM B pack type: " << (int)req_pack_type << std::endl;

    std::vector<int8_t> B_packed;
    bool use_packing = (req_pack_type != pack_type::no_trans);

    if (use_packing)
    {
        // Just allocate 2x size to be safe.
        B_packed.resize(K * N * 2);

        // Create transform to pack B
        // We need to transform from our layout (row-major) to the required pack type.
        // The 3rd argument is the TARGET pack type.
        try
        {
            // Transform seems to have a limit on N size (found via probing to be 64).
            // However, in_ld can be large (stride), but out_ld MUST be the block size (64).
            // So we can read directly from B_unpacked (strided) but must write to B_packed in tiles.
            const int N_blk_pack = 64;

            // Create transform for strided input -> packed block
            // in_ld = N (full width of B), out_ld = N_blk_pack (dense packed block)
            transform tr(K_blk, N_blk_pack, req_pack_type, N, N_blk_pack, memory::data_type::s8, memory::data_type::s8, false);
            tr.generate();

// Pack each K-slice, tiled along N
#pragma omp parallel for
            for (int k_blk = 0; k_blk < num_blocks; ++k_blk)
            {
                for (int n = 0; n < N; n += N_blk_pack)
                {
                    // Source: B_unpacked at [k_blk*K_blk, n]
                    const int8_t *src = &B_unpacked[(k_blk * K_blk) * N + n];

                    // Destination: B_packed is tiled.
                    // We need to calculate where this tile goes.
                    // Assuming VNNI packing [K/4, N, 4]
                    // The destination pointer calculation in the previous double-copy version was manual.
                    // Here we can just use a temp buffer for the OUTPUT of the transform if B_packed layout is complex,
                    // OR write directly if we understand the layout.
                    // The transform kernel writes a dense packed block.
                    // But B_packed is a single large buffer.
                    // If we want B_packed to be a single large VNNI buffer, we need to be careful.
                    // The transform kernel writes [K_blk/4, N_blk_pack, 4] (for s8).
                    // We want to place this into the larger [K/4, N, 4] buffer.
                    // This requires strided OUTPUT, which transform does NOT support (out_ld is fixed to N_blk).
                    // So we STILL need a temporary buffer for the output, or we need to copy line-by-line.

                    // Optimization: Read directly from src (strided), write to temp dense packed buffer, then copy to dst.
                    // This saves the input copy.

                    std::vector<int8_t> t_out(K_blk * N_blk_pack);
                    tr.execute(src, t_out.data());

                    // Copy to strided output
                    const int packed_row_size = N_blk_pack * 4; // 4 bytes per column per K-group
                    const int full_stride = N * 4;

                    for (int r = 0; r < K_blk / 4; ++r)
                    {
                        std::memcpy(&B_packed[(k_blk * K_blk / 4 + r) * full_stride + n * 4],
                                    &t_out[r * packed_row_size],
                                    packed_row_size);
                    }
                }
            }
            // std::cout << "DEBUG: B packing completed successfully." << std::endl;
        }
        catch (const dnnl::error &e)
        {
            std::cout << "ERROR: transform creation failed with status: " << e.status << " message: " << e.message << std::endl;
            use_packing = false;
        }
        catch (const std::exception &e)
        {
            std::cout << "ERROR: transform creation threw exception: " << e.what() << std::endl;
            use_packing = false;
        }
    }

    // Buffers
    std::vector<int32_t> C_accum(N, 0);
    std::vector<uint8_t> A_u8_full(K); // Full A buffer
    std::vector<std::pair<memory::dim, memory::dim>> offsets = {{0, 0}};

    // ========================================================================
    // Benchmark Loop: Custom BRGEMM
    // ========================================================================
    auto start_custom = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters + warmup; ++iter)
    {
        if (iter == warmup)
            start_custom = std::chrono::high_resolution_clock::now();

        // Reset C
        std::memset(C_accum.data(), 0, N * sizeof(int32_t));

// 1. Convert A (Whole vector)
// This simulates converting the activation vector once per token
#pragma omp parallel for
        for (int k = 0; k < K; ++k)
        {
            // A_blocks is [num_blocks] of Q8_1Block.
            // Each block has 32 quants.
            int blk_idx = k / K_blk;
            int off = k % K_blk;
            A_u8_full[k] = (uint8_t)(A_blocks[blk_idx].qs[off] + 128);
        }

// 2. Parallel BRGEMM over N
#pragma omp parallel for
        for (int n = 0; n < N; n += N_blk_gemm)
        {
            // Loop over K blocks
            for (int k_blk = 0; k_blk < num_blocks; ++k_blk)
            {
                const uint8_t *A_ptr = &A_u8_full[k_blk * K_blk];

                // B_ptr calculation
                // Packed B is effectively [K/4, N, 4] in bytes if we consider strides.
                // But we access it as a flat buffer.
                // Offset = (k_blk * K_blk) * N + n * 4
                const int8_t *B_ptr = use_packing ? &B_packed[(k_blk * K_blk) * N + n * 4] : &B_unpacked[(k_blk * K_blk) * N + n]; // Unpacked is just row-major [K, N]

                brg.execute(A_ptr, B_ptr, offsets, &C_accum[n], nullptr);
            }
        }

// 3. Apply Compensation (Runtime Cost)
// C = C - 128 * comp
#pragma omp parallel for
        for (int n = 0; n < N; ++n)
        {
            C_accum[n] -= 128 * comp_vec[n];
        }
    }

    auto end_custom = std::chrono::high_resolution_clock::now();
    double time_custom = std::chrono::duration<double, std::milli>(end_custom - start_custom).count() / iters;

    // ========================================================================
    // 2. Flat Int8 Setup (OneDNN MatMul)
    // ========================================================================
    engine eng(engine::kind::cpu, 0);
    stream strm(eng);

    memory::dims a_dims = {M, K};
    memory::dims b_dims = {K, N};
    memory::dims c_dims = {M, N};

    auto a_md = memory::desc(a_dims, memory::data_type::s8, memory::format_tag::ab);
    auto b_md = memory::desc(b_dims, memory::data_type::s8, memory::format_tag::ab); // KxN row major? 'ab' is usually row major
    auto c_md = memory::desc(c_dims, memory::data_type::s32, memory::format_tag::ab);

    // Create primitive
    matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
    matmul matmul_prim(matmul_pd);

    // Allocate memories
    memory a_mem(a_md, eng);
    memory b_mem(b_md, eng);
    memory c_mem(c_md, eng);

    // Fill data
    fill_random(*(std::vector<int8_t> *)a_mem.get_data_handle()); // Hacky but works for bench
    // Reuse B_unpacked for B
    std::memcpy(b_mem.get_data_handle(), B_unpacked.data(), K * N);

    // Args
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});

    // ========================================================================
    // Benchmark Loop: Flat Int8
    // ========================================================================
    auto start_flat = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters + warmup; ++iter)
    {
        if (iter == warmup)
            start_flat = std::chrono::high_resolution_clock::now();

        matmul_prim.execute(strm, matmul_args);
        strm.wait();
    }

    auto end_flat = std::chrono::high_resolution_clock::now();
    double time_flat = std::chrono::duration<double, std::milli>(end_flat - start_flat).count() / iters;

    // ========================================================================
    // Results
    // ========================================================================
    double gflops = 2.0 * M * N * K * 1e-9;

    std::cout << "\nResults (M=" << M << ", N=" << N << ", K=" << K << "):" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Custom BRGEMM (u8 shift): " << time_custom << " ms (" << gflops / (time_custom / 1000.0) << " GFLOPS)" << std::endl;
    std::cout << "Flat Int8 (OneDNN MatMul): " << time_flat << " ms (" << gflops / (time_flat / 1000.0) << " GFLOPS)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Overhead: " << (time_custom / time_flat) << "x slower" << std::endl;
}

// Test blockwise GEMM with real Q4_0 tensor using IINT8Unpackable interface
TEST_F(OneDNN_Perf, Q4_0_IINT8Unpackable_Throughput)
{
    using namespace llaminar2::gemm_v4;

    // Test dimensions (realistic inference shapes)
    const int M = 1;    // Single token decode
    const int N = 4096; // Model dimension (Qwen 2.5 0.5B)
    const int K = 4096; // Hidden dimension
    const int K_blocks = K / 32;

    std::cout << "\n[Q4_0_IINT8Unpackable_Throughput] Testing M=" << M << " N=" << N << " K=" << K << std::endl;

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

    std::cout << "[Q4_0_IINT8Unpackable_Throughput] Q4_0Tensor implements IINT8Unpackable: YES" << std::endl;
    std::cout << "[Q4_0_IINT8Unpackable_Throughput] Tensor shape: " << N << " × " << K << std::endl;

    // 3. Create OneDNNGemmKernel and pack weights
    OneDNNGemmKernel kernel(q4_0_tensor.get());

    std::cout << "[Q4_0_IINT8Unpackable_Throughput] Packing weights via IINT8Unpackable..." << std::endl;
    auto weight_pack_opt = kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack_opt.has_value()) << "Failed to pack Q4_0 weights";

    const auto &weight_pack = weight_pack_opt.value();
    std::cout << "[Q4_0_IINT8Unpackable_Throughput] Weight pack: K=" << weight_pack.K
              << " N=" << weight_pack.N << " K_blocks=" << weight_pack.K_blocks << std::endl;

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
    size_t bytes_A = M * K_blocks * sizeof(Q8_1Block);
    size_t bytes_B = N * K * sizeof(int8_t);
    size_t bytes_C = M * N * sizeof(float);
    size_t total_bytes = bytes_A + bytes_B + bytes_C;
    double gb_per_sec = (total_bytes / (avg_ms * 1e6));

    std::cout << "\n[Q4_0_IINT8Unpackable_Throughput] Performance Results:" << std::endl;
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
    std::cout << "\n[Q4_0_IINT8Unpackable_Throughput] Output sanity check:" << std::endl;
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

    std::cout << "\n[Q4_0_IINT8Unpackable_Throughput] Test PASSED" << std::endl;
}
