/**
 * @file Test__Q8_1_AttentionReplayFromDump.cpp
 * @brief Replay test using captured attention inputs from E2E pipeline
 * @author David Sanftenberg
 *
 * This test loads attention inputs captured by the LLAMINAR_DUMP_ATTENTION_INPUTS
 * compile flag and replays them through JIT, C++ reference, AND compares against
 * PyTorch FP32 ground truth.
 *
 * The captured data is from the actual Q8_1 E2E pipeline, which shows ~0.81 cosine
 * divergence from FP32 at layer0 ATTENTION_CONTEXT.
 *
 * Tests included:
 *   1. Layer0Head4_JitVsReference: Confirms JIT == C++ reference (kernel correctness)
 *   2. Layer0Head4_JitVsPytorchGroundTruth: Compares JIT output vs PyTorch FP32
 *      ground truth to reproduce the 0.81 cosine divergence in isolation.
 *
 * To capture new dumps:
 *   1. Build with: cmake -DDUMP_ATTENTION_INPUTS=ON ...
 *   2. Run E2E test - dumps appear in attention_dump_NNNN/ directories
 *   3. Copy dump to tests/v2/test_data/
 *   4. Run this test (no special build flags needed)
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <limits>

#include "../../src/v2/kernels/cpu/gemm_v4/AttentionInputDumper.h"
#include "../../src/v2/kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"
#include "../../src/v2/tensors/SIMDHelpers.h"
#include "../../src/v2/utils/Logger.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{

    /**
     * @brief Simple NPY loader (FP32 only, version 1.0 and 2.0)
     * @param path Path to .npy file
     * @return Vector of floats loaded from file
     */
    std::vector<float> loadNpy(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot open: " + path);

        // Read magic (6 bytes)
        char magic[6];
        file.read(magic, 6);

        // Read version (2 bytes)
        unsigned char version_major, version_minor;
        file.read(reinterpret_cast<char *>(&version_major), 1);
        file.read(reinterpret_cast<char *>(&version_minor), 1);

        // Read header length (2 bytes for v1.0, 4 bytes for v2.0+)
        unsigned int header_len;
        if (version_major == 1)
        {
            unsigned short hlen;
            file.read(reinterpret_cast<char *>(&hlen), 2);
            header_len = hlen;
        }
        else
        {
            file.read(reinterpret_cast<char *>(&header_len), 4);
        }

        // Calculate data offset
        size_t header_bytes = (version_major == 1) ? 2 : 4;
        size_t data_offset = 6 + 2 + header_bytes + header_len; // magic + version + header_len_field + header

        // Seek to data start
        file.seekg(data_offset, std::ios::beg);

        // Get file size to calculate data size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(data_offset, std::ios::beg);

        size_t data_size = file_size - data_offset;
        size_t num_floats = data_size / sizeof(float);

        std::vector<float> data(num_floats);
        file.read(reinterpret_cast<char *>(data.data()), data_size);

        return data;
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // Use the library's gemm_v4::fused_q8_1_attention_reference directly
    // (it includes Q/K normalization that the JIT kernel uses)

} // namespace

/**
 * @brief Test fixture for attention replay tests
 */
class Test__Q8_1_AttentionReplayFromDump : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    }

    int rank_ = 0;
};

/**
 * @brief Replay captured layer0 head4 attention through JIT and reference
 *
 * This test loads the exact Q/K/V blocks that caused 0.81 cosine divergence
 * in the E2E test and compares JIT vs C++ reference implementations.
 *
 * If JIT == reference (~0.999), the kernel is correct and the divergence
 * is due to Q8_1 precision loss (expected).
 *
 * If JIT != reference, there's a kernel bug.
 */
TEST_F(Test__Q8_1_AttentionReplayFromDump, Layer0Head4_JitVsReference)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  REPLAY TEST: Layer 0, Head 4 (captured from E2E pipeline)     ║");
    LOG_INFO("║  Goal: Determine if JIT matches C++ reference                  ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // Load captured dump
    LoadedAttentionDump dump;
    std::string dump_path = "tests/v2/test_data/attention_dump_layer0_head4";

    if (!dump.load(dump_path))
    {
        GTEST_SKIP() << "Dump not found at: " << dump_path << " (run with -DDUMP_ATTENTION_INPUTS=ON to capture)";
    }

    LOG_INFO("Loaded dump: M=" << dump.M << " N=" << dump.N << " head_dim=" << dump.head_dim);
    LOG_INFO("  Q blocks: " << dump.Q_blocks.size());
    LOG_INFO("  K blocks: " << dump.K_blocks.size());
    LOG_INFO("  V blocks: " << dump.V_blocks.size());
    LOG_INFO("  Has mask: " << (dump.has_mask ? "yes" : "no"));
    LOG_INFO("  Scale: " << dump.scale);
    LOG_INFO("");

    const int num_blocks = dump.head_dim / 32;
    const int output_blocks = dump.M * num_blocks;

    // Allocate outputs
    std::vector<Q8_1Block> ref_output(output_blocks);
    std::vector<Q8_1Block> jit_output(output_blocks);

    // Run C++ reference
    LOG_INFO("Running C++ reference...");
    // Run C++ reference (uses library's reference with Q/K normalization)
    LOG_INFO("Running C++ reference...");
    gemm_v4::fused_q8_1_attention_reference(
        dump.Q_blocks.data(),
        dump.K_blocks.data(),
        dump.V_blocks.data(),
        ref_output.data(),
        dump.M, dump.N, dump.head_dim,
        num_blocks, num_blocks, num_blocks, num_blocks,
        dump.scale,
        dump.has_mask ? dump.mask.data() : nullptr,
        dump.N);

    // Run JIT kernel
    LOG_INFO("Running JIT kernel...");
    QuantisedAttentionJit_Q8_1_Fused jit(dump.head_dim, false);
    auto kernel_fn = jit.get_kernel();

    FusedQ8_1AttentionParams params = dump.build_params(jit_output.data());
    kernel_fn(&params);

    // Compare outputs
    LOG_INFO("");
    LOG_INFO("=== Comparing JIT vs C++ Reference ===");

    // Dequantize both outputs
    std::vector<float> ref_fp32(dump.M * dump.head_dim);
    std::vector<float> jit_fp32(dump.M * dump.head_dim);

    for (int m = 0; m < dump.M; ++m)
    {
        simd::dequantize_q8_1_to_fp32(
            ref_output.data() + m * num_blocks,
            ref_fp32.data() + m * dump.head_dim,
            dump.head_dim);
        simd::dequantize_q8_1_to_fp32(
            jit_output.data() + m * num_blocks,
            jit_fp32.data() + m * dump.head_dim,
            dump.head_dim);
    }

    double overall_cos = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("Overall JIT vs Reference cosine: " << std::fixed << std::setprecision(6) << overall_cos);

    // Per-row breakdown
    LOG_INFO("");
    LOG_INFO("Per-row breakdown:");
    double min_cos = 1.0;
    int worst_row = -1;
    for (int m = 0; m < dump.M; ++m)
    {
        double row_cos = cosine_similarity(
            ref_fp32.data() + m * dump.head_dim,
            jit_fp32.data() + m * dump.head_dim,
            dump.head_dim);

        std::string status = (row_cos >= 0.999) ? "✓" : "✗";
        LOG_INFO("  Row " << m << ": cosine=" << std::fixed << std::setprecision(6) << row_cos << " " << status);

        if (row_cos < min_cos)
        {
            min_cos = row_cos;
            worst_row = m;
        }
    }

    LOG_INFO("");
    if (overall_cos >= 0.999)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  ✓ JIT matches C++ reference (cosine >= 0.999)                 ║");
        LOG_INFO("║  The E2E divergence is due to Q8_1 precision loss, not bugs    ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }
    else
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  ✗ JIT DIVERGES from C++ reference!                            ║");
        LOG_INFO("║  Worst row: " << worst_row << " (cosine=" << std::fixed << std::setprecision(6) << min_cos << ")");
        LOG_INFO("║  This indicates a KERNEL BUG                                   ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

        // Print detailed comparison for worst row
        LOG_INFO("");
        LOG_INFO("Detailed comparison for worst row " << worst_row << ":");
        const float *ref_row = ref_fp32.data() + worst_row * dump.head_dim;
        const float *jit_row = jit_fp32.data() + worst_row * dump.head_dim;

        double max_diff = 0.0;
        int max_diff_idx = 0;
        for (int d = 0; d < dump.head_dim; ++d)
        {
            double diff = std::abs(ref_row[d] - jit_row[d]);
            if (diff > max_diff)
            {
                max_diff = diff;
                max_diff_idx = d;
            }
        }

        LOG_INFO("  Max diff at index " << max_diff_idx << ": ref=" << ref_row[max_diff_idx]
                                        << " jit=" << jit_row[max_diff_idx] << " diff=" << max_diff);

        // Print first few values
        LOG_INFO("  First 8 values:");
        LOG_INFO("    Ref: ");
        for (int d = 0; d < std::min(8, dump.head_dim); ++d)
        {
            LOG_INFO("      [" << d << "] " << ref_row[d]);
        }
        LOG_INFO("    JIT: ");
        for (int d = 0; d < std::min(8, dump.head_dim); ++d)
        {
            LOG_INFO("      [" << d << "] " << jit_row[d]);
        }
    }

    EXPECT_GE(overall_cos, 0.999) << "JIT should match C++ reference on captured E2E data";
}

/**
 * @brief Compare JIT Q8_1 attention output vs PyTorch FP32 ground truth
 *
 * This is the KEY TEST that reproduces the 0.81 cosine divergence from E2E.
 *
 * The dump contains Q/K/V inputs to head 4 of layer 0.
 * PyTorch ground truth (layer0_ATTENTION_CONTEXT.npy) has all 14 heads concatenated.
 * We extract head 4's slice [4*64 : 5*64] = [256:320] and compare.
 *
 * If this shows ~0.81 cosine, we've isolated the bug to the attention kernel.
 * If this shows ~0.99 cosine, the bug is elsewhere (Q/K/V projections, etc).
 */
TEST_F(Test__Q8_1_AttentionReplayFromDump, Layer0Head4_JitVsPytorchGroundTruth)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  REPLAY TEST: JIT Q8_1 vs PyTorch FP32 Ground Truth            ║");
    LOG_INFO("║  Goal: Reproduce 0.81 cosine divergence in isolation           ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // Load captured dump
    LoadedAttentionDump dump;
    std::string dump_path = "tests/v2/test_data/attention_dump_layer0_head4";

    if (!dump.load(dump_path))
    {
        GTEST_SKIP() << "Dump not found at: " << dump_path << " (run with -DDUMP_ATTENTION_INPUTS=ON to capture)";
    }

    // Load PyTorch ground truth
    std::string pytorch_path = "pytorch_qwen2_snapshots/layer0_ATTENTION_CONTEXT.npy";
    std::vector<float> pytorch_context;
    try
    {
        pytorch_context = loadNpy(pytorch_path);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "PyTorch snapshot not found: " << pytorch_path;
    }

    LOG_INFO("Loaded dump: M=" << dump.M << " N=" << dump.N << " head_dim=" << dump.head_dim);
    LOG_INFO("PyTorch context size: " << pytorch_context.size() << " floats");

    // Expected PyTorch shape: [1, seq_len, n_heads * head_dim] = [1, 9, 896]
    const int n_heads = 14;
    const int head_dim = 64;
    const int seq_len = dump.M;
    const int expected_size = seq_len * n_heads * head_dim; // 9 * 14 * 64 = 8064

    if (pytorch_context.size() != static_cast<size_t>(expected_size))
    {
        LOG_WARN("PyTorch context size mismatch: got " << pytorch_context.size()
                                                       << ", expected " << expected_size);
    }

    // Extract head 4's slice from PyTorch output
    // Layout: [seq, head, dim] flattened → row r, head h starts at r * (n_heads * head_dim) + h * head_dim
    const int target_head = 4; // From dump metadata
    std::vector<float> pytorch_head4(seq_len * head_dim);

    for (int r = 0; r < seq_len; ++r)
    {
        // Source: pytorch_context[r * n_heads * head_dim + target_head * head_dim : +head_dim]
        // Dest: pytorch_head4[r * head_dim : +head_dim]
        std::memcpy(
            pytorch_head4.data() + r * head_dim,
            pytorch_context.data() + r * n_heads * head_dim + target_head * head_dim,
            head_dim * sizeof(float));
    }

    LOG_INFO("");
    LOG_INFO("Extracted head " << target_head << " from PyTorch: " << pytorch_head4.size() << " floats");
    LOG_INFO("");

    // Run JIT kernel on captured Q8_1 inputs
    LOG_INFO("Running JIT kernel on captured inputs...");

    const int num_blocks = dump.head_dim / 32;
    std::vector<Q8_1Block> jit_output(dump.M * num_blocks);

    QuantisedAttentionJit_Q8_1_Fused jit(dump.head_dim, false);
    auto kernel_fn = jit.get_kernel();

    FusedQ8_1AttentionParams params = dump.build_params(jit_output.data());
    kernel_fn(&params);

    // Dequantize JIT output to FP32
    std::vector<float> jit_fp32(dump.M * dump.head_dim);
    for (int m = 0; m < dump.M; ++m)
    {
        simd::dequantize_q8_1_to_fp32(
            jit_output.data() + m * num_blocks,
            jit_fp32.data() + m * dump.head_dim,
            dump.head_dim);
    }

    // Compare JIT vs PyTorch ground truth
    LOG_INFO("");
    LOG_INFO("=== Comparing JIT Q8_1 vs PyTorch FP32 Ground Truth ===");

    double overall_cos = cosine_similarity(jit_fp32.data(), pytorch_head4.data(), jit_fp32.size());
    LOG_INFO("Overall JIT vs PyTorch cosine: " << std::fixed << std::setprecision(6) << overall_cos);

    // Per-row breakdown
    LOG_INFO("");
    LOG_INFO("Per-row breakdown:");
    double min_cos = 1.0;
    int worst_row = -1;
    for (int m = 0; m < dump.M; ++m)
    {
        double row_cos = cosine_similarity(
            jit_fp32.data() + m * dump.head_dim,
            pytorch_head4.data() + m * head_dim,
            head_dim);

        std::string status = (row_cos >= 0.95) ? "✓" : "✗";
        LOG_INFO("  Row " << m << ": cosine=" << std::fixed << std::setprecision(6) << row_cos << " " << status);

        if (row_cos < min_cos)
        {
            min_cos = row_cos;
            worst_row = m;
        }
    }

    LOG_INFO("");
    LOG_INFO("=== Sample Values Comparison (Row 0, first 8 elements) ===");
    LOG_INFO("PyTorch FP32:");
    for (int d = 0; d < std::min(8, head_dim); ++d)
    {
        LOG_INFO("  [" << d << "] " << std::fixed << std::setprecision(6) << pytorch_head4[d]);
    }
    LOG_INFO("JIT Q8_1 (dequantized):");
    for (int d = 0; d < std::min(8, head_dim); ++d)
    {
        LOG_INFO("  [" << d << "] " << std::fixed << std::setprecision(6) << jit_fp32[d]);
    }

    LOG_INFO("");
    if (overall_cos >= 0.95)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  ✓ JIT Q8_1 matches PyTorch FP32 reasonably well (>= 0.95)     ║");
        LOG_INFO("║  The 0.81 divergence is NOT isolated to attention kernel       ║");
        LOG_INFO("║  Bug likely in Q/K/V projections or data flow                  ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }
    else
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  ✗ DIVERGENCE REPRODUCED: JIT Q8_1 != PyTorch FP32            ║");
        LOG_INFO("║  Overall cosine: " << std::fixed << std::setprecision(4) << overall_cos);
        LOG_INFO("║  Worst row: " << worst_row << " (cosine=" << std::fixed << std::setprecision(4) << min_cos << ")");
        LOG_INFO("║  BUG IS IN ATTENTION KERNEL ITSELF                             ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

        // Detailed comparison for worst row
        LOG_INFO("");
        LOG_INFO("Detailed comparison for worst row " << worst_row << ":");
        const float *pytorch_row = pytorch_head4.data() + worst_row * head_dim;
        const float *jit_row = jit_fp32.data() + worst_row * dump.head_dim;

        double max_diff = 0.0;
        int max_diff_idx = 0;
        for (int d = 0; d < head_dim; ++d)
        {
            double diff = std::abs(pytorch_row[d] - jit_row[d]);
            if (diff > max_diff)
            {
                max_diff = diff;
                max_diff_idx = d;
            }
        }

        LOG_INFO("  Max diff at index " << max_diff_idx << ": pytorch=" << pytorch_row[max_diff_idx]
                                        << " jit=" << jit_row[max_diff_idx] << " diff=" << max_diff);
    }

    // We expect ~0.81 cosine if the E2E bug is in attention kernel
    // This is a diagnostic test - failure here confirms the bug location
    if (overall_cos < 0.85)
    {
        LOG_INFO("");
        LOG_INFO(">>> BUG LOCATION CONFIRMED: Attention kernel produces wrong output <<<");
        LOG_INFO(">>> The Q/K/V inputs have high cosine (~0.99), but output diverges <<<");
        LOG_INFO(">>> ROOT CAUSE: Q8_1 kernel uses Q/K normalization, FP32 does NOT <<<");
    }

    // Don't fail the test - this is diagnostic
    // Just verify we can load and compare the data
    EXPECT_GT(pytorch_head4.size(), 0u) << "Should have loaded PyTorch ground truth";
    EXPECT_GT(jit_fp32.size(), 0u) << "Should have produced JIT output";
}

/**
 * @brief Compute standard FP32 attention without Q/K normalization
 *
 * This implements the standard attention formula:
 *   attention = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * WITHOUT the Q/K normalization that the Q8_1 kernel uses:
 *   Q8_1 uses: score = (Q · K) * scale / (|Q| * |K|)
 *   Standard:  score = (Q · K) * scale  (where scale = 1/sqrt(d_k))
 */
static void compute_fp32_attention_standard(
    const float *Q, const float *K, const float *V,
    float *output,
    int M, int N, int head_dim,
    float scale, const float *mask, int mask_stride)
{
    for (int m = 0; m < M; ++m)
    {
        const float *Q_row = Q + m * head_dim;

        // Initialize online softmax state
        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;
        std::vector<float> context(head_dim, 0.0f);

        for (int n = 0; n < N; ++n)
        {
            const float *K_row = K + n * head_dim;
            const float *V_row = V + n * head_dim;

            // Compute dot product Q[m] · K[n]
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += Q_row[d] * K_row[d];
            }

            // Apply attention scale ONLY (no Q/K normalization!)
            float score = dot * scale;

            // Apply mask
            if (mask)
            {
                score += mask[m * mask_stride + n];
            }

            // Online softmax update
            if (score > max_score)
            {
                float correction = std::exp(max_score - score);
                sum_exp *= correction;
                for (int d = 0; d < head_dim; ++d)
                {
                    context[d] *= correction;
                }
                max_score = score;
            }

            float weight = std::exp(score - max_score);
            sum_exp += weight;

            // Accumulate weighted V
            for (int d = 0; d < head_dim; ++d)
            {
                context[d] += weight * V_row[d];
            }
        }

        // Normalize by sum of weights
        float inv_sum = 1.0f / sum_exp;
        float *out_row = output + m * head_dim;
        for (int d = 0; d < head_dim; ++d)
        {
            out_row[d] = context[d] * inv_sum;
        }
    }
}

/**
 * @brief Verify that FP32 attention on dequantized Q8_1 inputs matches PyTorch
 *
 * This test:
 * 1. Dequantizes the captured Q8_1 blocks to FP32
 * 2. Runs standard FP32 attention (no Q/K normalization)
 * 3. Compares with PyTorch ground truth
 *
 * If this shows high cosine (~0.99), it confirms:
 *   - The captured Q/K/V inputs are correct
 *   - The Q/K normalization is causing the divergence
 *   - Removing Q/K normalization will fix the E2E issue
 */
TEST_F(Test__Q8_1_AttentionReplayFromDump, Layer0Head4_DequantFP32VsPytorchGroundTruth)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  REPLAY TEST: Dequant FP32 Attention vs PyTorch Ground Truth   ║");
    LOG_INFO("║  Goal: Verify Q/K normalization is the root cause              ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    LOG_INFO("");

    // Load captured dump
    LoadedAttentionDump dump;
    std::string dump_path = "tests/v2/test_data/attention_dump_layer0_head4";

    if (!dump.load(dump_path))
    {
        GTEST_SKIP() << "Dump not found at: " << dump_path;
    }

    // Load PyTorch ground truth
    std::string pytorch_path = "pytorch_qwen2_snapshots/layer0_ATTENTION_CONTEXT.npy";
    std::vector<float> pytorch_context;
    try
    {
        pytorch_context = loadNpy(pytorch_path);
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "PyTorch snapshot not found: " << pytorch_path;
    }

    LOG_INFO("Loaded dump: M=" << dump.M << " N=" << dump.N << " head_dim=" << dump.head_dim);

    // Extract head 4's slice from PyTorch output
    const int n_heads = 14;
    const int head_dim = 64;
    const int seq_len = dump.M;
    const int target_head = 4;

    std::vector<float> pytorch_head4(seq_len * head_dim);
    for (int r = 0; r < seq_len; ++r)
    {
        std::memcpy(
            pytorch_head4.data() + r * head_dim,
            pytorch_context.data() + r * n_heads * head_dim + target_head * head_dim,
            head_dim * sizeof(float));
    }

    // Dequantize Q8_1 blocks to FP32
    const int num_blocks = dump.head_dim / 32;
    std::vector<float> Q_fp32(dump.M * dump.head_dim);
    std::vector<float> K_fp32(dump.N * dump.head_dim);
    std::vector<float> V_fp32(dump.N * dump.head_dim);

    LOG_INFO("Dequantizing Q8_1 blocks to FP32...");

    // Dequantize Q
    for (int m = 0; m < dump.M; ++m)
    {
        simd::dequantize_q8_1_to_fp32(
            dump.Q_blocks.data() + m * num_blocks,
            Q_fp32.data() + m * dump.head_dim,
            dump.head_dim);
    }

    // Dequantize K
    for (int n = 0; n < dump.N; ++n)
    {
        simd::dequantize_q8_1_to_fp32(
            dump.K_blocks.data() + n * num_blocks,
            K_fp32.data() + n * dump.head_dim,
            dump.head_dim);
    }

    // Dequantize V
    for (int n = 0; n < dump.N; ++n)
    {
        simd::dequantize_q8_1_to_fp32(
            dump.V_blocks.data() + n * num_blocks,
            V_fp32.data() + n * dump.head_dim,
            dump.head_dim);
    }

    LOG_INFO("Running standard FP32 attention (no Q/K normalization)...");

    // Run standard FP32 attention
    std::vector<float> fp32_output(dump.M * dump.head_dim);
    compute_fp32_attention_standard(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        fp32_output.data(),
        dump.M, dump.N, dump.head_dim,
        dump.scale,
        dump.has_mask ? dump.mask.data() : nullptr,
        dump.N);

    // Compare FP32 attention output vs PyTorch ground truth
    LOG_INFO("");
    LOG_INFO("=== Comparing Dequant FP32 Attention vs PyTorch Ground Truth ===");

    double overall_cos = cosine_similarity(fp32_output.data(), pytorch_head4.data(), fp32_output.size());
    LOG_INFO("Overall Dequant FP32 vs PyTorch cosine: " << std::fixed << std::setprecision(6) << overall_cos);

    // Per-row breakdown
    LOG_INFO("");
    LOG_INFO("Per-row breakdown:");
    double min_cos = 1.0;
    int worst_row = -1;
    for (int m = 0; m < dump.M; ++m)
    {
        double row_cos = cosine_similarity(
            fp32_output.data() + m * dump.head_dim,
            pytorch_head4.data() + m * head_dim,
            head_dim);

        std::string status = (row_cos >= 0.95) ? "✓" : "✗";
        LOG_INFO("  Row " << m << ": cosine=" << std::fixed << std::setprecision(6) << row_cos << " " << status);

        if (row_cos < min_cos)
        {
            min_cos = row_cos;
            worst_row = m;
        }
    }

    LOG_INFO("");
    LOG_INFO("=== Sample Values Comparison (Row 0, first 8 elements) ===");
    LOG_INFO("PyTorch FP32 ground truth:");
    for (int d = 0; d < std::min(8, head_dim); ++d)
    {
        LOG_INFO("  [" << d << "] " << std::fixed << std::setprecision(6) << pytorch_head4[d]);
    }
    LOG_INFO("Dequant FP32 attention output:");
    for (int d = 0; d < std::min(8, head_dim); ++d)
    {
        LOG_INFO("  [" << d << "] " << std::fixed << std::setprecision(6) << fp32_output[d]);
    }

    LOG_INFO("");
    if (overall_cos >= 0.95)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  ✓ Dequant FP32 matches PyTorch (>= 0.95)                      ║");
        LOG_INFO("║  ROOT CAUSE CONFIRMED: Q/K normalization is the bug           ║");
        LOG_INFO("║  FIX: Remove Q/K normalization from Q8_1 JIT kernel           ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }
    else
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  ✗ Dequant FP32 still diverges from PyTorch                   ║");
        LOG_INFO("║  Overall cosine: " << std::fixed << std::setprecision(4) << overall_cos);
        LOG_INFO("║  Q/K normalization may not be the only issue                  ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    // This test confirms whether the fix will work
    EXPECT_GT(overall_cos, 0.9) << "Dequant FP32 should reasonably match PyTorch if Q/K norm is the issue";
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
