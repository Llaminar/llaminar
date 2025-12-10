/**
 * @file Test__Q8_1_AttentionJitReplay.cpp
 * @brief Replay-based test for Q8_1 JIT attention kernel debugging
 *
 * This test creates a fully reproducible harness for the Q8_1 attention kernel:
 * 1. DumpAttentionInputs - Runs Q8_1 pipeline, captures raw Q8_1 blocks before attention
 * 2. ReplayJitAttention - Loads saved blocks, replays JIT kernel in isolation
 *
 * The goal is to have a deterministic, isolated test where we can add debug
 * prints to the JIT kernel to trace exactly where computations diverge.
 *
 * Key difference from Test__Q8_1_AttentionKernel.cpp:
 * - That test quantizes FP32 inputs to Q8_1 → gets 0.999+ cosine (kernel works!)
 * - This test uses ACTUAL Q8_1 blocks from the pipeline → reproduces 0.83 cosine
 *
 * @author David Sanftenberg
 * @date 2025-12-10
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>

// Core includes
#include "pipelines/qwen/Qwen2Pipeline.h"
#include "pipelines/PipelineConfig.h"
#include "loaders/ModelContext.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "kernels/cpu/attention/CPUAttentionKernelTyped.h"
#include "kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"

using namespace llaminar2;

namespace
{
    constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    constexpr const char *DUMP_DIR = "/tmp/q8_1_attention_replay";

    // Qwen 2.5 0.5B model parameters
    constexpr int N_HEADS = 14;
    constexpr int N_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int D_MODEL = 896;

    /**
     * @brief Compute cosine similarity
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Metadata for saved attention inputs
     */
    struct AttentionReplayData
    {
        int seq_len;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        size_t q_blocks;  // Number of Q8_1 blocks for Q
        size_t kv_blocks; // Number of Q8_1 blocks for K/V

        // Raw Q8_1 block data
        std::vector<Q8_1Block> Q_blocks;
        std::vector<Q8_1Block> K_blocks;
        std::vector<Q8_1Block> V_blocks;

        // FP32 reference attention output from FP32 pipeline
        std::vector<float> fp32_attention_output;
    };

    /**
     * @brief Save attention replay data to disk
     */
    void save_replay_data(const std::string &dir, const AttentionReplayData &data)
    {
        std::filesystem::create_directories(dir);

        // Save metadata
        std::ofstream meta(dir + "/metadata.bin", std::ios::binary);
        meta.write(reinterpret_cast<const char *>(&data.seq_len), sizeof(data.seq_len));
        meta.write(reinterpret_cast<const char *>(&data.n_heads), sizeof(data.n_heads));
        meta.write(reinterpret_cast<const char *>(&data.n_kv_heads), sizeof(data.n_kv_heads));
        meta.write(reinterpret_cast<const char *>(&data.head_dim), sizeof(data.head_dim));
        meta.write(reinterpret_cast<const char *>(&data.q_blocks), sizeof(data.q_blocks));
        meta.write(reinterpret_cast<const char *>(&data.kv_blocks), sizeof(data.kv_blocks));

        // Save Q8_1 blocks as raw binary
        std::ofstream q_file(dir + "/Q_blocks.bin", std::ios::binary);
        q_file.write(reinterpret_cast<const char *>(data.Q_blocks.data()),
                     data.Q_blocks.size() * sizeof(Q8_1Block));

        std::ofstream k_file(dir + "/K_blocks.bin", std::ios::binary);
        k_file.write(reinterpret_cast<const char *>(data.K_blocks.data()),
                     data.K_blocks.size() * sizeof(Q8_1Block));

        std::ofstream v_file(dir + "/V_blocks.bin", std::ios::binary);
        v_file.write(reinterpret_cast<const char *>(data.V_blocks.data()),
                     data.V_blocks.size() * sizeof(Q8_1Block));

        // Save FP32 reference output
        std::ofstream ref_file(dir + "/fp32_reference.bin", std::ios::binary);
        uint64_t ref_size = data.fp32_attention_output.size();
        ref_file.write(reinterpret_cast<const char *>(&ref_size), sizeof(ref_size));
        ref_file.write(reinterpret_cast<const char *>(data.fp32_attention_output.data()),
                       ref_size * sizeof(float));

        LOG_INFO("[save_replay_data] Saved to " << dir);
        LOG_INFO("  Q blocks: " << data.Q_blocks.size());
        LOG_INFO("  K blocks: " << data.K_blocks.size());
        LOG_INFO("  V blocks: " << data.V_blocks.size());
        LOG_INFO("  FP32 reference: " << data.fp32_attention_output.size() << " elements");
    }

    /**
     * @brief Load attention replay data from disk
     */
    bool load_replay_data(const std::string &dir, AttentionReplayData &data)
    {
        std::ifstream meta(dir + "/metadata.bin", std::ios::binary);
        if (!meta.is_open())
        {
            LOG_WARN("[load_replay_data] No metadata found at " << dir);
            return false;
        }

        meta.read(reinterpret_cast<char *>(&data.seq_len), sizeof(data.seq_len));
        meta.read(reinterpret_cast<char *>(&data.n_heads), sizeof(data.n_heads));
        meta.read(reinterpret_cast<char *>(&data.n_kv_heads), sizeof(data.n_kv_heads));
        meta.read(reinterpret_cast<char *>(&data.head_dim), sizeof(data.head_dim));
        meta.read(reinterpret_cast<char *>(&data.q_blocks), sizeof(data.q_blocks));
        meta.read(reinterpret_cast<char *>(&data.kv_blocks), sizeof(data.kv_blocks));

        // Load Q8_1 blocks
        data.Q_blocks.resize(data.q_blocks);
        std::ifstream q_file(dir + "/Q_blocks.bin", std::ios::binary);
        q_file.read(reinterpret_cast<char *>(data.Q_blocks.data()),
                    data.q_blocks * sizeof(Q8_1Block));

        data.K_blocks.resize(data.kv_blocks);
        std::ifstream k_file(dir + "/K_blocks.bin", std::ios::binary);
        k_file.read(reinterpret_cast<char *>(data.K_blocks.data()),
                    data.kv_blocks * sizeof(Q8_1Block));

        data.V_blocks.resize(data.kv_blocks);
        std::ifstream v_file(dir + "/V_blocks.bin", std::ios::binary);
        v_file.read(reinterpret_cast<char *>(data.V_blocks.data()),
                    data.kv_blocks * sizeof(Q8_1Block));

        // Load FP32 reference
        std::ifstream ref_file(dir + "/fp32_reference.bin", std::ios::binary);
        uint64_t ref_size;
        ref_file.read(reinterpret_cast<char *>(&ref_size), sizeof(ref_size));
        data.fp32_attention_output.resize(ref_size);
        ref_file.read(reinterpret_cast<char *>(data.fp32_attention_output.data()),
                      ref_size * sizeof(float));

        LOG_INFO("[load_replay_data] Loaded from " << dir);
        LOG_INFO("  seq_len=" << data.seq_len << ", n_heads=" << data.n_heads
                              << ", n_kv_heads=" << data.n_kv_heads << ", head_dim=" << data.head_dim);
        LOG_INFO("  Q blocks: " << data.Q_blocks.size());
        LOG_INFO("  K/V blocks: " << data.K_blocks.size());

        return true;
    }

} // namespace

/**
 * @brief Test fixture for Q8_1 attention JIT replay
 */
class Test__Q8_1_AttentionJitReplay : public ::testing::Test
{
protected:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        std::cout << "RUNNING MODIFIED TEST" << std::endl;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

        model_ctx_ = ModelContext::create(MODEL_PATH, mpi_ctx_);
        if (!model_ctx_)
        {
            GTEST_SKIP() << "Model not found: " << MODEL_PATH;
        }
    }

    void TearDown() override
    {
        model_ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a causal mask for attention
     */
    std::vector<float> createCausalMask(int seq_len)
    {
        std::vector<float> mask(seq_len * seq_len);
        constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

        for (int m = 0; m < seq_len; ++m)
        {
            for (int n = 0; n < seq_len; ++n)
            {
                mask[m * seq_len + n] = (n <= m) ? 0.0f : NEG_INF;
            }
        }
        return mask;
    }

    /**
     * @brief Run the FP32 reference attention computation
     */
    std::vector<float> runFP32ReferenceAttention(
        const std::vector<float> &Q_fp32,
        const std::vector<float> &K_fp32,
        const std::vector<float> &V_fp32,
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

        CPUAttentionKernelTyped<ActivationPrecision::FP32> kernel;

        auto workspace_scores = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});

        bool success = kernel.compute(
            Q_fp32.data(), K_fp32.data(), V_fp32.data(), output.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            /*causal=*/true, /*window_size=*/-1,
            workspace_scores.get(),
            /*mask=*/nullptr, /*K_cache=*/nullptr, /*V_cache=*/nullptr,
            /*is_batched=*/false, mpi_ctx_.get(), /*device_idx=*/-1);

        if (!success)
        {
            LOG_ERROR("FP32 reference attention failed");
        }

        return output;
    }

    /**
     * @brief Run Q8_1 JIT attention directly on Q8_1 blocks
     *
     * This bypasses all the higher-level abstractions and calls the JIT kernel
     * directly with the raw Q8_1 block data.
     */
    std::vector<float> runQ8_1JitAttention(
        const std::vector<Q8_1Block> &Q_blocks,
        const std::vector<Q8_1Block> &K_blocks,
        const std::vector<Q8_1Block> &V_blocks,
        const std::vector<float> &mask,
        int seq_len, int n_heads, int n_kv_heads, int head_dim)
    {
        // Output buffer - Q8_1 blocks that will be dequantized at the end
        size_t output_blocks_per_row = (n_heads * head_dim + 31) / 32;
        std::vector<Q8_1Block> output_blocks(seq_len * output_blocks_per_row);

        // Strides in bytes
        int q_blocks_per_row = (n_heads * head_dim + 31) / 32;
        int kv_blocks_per_row = (n_kv_heads * head_dim + 31) / 32;

        int Q_stride_bytes = q_blocks_per_row * sizeof(Q8_1Block);
        int K_stride_bytes = kv_blocks_per_row * sizeof(Q8_1Block);
        int V_stride_bytes = kv_blocks_per_row * sizeof(Q8_1Block);
        int out_stride_bytes = q_blocks_per_row * sizeof(Q8_1Block);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // GQA: n_heads / n_kv_heads = heads per KV group
        int heads_per_kv = n_heads / n_kv_heads;

        LOG_INFO("[runQ8_1JitAttention] Parameters:");
        LOG_INFO("  seq_len=" << seq_len << ", n_heads=" << n_heads << ", n_kv_heads=" << n_kv_heads);
        LOG_INFO("  head_dim=" << head_dim << ", scale=" << scale);
        LOG_INFO("  Q_stride=" << Q_stride_bytes << " bytes, K/V_stride=" << K_stride_bytes << " bytes");
        LOG_INFO("  heads_per_kv=" << heads_per_kv);

        // Create the JIT kernel with debug output
        gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit_kernel(head_dim, /*debug_gen=*/false);
        auto kernel_fn = jit_kernel.get_kernel();

        LOG_INFO("  JIT kernel code size: " << jit_kernel.get_code_size() << " bytes");

        // Process each head
        // GQA layout: Q is [seq_len, n_heads, head_dim], K/V is [seq_len, n_kv_heads, head_dim]
        // We need to call the kernel per head, mapping Q heads to KV heads

        for (int h = 0; h < n_heads; ++h)
        {
            int kv_head = h / heads_per_kv; // Which KV head this Q head uses

            // Q pointer for head h: offset by h * head_dim elements
            // Since Q is stored as [seq_len, n_heads * head_dim], head h starts at offset h * head_dim
            // In Q8_1 blocks: each block has 32 elements, head_dim = 64, so 2 blocks per head
            int q_block_offset = h * (head_dim / 32);
            const Q8_1Block *q_ptr = Q_blocks.data() + q_block_offset;

            // K/V pointer for kv_head: offset similarly
            int kv_block_offset = kv_head * (head_dim / 32);
            const Q8_1Block *k_ptr = K_blocks.data() + kv_block_offset;
            const Q8_1Block *v_ptr = V_blocks.data() + kv_block_offset;

            // Output pointer for head h
            int out_block_offset = h * (head_dim / 32);
            Q8_1Block *out_ptr = output_blocks.data() + out_block_offset;

            // Mask pointer (mask is [seq_len, seq_len], same for all heads)
            const float *mask_ptr = mask.data();

            gemm_v4::FusedQ8_1AttentionParams params;
            params.Q = q_ptr;
            params.K = k_ptr;
            params.V = v_ptr;
            params.output = out_ptr;
            params.M = seq_len; // Number of Q rows
            params.N = seq_len; // Number of K/V rows (same as Q for prefill)
            params.head_dim = head_dim;
            params.Q_stride_bytes = Q_stride_bytes; // Full row stride
            params.K_stride_bytes = K_stride_bytes;
            params.V_stride_bytes = V_stride_bytes;
            params.output_stride_bytes = out_stride_bytes;
            params.scale = scale;
            params.mask = mask_ptr;
            params.mask_stride = seq_len;

            // Run JIT kernel
            kernel_fn(&params);
        }

        // Dequantize output to FP32
        std::vector<float> output_fp32(seq_len * n_heads * head_dim);
        simd::dequantize_q8_1_to_fp32(output_blocks.data(), output_fp32.data(), output_fp32.size());

        return output_fp32;
    }
};

#ifdef ENABLE_PIPELINE_SNAPSHOTS
/**
 * @brief Dump attention inputs from FP32 pipeline for replay testing
 *
 * This test:
 * 1. Runs FP32 pipeline to capture FP32 Q/K/V (after RoPE, before attention)
 * 2. Uses FP32 attention output as ground truth
 * 3. Quantizes Q/K/V to Q8_1 and saves for replay
 *
 * KEY INSIGHT: We want to test the Q8_1 ATTENTION KERNEL in isolation.
 * So we use the SAME Q/K/V inputs (from FP32, then quantized) for both:
 *   - FP32 reference attention (ground truth from pipeline)
 *   - Q8_1 JIT attention (in replay test)
 *
 * This isolates the attention kernel specifically.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, DumpAttentionInputs)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  DUMPING ATTENTION INPUTS FOR Q8_1 KERNEL REPLAY               ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    AttentionReplayData data;
    data.seq_len = seq_len;
    data.n_heads = N_HEADS;
    data.n_kv_heads = N_KV_HEADS;
    data.head_dim = HEAD_DIM;

    // Run FP32 pipeline to capture Q/K/V and attention output
    LOG_INFO("");
    LOG_INFO("Running FP32 pipeline to capture Q/K/V and attention output...");

    PipelineConfig fp32_config;
    fp32_config.activation_precision = ActivationPrecision::FP32;
    fp32_config.max_seq_len = 512;

    auto fp32_pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, fp32_config, 1);
    fp32_pipeline->enableSnapshotCapture();

    bool success = fp32_pipeline->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success) << "FP32 pipeline forward failed";

    // Get FP32 Q/K/V snapshots (after RoPE, before attention)
    size_t q_size = 0, k_size = 0, v_size = 0;
    const float *q_fp32 = fp32_pipeline->getSnapshot("layer0_Q_ROPE", q_size);
    const float *k_fp32 = fp32_pipeline->getSnapshot("layer0_K_ROPE", k_size);
    const float *v_fp32 = fp32_pipeline->getSnapshot("layer0_V_PROJECTION", v_size);

    ASSERT_NE(q_fp32, nullptr) << "Q snapshot not found";
    ASSERT_NE(k_fp32, nullptr) << "K snapshot not found";
    ASSERT_NE(v_fp32, nullptr) << "V snapshot not found";

    LOG_INFO("  Q (FP32): " << q_size << " elements = [" << seq_len << " x " << N_HEADS << " x " << HEAD_DIM << "]");
    LOG_INFO("  K (FP32): " << k_size << " elements = [" << seq_len << " x " << N_KV_HEADS << " x " << HEAD_DIM << "]");
    LOG_INFO("  V (FP32): " << v_size << " elements = [" << seq_len << " x " << N_KV_HEADS << " x " << HEAD_DIM << "]");

    // Get FP32 attention output (this is our ground truth)
    size_t attn_size = 0;
    const float *attn_fp32 = fp32_pipeline->getSnapshot("layer0_ATTENTION_CONTEXT", attn_size);
    ASSERT_NE(attn_fp32, nullptr) << "Attention snapshot not found";
    LOG_INFO("  Attention (FP32): " << attn_size << " elements");

    data.fp32_attention_output.assign(attn_fp32, attn_fp32 + attn_size);

    // Quantize FP32 Q/K/V to Q8_1 blocks for replay
    LOG_INFO("");
    LOG_INFO("Quantizing Q/K/V to Q8_1 blocks...");

    size_t q_blocks_count = (q_size + 31) / 32;
    size_t kv_blocks_count = (k_size + 31) / 32;

    data.q_blocks = q_blocks_count;
    data.kv_blocks = kv_blocks_count;

    data.Q_blocks.resize(q_blocks_count);
    data.K_blocks.resize(kv_blocks_count);
    data.V_blocks.resize(kv_blocks_count);

    simd::quantize_fp32_to_q8_1_blocks(q_fp32, data.Q_blocks.data(), q_size);
    simd::quantize_fp32_to_q8_1_blocks(k_fp32, data.K_blocks.data(), k_size);
    simd::quantize_fp32_to_q8_1_blocks(v_fp32, data.V_blocks.data(), v_size);

    LOG_INFO("  Q8_1 blocks: Q=" << q_blocks_count << ", K/V=" << kv_blocks_count);

    // Verify quantization quality
    std::vector<float> q_roundtrip(q_size);
    for (size_t i = 0; i < q_blocks_count; ++i)
    {
        float decoded[32];
        const auto &block = data.Q_blocks[i];
        float scale = fp16_to_fp32(block.d);
        for (int j = 0; j < 32; ++j)
        {
            decoded[j] = scale * static_cast<float>(block.qs[j]);
        }
        size_t copy_count = std::min(size_t(32), q_size - i * 32);
        std::memcpy(&q_roundtrip[i * 32], decoded, copy_count * sizeof(float));
    }
    double quant_cos = cosine_similarity(q_fp32, q_roundtrip.data(), q_size);
    LOG_INFO("  Q quantization fidelity (roundtrip cosine): " << std::fixed << std::setprecision(6) << quant_cos);

    // Save to disk
    save_replay_data(DUMP_DIR, data);

    LOG_INFO("");
    LOG_INFO("✓ Replay data saved to " << DUMP_DIR);
    LOG_INFO("");
    LOG_INFO("Now run ReplayJitAttention to test the Q8_1 JIT kernel against this data.");
}

/**
 * @brief Replay Q8_1 attention kernel on saved inputs
 *
 * This test loads saved Q8_1 blocks and runs the JIT kernel directly,
 * comparing against the FP32 reference. This is the test where we can
 * add debug prints to the JIT kernel.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, ReplayJitAttention)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  REPLAYING Q8_1 JIT ATTENTION KERNEL                           ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    ASSERT_EQ(data.n_heads, N_HEADS);
    ASSERT_EQ(data.n_kv_heads, N_KV_HEADS);
    ASSERT_EQ(data.head_dim, HEAD_DIM);

    // Create causal mask
    auto mask = createCausalMask(data.seq_len);

    LOG_INFO("");
    LOG_INFO("Running Q8_1 JIT attention kernel...");

    // Run JIT kernel
    auto q8_output = runQ8_1JitAttention(
        data.Q_blocks, data.K_blocks, data.V_blocks, mask,
        data.seq_len, data.n_heads, data.n_kv_heads, data.head_dim);

    // Compare with FP32 reference
    ASSERT_EQ(q8_output.size(), data.fp32_attention_output.size());

    double cos_sim = cosine_similarity(
        data.fp32_attention_output.data(), q8_output.data(), q8_output.size());

    LOG_INFO("");
    LOG_INFO("=== COMPARISON WITH FP32 REFERENCE ===");
    LOG_INFO("  Overall cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim);

    // Per-position breakdown
    LOG_INFO("");
    LOG_INFO("Per-position cosine similarity:");
    for (int pos = 0; pos < data.seq_len; ++pos)
    {
        const float *ref_row = data.fp32_attention_output.data() + pos * data.n_heads * data.head_dim;
        const float *q8_row = q8_output.data() + pos * data.n_heads * data.head_dim;
        double pos_cos = cosine_similarity(ref_row, q8_row, data.n_heads * data.head_dim);
        LOG_INFO("  Position " << pos << ": " << std::fixed << std::setprecision(6) << pos_cos);
    }

    // Print some sample values for debugging
    LOG_INFO("");
    LOG_INFO("Sample output values (first 10):");
    for (int i = 0; i < std::min(10, static_cast<int>(q8_output.size())); ++i)
    {
        LOG_INFO("  [" << i << "] FP32_ref=" << std::setprecision(6) << data.fp32_attention_output[i]
                       << " Q8_1=" << q8_output[i]
                       << " diff=" << std::abs(data.fp32_attention_output[i] - q8_output[i]));
    }

    // The bug manifests as ~0.83 cosine - verify we reproduce it
    LOG_INFO("");
    if (cos_sim < 0.90)
    {
        LOG_INFO("✓ Successfully reproduced divergence (cosine=" << cos_sim << ")");
        LOG_INFO("  This harness can now be used to debug the JIT kernel.");
    }
    else
    {
        LOG_WARN("⚠ Divergence NOT reproduced (cosine=" << cos_sim << ")");
        LOG_WARN("  This may indicate the bug is elsewhere or the replay data doesn't match.");
    }

    // For now, we don't assert pass/fail - this is a diagnostic test
    // Once we fix the kernel, change this to EXPECT_GE(cos_sim, 0.95)
}

/**
 * @brief Debug test: Run attention on a single head with detailed output
 */
TEST_F(Test__Q8_1_AttentionJitReplay, DebugSingleHead)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  DEBUG: SINGLE HEAD ATTENTION ANALYSIS                         ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int head_dim = data.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Extract Q/K/V for head 0 only
    LOG_INFO("");
    LOG_INFO("Analyzing head 0...");

    // Dequantize Q/K/V to FP32 for analysis
    size_t q_elements = seq_len * data.n_heads * head_dim;
    size_t kv_elements = seq_len * data.n_kv_heads * head_dim;

    std::vector<float> Q_fp32(q_elements);
    std::vector<float> K_fp32(kv_elements);
    std::vector<float> V_fp32(kv_elements);

    simd::dequantize_q8_1_to_fp32(data.Q_blocks.data(), Q_fp32.data(), q_elements);
    simd::dequantize_q8_1_to_fp32(data.K_blocks.data(), K_fp32.data(), kv_elements);
    simd::dequantize_q8_1_to_fp32(data.V_blocks.data(), V_fp32.data(), kv_elements);

    // Compute reference attention for head 0 manually
    // Q[m] @ K^T -> scores[m, n]
    LOG_INFO("");
    LOG_INFO("Computing Q @ K^T scores for head 0...");

    std::vector<float> scores(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        const float *q_row = Q_fp32.data() + m * data.n_heads * head_dim; // Head 0
        for (int n = 0; n < seq_len; ++n)
        {
            const float *k_row = K_fp32.data() + n * data.n_kv_heads * head_dim; // KV head 0
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += q_row[d] * k_row[d];
            }
            scores[m * seq_len + n] = dot * scale;
        }
    }

    // Apply causal mask
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    for (int m = 0; m < seq_len; ++m)
    {
        for (int n = m + 1; n < seq_len; ++n)
        {
            scores[m * seq_len + n] = NEG_INF;
        }
    }

    // Print scores for first few positions
    LOG_INFO("");
    LOG_INFO("Scores[0, :] (first position, should be [value, -inf, -inf, ...]):");
    for (int n = 0; n < std::min(5, seq_len); ++n)
    {
        LOG_INFO("  scores[0, " << n << "] = " << scores[n]);
    }

    LOG_INFO("");
    LOG_INFO("Scores[1, :] (second position, should be [value, value, -inf, ...]):");
    for (int n = 0; n < std::min(5, seq_len); ++n)
    {
        LOG_INFO("  scores[1, " << n << "] = " << scores[seq_len + n]);
    }

    // Compute softmax
    std::vector<float> attn_weights(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int n = 0; n <= m; ++n)
        {
            max_val = std::max(max_val, scores[m * seq_len + n]);
        }

        float sum = 0.0f;
        for (int n = 0; n <= m; ++n)
        {
            attn_weights[m * seq_len + n] = std::exp(scores[m * seq_len + n] - max_val);
            sum += attn_weights[m * seq_len + n];
        }

        for (int n = 0; n <= m; ++n)
        {
            attn_weights[m * seq_len + n] /= sum;
        }
    }

    // Print attention weights for first few positions
    LOG_INFO("");
    LOG_INFO("Attention weights[0, :] (should be [1.0, 0, 0, ...]):");
    for (int n = 0; n < std::min(5, seq_len); ++n)
    {
        LOG_INFO("  attn[0, " << n << "] = " << attn_weights[n]);
    }

    // Compute context for head 0
    std::vector<float> context(seq_len * head_dim, 0.0f);
    for (int m = 0; m < seq_len; ++m)
    {
        for (int n = 0; n <= m; ++n)
        {
            float w = attn_weights[m * seq_len + n];
            const float *v_row = V_fp32.data() + n * data.n_kv_heads * head_dim;
            for (int d = 0; d < head_dim; ++d)
            {
                context[m * head_dim + d] += w * v_row[d];
            }
        }
    }

    // Compare with FP32 reference (head 0 only)
    LOG_INFO("");
    LOG_INFO("Computed context vs FP32 reference (head 0):");
    for (int m = 0; m < seq_len; ++m)
    {
        const float *ref_head0 = data.fp32_attention_output.data() + m * data.n_heads * head_dim;
        const float *comp_head0 = context.data() + m * head_dim;
        double pos_cos = cosine_similarity(ref_head0, comp_head0, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << pos_cos);
    }

    LOG_INFO("");
    LOG_INFO("This manual computation should match FP32 reference.");
    LOG_INFO("If it does but JIT doesn't, the bug is in the JIT kernel.");
}

/**
 * @brief Test that directly compares JIT kernel output vs manual FP32 computation
 *
 * This eliminates any ambiguity about the reference - we compare JIT to our own
 * manual FP32 computation on the SAME dequantized Q8_1 data.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, JitVsManualFP32)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  JIT KERNEL vs MANUAL FP32 ATTENTION                           ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int head_dim = data.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Dequantize Q/K/V to FP32
    size_t q_elements = seq_len * data.n_heads * head_dim;
    size_t kv_elements = seq_len * data.n_kv_heads * head_dim;

    std::vector<float> Q_fp32(q_elements);
    std::vector<float> K_fp32(kv_elements);
    std::vector<float> V_fp32(kv_elements);

    simd::dequantize_q8_1_to_fp32(data.Q_blocks.data(), Q_fp32.data(), q_elements);
    simd::dequantize_q8_1_to_fp32(data.K_blocks.data(), K_fp32.data(), kv_elements);
    simd::dequantize_q8_1_to_fp32(data.V_blocks.data(), V_fp32.data(), kv_elements);

    // Create causal mask
    auto mask = createCausalMask(seq_len);

    // === RUN JIT KERNEL ===
    LOG_INFO("");
    LOG_INFO("Running Q8_1 JIT attention kernel...");

    auto jit_output = runQ8_1JitAttention(
        data.Q_blocks, data.K_blocks, data.V_blocks, mask,
        seq_len, data.n_heads, data.n_kv_heads, head_dim);

    // === RUN MANUAL FP32 ATTENTION ===
    LOG_INFO("");
    LOG_INFO("Running manual FP32 attention (for comparison)...");

    std::vector<float> manual_output(seq_len * data.n_heads * head_dim, 0.0f);

    // Process each head
    for (int h = 0; h < data.n_heads; ++h)
    {
        int kv_head = h / (data.n_heads / data.n_kv_heads);

        // Get Q/K/V pointers for this head
        // Q: [seq_len, n_heads, head_dim]
        // K/V: [seq_len, n_kv_heads, head_dim]
        for (int m = 0; m < seq_len; ++m)
        {
            // Compute scores for row m
            std::vector<float> scores(seq_len);
            for (int n = 0; n < seq_len; ++n)
            {
                // Q[m, h, :] @ K[n, kv_head, :].T
                const float *q_vec = Q_fp32.data() + m * data.n_heads * head_dim + h * head_dim;
                const float *k_vec = K_fp32.data() + n * data.n_kv_heads * head_dim + kv_head * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[n] = dot * scale;
            }

            // Apply causal mask
            for (int n = m + 1; n < seq_len; ++n)
            {
                scores[n] = -std::numeric_limits<float>::infinity();
            }

            // Softmax
            float max_val = *std::max_element(scores.begin(), scores.begin() + m + 1);
            float sum = 0.0f;
            for (int n = 0; n <= m; ++n)
            {
                scores[n] = std::exp(scores[n] - max_val);
                sum += scores[n];
            }
            for (int n = 0; n <= m; ++n)
            {
                scores[n] /= sum;
            }

            // Weighted sum of V
            float *out_vec = manual_output.data() + m * data.n_heads * head_dim + h * head_dim;
            for (int n = 0; n <= m; ++n)
            {
                const float *v_vec = V_fp32.data() + n * data.n_kv_heads * head_dim + kv_head * head_dim;
                for (int d = 0; d < head_dim; ++d)
                {
                    out_vec[d] += scores[n] * v_vec[d];
                }
            }
        }
    }

    // === COMPARE JIT vs MANUAL ===
    LOG_INFO("");
    LOG_INFO("=== JIT vs MANUAL FP32 COMPARISON ===");

    double overall_cos = cosine_similarity(manual_output.data(), jit_output.data(), jit_output.size());
    LOG_INFO("  Overall cosine similarity: " << std::fixed << std::setprecision(6) << overall_cos);

    LOG_INFO("");
    LOG_INFO("Per-position cosine similarity:");
    for (int pos = 0; pos < seq_len; ++pos)
    {
        const float *manual_row = manual_output.data() + pos * data.n_heads * head_dim;
        const float *jit_row = jit_output.data() + pos * data.n_heads * head_dim;
        double pos_cos = cosine_similarity(manual_row, jit_row, data.n_heads * head_dim);
        LOG_INFO("  Position " << pos << ": " << std::fixed << std::setprecision(6) << pos_cos);
    }

    // Per-head analysis for position 1 (first multi-position attention)
    LOG_INFO("");
    LOG_INFO("Per-head analysis for position 1 (first multi-position attention):");
    for (int h = 0; h < data.n_heads; ++h)
    {
        int pos = 1;
        const float *manual_head = manual_output.data() + pos * data.n_heads * head_dim + h * head_dim;
        const float *jit_head = jit_output.data() + pos * data.n_heads * head_dim + h * head_dim;
        double head_cos = cosine_similarity(manual_head, jit_head, head_dim);
        LOG_INFO("  Head " << h << ": cosine=" << std::fixed << std::setprecision(6) << head_cos);
    }

    LOG_INFO("");
    if (overall_cos < 0.95)
    {
        LOG_ERROR("✗ JIT kernel diverges from manual FP32 computation!");
        LOG_ERROR("  The bug is definitively in the JIT kernel.");
    }
    else
    {
        LOG_INFO("✓ JIT kernel matches manual FP32 computation.");
    }

    // Now compare manual output (computed on roundtrip Q/K/V) to FP32 reference
    LOG_INFO("");
    LOG_INFO("=== MANUAL (roundtrip Q/K/V) vs FP32 PIPELINE REFERENCE ===");
    double manual_vs_ref = cosine_similarity(manual_output.data(), data.fp32_attention_output.data(), manual_output.size());
    LOG_INFO("  Overall cosine similarity: " << std::fixed << std::setprecision(6) << manual_vs_ref);

    LOG_INFO("");
    LOG_INFO("Per-position cosine (Manual vs FP32 Pipeline):");
    for (int pos = 0; pos < seq_len; ++pos)
    {
        const float *manual_row = manual_output.data() + pos * data.n_heads * head_dim;
        const float *ref_row = data.fp32_attention_output.data() + pos * data.n_heads * head_dim;
        double pos_cos = cosine_similarity(manual_row, ref_row, data.n_heads * head_dim);
        LOG_INFO("  Position " << pos << ": " << std::fixed << std::setprecision(6) << pos_cos);
    }

    LOG_INFO("");
    LOG_INFO("=== SUMMARY ===");
    LOG_INFO("  JIT vs Manual (same dequantized input): " << std::fixed << std::setprecision(6) << overall_cos);
    LOG_INFO("  Manual vs FP32 Pipeline (quantization error): " << std::fixed << std::setprecision(6) << manual_vs_ref);
}

/**
 * @brief Compare JIT kernel vs C++ reference implementation
 *
 * Both operate on the exact same Q8_1 blocks - this isolates any JIT bugs
 * from quantization differences.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, JitVsCppReference)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  JIT KERNEL vs C++ REFERENCE IMPLEMENTATION                    ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int head_dim = data.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create causal mask
    auto mask = createCausalMask(seq_len);

    // === RUN JIT KERNEL ===
    LOG_INFO("");
    LOG_INFO("Running Q8_1 JIT attention kernel...");

    auto jit_output = runQ8_1JitAttention(
        data.Q_blocks, data.K_blocks, data.V_blocks, mask,
        seq_len, data.n_heads, data.n_kv_heads, head_dim);

    // === RUN C++ REFERENCE ===
    LOG_INFO("");
    LOG_INFO("Running C++ reference attention...");

    // Output buffer
    size_t output_blocks_per_row = (data.n_heads * head_dim + 31) / 32;
    std::vector<Q8_1Block> ref_output_blocks(seq_len * output_blocks_per_row);

    // Strides in blocks (not bytes)
    int q_blocks_per_row = (data.n_heads * head_dim + 31) / 32;
    int kv_blocks_per_row = (data.n_kv_heads * head_dim + 31) / 32;

    // Process each head using C++ reference
    int heads_per_kv = data.n_heads / data.n_kv_heads;

    for (int h = 0; h < data.n_heads; ++h)
    {
        int kv_head = h / heads_per_kv;

        int q_block_offset = h * (head_dim / 32);
        int kv_block_offset = kv_head * (head_dim / 32);
        int out_block_offset = h * (head_dim / 32);

        const Q8_1Block *q_ptr = data.Q_blocks.data() + q_block_offset;
        const Q8_1Block *k_ptr = data.K_blocks.data() + kv_block_offset;
        const Q8_1Block *v_ptr = data.V_blocks.data() + kv_block_offset;
        Q8_1Block *out_ptr = ref_output_blocks.data() + out_block_offset;

        gemm_v4::fused_q8_1_attention_reference(
            q_ptr, k_ptr, v_ptr, out_ptr,
            seq_len, seq_len, head_dim,
            q_blocks_per_row, kv_blocks_per_row, kv_blocks_per_row, q_blocks_per_row,
            scale, mask.data(), seq_len);
    }

    // Dequantize reference output
    std::vector<float> ref_output_fp32(seq_len * data.n_heads * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_output_blocks.data(), ref_output_fp32.data(), ref_output_fp32.size());

    // === COMPARE JIT vs C++ REFERENCE ===
    LOG_INFO("");
    LOG_INFO("=== JIT vs C++ REFERENCE COMPARISON ===");

    double overall_cos = cosine_similarity(ref_output_fp32.data(), jit_output.data(), jit_output.size());
    LOG_INFO("  Overall cosine similarity: " << std::fixed << std::setprecision(6) << overall_cos);

    LOG_INFO("");
    LOG_INFO("Per-position cosine similarity:");
    for (int pos = 0; pos < seq_len; ++pos)
    {
        const float *ref_row = ref_output_fp32.data() + pos * data.n_heads * head_dim;
        const float *jit_row = jit_output.data() + pos * data.n_heads * head_dim;
        double pos_cos = cosine_similarity(ref_row, jit_row, data.n_heads * head_dim);
        LOG_INFO("  Position " << pos << ": " << std::fixed << std::setprecision(6) << pos_cos);
    }

    // Per-head analysis for position 1
    LOG_INFO("");
    LOG_INFO("Per-head analysis for position 1:");
    for (int h = 0; h < data.n_heads; ++h)
    {
        int pos = 1;
        const float *ref_head = ref_output_fp32.data() + pos * data.n_heads * head_dim + h * head_dim;
        const float *jit_head = jit_output.data() + pos * data.n_heads * head_dim + h * head_dim;
        double head_cos = cosine_similarity(ref_head, jit_head, head_dim);
        LOG_INFO("  Head " << h << ": cosine=" << std::fixed << std::setprecision(6) << head_cos);
    }

    LOG_INFO("");
    if (overall_cos < 0.999)
    {
        LOG_ERROR("✗ JIT kernel diverges from C++ reference!");
        LOG_ERROR("  This indicates a bug in the JIT kernel implementation.");

        // Print detailed comparison for debugging
        LOG_INFO("");
        LOG_INFO("Sample values for head 0, position 1:");
        int pos = 1, h = 0;
        const float *ref_vec = ref_output_fp32.data() + pos * data.n_heads * head_dim + h * head_dim;
        const float *jit_vec = jit_output.data() + pos * data.n_heads * head_dim + h * head_dim;
        for (int i = 0; i < std::min(10, head_dim); ++i)
        {
            LOG_INFO("  [" << i << "] C++_ref=" << std::setprecision(6) << ref_vec[i]
                           << " JIT=" << jit_vec[i]
                           << " diff=" << std::abs(ref_vec[i] - jit_vec[i]));
        }
    }
    else
    {
        LOG_INFO("✓ JIT kernel matches C++ reference implementation.");
    }

    EXPECT_GE(overall_cos, 0.999) << "JIT kernel should match C++ reference";
}

/**
 * @brief Diagnostic test: Compare Q*K^T scores between Q8_1 and FP32 computation
 *
 * This test isolates the score computation to see if the issue is in:
 * 1. The dot product itself (Q8_1 vs FP32 matmul)
 * 2. The softmax computation
 * 3. The V accumulation
 */
TEST_F(Test__Q8_1_AttentionJitReplay, DiagnoseScoreDivergence)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  DIAGNOSING SCORE DIVERGENCE: Q8_1 vs FP32                     ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int head_dim = data.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int num_blocks = head_dim / 32;

    // Dequantize Q/K to FP32
    size_t q_elements = seq_len * data.n_heads * head_dim;
    size_t kv_elements = seq_len * data.n_kv_heads * head_dim;

    std::vector<float> Q_fp32(q_elements);
    std::vector<float> K_fp32(kv_elements);

    simd::dequantize_q8_1_to_fp32(data.Q_blocks.data(), Q_fp32.data(), q_elements);
    simd::dequantize_q8_1_to_fp32(data.K_blocks.data(), K_fp32.data(), kv_elements);

    LOG_INFO("");
    LOG_INFO("Comparing Q*K^T scores for head 0...");

    // For head 0, compute scores both ways
    int h = 0;
    int kv_head = 0;

    // Q8_1 scores (using C++ reference dot product)
    std::vector<float> q8_scores(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        const Q8_1Block *Q_row = data.Q_blocks.data() + m * (data.n_heads * head_dim / 32) + h * (head_dim / 32);
        for (int n = 0; n < seq_len; ++n)
        {
            const Q8_1Block *K_row = data.K_blocks.data() + n * (data.n_kv_heads * head_dim / 32) + kv_head * (head_dim / 32);

            float dot = 0.0f;
            for (int b = 0; b < num_blocks; ++b)
            {
                float d_q = simd::fp16_to_fp32(Q_row[b].d);
                float d_k = simd::fp16_to_fp32(K_row[b].d);
                int32_t acc = 0;
                for (int i = 0; i < 32; ++i)
                {
                    acc += static_cast<int32_t>(Q_row[b].qs[i]) * static_cast<int32_t>(K_row[b].qs[i]);
                }
                dot += static_cast<float>(acc) * d_q * d_k;
            }
            q8_scores[m * seq_len + n] = dot * scale;
        }
    }

    // FP32 scores (using dequantized data)
    std::vector<float> fp32_scores(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        const float *q_row = Q_fp32.data() + m * data.n_heads * head_dim + h * head_dim;
        for (int n = 0; n < seq_len; ++n)
        {
            const float *k_row = K_fp32.data() + n * data.n_kv_heads * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += q_row[d] * k_row[d];
            }
            fp32_scores[m * seq_len + n] = dot * scale;
        }
    }

    // Compare scores
    LOG_INFO("");
    LOG_INFO("Score comparison (Q8_1 vs FP32) for head 0:");
    LOG_INFO("  Format: [m, n]: Q8_1 vs FP32 (diff)");

    double total_diff = 0.0;
    double max_diff = 0.0;
    for (int m = 0; m < std::min(3, seq_len); ++m)
    {
        for (int n = 0; n <= m && n < std::min(3, seq_len); ++n)
        {
            float q8 = q8_scores[m * seq_len + n];
            float fp = fp32_scores[m * seq_len + n];
            float diff = std::abs(q8 - fp);
            total_diff += diff;
            max_diff = std::max(max_diff, (double)diff);
            LOG_INFO("  [" << m << ", " << n << "]: " << std::fixed << std::setprecision(4)
                           << q8 << " vs " << fp << " (diff=" << diff << ")");
        }
    }

    LOG_INFO("");
    LOG_INFO("Score statistics:");
    LOG_INFO("  Max diff: " << max_diff);
    LOG_INFO("  Mean diff: " << total_diff / (seq_len * seq_len));

    // The key question: are these differences enough to cause 0.90 cosine after softmax?
    // Let's compute softmax for position 1 and compare
    LOG_INFO("");
    LOG_INFO("Softmax weights for position 1 (attends to positions 0,1):");

    int m = 1;

    // Q8_1 softmax
    float q8_max = std::max(q8_scores[m * seq_len + 0], q8_scores[m * seq_len + 1]);
    float q8_exp0 = std::exp(q8_scores[m * seq_len + 0] - q8_max);
    float q8_exp1 = std::exp(q8_scores[m * seq_len + 1] - q8_max);
    float q8_sum = q8_exp0 + q8_exp1;
    float q8_w0 = q8_exp0 / q8_sum;
    float q8_w1 = q8_exp1 / q8_sum;

    // FP32 softmax
    float fp_max = std::max(fp32_scores[m * seq_len + 0], fp32_scores[m * seq_len + 1]);
    float fp_exp0 = std::exp(fp32_scores[m * seq_len + 0] - fp_max);
    float fp_exp1 = std::exp(fp32_scores[m * seq_len + 1] - fp_max);
    float fp_sum = fp_exp0 + fp_exp1;
    float fp_w0 = fp_exp0 / fp_sum;
    float fp_w1 = fp_exp1 / fp_sum;

    LOG_INFO("  Q8_1: w[0]=" << std::fixed << std::setprecision(6) << q8_w0
                             << ", w[1]=" << q8_w1);
    LOG_INFO("  FP32: w[0]=" << std::fixed << std::setprecision(6) << fp_w0
                             << ", w[1]=" << fp_w1);
    LOG_INFO("  Weight diff: w[0]=" << std::abs(q8_w0 - fp_w0)
                                    << ", w[1]=" << std::abs(q8_w1 - fp_w1));

    LOG_INFO("");
    LOG_INFO("Raw scores for position 1:");
    LOG_INFO("  Q8_1: score[0]=" << q8_scores[m * seq_len + 0]
                                 << ", score[1]=" << q8_scores[m * seq_len + 1]);
    LOG_INFO("  FP32: score[0]=" << fp32_scores[m * seq_len + 0]
                                 << ", score[1]=" << fp32_scores[m * seq_len + 1]);
    LOG_INFO("  Score diff: s[0]=" << std::abs(q8_scores[m * seq_len + 0] - fp32_scores[m * seq_len + 0])
                                   << ", s[1]=" << std::abs(q8_scores[m * seq_len + 1] - fp32_scores[m * seq_len + 1]));
}

/**
 * @brief Test to verify the ACTUAL Q/K/V values being compared
 *
 * This loads the original FP32 Q/K/V from the pipeline, and checks what
 * values we're actually comparing after quantization roundtrip.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, VerifyQKVRoundtrip)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  VERIFYING Q/K/V QUANTIZATION ROUNDTRIP                        ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    // Run FP32 pipeline to get original Q/K/V
    LOG_INFO("");
    LOG_INFO("Running FP32 pipeline to capture original Q/K/V...");

    PipelineConfig fp32_config;
    fp32_config.activation_precision = ActivationPrecision::FP32;
    fp32_config.max_seq_len = 512;

    auto fp32_pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, fp32_config, 1);
    fp32_pipeline->enableSnapshotCapture();

    bool success = fp32_pipeline->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success);

    // Get FP32 Q/K/V snapshots
    size_t q_size = 0, k_size = 0, v_size = 0;
    const float *Q_orig = fp32_pipeline->getSnapshot("layer0_Q_ROPE", q_size);
    const float *K_orig = fp32_pipeline->getSnapshot("layer0_K_ROPE", k_size);
    const float *V_orig = fp32_pipeline->getSnapshot("layer0_V_PROJECTION", v_size);

    LOG_INFO("  Original Q: " << q_size << " elements");
    LOG_INFO("  Original K: " << k_size << " elements");
    LOG_INFO("  Original V: " << v_size << " elements");

    // Quantize to Q8_1
    size_t q_blocks = (q_size + 31) / 32;
    size_t k_blocks = (k_size + 31) / 32;
    size_t v_blocks = (v_size + 31) / 32;

    std::vector<Q8_1Block> Q_q8(q_blocks);
    std::vector<Q8_1Block> K_q8(k_blocks);
    std::vector<Q8_1Block> V_q8(v_blocks);

    simd::quantize_fp32_to_q8_1_blocks(Q_orig, Q_q8.data(), q_size);
    simd::quantize_fp32_to_q8_1_blocks(K_orig, K_q8.data(), k_size);
    simd::quantize_fp32_to_q8_1_blocks(V_orig, V_q8.data(), v_size);

    // Dequantize back to FP32
    std::vector<float> Q_roundtrip(q_size);
    std::vector<float> K_roundtrip(k_size);
    std::vector<float> V_roundtrip(v_size);

    simd::dequantize_q8_1_to_fp32(Q_q8.data(), Q_roundtrip.data(), q_size);
    simd::dequantize_q8_1_to_fp32(K_q8.data(), K_roundtrip.data(), k_size);
    simd::dequantize_q8_1_to_fp32(V_q8.data(), V_roundtrip.data(), v_size);

    // Compare original vs roundtrip
    double q_cos = cosine_similarity(Q_orig, Q_roundtrip.data(), q_size);
    double k_cos = cosine_similarity(K_orig, K_roundtrip.data(), k_size);
    double v_cos = cosine_similarity(V_orig, V_roundtrip.data(), v_size);

    LOG_INFO("");
    LOG_INFO("Q/K/V quantization roundtrip fidelity (cosine):");
    LOG_INFO("  Q: " << std::fixed << std::setprecision(6) << q_cos);
    LOG_INFO("  K: " << std::fixed << std::setprecision(6) << k_cos);
    LOG_INFO("  V: " << std::fixed << std::setprecision(6) << v_cos);

    // Now compute attention on ORIGINAL FP32 Q/K/V
    LOG_INFO("");
    LOG_INFO("Computing attention on ORIGINAL FP32 Q/K/V...");

    float scale = 1.0f / std::sqrt(64.0f);
    int head_dim = 64;
    int n_heads = 14;
    int n_kv_heads = 2;
    int heads_per_kv = n_heads / n_kv_heads;

    std::vector<float> attn_orig(seq_len * n_heads * head_dim, 0.0f);

    for (int h = 0; h < n_heads; ++h)
    {
        int kv_head = h / heads_per_kv;
        for (int m = 0; m < seq_len; ++m)
        {
            // Compute scores
            std::vector<float> scores(seq_len);
            for (int n = 0; n < seq_len; ++n)
            {
                const float *q_vec = Q_orig + m * n_heads * head_dim + h * head_dim;
                const float *k_vec = K_orig + n * n_kv_heads * head_dim + kv_head * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += q_vec[d] * k_vec[d];
                scores[n] = dot * scale;
            }

            // Causal mask
            for (int n = m + 1; n < seq_len; ++n)
                scores[n] = -std::numeric_limits<float>::infinity();

            // Softmax
            float max_val = *std::max_element(scores.begin(), scores.begin() + m + 1);
            float sum = 0.0f;
            for (int n = 0; n <= m; ++n)
            {
                scores[n] = std::exp(scores[n] - max_val);
                sum += scores[n];
            }
            for (int n = 0; n <= m; ++n)
                scores[n] /= sum;

            // Weighted V
            float *out = attn_orig.data() + m * n_heads * head_dim + h * head_dim;
            for (int n = 0; n <= m; ++n)
            {
                const float *v_vec = V_orig + n * n_kv_heads * head_dim + kv_head * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    out[d] += scores[n] * v_vec[d];
            }
        }
    }

    // Compute attention on ROUNDTRIP Q/K/V
    LOG_INFO("Computing attention on ROUNDTRIP Q/K/V...");

    std::vector<float> attn_roundtrip(seq_len * n_heads * head_dim, 0.0f);

    for (int h = 0; h < n_heads; ++h)
    {
        int kv_head = h / heads_per_kv;
        for (int m = 0; m < seq_len; ++m)
        {
            std::vector<float> scores(seq_len);
            for (int n = 0; n < seq_len; ++n)
            {
                const float *q_vec = Q_roundtrip.data() + m * n_heads * head_dim + h * head_dim;
                const float *k_vec = K_roundtrip.data() + n * n_kv_heads * head_dim + kv_head * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += q_vec[d] * k_vec[d];
                scores[n] = dot * scale;
            }

            for (int n = m + 1; n < seq_len; ++n)
                scores[n] = -std::numeric_limits<float>::infinity();

            float max_val = *std::max_element(scores.begin(), scores.begin() + m + 1);
            float sum = 0.0f;
            for (int n = 0; n <= m; ++n)
            {
                scores[n] = std::exp(scores[n] - max_val);
                sum += scores[n];
            }
            for (int n = 0; n <= m; ++n)
                scores[n] /= sum;

            float *out = attn_roundtrip.data() + m * n_heads * head_dim + h * head_dim;
            for (int n = 0; n <= m; ++n)
            {
                const float *v_vec = V_roundtrip.data() + n * n_kv_heads * head_dim + kv_head * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    out[d] += scores[n] * v_vec[d];
            }
        }
    }

    // Compare attention outputs
    double attn_cos = cosine_similarity(attn_orig.data(), attn_roundtrip.data(), attn_orig.size());

    LOG_INFO("");
    LOG_INFO("=== ATTENTION OUTPUT COMPARISON ===");
    LOG_INFO("  Attention (original vs roundtrip): " << std::fixed << std::setprecision(6) << attn_cos);

    LOG_INFO("");
    LOG_INFO("Per-position cosine:");
    for (int pos = 0; pos < seq_len; ++pos)
    {
        const float *orig_row = attn_orig.data() + pos * n_heads * head_dim;
        const float *rt_row = attn_roundtrip.data() + pos * n_heads * head_dim;
        double pos_cos = cosine_similarity(orig_row, rt_row, n_heads * head_dim);
        LOG_INFO("  Position " << pos << ": " << std::fixed << std::setprecision(6) << pos_cos);
    }

    LOG_INFO("");
    LOG_INFO("=== SUMMARY ===");
    LOG_INFO("  Q roundtrip cosine: " << std::fixed << std::setprecision(6) << q_cos);
    LOG_INFO("  K roundtrip cosine: " << std::fixed << std::setprecision(6) << k_cos);
    LOG_INFO("  V roundtrip cosine: " << std::fixed << std::setprecision(6) << v_cos);
    LOG_INFO("  Attention output cosine: " << std::fixed << std::setprecision(6) << attn_cos);

    if (q_cos > 0.999 && k_cos > 0.999 && v_cos > 0.999 && attn_cos < 0.95)
    {
        LOG_ERROR("");
        LOG_ERROR("!!! Q/K/V roundtrip is 0.999+ but attention output is only " << attn_cos << " !!!");
        LOG_ERROR("This indicates quantization error is being AMPLIFIED through attention.");
        LOG_ERROR("This is NOT expected behavior - the error amplification is too high.");
    }

    // Detailed V analysis
    LOG_INFO("");
    LOG_INFO("=== DETAILED V ANALYSIS ===");

    // Sample some V values
    LOG_INFO("Sample V values (position 0, KV head 0, first 10 dims):");
    for (int d = 0; d < std::min(10, head_dim); ++d)
    {
        float orig_v = V_orig[0 * n_kv_heads * head_dim + 0 * head_dim + d];
        float rt_v = V_roundtrip[0 * n_kv_heads * head_dim + 0 * head_dim + d];
        LOG_INFO("  d=" << d << ": orig=" << std::fixed << std::setprecision(6) << orig_v
                        << ", roundtrip=" << rt_v << ", diff=" << std::abs(orig_v - rt_v));
    }

    // Check V block scales
    LOG_INFO("");
    LOG_INFO("V block scales (first few blocks):");
    for (size_t b = 0; b < std::min(size_t(5), v_blocks); ++b)
    {
        float d_v = simd::fp16_to_fp32(V_q8[b].d);
        LOG_INFO("  Block " << b << ": d=" << std::scientific << std::setprecision(4) << d_v);
    }

    // V value statistics
    float v_min = *std::min_element(V_orig, V_orig + v_size);
    float v_max = *std::max_element(V_orig, V_orig + v_size);
    float v_mean = 0.0f;
    for (size_t i = 0; i < v_size; ++i)
        v_mean += V_orig[i];
    v_mean /= v_size;

    LOG_INFO("");
    LOG_INFO("V statistics:");
    LOG_INFO("  Min: " << v_min);
    LOG_INFO("  Max: " << v_max);
    LOG_INFO("  Mean: " << v_mean);
    LOG_INFO("  Range: " << (v_max - v_min));

    // Check if V has large values that might cause quantization issues
    float v_abs_max = std::max(std::abs(v_min), std::abs(v_max));
    LOG_INFO("  Abs max: " << v_abs_max);
    LOG_INFO("  Expected Q8_1 scale: " << v_abs_max / 127.0f);

    // Deep dive: Compare scores between original and roundtrip for position 7 (worst case)
    LOG_INFO("");
    LOG_INFO("=== DEEP DIVE: POSITION 7 (worst cosine 0.795) ===");
    int worst_pos = 7;
    int h = 0; // head 0
    int kv_head = 0;

    // First check Q and K magnitudes
    const float *q_check = Q_orig + worst_pos * n_heads * head_dim + h * head_dim;
    const float *k_check = K_orig + 0 * n_kv_heads * head_dim + kv_head * head_dim;

    float q_mag = 0.0f, k_mag = 0.0f;
    for (int d = 0; d < head_dim; ++d)
    {
        q_mag += q_check[d] * q_check[d];
        k_mag += k_check[d] * k_check[d];
    }
    LOG_INFO("Q magnitude (pos 7, head 0): " << std::sqrt(q_mag));
    LOG_INFO("K magnitude (pos 0, kv_head 0): " << std::sqrt(k_mag));
    LOG_INFO("Q[0..4]: " << q_check[0] << ", " << q_check[1] << ", " << q_check[2] << ", " << q_check[3] << ", " << q_check[4]);
    LOG_INFO("K[0..4]: " << k_check[0] << ", " << k_check[1] << ", " << k_check[2] << ", " << k_check[3] << ", " << k_check[4]);
    LOG_INFO("Scale: " << scale);

    // Compute scores using original Q/K
    LOG_INFO("Scores (Q*K^T) for position " << worst_pos << ", head 0:");
    std::vector<float> scores_orig(worst_pos + 1);
    std::vector<float> scores_rt(worst_pos + 1);

    for (int n = 0; n <= worst_pos; ++n)
    {
        const float *q_orig = Q_orig + worst_pos * n_heads * head_dim + h * head_dim;
        const float *k_orig = K_orig + n * n_kv_heads * head_dim + kv_head * head_dim;
        float dot_orig = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot_orig += q_orig[d] * k_orig[d];
        scores_orig[n] = dot_orig * scale;

        const float *q_rt = Q_roundtrip.data() + worst_pos * n_heads * head_dim + h * head_dim;
        const float *k_rt = K_roundtrip.data() + n * n_kv_heads * head_dim + kv_head * head_dim;
        float dot_rt = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot_rt += q_rt[d] * k_rt[d];
        scores_rt[n] = dot_rt * scale;

        LOG_INFO("  n=" << n << ": orig=" << std::fixed << std::setprecision(4) << scores_orig[n]
                        << ", roundtrip=" << scores_rt[n]
                        << ", diff=" << std::abs(scores_orig[n] - scores_rt[n]));
    }

    // Compute softmax weights
    LOG_INFO("");
    LOG_INFO("Softmax weights:");
    float max_orig = *std::max_element(scores_orig.begin(), scores_orig.end());
    float max_rt = *std::max_element(scores_rt.begin(), scores_rt.end());

    std::vector<float> weights_orig(worst_pos + 1);
    std::vector<float> weights_rt(worst_pos + 1);
    float sum_orig = 0.0f, sum_rt = 0.0f;

    for (int n = 0; n <= worst_pos; ++n)
    {
        weights_orig[n] = std::exp(scores_orig[n] - max_orig);
        weights_rt[n] = std::exp(scores_rt[n] - max_rt);
        sum_orig += weights_orig[n];
        sum_rt += weights_rt[n];
    }

    for (int n = 0; n <= worst_pos; ++n)
    {
        weights_orig[n] /= sum_orig;
        weights_rt[n] /= sum_rt;
        LOG_INFO("  n=" << n << ": orig=" << std::fixed << std::setprecision(6) << weights_orig[n]
                        << ", roundtrip=" << weights_rt[n]
                        << ", diff=" << std::abs(weights_orig[n] - weights_rt[n]));
    }

    // Compute weighted V sum
    LOG_INFO("");
    LOG_INFO("Weighted V output (first 5 dims):");
    std::vector<float> out_orig(head_dim, 0.0f);
    std::vector<float> out_rt(head_dim, 0.0f);

    for (int n = 0; n <= worst_pos; ++n)
    {
        const float *v_orig = V_orig + n * n_kv_heads * head_dim + kv_head * head_dim;
        const float *v_rt = V_roundtrip.data() + n * n_kv_heads * head_dim + kv_head * head_dim;
        for (int d = 0; d < head_dim; ++d)
        {
            out_orig[d] += weights_orig[n] * v_orig[d];
            out_rt[d] += weights_rt[n] * v_rt[d];
        }
    }

    for (int d = 0; d < std::min(5, head_dim); ++d)
    {
        LOG_INFO("  d=" << d << ": orig=" << std::fixed << std::setprecision(6) << out_orig[d]
                        << ", roundtrip=" << out_rt[d]
                        << ", diff=" << std::abs(out_orig[d] - out_rt[d]));
    }

    double pos7_cos = cosine_similarity(out_orig.data(), out_rt.data(), head_dim);
    LOG_INFO("");
    LOG_INFO("Position 7 head 0 output cosine: " << pos7_cos);
}
#endif // ENABLE_PIPELINE_SNAPSHOTS

/**
 * @brief Diagnostic test: Compare |Q|² and |K|² computation between JIT and C++ reference
 *
 * This test isolates the norm computation to identify where JIT diverges from reference.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, DiagnoseNormComputation)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  DIAGNOSING NORM COMPUTATION: JIT vs Reference                 ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;

    LOG_INFO("");
    LOG_INFO("Computing |Q|² and |K|² using different methods...");
    LOG_INFO("  head_dim=" << head_dim << ", num_blocks=" << num_blocks);

    // Get Q/K block pointers for head 0, position 0
    int h = 0;
    int m = 0; // Query position
    int n = 0; // Key position
    int kv_head = 0;

    const Q8_1Block *Q_row = data.Q_blocks.data() + m * (data.n_heads * num_blocks) + h * num_blocks;
    const Q8_1Block *K_row = data.K_blocks.data() + n * (data.n_kv_heads * num_blocks) + kv_head * num_blocks;

    // Method 1: Simple reference (signed×signed)
    float Q_norm_sq_ref = 0.0f;
    float K_norm_sq_ref = 0.0f;

    for (int b = 0; b < num_blocks; ++b)
    {
        float d_q = simd::fp16_to_fp32(Q_row[b].d);
        float d_k = simd::fp16_to_fp32(K_row[b].d);
        int32_t q_sq = 0;
        int32_t k_sq = 0;
        for (int i = 0; i < 32; ++i)
        {
            q_sq += static_cast<int32_t>(Q_row[b].qs[i]) * static_cast<int32_t>(Q_row[b].qs[i]);
            k_sq += static_cast<int32_t>(K_row[b].qs[i]) * static_cast<int32_t>(K_row[b].qs[i]);
        }
        Q_norm_sq_ref += static_cast<float>(q_sq) * d_q * d_q;
        K_norm_sq_ref += static_cast<float>(k_sq) * d_k * d_k;

        LOG_INFO("  Block " << b << ": d_q=" << std::fixed << std::setprecision(6) << d_q
                            << ", d_k=" << d_k << ", q_sq=" << q_sq << ", k_sq=" << k_sq
                            << ", sum_qs_q=" << Q_row[b].sum_qs << ", sum_qs_k=" << K_row[b].sum_qs);
    }

    LOG_INFO("");
    LOG_INFO("Reference norms: |Q|²=" << Q_norm_sq_ref << ", |K|²=" << K_norm_sq_ref);
    LOG_INFO("Reference |Q|=" << std::sqrt(Q_norm_sq_ref) << ", |K|=" << std::sqrt(K_norm_sq_ref));

    // Method 2: Using vpdpbusd formula (Q+128)*Q - 128*sum_qs
    float Q_norm_sq_vpdpbusd = 0.0f;
    float K_norm_sq_vpdpbusd = 0.0f;

    for (int b = 0; b < num_blocks; ++b)
    {
        float d_q = simd::fp16_to_fp32(Q_row[b].d);
        float d_k = simd::fp16_to_fp32(K_row[b].d);

        // Simulate vpdpbusd: (Q+128) * Q = Q² + 128*Q
        int32_t q_raw = 0;
        int32_t k_raw = 0;
        for (int i = 0; i < 32; ++i)
        {
            int8_t q = Q_row[b].qs[i];
            int8_t k = K_row[b].qs[i];
            uint8_t q_unsigned = static_cast<uint8_t>(static_cast<int16_t>(q) + 128);
            uint8_t k_unsigned = static_cast<uint8_t>(static_cast<int16_t>(k) + 128);
            q_raw += static_cast<int32_t>(q_unsigned) * static_cast<int32_t>(q); // unsigned × signed
            k_raw += static_cast<int32_t>(k_unsigned) * static_cast<int32_t>(k);
        }

        // Apply correction: Q² = raw - 128*sum_qs
        float q_correction = 128.0f * static_cast<float>(Q_row[b].sum_qs);
        float k_correction = 128.0f * static_cast<float>(K_row[b].sum_qs);
        float q_sq = static_cast<float>(q_raw) - q_correction;
        float k_sq = static_cast<float>(k_raw) - k_correction;

        Q_norm_sq_vpdpbusd += q_sq * d_q * d_q;
        K_norm_sq_vpdpbusd += k_sq * d_k * d_k;

        LOG_INFO("  Block " << b << " (vpdpbusd): q_raw=" << q_raw << ", k_raw=" << k_raw
                            << ", q_corr=" << q_correction << ", k_corr=" << k_correction
                            << ", q_sq=" << q_sq << ", k_sq=" << k_sq);
    }

    LOG_INFO("");
    LOG_INFO("vpdpbusd norms: |Q|²=" << Q_norm_sq_vpdpbusd << ", |K|²=" << K_norm_sq_vpdpbusd);

    // Compare
    float Q_diff = std::abs(Q_norm_sq_ref - Q_norm_sq_vpdpbusd);
    float K_diff = std::abs(K_norm_sq_ref - K_norm_sq_vpdpbusd);
    LOG_INFO("");
    LOG_INFO("Difference: |Q|² diff=" << Q_diff << ", |K|² diff=" << K_diff);

    EXPECT_LT(Q_diff, 1e-3f) << "Q norm computation should match";
    EXPECT_LT(K_diff, 1e-3f) << "K norm computation should match";

    // Now check what the normalized score would be
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Compute Q·K using reference
    float dot_ref = 0.0f;
    for (int b = 0; b < num_blocks; ++b)
    {
        float d_q = simd::fp16_to_fp32(Q_row[b].d);
        float d_k = simd::fp16_to_fp32(K_row[b].d);
        int32_t acc = 0;
        for (int i = 0; i < 32; ++i)
        {
            acc += static_cast<int32_t>(Q_row[b].qs[i]) * static_cast<int32_t>(K_row[b].qs[i]);
        }
        dot_ref += static_cast<float>(acc) * d_q * d_k;
    }

    float score_unnorm = dot_ref * scale;
    float norm_product = std::sqrt(Q_norm_sq_ref * K_norm_sq_ref + 1e-8f);
    float score_norm = score_unnorm / norm_product;

    LOG_INFO("");
    LOG_INFO("Position [0,0] score:");
    LOG_INFO("  Unnormalized: " << score_unnorm);
    LOG_INFO("  Norm product: " << norm_product);
    LOG_INFO("  Normalized:   " << score_norm);
}

/**
 * @brief Minimal test: Run JIT and C++ reference on a single head with 2 positions
 *
 * This creates the simplest possible test case to isolate the normalization bug.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, MinimalNormalizationTest)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  MINIMAL NORMALIZATION TEST: Single Head, 2 Positions          ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Create a minimal test case: 2 positions, 1 head, head_dim=64
    const int M = 2; // seq_len
    const int N = 2; // kv_len (same for self-attention)
    const int head_dim = 64;
    const int num_blocks = head_dim / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create simple Q8_1 blocks with known values
    std::vector<Q8_1Block> Q_blocks(M * num_blocks);
    std::vector<Q8_1Block> K_blocks(N * num_blocks);
    std::vector<Q8_1Block> V_blocks(N * num_blocks);

    // Initialize with simple values: ascending integers, scale = 1.0
    auto fill_blocks = [](std::vector<Q8_1Block> &blocks, int num_rows, int num_blocks, int8_t base_val, float d)
    {
        for (int m = 0; m < num_rows; ++m)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                Q8_1Block &blk = blocks[m * num_blocks + b];
                blk.d = simd::fp32_to_fp16(d);
                int16_t sum = 0;
                for (int i = 0; i < 32; ++i)
                {
                    // Use values that create predictable norms
                    int8_t val = static_cast<int8_t>(((m * 32 + b * 32 + i) % 127) - 64 + base_val);
                    blk.qs[i] = val;
                    sum += val;
                }
                blk.sum_qs = sum;
            }
        }
    };

    fill_blocks(Q_blocks, M, num_blocks, 10, 0.3f);
    fill_blocks(K_blocks, N, num_blocks, 20, 1.0f); // K with larger scale (like real data)
    fill_blocks(V_blocks, N, num_blocks, 5, 0.5f);

    // Compute expected norms manually
    LOG_INFO("");
    LOG_INFO("Input block statistics:");
    for (int m = 0; m < M; ++m)
    {
        float Q_norm_sq = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            const Q8_1Block &blk = Q_blocks[m * num_blocks + b];
            float d = simd::fp16_to_fp32(blk.d);
            int32_t sq = 0;
            for (int i = 0; i < 32; ++i)
            {
                sq += static_cast<int32_t>(blk.qs[i]) * static_cast<int32_t>(blk.qs[i]);
            }
            Q_norm_sq += static_cast<float>(sq) * d * d;
        }
        LOG_INFO("  Q[" << m << "] |Q|²=" << Q_norm_sq << ", |Q|=" << std::sqrt(Q_norm_sq));
    }

    for (int n = 0; n < N; ++n)
    {
        float K_norm_sq = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            const Q8_1Block &blk = K_blocks[n * num_blocks + b];
            float d = simd::fp16_to_fp32(blk.d);
            int32_t sq = 0;
            for (int i = 0; i < 32; ++i)
            {
                sq += static_cast<int32_t>(blk.qs[i]) * static_cast<int32_t>(blk.qs[i]);
            }
            K_norm_sq += static_cast<float>(sq) * d * d;
        }
        LOG_INFO("  K[" << n << "] |K|²=" << K_norm_sq << ", |K|=" << std::sqrt(K_norm_sq));
    }

    // Create causal mask
    std::vector<float> mask(M * N);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            mask[m * N + n] = (n <= m) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    // === RUN C++ REFERENCE ===
    LOG_INFO("");
    LOG_INFO("Running C++ reference with normalization...");

    std::vector<Q8_1Block> ref_output(M * num_blocks);
    gemm_v4::fused_q8_1_attention_reference(
        Q_blocks.data(), K_blocks.data(), V_blocks.data(), ref_output.data(),
        M, N, head_dim,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, mask.data(), N);

    // Dequantize reference output
    std::vector<float> ref_fp32(M * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_output.data(), ref_fp32.data(), ref_fp32.size());

    // === RUN JIT KERNEL ===
    LOG_INFO("Running JIT kernel...");

    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    std::vector<Q8_1Block> jit_output(M * num_blocks);

    // Pack parameters
    gemm_v4::FusedQ8_1AttentionParams params;
    params.Q = Q_blocks.data();
    params.K = K_blocks.data();
    params.V = V_blocks.data();
    params.output = jit_output.data();
    params.M = M;
    params.N = N;
    params.head_dim = head_dim;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = mask.data();
    params.mask_stride = N;

    kernel_fn(&params);

    // Dequantize JIT output
    std::vector<float> jit_fp32(M * head_dim);
    simd::dequantize_q8_1_to_fp32(jit_output.data(), jit_fp32.data(), jit_fp32.size());

    // === COMPARE ===
    LOG_INFO("");
    LOG_INFO("Comparison:");

    for (int m = 0; m < M; ++m)
    {
        const float *ref_row = ref_fp32.data() + m * head_dim;
        const float *jit_row = jit_fp32.data() + m * head_dim;
        double cos = cosine_similarity(ref_row, jit_row, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << cos);

        // Print first few values
        LOG_INFO("    First 5 values:");
        for (int i = 0; i < 5; ++i)
        {
            LOG_INFO("      [" << i << "] ref=" << std::setprecision(4) << ref_row[i]
                               << ", jit=" << jit_row[i]
                               << ", diff=" << std::abs(ref_row[i] - jit_row[i]));
        }
    }

    // Overall cosine
    double overall = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("");
    LOG_INFO("Overall cosine: " << std::fixed << std::setprecision(6) << overall);

    EXPECT_GE(overall, 0.999) << "JIT should match C++ reference for this minimal test case";
}

/**
 * @brief Debug single head with normalization: Run JIT and reference on just head 0 from real data
 */
/**
 * @brief Debug test: Extract ALL positions from real data with COMPACT strides
 *
 * If this passes but DebugSingleHeadNormalized fails, the issue is stride handling.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, AllPositionsCompactStride)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  ALL POSITIONS COMPACT STRIDE TEST                             ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Extract ALL positions for head 0 into COMPACT arrays
    int h = 0;
    int kv_head = 0;
    int q_blocks_per_row = data.n_heads * num_blocks;
    int kv_blocks_per_row = data.n_kv_heads * num_blocks;

    std::vector<Q8_1Block> Q_compact(seq_len * num_blocks);
    std::vector<Q8_1Block> K_compact(seq_len * num_blocks);
    std::vector<Q8_1Block> V_compact(seq_len * num_blocks);

    for (int m = 0; m < seq_len; ++m)
    {
        const Q8_1Block *src_q = data.Q_blocks.data() + m * q_blocks_per_row + h * num_blocks;
        const Q8_1Block *src_k = data.K_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;
        const Q8_1Block *src_v = data.V_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;

        memcpy(&Q_compact[m * num_blocks], src_q, num_blocks * sizeof(Q8_1Block));
        memcpy(&K_compact[m * num_blocks], src_k, num_blocks * sizeof(Q8_1Block));
        memcpy(&V_compact[m * num_blocks], src_v, num_blocks * sizeof(Q8_1Block));
    }

    LOG_INFO("Extracted " << seq_len << " positions for head 0 with compact strides");

    // Create causal mask
    std::vector<float> mask(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        for (int n = 0; n < seq_len; ++n)
        {
            mask[m * seq_len + n] = (n <= m) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    // Run reference
    LOG_INFO("");
    LOG_INFO("Running C++ reference...");
    std::vector<Q8_1Block> ref_out(seq_len * num_blocks);
    gemm_v4::fused_q8_1_attention_reference(
        Q_compact.data(), K_compact.data(), V_compact.data(), ref_out.data(),
        seq_len, seq_len, head_dim,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, mask.data(), seq_len);

    std::vector<float> ref_fp32(seq_len * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_out.data(), ref_fp32.data(), ref_fp32.size());

    // Run JIT
    LOG_INFO("Running JIT...");
    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    std::vector<Q8_1Block> jit_out(seq_len * num_blocks);
    gemm_v4::FusedQ8_1AttentionParams params;
    params.Q = Q_compact.data();
    params.K = K_compact.data();
    params.V = V_compact.data();
    params.output = jit_out.data();
    params.M = seq_len;
    params.N = seq_len;
    params.head_dim = head_dim;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block); // COMPACT!
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block); // COMPACT!
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block); // COMPACT!
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = mask.data();
    params.mask_stride = seq_len;

    kernel_fn(&params);

    std::vector<float> jit_fp32(seq_len * head_dim);
    simd::dequantize_q8_1_to_fp32(jit_out.data(), jit_fp32.data(), jit_fp32.size());

    // Compare
    LOG_INFO("");
    LOG_INFO("Comparison (compact strides):");
    for (int m = 0; m < seq_len; ++m)
    {
        double cos = cosine_similarity(ref_fp32.data() + m * head_dim, jit_fp32.data() + m * head_dim, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << cos);

        if (m == 1)
        {
            LOG_INFO("  Position 1 values (Block 0):");
            for (int i = 0; i < 8; ++i)
            {
                LOG_INFO("    [" << i << "] Ref=" << ref_fp32[m * head_dim + i] << " Jit=" << jit_fp32[m * head_dim + i]);
            }
            LOG_INFO("  Position 1 values (Block 1):");
            for (int i = 32; i < 40; ++i)
            {
                LOG_INFO("    [" << i << "] Ref=" << ref_fp32[m * head_dim + i] << " Jit=" << jit_fp32[m * head_dim + i]);
            }
        }
    }

    double overall = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("");
    LOG_INFO("Overall cosine (compact): " << std::fixed << std::setprecision(6) << overall);

    EXPECT_GE(overall, 0.999) << "Compact stride test should match";
}

/**
 * @brief Debug test: Extract just positions 0,1 from real data to create minimal repro
 *
 * This tests the exact same data as DebugSingleHeadNormalized but with only 2 positions.
 */
TEST_F(Test__Q8_1_AttentionJitReplay, TwoPositionRealData)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  TWO POSITION REAL DATA TEST                                   ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Extract positions 0 and 1 for head 0
    int h = 0;
    int kv_head = 0;
    int q_blocks_per_row = data.n_heads * num_blocks;
    int kv_blocks_per_row = data.n_kv_heads * num_blocks;

    // Copy just positions 0 and 1 to contiguous arrays
    std::vector<Q8_1Block> Q_compact(2 * num_blocks);
    std::vector<Q8_1Block> K_compact(2 * num_blocks);
    std::vector<Q8_1Block> V_compact(2 * num_blocks);

    for (int m = 0; m < 2; ++m)
    {
        const Q8_1Block *src_q = data.Q_blocks.data() + m * q_blocks_per_row + h * num_blocks;
        const Q8_1Block *src_k = data.K_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;
        const Q8_1Block *src_v = data.V_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;

        memcpy(&Q_compact[m * num_blocks], src_q, num_blocks * sizeof(Q8_1Block));
        memcpy(&K_compact[m * num_blocks], src_k, num_blocks * sizeof(Q8_1Block));
        memcpy(&V_compact[m * num_blocks], src_v, num_blocks * sizeof(Q8_1Block));
    }

    LOG_INFO("Extracted 2 positions for head 0");

    // Print norms
    for (int m = 0; m < 2; ++m)
    {
        float Q_norm_sq = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            const Q8_1Block &blk = Q_compact[m * num_blocks + b];
            float d = simd::fp16_to_fp32(blk.d);
            int32_t sq = 0;
            for (int i = 0; i < 32; ++i)
            {
                sq += static_cast<int32_t>(blk.qs[i]) * static_cast<int32_t>(blk.qs[i]);
            }
            Q_norm_sq += sq * d * d;
        }
        LOG_INFO("  Q[" << m << "] |Q|²=" << Q_norm_sq << ", |Q|=" << std::sqrt(Q_norm_sq));
    }

    for (int n = 0; n < 2; ++n)
    {
        float K_norm_sq = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            const Q8_1Block &blk = K_compact[n * num_blocks + b];
            float d = simd::fp16_to_fp32(blk.d);
            int32_t sq = 0;
            for (int i = 0; i < 32; ++i)
            {
                sq += static_cast<int32_t>(blk.qs[i]) * static_cast<int32_t>(blk.qs[i]);
            }
            K_norm_sq += sq * d * d;
        }
        LOG_INFO("  K[" << n << "] |K|²=" << K_norm_sq << ", |K|=" << std::sqrt(K_norm_sq));
    }

    // Create causal mask
    std::vector<float> mask(4);
    mask[0 * 2 + 0] = 0.0f;
    mask[0 * 2 + 1] = -std::numeric_limits<float>::infinity();
    mask[1 * 2 + 0] = 0.0f;
    mask[1 * 2 + 1] = 0.0f;

    // Run reference
    LOG_INFO("");
    LOG_INFO("Running C++ reference...");
    std::vector<Q8_1Block> ref_out(2 * num_blocks);
    gemm_v4::fused_q8_1_attention_reference(
        Q_compact.data(), K_compact.data(), V_compact.data(), ref_out.data(),
        2, 2, head_dim,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, mask.data(), 2);

    std::vector<float> ref_fp32(2 * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_out.data(), ref_fp32.data(), ref_fp32.size());

    // Run JIT
    LOG_INFO("Running JIT...");
    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    std::vector<Q8_1Block> jit_out(2 * num_blocks);
    gemm_v4::FusedQ8_1AttentionParams params;
    params.Q = Q_compact.data();
    params.K = K_compact.data();
    params.V = V_compact.data();
    params.output = jit_out.data();
    params.M = 2;
    params.N = 2;
    params.head_dim = head_dim;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = mask.data();
    params.mask_stride = 2;

    kernel_fn(&params);

    std::vector<float> jit_fp32(2 * head_dim);
    simd::dequantize_q8_1_to_fp32(jit_out.data(), jit_fp32.data(), jit_fp32.size());

    // Compare
    LOG_INFO("");
    LOG_INFO("Comparison:");
    for (int m = 0; m < 2; ++m)
    {
        double cos = cosine_similarity(ref_fp32.data() + m * head_dim, jit_fp32.data() + m * head_dim, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << cos);
    }

    // Print position 1 values (Block 0)
    LOG_INFO("");
    LOG_INFO("Position 1 values (Block 0):");
    for (int i = 0; i < 10; ++i)
    {
        LOG_INFO("  [" << i << "] ref=" << std::setprecision(6) << ref_fp32[head_dim + i]
                       << ", jit=" << jit_fp32[head_dim + i]);
    }

    // Print position 1 values (Block 1)
    LOG_INFO("");
    LOG_INFO("Position 1 values (Block 1):");
    for (int i = 32; i < 42; ++i)
    {
        LOG_INFO("  [" << i << "] ref=" << std::setprecision(6) << ref_fp32[head_dim + i]
                       << ", jit=" << jit_fp32[head_dim + i]);
    }

    double overall = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("");
    LOG_INFO("Overall cosine: " << std::fixed << std::setprecision(6) << overall);

    EXPECT_GE(overall, 0.999) << "Two position test should match";
}

TEST_F(Test__Q8_1_AttentionJitReplay, ThreePositionMasked)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  THREE POSITION MASKED TEST                                    ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Extract positions 0, 1, 2 for head 0
    int h = 0;
    int kv_head = 0;
    int q_blocks_per_row = data.n_heads * num_blocks;
    int kv_blocks_per_row = data.n_kv_heads * num_blocks;

    // Copy positions 0, 1, 2 to contiguous arrays
    std::vector<Q8_1Block> Q_compact(3 * num_blocks);
    std::vector<Q8_1Block> K_compact(3 * num_blocks);
    std::vector<Q8_1Block> V_compact(3 * num_blocks);

    for (int m = 0; m < 3; ++m)
    {
        const Q8_1Block *src_q = data.Q_blocks.data() + m * q_blocks_per_row + h * num_blocks;
        const Q8_1Block *src_k = data.K_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;
        const Q8_1Block *src_v = data.V_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;

        memcpy(&Q_compact[m * num_blocks], src_q, num_blocks * sizeof(Q8_1Block));
        memcpy(&K_compact[m * num_blocks], src_k, num_blocks * sizeof(Q8_1Block));
        memcpy(&V_compact[m * num_blocks], src_v, num_blocks * sizeof(Q8_1Block));
    }

    LOG_INFO("Extracted 3 positions for head 0");

    // Create causal mask for 3x3
    // But we only run M=2 (pos 0, 1) against N=3 (pos 0, 1, 2)
    // Pos 0: attends to 0. 1, 2 masked.
    // Pos 1: attends to 0, 1. 2 masked.
    std::vector<float> mask(2 * 3);
    // m=0
    mask[0 * 3 + 0] = 0.0f;
    mask[0 * 3 + 1] = -std::numeric_limits<float>::infinity();
    mask[0 * 3 + 2] = -std::numeric_limits<float>::infinity();
    // m=1
    mask[1 * 3 + 0] = 0.0f;
    mask[1 * 3 + 1] = 0.0f;
    mask[1 * 3 + 2] = -std::numeric_limits<float>::infinity();

    // Run reference
    LOG_INFO("");
    LOG_INFO("Running C++ reference...");
    std::vector<Q8_1Block> ref_out(2 * num_blocks);
    gemm_v4::fused_q8_1_attention_reference(
        Q_compact.data(), K_compact.data(), V_compact.data(), ref_out.data(),
        2, 3, head_dim,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, mask.data(), 3);

    std::vector<float> ref_fp32(2 * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_out.data(), ref_fp32.data(), ref_fp32.size());

    // Run JIT
    LOG_INFO("Running JIT...");
    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    std::vector<Q8_1Block> jit_out(2 * num_blocks);
    gemm_v4::FusedQ8_1AttentionParams params;
    params.Q = Q_compact.data();
    params.K = K_compact.data();
    params.V = V_compact.data();
    params.output = jit_out.data();
    params.M = 2;
    params.N = 3;
    params.head_dim = head_dim;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = mask.data();
    params.mask_stride = 3;

    kernel_fn(&params);

    std::vector<float> jit_fp32(2 * head_dim);
    simd::dequantize_q8_1_to_fp32(jit_out.data(), jit_fp32.data(), jit_fp32.size());

    // Compare
    LOG_INFO("");
    LOG_INFO("Comparison:");
    for (int m = 0; m < 2; ++m)
    {
        double cos = cosine_similarity(ref_fp32.data() + m * head_dim, jit_fp32.data() + m * head_dim, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << cos);
    }

    double overall = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("");
    LOG_INFO("Overall cosine: " << std::fixed << std::setprecision(6) << overall);

    EXPECT_GE(overall, 0.999) << "Three position masked test should match";
}

TEST_F(Test__Q8_1_AttentionJitReplay, ThreePositionUnmasked)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  THREE POSITION UNMASKED TEST                                  ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Extract positions 0, 1, 2 for head 0
    int h = 0;
    int kv_head = 0;
    int q_blocks_per_row = data.n_heads * num_blocks;
    int kv_blocks_per_row = data.n_kv_heads * num_blocks;

    // Copy positions 0, 1, 2 to contiguous arrays
    std::vector<Q8_1Block> Q_compact(3 * num_blocks);
    std::vector<Q8_1Block> K_compact(3 * num_blocks);
    std::vector<Q8_1Block> V_compact(3 * num_blocks);

    for (int m = 0; m < 3; ++m)
    {
        const Q8_1Block *src_q = data.Q_blocks.data() + m * q_blocks_per_row + h * num_blocks;
        const Q8_1Block *src_k = data.K_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;
        const Q8_1Block *src_v = data.V_blocks.data() + m * kv_blocks_per_row + kv_head * num_blocks;

        memcpy(&Q_compact[m * num_blocks], src_q, num_blocks * sizeof(Q8_1Block));
        memcpy(&K_compact[m * num_blocks], src_k, num_blocks * sizeof(Q8_1Block));
        memcpy(&V_compact[m * num_blocks], src_v, num_blocks * sizeof(Q8_1Block));
    }

    LOG_INFO("Extracted 3 positions for head 0");

    // No mask (all zeros)
    std::vector<float> mask(2 * 3, 0.0f);

    // Run reference
    LOG_INFO("");
    LOG_INFO("Running C++ reference...");
    std::vector<Q8_1Block> ref_out(2 * num_blocks);
    gemm_v4::fused_q8_1_attention_reference(
        Q_compact.data(), K_compact.data(), V_compact.data(), ref_out.data(),
        2, 3, head_dim,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, mask.data(), 3);

    std::vector<float> ref_fp32(2 * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_out.data(), ref_fp32.data(), ref_fp32.size());

    // Run JIT
    LOG_INFO("Running JIT...");
    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    std::vector<Q8_1Block> jit_out(2 * num_blocks);
    gemm_v4::FusedQ8_1AttentionParams params;
    params.Q = Q_compact.data();
    params.K = K_compact.data();
    params.V = V_compact.data();
    params.output = jit_out.data();
    params.M = 2;
    params.N = 3;
    params.head_dim = head_dim;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = mask.data();
    params.mask_stride = 3;

    kernel_fn(&params);

    std::vector<float> jit_fp32(2 * head_dim);
    simd::dequantize_q8_1_to_fp32(jit_out.data(), jit_fp32.data(), jit_fp32.size());

    // Compare
    LOG_INFO("");
    LOG_INFO("Comparison:");
    for (int m = 0; m < 2; ++m)
    {
        double cos = cosine_similarity(ref_fp32.data() + m * head_dim, jit_fp32.data() + m * head_dim, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << cos);
    }

    double overall = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("");
    LOG_INFO("Overall cosine: " << std::fixed << std::setprecision(6) << overall);

    EXPECT_GE(overall, 0.999) << "Three position unmasked test should match";
}

TEST_F(Test__Q8_1_AttentionJitReplay, DebugSingleHeadNormalized)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  DEBUG SINGLE HEAD: Head 0 from real data                      ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create causal mask
    std::vector<float> mask(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        for (int n = 0; n < seq_len; ++n)
        {
            mask[m * seq_len + n] = (n <= m) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    // Extract data for head 0 / KV head 0
    int h = 0;
    int kv_head = 0;
    int q_blocks_per_row = data.n_heads * num_blocks; // Full row stride
    int kv_blocks_per_row = data.n_kv_heads * num_blocks;

    int q_block_offset = h * num_blocks;
    int kv_block_offset = kv_head * num_blocks;

    const Q8_1Block *Q_ptr = data.Q_blocks.data() + q_block_offset;
    const Q8_1Block *K_ptr = data.K_blocks.data() + kv_block_offset;
    const Q8_1Block *V_ptr = data.V_blocks.data() + kv_block_offset;

    LOG_INFO("");
    LOG_INFO("Data layout:");
    LOG_INFO("  seq_len=" << seq_len << ", head_dim=" << head_dim << ", num_blocks=" << num_blocks);
    LOG_INFO("  q_blocks_per_row=" << q_blocks_per_row << ", kv_blocks_per_row=" << kv_blocks_per_row);
    LOG_INFO("  Q stride bytes=" << q_blocks_per_row * sizeof(Q8_1Block));
    LOG_INFO("  K/V stride bytes=" << kv_blocks_per_row * sizeof(Q8_1Block));

    // === Manually compute |Q|² and |K|² for debugging ===
    LOG_INFO("");
    LOG_INFO("Norms for head 0:");
    for (int m = 0; m < std::min(3, seq_len); ++m)
    {
        const Q8_1Block *Q_row = Q_ptr + m * q_blocks_per_row;
        float Q_norm_sq = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            float d = simd::fp16_to_fp32(Q_row[b].d);
            int32_t sq = 0;
            for (int i = 0; i < 32; ++i)
            {
                sq += static_cast<int32_t>(Q_row[b].qs[i]) * static_cast<int32_t>(Q_row[b].qs[i]);
            }
            Q_norm_sq += static_cast<float>(sq) * d * d;
        }
        LOG_INFO("  Q[m=" << m << "] |Q|²=" << Q_norm_sq << ", |Q|=" << std::sqrt(Q_norm_sq));
    }

    for (int n = 0; n < std::min(3, seq_len); ++n)
    {
        const Q8_1Block *K_row = K_ptr + n * kv_blocks_per_row;
        float K_norm_sq = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            float d = simd::fp16_to_fp32(K_row[b].d);
            int32_t sq = 0;
            for (int i = 0; i < 32; ++i)
            {
                sq += static_cast<int32_t>(K_row[b].qs[i]) * static_cast<int32_t>(K_row[b].qs[i]);
            }
            K_norm_sq += static_cast<float>(sq) * d * d;
        }
        LOG_INFO("  K[n=" << n << "] |K|²=" << K_norm_sq << ", |K|=" << std::sqrt(K_norm_sq));
    }

    // === Run C++ reference for head 0 ===
    LOG_INFO("");
    LOG_INFO("Running C++ reference for head 0...");

    std::vector<Q8_1Block> ref_output(seq_len * num_blocks);
    gemm_v4::fused_q8_1_attention_reference(
        Q_ptr, K_ptr, V_ptr, ref_output.data(),
        seq_len, seq_len, head_dim,
        q_blocks_per_row, kv_blocks_per_row, kv_blocks_per_row, num_blocks, // out stride = just head_dim
        scale, mask.data(), seq_len);

    std::vector<float> ref_fp32(seq_len * head_dim);
    simd::dequantize_q8_1_to_fp32(ref_output.data(), ref_fp32.data(), ref_fp32.size());

    // === Run JIT for head 0 ===
    LOG_INFO("Running JIT for head 0...");

    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    std::vector<Q8_1Block> jit_output(seq_len * num_blocks);

    gemm_v4::FusedQ8_1AttentionParams params;
    params.Q = Q_ptr;
    params.K = K_ptr;
    params.V = V_ptr;
    params.output = jit_output.data();
    params.M = seq_len;
    params.N = seq_len;
    params.head_dim = head_dim;
    params.Q_stride_bytes = q_blocks_per_row * sizeof(Q8_1Block);
    params.K_stride_bytes = kv_blocks_per_row * sizeof(Q8_1Block);
    params.V_stride_bytes = kv_blocks_per_row * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block); // out stride = just head_dim
    params.scale = scale;
    params.mask = mask.data();
    params.mask_stride = seq_len;

    kernel_fn(&params);

    std::vector<float> jit_fp32(seq_len * head_dim);
    simd::dequantize_q8_1_to_fp32(jit_output.data(), jit_fp32.data(), jit_fp32.size());

    // === Compare ===
    LOG_INFO("");
    LOG_INFO("Comparison for head 0:");

    for (int m = 0; m < seq_len; ++m)
    {
        const float *ref_row = ref_fp32.data() + m * head_dim;
        const float *jit_row = jit_fp32.data() + m * head_dim;
        double cos = cosine_similarity(ref_row, jit_row, head_dim);
        LOG_INFO("  Position " << m << ": cosine=" << std::fixed << std::setprecision(6) << cos);
    }

    // Print detailed values for position 1
    LOG_INFO("");
    LOG_INFO("Position 1 values:");
    for (int i = 0; i < 10; ++i)
    {
        LOG_INFO("  [" << i << "] ref=" << std::setprecision(6) << ref_fp32[1 * head_dim + i]
                       << ", jit=" << jit_fp32[1 * head_dim + i]
                       << ", diff=" << std::abs(ref_fp32[1 * head_dim + i] - jit_fp32[1 * head_dim + i]));
    }

    double overall = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("");
    LOG_INFO("Overall cosine for head 0: " << std::fixed << std::setprecision(6) << overall);

    EXPECT_GE(overall, 0.999) << "JIT should match reference for single head";
}

/**
 * @brief Test JIT with NON-compact strides (E2E pipeline scenario)
 *
 * This test reproduces the exact memory layout used by the E2E Qwen2Pipeline:
 * - Q/K/V are laid out with ALL heads in each row
 * - Output also uses the full multi-head stride
 *
 * The AllPositionsCompactStride test passes because it extracts a single head's
 * data into compact arrays. But the E2E pipeline passes pointers into the middle
 * of multi-head rows with larger strides.
 *
 * This test specifically validates that the JIT kernel handles:
 * 1. Q_stride_bytes = n_heads * head_dim_blocks * 36 (not just head_dim_blocks * 36)
 * 2. K/V_stride_bytes = n_kv_heads * head_dim_blocks * 36
 * 3. Output stride = n_heads * head_dim_blocks * 36
 */
TEST_F(Test__Q8_1_AttentionJitReplay, NonCompactStridesE2ELayout)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  NON-COMPACT STRIDES TEST (E2E PIPELINE LAYOUT)                ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    // Load saved data
    AttentionReplayData data;
    if (!load_replay_data(DUMP_DIR, data))
    {
        GTEST_SKIP() << "No replay data found. Run DumpAttentionInputs first.";
    }

    int seq_len = data.seq_len;
    int n_heads = data.n_heads;
    int n_kv_heads = data.n_kv_heads;
    int head_dim = data.head_dim;
    int num_blocks = head_dim / 32;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    LOG_INFO("Layout: seq_len=" << seq_len << ", n_heads=" << n_heads
                                << ", n_kv_heads=" << n_kv_heads << ", head_dim=" << head_dim);

    // Calculate strides (E2E layout - all heads per row)
    int q_blocks_per_row = n_heads * num_blocks;     // 14 * 2 = 28 for Qwen 0.5B
    int kv_blocks_per_row = n_kv_heads * num_blocks; // 2 * 2 = 4 for Qwen 0.5B
    int out_blocks_per_row = n_heads * num_blocks;   // 14 * 2 = 28 for Qwen 0.5B

    int q_stride_bytes = q_blocks_per_row * sizeof(Q8_1Block);
    int kv_stride_bytes = kv_blocks_per_row * sizeof(Q8_1Block);
    int out_stride_bytes = out_blocks_per_row * sizeof(Q8_1Block);

    LOG_INFO("Strides: Q=" << q_stride_bytes << "B, K/V=" << kv_stride_bytes
                           << "B, out=" << out_stride_bytes << "B");
    LOG_INFO("This matches E2E pipeline layout (non-compact)");

    // Create causal mask
    std::vector<float> mask(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        for (int n = 0; n < seq_len; ++n)
        {
            mask[m * seq_len + n] = (n <= m) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    // Get JIT kernel
    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    // Allocate output buffer with E2E layout (all heads per row)
    std::vector<Q8_1Block> jit_output(seq_len * out_blocks_per_row);
    std::vector<Q8_1Block> ref_output(seq_len * out_blocks_per_row);

    LOG_INFO("");
    LOG_INFO("Testing each head with non-compact strides:");

    int heads_per_kv = n_heads / n_kv_heads;
    bool any_divergence = false;

    for (int h = 0; h < n_heads; ++h)
    {
        int kv_h = h / heads_per_kv;

        // Pointers into the multi-head arrays (E2E style)
        const Q8_1Block *Q_h = data.Q_blocks.data() + h * num_blocks;
        const Q8_1Block *K_h = data.K_blocks.data() + kv_h * num_blocks;
        const Q8_1Block *V_h = data.V_blocks.data() + kv_h * num_blocks;
        Q8_1Block *jit_out_h = jit_output.data() + h * num_blocks;
        Q8_1Block *ref_out_h = ref_output.data() + h * num_blocks;

        // Run C++ reference
        gemm_v4::fused_q8_1_attention_reference(
            Q_h, K_h, V_h, ref_out_h,
            seq_len, seq_len, head_dim,
            q_stride_bytes / sizeof(Q8_1Block),   // Q stride in blocks
            kv_stride_bytes / sizeof(Q8_1Block),  // K stride in blocks
            kv_stride_bytes / sizeof(Q8_1Block),  // V stride in blocks
            out_stride_bytes / sizeof(Q8_1Block), // out stride in blocks
            scale, mask.data(), seq_len);

        // Run JIT kernel
        gemm_v4::FusedQ8_1AttentionParams params;
        params.Q = Q_h;
        params.K = K_h;
        params.V = V_h;
        params.output = jit_out_h;
        params.M = seq_len;
        params.N = seq_len;
        params.head_dim = head_dim;
        params.Q_stride_bytes = q_stride_bytes; // E2E stride!
        params.K_stride_bytes = kv_stride_bytes;
        params.V_stride_bytes = kv_stride_bytes;
        params.output_stride_bytes = out_stride_bytes; // E2E stride!
        params.scale = scale;
        params.mask = mask.data();
        params.mask_stride = seq_len;

        kernel_fn(&params);

        // Compare this head's output
        // Extract and dequantize just this head's output for comparison
        std::vector<float> ref_fp32(seq_len * head_dim);
        std::vector<float> jit_fp32(seq_len * head_dim);

        // Dequantize with strides
        for (int m = 0; m < seq_len; ++m)
        {
            const Q8_1Block *ref_row = ref_out_h + m * (out_stride_bytes / sizeof(Q8_1Block));
            const Q8_1Block *jit_row = jit_out_h + m * (out_stride_bytes / sizeof(Q8_1Block));
            simd::dequantize_q8_1_to_fp32(ref_row, ref_fp32.data() + m * head_dim, head_dim);
            simd::dequantize_q8_1_to_fp32(jit_row, jit_fp32.data() + m * head_dim, head_dim);
        }

        double cos = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
        std::string status = (cos >= 0.999) ? "✓ GOOD" : (cos >= 0.99 ? "~ OK" : "✗ DIVERGED");

        if (cos < 0.99)
        {
            any_divergence = true;
            LOG_INFO("  Head " << h << " (kv=" << kv_h << "): cosine=" << std::fixed
                               << std::setprecision(6) << cos << " " << status);

            // Show per-position breakdown for divergent heads
            LOG_INFO("    Per-position breakdown:");
            for (int m = 0; m < std::min(seq_len, 5); ++m)
            {
                double pos_cos = cosine_similarity(
                    ref_fp32.data() + m * head_dim,
                    jit_fp32.data() + m * head_dim,
                    head_dim);
                LOG_INFO("      Position " << m << ": cosine=" << std::fixed
                                           << std::setprecision(6) << pos_cos);
            }
        }
        else
        {
            LOG_INFO("  Head " << h << " (kv=" << kv_h << "): cosine=" << std::fixed
                               << std::setprecision(6) << cos << " " << status);
        }
    }

    LOG_INFO("");
    if (any_divergence)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  DIVERGENCE DETECTED WITH NON-COMPACT STRIDES                  ║");
        LOG_INFO("║  This reproduces the E2E pipeline issue!                       ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }
    else
    {
        LOG_INFO("All heads passed with non-compact strides!");
    }

    EXPECT_FALSE(any_divergence) << "JIT should match reference with non-compact (E2E) strides";
}

/**
 * @brief Test with actual Q8_1 pipeline's Q/K/V (not quantized-from-FP32)
 *
 * CRITICAL DIFFERENCE from other tests:
 * - Other tests: FP32 Q/K/V → quantize to Q8_1 → run JIT → compare to FP32 attention
 * - This test: Run actual Q8_1 pipeline → capture Q8_1 Q/K/V → run JIT → compare to C++ reference
 *
 * This isolates whether the issue is:
 * A) JIT kernel bug (JIT != C++ reference on same inputs)
 * B) Q8_1 vs FP32 precision difference (expected divergence)
 *
 * If JIT == C++ reference but both diverge from FP32, that's expected precision loss.
 * If JIT != C++ reference, we have a kernel bug.
 */
#ifdef ENABLE_PIPELINE_SNAPSHOTS
TEST_F(Test__Q8_1_AttentionJitReplay, ActualQ8_1PipelineQKV)
{
    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║  TESTING JIT WITH ACTUAL Q8_1 PIPELINE Q/K/V                   ║");
    LOG_INFO("║  (Isolates JIT bug vs expected precision loss)                 ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    std::vector<int> tokens = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};
    int seq_len = static_cast<int>(tokens.size());

    // Run Q8_1 pipeline with snapshot capture
    LOG_INFO("");
    LOG_INFO("Running Q8_1 pipeline to capture actual Q8_1 Q/K/V...");

    PipelineConfig q8_1_config;
    q8_1_config.activation_precision = ActivationPrecision::Q8_1;
    q8_1_config.max_seq_len = 512;

    auto q8_1_pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_, mpi_ctx_, -1, nullptr, q8_1_config, 1);
    q8_1_pipeline->enableSnapshotCapture();

    bool success = q8_1_pipeline->forward(tokens.data(), seq_len);
    ASSERT_TRUE(success) << "Q8_1 pipeline forward failed";

    // Get Q8_1 Q/K/V snapshots (after RoPE for Q/K, projection for V)
    // Note: Snapshots are stored as FP32 (dequantized), but we need the raw Q8_1 blocks
    // The Q8_1 tensors are stored internally in the pipeline's buffers

    size_t q_size = 0, k_size = 0, v_size = 0, attn_size = 0;
    const float *q_snap = q8_1_pipeline->getSnapshot("layer0_Q_ROPE", q_size);
    const float *k_snap = q8_1_pipeline->getSnapshot("layer0_K_ROPE", k_size);
    const float *v_snap = q8_1_pipeline->getSnapshot("layer0_V_PROJECTION", v_size);
    const float *attn_snap = q8_1_pipeline->getSnapshot("layer0_ATTENTION_CONTEXT", attn_size);

    ASSERT_NE(q_snap, nullptr) << "Q snapshot not found";
    ASSERT_NE(k_snap, nullptr) << "K snapshot not found";
    ASSERT_NE(v_snap, nullptr) << "V snapshot not found";
    ASSERT_NE(attn_snap, nullptr) << "Attention context snapshot not found";

    LOG_INFO("  Q (dequantized): " << q_size << " elements");
    LOG_INFO("  K (dequantized): " << k_size << " elements");
    LOG_INFO("  V (dequantized): " << v_size << " elements");
    LOG_INFO("  Attention context (dequantized): " << attn_size << " elements");

    // Re-quantize to Q8_1 blocks (simulating the internal Q8_1 format)
    // This is necessary because snapshots are stored dequantized
    int n_heads = N_HEADS;
    int n_kv_heads = N_KV_HEADS;
    int head_dim = HEAD_DIM;
    int num_blocks = head_dim / 32;

    int q_blocks_per_row = n_heads * num_blocks;
    int kv_blocks_per_row = n_kv_heads * num_blocks;

    std::vector<Q8_1Block> Q_blocks(seq_len * q_blocks_per_row);
    std::vector<Q8_1Block> K_blocks(seq_len * kv_blocks_per_row);
    std::vector<Q8_1Block> V_blocks(seq_len * kv_blocks_per_row);

    simd::quantize_fp32_to_q8_1_blocks(q_snap, Q_blocks.data(), q_size);
    simd::quantize_fp32_to_q8_1_blocks(k_snap, K_blocks.data(), k_size);
    simd::quantize_fp32_to_q8_1_blocks(v_snap, V_blocks.data(), v_size);

    // Create causal mask
    std::vector<float> mask(seq_len * seq_len);
    for (int m = 0; m < seq_len; ++m)
    {
        for (int n = 0; n < seq_len; ++n)
        {
            mask[m * seq_len + n] = (n <= m) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Run BOTH C++ reference AND JIT on the same Q8_1 blocks
    LOG_INFO("");
    LOG_INFO("Running C++ reference and JIT on actual Q8_1 pipeline data...");

    int q_stride_bytes = q_blocks_per_row * sizeof(Q8_1Block);
    int kv_stride_bytes = kv_blocks_per_row * sizeof(Q8_1Block);
    int out_stride_bytes = q_blocks_per_row * sizeof(Q8_1Block);

    std::vector<Q8_1Block> ref_output(seq_len * q_blocks_per_row);
    std::vector<Q8_1Block> jit_output(seq_len * q_blocks_per_row);

    // Get JIT kernel
    gemm_v4::QuantisedAttentionJit_Q8_1_Fused jit(head_dim, false);
    auto kernel_fn = jit.get_kernel();

    int heads_per_kv = n_heads / n_kv_heads;

    for (int h = 0; h < n_heads; ++h)
    {
        int kv_h = h / heads_per_kv;

        const Q8_1Block *Q_h = Q_blocks.data() + h * num_blocks;
        const Q8_1Block *K_h = K_blocks.data() + kv_h * num_blocks;
        const Q8_1Block *V_h = V_blocks.data() + kv_h * num_blocks;
        Q8_1Block *ref_out_h = ref_output.data() + h * num_blocks;
        Q8_1Block *jit_out_h = jit_output.data() + h * num_blocks;

        // C++ reference
        gemm_v4::fused_q8_1_attention_reference(
            Q_h, K_h, V_h, ref_out_h,
            seq_len, seq_len, head_dim,
            q_stride_bytes / sizeof(Q8_1Block),
            kv_stride_bytes / sizeof(Q8_1Block),
            kv_stride_bytes / sizeof(Q8_1Block),
            out_stride_bytes / sizeof(Q8_1Block),
            scale, mask.data(), seq_len);

        // JIT kernel
        gemm_v4::FusedQ8_1AttentionParams params;
        params.Q = Q_h;
        params.K = K_h;
        params.V = V_h;
        params.output = jit_out_h;
        params.M = seq_len;
        params.N = seq_len;
        params.head_dim = head_dim;
        params.Q_stride_bytes = q_stride_bytes;
        params.K_stride_bytes = kv_stride_bytes;
        params.V_stride_bytes = kv_stride_bytes;
        params.output_stride_bytes = out_stride_bytes;
        params.scale = scale;
        params.mask = mask.data();
        params.mask_stride = seq_len;

        kernel_fn(&params);
    }

    // Compare JIT vs C++ reference (should be nearly identical)
    LOG_INFO("");
    LOG_INFO("=== JIT vs C++ Reference (should be ~0.999+) ===");

    std::vector<float> ref_fp32(seq_len * n_heads * head_dim);
    std::vector<float> jit_fp32(seq_len * n_heads * head_dim);

    for (int m = 0; m < seq_len; ++m)
    {
        const Q8_1Block *ref_row = ref_output.data() + m * q_blocks_per_row;
        const Q8_1Block *jit_row = jit_output.data() + m * q_blocks_per_row;
        simd::dequantize_q8_1_to_fp32(ref_row, ref_fp32.data() + m * n_heads * head_dim, n_heads * head_dim);
        simd::dequantize_q8_1_to_fp32(jit_row, jit_fp32.data() + m * n_heads * head_dim, n_heads * head_dim);
    }

    double jit_vs_ref_cos = cosine_similarity(ref_fp32.data(), jit_fp32.data(), ref_fp32.size());
    LOG_INFO("JIT vs C++ Reference cosine: " << std::fixed << std::setprecision(6) << jit_vs_ref_cos);

    // Also compare to the pipeline's attention output (should show expected Q8_1 divergence)
    LOG_INFO("");
    LOG_INFO("=== JIT vs Pipeline Attention Output (shows precision loss) ===");

    // Re-quantize pipeline attention output for fair comparison
    std::vector<Q8_1Block> pipeline_attn_blocks(seq_len * q_blocks_per_row);
    simd::quantize_fp32_to_q8_1_blocks(attn_snap, pipeline_attn_blocks.data(), attn_size);

    std::vector<float> pipeline_fp32(attn_size);
    simd::dequantize_q8_1_to_fp32(pipeline_attn_blocks.data(), pipeline_fp32.data(), attn_size);

    double jit_vs_pipeline_cos = cosine_similarity(jit_fp32.data(), pipeline_fp32.data(), jit_fp32.size());
    LOG_INFO("JIT vs Pipeline cosine: " << std::fixed << std::setprecision(6) << jit_vs_pipeline_cos);

    // Per-head breakdown
    LOG_INFO("");
    LOG_INFO("Per-head JIT vs Reference:");
    bool any_divergence = false;
    for (int h = 0; h < n_heads; ++h)
    {
        std::vector<float> ref_head_fp32(seq_len * head_dim);
        std::vector<float> jit_head_fp32(seq_len * head_dim);

        for (int m = 0; m < seq_len; ++m)
        {
            const Q8_1Block *ref_row = ref_output.data() + m * q_blocks_per_row + h * num_blocks;
            const Q8_1Block *jit_row = jit_output.data() + m * q_blocks_per_row + h * num_blocks;
            simd::dequantize_q8_1_to_fp32(ref_row, ref_head_fp32.data() + m * head_dim, head_dim);
            simd::dequantize_q8_1_to_fp32(jit_row, jit_head_fp32.data() + m * head_dim, head_dim);
        }

        double head_cos = cosine_similarity(ref_head_fp32.data(), jit_head_fp32.data(), ref_head_fp32.size());
        std::string status = (head_cos >= 0.999) ? "✓" : "✗";
        LOG_INFO("  Head " << h << ": cosine=" << std::fixed << std::setprecision(6) << head_cos << " " << status);

        if (head_cos < 0.999)
            any_divergence = true;
    }

    LOG_INFO("");
    if (jit_vs_ref_cos >= 0.999)
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  JIT matches C++ reference - kernel is CORRECT                 ║");
        LOG_INFO("║  Any E2E divergence is due to Q8_1 precision loss, not bugs    ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }
    else
    {
        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  JIT DIVERGES from C++ reference - KERNEL BUG DETECTED!        ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");
    }

    EXPECT_GE(jit_vs_ref_cos, 0.999) << "JIT should match C++ reference on actual Q8_1 pipeline data";
    EXPECT_FALSE(any_divergence) << "All heads should match";
}
#endif // ENABLE_PIPELINE_SNAPSHOTS

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
