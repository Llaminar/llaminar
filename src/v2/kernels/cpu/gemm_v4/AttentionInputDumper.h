/**
 * @file AttentionInputDumper.h
 * @brief Compile-time optional input dumper for Q8_1 attention kernel debugging
 * @author David Sanftenberg
 *
 * When LLAMINAR_DUMP_ATTENTION_INPUTS is defined at compile time, this will
 * save the exact inputs to the JIT attention kernel to disk. This allows
 * reproducing issues in isolation tests with static test data.
 *
 * Usage:
 *   cmake -DLLAMINAR_DUMP_ATTENTION_INPUTS=ON ...
 *
 * Output:
 *   Creates attention_dump_NNNN/ directories with:
 *     - params.bin: FusedQ8_1AttentionParams struct
 *     - Q_blocks.bin: Raw Q8_1Block data
 *     - K_blocks.bin: Raw Q8_1Block data
 *     - V_blocks.bin: Raw Q8_1Block data
 *     - mask.bin: Optional attention mask (FP32)
 *     - metadata.txt: Human-readable parameters
 */

#pragma once

#include "../../../tensors/BlockStructures.h"
#include "QuantisedAttentionJit_Q8_1_Fused.h"

#include <cstdint>
#include <cstdio>
#include <atomic>
#include <string>
#include <sys/stat.h>
#include <fstream>

namespace llaminar2::gemm_v4
{

/**
 * @brief Dump attention inputs to disk for debugging
 *
 * When LLAMINAR_DUMP_ATTENTION_INPUTS is defined, this function saves
 * the exact inputs to a JIT kernel invocation for later replay.
 *
 * @param params Kernel parameters
 * @param head_idx Head index (for multi-head context)
 * @param layer_idx Layer index (-1 if unknown)
 */
#ifdef LLAMINAR_DUMP_ATTENTION_INPUTS

    inline void dump_attention_inputs(
        const FusedQ8_1AttentionParams &params,
        int head_idx = 0,
        int layer_idx = -1)
    {
        // Thread-safe dump counter
        static std::atomic<int> dump_counter{0};
        int dump_id = dump_counter.fetch_add(1);

        // Only dump first few invocations to avoid disk explosion
        constexpr int MAX_DUMPS = 100;
        if (dump_id >= MAX_DUMPS)
        {
            return;
        }

        // Create output directory
        char dir_name[256];
        snprintf(dir_name, sizeof(dir_name), "attention_dump_%04d", dump_id);
        mkdir(dir_name, 0755);

        // Calculate sizes
        const int num_blocks_per_head = params.head_dim / 32;
        const int q_rows = params.M;
        const int kv_rows = params.N;

        // Q stride is per-row, but we receive a pointer to a specific head's start
        // The stride tells us how far to jump to get to the same head's next row
        const int q_stride_blocks = params.Q_stride_bytes / static_cast<int>(sizeof(Q8_1Block));
        const int k_stride_blocks = params.K_stride_bytes / static_cast<int>(sizeof(Q8_1Block));
        const int v_stride_blocks = params.V_stride_bytes / static_cast<int>(sizeof(Q8_1Block));

        // Calculate total blocks to save
        // For Q: we have q_rows rows, each row has num_blocks_per_head blocks for THIS head
        // The data is non-contiguous if stride > num_blocks_per_head
        const size_t q_block_count = static_cast<size_t>(q_rows) * num_blocks_per_head;
        const size_t kv_block_count = static_cast<size_t>(kv_rows) * num_blocks_per_head;

        // Write metadata
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/metadata.txt", dir_name);
            FILE *f = fopen(path, "w");
            if (f)
            {
                fprintf(f, "# Attention Input Dump #%04d\n", dump_id);
                fprintf(f, "layer_idx=%d\n", layer_idx);
                fprintf(f, "head_idx=%d\n", head_idx);
                fprintf(f, "M=%d  # seq_len (Q rows)\n", params.M);
                fprintf(f, "N=%d  # kv_len (K/V rows)\n", params.N);
                fprintf(f, "head_dim=%d\n", params.head_dim);
                fprintf(f, "num_blocks_per_head=%d\n", num_blocks_per_head);
                fprintf(f, "Q_stride_bytes=%d\n", params.Q_stride_bytes);
                fprintf(f, "K_stride_bytes=%d\n", params.K_stride_bytes);
                fprintf(f, "V_stride_bytes=%d\n", params.V_stride_bytes);
                fprintf(f, "output_stride_bytes=%d\n", params.output_stride_bytes);
                fprintf(f, "scale=%f\n", params.scale);
                fprintf(f, "has_mask=%d\n", params.mask != nullptr ? 1 : 0);
                fprintf(f, "mask_stride=%d\n", params.mask_stride);
                fprintf(f, "q_stride_blocks=%d\n", q_stride_blocks);
                fprintf(f, "k_stride_blocks=%d\n", k_stride_blocks);
                fprintf(f, "v_stride_blocks=%d\n", v_stride_blocks);
                fprintf(f, "q_block_count=%zu\n", q_block_count);
                fprintf(f, "kv_block_count=%zu\n", kv_block_count);
                fclose(f);
            }
        }

        // Write params struct (for binary replay)
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/params.bin", dir_name);
            FILE *f = fopen(path, "wb");
            if (f)
            {
                // Write just the scalar params (pointers will be re-set on load)
                struct SavedParams
                {
                    int M, N, head_dim;
                    int Q_stride_bytes, K_stride_bytes, V_stride_bytes, output_stride_bytes;
                    float scale;
                    int has_mask;
                    int mask_stride;
                } saved;
                saved.M = params.M;
                saved.N = params.N;
                saved.head_dim = params.head_dim;
                saved.Q_stride_bytes = params.Q_stride_bytes;
                saved.K_stride_bytes = params.K_stride_bytes;
                saved.V_stride_bytes = params.V_stride_bytes;
                saved.output_stride_bytes = params.output_stride_bytes;
                saved.scale = params.scale;
                saved.has_mask = params.mask != nullptr ? 1 : 0;
                saved.mask_stride = params.mask_stride;
                fwrite(&saved, sizeof(saved), 1, f);
                fclose(f);
            }
        }

        // Helper to write blocks with stride handling
        auto write_blocks_strided = [&](const char *filename, const void *base_ptr,
                                        int rows, int blocks_per_head, int stride_blocks)
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s", dir_name, filename);
            FILE *f = fopen(path, "wb");
            if (f)
            {
                const Q8_1Block *blocks = static_cast<const Q8_1Block *>(base_ptr);
                for (int row = 0; row < rows; ++row)
                {
                    // Write blocks for this head's row
                    const Q8_1Block *row_start = blocks + row * stride_blocks;
                    fwrite(row_start, sizeof(Q8_1Block), blocks_per_head, f);
                }
                fclose(f);
            }
        };

        // Write Q blocks
        write_blocks_strided("Q_blocks.bin", params.Q, q_rows, num_blocks_per_head, q_stride_blocks);

        // Write K blocks
        write_blocks_strided("K_blocks.bin", params.K, kv_rows, num_blocks_per_head, k_stride_blocks);

        // Write V blocks
        write_blocks_strided("V_blocks.bin", params.V, kv_rows, num_blocks_per_head, v_stride_blocks);

        // Write mask if present
        if (params.mask)
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/mask.bin", dir_name);
            FILE *f = fopen(path, "wb");
            if (f)
            {
                // Mask is [M, N] with stride mask_stride
                for (int m = 0; m < params.M; ++m)
                {
                    fwrite(params.mask + m * params.mask_stride, sizeof(float), params.N, f);
                }
                fclose(f);
            }
        }

        // Log that we dumped
        fprintf(stderr, "[ATTENTION_DUMP] Saved dump #%04d: layer=%d head=%d M=%d N=%d head_dim=%d to %s/\n",
                dump_id, layer_idx, head_idx, params.M, params.N, params.head_dim, dir_name);
    }

#else // !LLAMINAR_DUMP_ATTENTION_INPUTS

    // No-op when dumping is disabled
    inline void dump_attention_inputs(
        const FusedQ8_1AttentionParams & /*params*/,
        int /*head_idx*/ = 0,
        int /*layer_idx*/ = -1)
    {
        // Compiled out
    }

#endif // LLAMINAR_DUMP_ATTENTION_INPUTS

    /**
     * @brief Load dumped attention inputs for replay testing
     *
     * This is always available (not compile-time gated) so tests can load
     * previously-dumped data regardless of whether dumping is currently enabled.
     */
    struct LoadedAttentionDump
    {
        // Scalar params
        int M = 0;
        int N = 0;
        int head_dim = 0;
        int Q_stride_bytes = 0;
        int K_stride_bytes = 0;
        int V_stride_bytes = 0;
        int output_stride_bytes = 0;
        float scale = 0.0f;
        int mask_stride = 0;

        // Block data (compacted - no stride gaps)
        std::vector<Q8_1Block> Q_blocks;
        std::vector<Q8_1Block> K_blocks;
        std::vector<Q8_1Block> V_blocks;
        std::vector<float> mask;

        bool has_mask = false;
        bool valid = false;

        /**
         * @brief Load attention dump from directory
         * @param dump_dir Path to attention_dump_NNNN directory
         * @return true if load succeeded
         */
        bool load(const std::string &dump_dir)
        {
            valid = false;

            // Read params.bin
            {
                std::string path = dump_dir + "/params.bin";
                FILE *f = fopen(path.c_str(), "rb");
                if (!f)
                    return false;

                struct SavedParams
                {
                    int M, N, head_dim;
                    int Q_stride_bytes, K_stride_bytes, V_stride_bytes, output_stride_bytes;
                    float scale;
                    int has_mask;
                    int mask_stride;
                } saved;

                if (fread(&saved, sizeof(saved), 1, f) != 1)
                {
                    fclose(f);
                    return false;
                }
                fclose(f);

                M = saved.M;
                N = saved.N;
                head_dim = saved.head_dim;
                Q_stride_bytes = saved.Q_stride_bytes;
                K_stride_bytes = saved.K_stride_bytes;
                V_stride_bytes = saved.V_stride_bytes;
                output_stride_bytes = saved.output_stride_bytes;
                scale = saved.scale;
                has_mask = saved.has_mask != 0;
                mask_stride = saved.mask_stride;
            }

            const int num_blocks_per_head = head_dim / 32;

            // Read Q blocks (already compacted by dump)
            {
                std::string path = dump_dir + "/Q_blocks.bin";
                FILE *f = fopen(path.c_str(), "rb");
                if (!f)
                    return false;

                size_t q_block_count = static_cast<size_t>(M) * num_blocks_per_head;
                Q_blocks.resize(q_block_count);
                if (fread(Q_blocks.data(), sizeof(Q8_1Block), q_block_count, f) != q_block_count)
                {
                    fclose(f);
                    return false;
                }
                fclose(f);
            }

            // Read K blocks
            {
                std::string path = dump_dir + "/K_blocks.bin";
                FILE *f = fopen(path.c_str(), "rb");
                if (!f)
                    return false;

                size_t kv_block_count = static_cast<size_t>(N) * num_blocks_per_head;
                K_blocks.resize(kv_block_count);
                if (fread(K_blocks.data(), sizeof(Q8_1Block), kv_block_count, f) != kv_block_count)
                {
                    fclose(f);
                    return false;
                }
                fclose(f);
            }

            // Read V blocks
            {
                std::string path = dump_dir + "/V_blocks.bin";
                FILE *f = fopen(path.c_str(), "rb");
                if (!f)
                    return false;

                size_t kv_block_count = static_cast<size_t>(N) * num_blocks_per_head;
                V_blocks.resize(kv_block_count);
                if (fread(V_blocks.data(), sizeof(Q8_1Block), kv_block_count, f) != kv_block_count)
                {
                    fclose(f);
                    return false;
                }
                fclose(f);
            }

            // Read mask if present
            if (has_mask)
            {
                std::string path = dump_dir + "/mask.bin";
                FILE *f = fopen(path.c_str(), "rb");
                if (!f)
                    return false;

                size_t mask_size = static_cast<size_t>(M) * N;
                mask.resize(mask_size);
                if (fread(mask.data(), sizeof(float), mask_size, f) != mask_size)
                {
                    fclose(f);
                    return false;
                }
                fclose(f);
            }

            valid = true;
            return true;
        }

        /**
         * @brief Build FusedQ8_1AttentionParams for kernel invocation
         *
         * Note: Q/K/V/mask pointers point to this object's vectors.
         * Output pointer must be provided by caller.
         */
        FusedQ8_1AttentionParams build_params(Q8_1Block *output) const
        {
            FusedQ8_1AttentionParams params;
            const int num_blocks_per_head = head_dim / 32;

            params.Q = Q_blocks.data();
            params.K = K_blocks.data();
            params.V = V_blocks.data();
            params.output = output;
            params.M = M;
            params.N = N;
            params.head_dim = head_dim;

            // For replaying, data is compacted (no multi-head stride)
            // So stride is just num_blocks_per_head * sizeof(Q8_1Block)
            params.Q_stride_bytes = num_blocks_per_head * static_cast<int>(sizeof(Q8_1Block));
            params.K_stride_bytes = num_blocks_per_head * static_cast<int>(sizeof(Q8_1Block));
            params.V_stride_bytes = num_blocks_per_head * static_cast<int>(sizeof(Q8_1Block));
            params.output_stride_bytes = num_blocks_per_head * static_cast<int>(sizeof(Q8_1Block));

            params.scale = scale;
            params.mask = has_mask ? mask.data() : nullptr;
            params.mask_stride = N; // Compacted mask stride

            return params;
        }
    };

} // namespace llaminar2::gemm_v4
