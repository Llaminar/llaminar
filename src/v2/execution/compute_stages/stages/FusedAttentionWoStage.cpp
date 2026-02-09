/**
 * @file FusedAttentionWoStage.cpp
 * @brief Implementation of FusedAttentionWoStage
 */

#include "FusedAttentionWoStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorVerification.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h"
#include "../../../kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llaminar2
{

    // =============================================================================
    // FusedAttentionWoStage Implementation
    // =============================================================================

    FusedAttentionWoStage::FusedAttentionWoStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
        if (params_.backend == FusedAttentionBackend::Q16_INTEGER)
        {
            // Create Q16_1 kernel via ITensorFusedAttentionWo interface
            q16_kernel_ = std::make_unique<kernels::q16_1::Q16FusedAttentionKernel>(/*use_jit=*/false);
            LOG_DEBUG("[FusedAttentionWoStage] Created Q16FusedAttentionKernel (Q16_INTEGER backend)");
        }
        else
        {
            // Create Q8_1 kernel (JIT/TILED/REFERENCE backends)
            FusedAttentionWoKernel::Config kernel_config;
            kernel_config.num_heads = params_.n_heads;
            kernel_config.num_kv_heads = params_.n_kv_heads;
            kernel_config.head_dim = params_.head_dim;
            kernel_config.d_model = params_.d_model;
            kernel_config.backend = params_.backend;
            kernel_config.use_hybrid_wo = params_.use_hybrid_wo;
            kernel_config.fuse_residual_add = params_.fuse_residual_add;

            kernel_ = std::make_unique<FusedAttentionWoKernel>(kernel_config);
        }
    }

    bool FusedAttentionWoStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[FusedAttentionWoStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        LOG_DEBUG("[FusedAttentionWoStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " d_model=" << params_.d_model
                                                            << " position_offset=" << params_.position_offset
                                                            << " backend=" << fusedAttentionBackendToString(params_.backend));

        // Validate inputs
        if (!params_.Q || !params_.K || !params_.V || !params_.Wo || !params_.output)
        {
            LOG_ERROR("[FusedAttentionWoStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 ||
            params_.head_dim <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[FusedAttentionWoStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[FusedAttentionWoStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Note: Layout validation is now handled declaratively via getLayoutExpectation()
        // and getBufferRequirements().withLayout() - GraphExecutor validates automatically

        // Compute position offset for decode mode
        int position_offset = params_.position_offset;
        if (position_offset == 0 && params_.seq_len < effective_kv_len)
        {
            // Auto-compute for decode mode: query is at end of cached context
            position_offset = effective_kv_len - params_.seq_len;
        }

        bool success = false;

        // Dispatch to appropriate kernel based on backend
        if (params_.backend == FusedAttentionBackend::Q16_INTEGER)
        {
            // Q16_1 kernel via ITensorFusedAttentionWo interface
            // NOTE: Q16_INTEGER backend requires fuse_residual_add=true (HybridQ16 mode)
            if (!params_.fuse_residual_add)
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER backend requires fuse_residual_add=true "
                          "(use JIT or REFERENCE backend for non-fused mode)");
                return false;
            }

            if (!q16_kernel_)
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16 kernel not initialized");
                return false;
            }
            q16_kernel_->setGPUStream(gpuStream());

            // Build params for Q16 kernel
            FusedAttentionWoParams q16_params;

            // Extract typed data from tensors - Q16 kernel expects Q16_1Block* for Q/K/V
            // Use dynamic_cast + typed_data() for type-safe extraction
            auto *q_q16 = dynamic_cast<Q16_1Tensor *>(params_.Q);
            LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER params_.Q ptr="
                      << static_cast<const void *>(params_.Q)
                      << " dtype=" << (params_.Q ? params_.Q->dtype_name() : "null"));
            if (q_q16 && q_q16->typed_data())
            {
                const auto &blk0 = q_q16->typed_data()[0];
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER Q input: blk0.d=" << blk0.d
                                                                                 << " blk0.qs[0]=" << blk0.qs[0]
                                                                                 << " typed_data ptr=" << static_cast<const void *>(q_q16->typed_data()));
            }

            // For Q16_INTEGER backend (HybridQ16 mode), fetch K/V from KV cache at runtime
            // The KV cache stores Q16_1 data (K from RoPE, V converted from Q8_1)
            // This is necessary because:
            // - During prefill: params_.K/V are Q8_1 from QKV GEMM, but cache has Q16_1
            // - During decode: params_.K/V point to current token, cache has full context
            TensorBase *k_tensor = asTensorBase(params_.K, "K");
            TensorBase *v_tensor = asTensorBase(params_.V, "V");
            LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER checking KV cache: kv_cache="
                      << static_cast<const void *>(params_.kv_cache)
                      << " layer_idx=" << params_.layer_idx
                      << " K=" << (params_.K ? params_.K->dtype_name() : "null")
                      << " V=" << (params_.V ? params_.V->dtype_name() : "null"));
            if (params_.kv_cache && params_.layer_idx >= 0)
            {
                // Always use cache for Q16_INTEGER - cache has properly typed Q16_1 tensors
                k_tensor = dynamic_cast<TensorBase *>(params_.kv_cache->get_k(params_.layer_idx, 0));
                v_tensor = dynamic_cast<TensorBase *>(params_.kv_cache->get_v(params_.layer_idx, 0));
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER using KV cache: K="
                          << (k_tensor ? k_tensor->dtype_name() : "null")
                          << " V=" << (v_tensor ? v_tensor->dtype_name() : "null"));
            }

            auto *k_q16 = dynamic_cast<Q16_1Tensor *>(k_tensor);
            auto *v_q16 = dynamic_cast<Q16_1Tensor *>(v_tensor);

            if (!q_q16 || !k_q16 || !v_q16)
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER backend requires Q16_1 tensors for Q/K/V, got: "
                          << "Q=" << (params_.Q ? params_.Q->dtype_name() : "null") << ", "
                          << "K=" << (k_tensor ? k_tensor->dtype_name() : "null") << ", "
                          << "V=" << (v_tensor ? v_tensor->dtype_name() : "null"));
                return false;
            }

            // Debug: spot-check quantization and saturation for layers that diverge.
            // Keep this extremely lightweight (small prefill seq_len) and only on selected layers.
            // NOTE: Uses raw_data() and q16_block_size() to handle variable block sizes (32/64/128).
            if (params_.layer_idx == 0 || params_.layer_idx == 22 || params_.layer_idx == 23)
            {
                // Template-based stats dump that handles variable Q16 block sizes
                auto dump_q16_stats_typed = [&](const char *label,
                                                const Q16_1Tensor *tensor,
                                                int rows,
                                                int dim_per_row,
                                                int max_rows)
                {
                    if (!tensor || rows <= 0 || dim_per_row <= 0)
                    {
                        LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx << " " << label
                                                                   << " stats: <invalid>");
                        return;
                    }

                    const Q16BlockSize block_size = tensor->q16_block_size();
                    const size_t block_elements = q16_block_size_elements(block_size);
                    const size_t block_bytes = q16_block_size_bytes(block_size);
                    const int blocks_per_row = (dim_per_row + static_cast<int>(block_elements) - 1) / static_cast<int>(block_elements);
                    const int rows_to_scan = std::min(rows, max_rows);

                    const uint8_t *raw = static_cast<const uint8_t *>(tensor->raw_data());

                    float d_min = 0.0f, d_max = 0.0f;
                    int16_t qs_min = 0, qs_max = 0;
                    int sat_count = 0;
                    int total_qs = 0;
                    bool first = true;

                    for (int r = 0; r < rows_to_scan; ++r)
                    {
                        for (int b = 0; b < blocks_per_row; ++b)
                        {
                            const size_t block_idx = static_cast<size_t>(r) * blocks_per_row + b;
                            const uint8_t *blk_ptr = raw + block_idx * block_bytes;

                            // All Q16 block types have: float d, int16_t qs[N], int32_t sum_qs
                            // d is at offset 0
                            float d;
                            std::memcpy(&d, blk_ptr, sizeof(float));

                            // qs starts at offset 4 (after float d)
                            const int16_t *qs = reinterpret_cast<const int16_t *>(blk_ptr + sizeof(float));

                            if (first)
                            {
                                d_min = d_max = d;
                                qs_min = qs_max = qs[0];
                                first = false;
                            }

                            d_min = std::min(d_min, d);
                            d_max = std::max(d_max, d);

                            for (size_t i = 0; i < block_elements; ++i)
                            {
                                const int16_t v = qs[i];
                                qs_min = std::min(qs_min, v);
                                qs_max = std::max(qs_max, v);
                                sat_count += (v == INT16_MIN || v == INT16_MAX) ? 1 : 0;
                                total_qs++;
                            }
                        }
                    }

                    LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx << " " << label
                                                               << " stats: rows_scanned=" << rows_to_scan
                                                               << " block_size=" << static_cast<int>(block_size)
                                                               << " d[min,max]=" << d_min << "," << d_max
                                                               << " qs[min,max]=" << qs_min << "," << qs_max
                                                               << " sat=" << sat_count << "/" << total_qs);
                };

                const int q_rows = params_.seq_len;
                const int kv_rows = effective_kv_len;
                const int q_dim_per_row = params_.n_heads * params_.head_dim;
                const int kv_dim_per_row = params_.n_kv_heads * params_.head_dim;

                // First block stats using tensor's actual block size
                const Q16BlockSize q_block_size = q_q16->q16_block_size();
                const uint8_t *q_raw = static_cast<const uint8_t *>(q_q16->raw_data());
                float q0_d;
                std::memcpy(&q0_d, q_raw, sizeof(float));
                const int16_t *q0_qs = reinterpret_cast<const int16_t *>(q_raw + sizeof(float));

                const Q16BlockSize k_block_size = k_q16->q16_block_size();
                const uint8_t *k_raw = static_cast<const uint8_t *>(k_q16->raw_data());
                float k0_d;
                std::memcpy(&k0_d, k_raw, sizeof(float));
                const int16_t *k0_qs = reinterpret_cast<const int16_t *>(k_raw + sizeof(float));

                const Q16BlockSize v_block_size = v_q16->q16_block_size();
                const uint8_t *v_raw = static_cast<const uint8_t *>(v_q16->raw_data());
                float v0_d;
                std::memcpy(&v0_d, v_raw, sizeof(float));
                const int16_t *v0_qs = reinterpret_cast<const int16_t *>(v_raw + sizeof(float));

                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " Q16_1 first scales (block_sizes: Q=" << static_cast<int>(q_block_size)
                                                           << ", K=" << static_cast<int>(k_block_size)
                                                           << ", V=" << static_cast<int>(v_block_size) << "):"
                                                           << " Q[0].d=" << q0_d
                                                           << " K[0].d=" << k0_d
                                                           << " V[0].d=" << v0_d);
                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " Q16_1 first qs: Q[0].qs[0]=" << q0_qs[0]
                                                           << " K[0].qs[0]=" << k0_qs[0]
                                                           << " V[0].qs[0]=" << v0_qs[0]);

                // Scan only the small prefill window (seq_len/kv_len are tiny in parity tests).
                dump_q16_stats_typed("Q", q_q16, q_rows, q_dim_per_row, /*max_rows=*/q_rows);
                dump_q16_stats_typed("K(cache)", k_q16, kv_rows, kv_dim_per_row, /*max_rows=*/kv_rows);
                dump_q16_stats_typed("V(cache)", v_q16, kv_rows, kv_dim_per_row, /*max_rows=*/kv_rows);

                // Diagnostic: compute a simple FP32 score/softmax summary from Q16_1 blocks
                // for the last query row (most sensitive) and head0.
                // This is for investigating late-layer attention divergence; it does not affect math.
                if (params_.seq_len == effective_kv_len && position_offset == 0 && params_.seq_len <= 16)
                {
                    const int head = 0;
                    const int row = params_.seq_len - 1;
                    const int group_size = params_.n_heads / params_.n_kv_heads;
                    const int kv_head = head / group_size;
                    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(params_.head_dim));

                    // Block size-aware element access for Q16 tensors
                    auto q16_at_raw = [](const uint8_t *raw_data, Q16BlockSize block_size, int total_dim, int row, int col) -> float
                    {
                        const size_t block_bytes = q16_block_size_bytes(block_size);
                        const int elems_per_block = q16_block_size_elements(block_size);
                        const int blocks_per_row = (total_dim + elems_per_block - 1) / elems_per_block;
                        const int b = col / elems_per_block;
                        const int i = col % elems_per_block;
                        const uint8_t *block_ptr = raw_data + (row * blocks_per_row + b) * block_bytes;
                        float d;
                        std::memcpy(&d, block_ptr, sizeof(float));
                        const int16_t *qs = reinterpret_cast<const int16_t *>(block_ptr + sizeof(float));
                        return d * static_cast<float>(qs[i]);
                    };

                    // Compute masked scores for kv positions [0..kv_len-1]
                    float max_score = -std::numeric_limits<float>::infinity();
                    float second_score = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;
                    for (int t = 0; t < effective_kv_len; ++t)
                    {
                        if (params_.causal && t > row)
                        {
                            continue;
                        }

                        float dot = 0.0f;
                        const int q_base = head * params_.head_dim;
                        const int k_base = kv_head * params_.head_dim;
                        for (int d = 0; d < params_.head_dim; ++d)
                        {
                            float q_val = q16_at_raw(q_raw, q_block_size, q_dim_per_row, row, q_base + d);
                            float k_val = q16_at_raw(k_raw, k_block_size, kv_dim_per_row, t, k_base + d);
                            dot += q_val * k_val;
                        }
                        const float score = dot * attention_scale;
                        if (score > max_score)
                        {
                            second_score = max_score;
                            max_score = score;
                            max_idx = t;
                        }
                        else if (score > second_score)
                        {
                            second_score = score;
                        }
                    }

                    // Softmax top weight estimate.
                    float weight_sum = 0.0f;
                    float max_weight = 0.0f;
                    for (int t = 0; t < effective_kv_len; ++t)
                    {
                        if (params_.causal && t > row)
                        {
                            continue;
                        }
                        float dot = 0.0f;
                        const int q_base = head * params_.head_dim;
                        const int k_base = kv_head * params_.head_dim;
                        for (int d = 0; d < params_.head_dim; ++d)
                        {
                            float q_val = q16_at_raw(q_raw, q_block_size, q_dim_per_row, row, q_base + d);
                            float k_val = q16_at_raw(k_raw, k_block_size, kv_dim_per_row, t, k_base + d);
                            dot += q_val * k_val;
                        }
                        const float score = dot * attention_scale;
                        const float w = std::exp(score - max_score);
                        weight_sum += w;
                        if (w > max_weight)
                        {
                            max_weight = w;
                        }
                    }

                    const float top_prob = (weight_sum > 0.0f) ? (max_weight / weight_sum) : 0.0f;
                    LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                               << " head0 row" << row
                                                               << " score_max=" << max_score
                                                               << " score_2nd=" << second_score
                                                               << " gap=" << (max_score - second_score)
                                                               << " argmax=" << max_idx
                                                               << " top_prob=" << top_prob);
                }
            }

            // ================================================================
            // LAYOUT-AWARE K/V ACCESS
            // ================================================================
            // Q16IntegerAttentionRef expects head-major layout:
            //   [n_kv_heads][kv_len][head_dim] -> block[h * kv_len + p]
            //
            // If KV cache is HEAD_MAJOR, use data directly.
            // If KV cache is POSITION_MAJOR, transpose on-the-fly.
            //
            const Q16BlockSize block_size = k_q16->q16_block_size();
            const size_t block_bytes = q16_block_size_bytes(block_size);
            const size_t block_elements = q16_block_size_elements(block_size);
            const int blocks_per_head = (params_.head_dim + static_cast<int>(block_elements) - 1) / static_cast<int>(block_elements);

            // Check if KV cache is already in HEAD_MAJOR layout
            const bool kv_is_head_major = params_.kv_cache &&
                                          params_.kv_cache->kv_layout() == TensorLayout::KV_HEAD_POS_DIM;

            // Storage for transposed data (only allocated if needed)
            std::vector<uint8_t> K_transposed_bytes;
            std::vector<uint8_t> V_transposed_bytes;

            // Use TYPE-SAFE Q16BlockPtr wrappers to ensure block type matches
            Q16BlockPtr K_ptr, V_ptr;

            if (kv_is_head_major)
            {
                // HEAD_MAJOR: use K/V data directly - already in correct layout
                // Create type-safe pointers based on actual block size
                switch (block_size)
                {
                case Q16BlockSize::BLOCK_64:
                    K_ptr = (k_q16->as_block_64());
                    V_ptr = (v_q16->as_block_64());
                    break;
                case Q16BlockSize::BLOCK_128:
                    K_ptr = (k_q16->as_block_128());
                    V_ptr = (v_q16->as_block_128());
                    break;
                case Q16BlockSize::BLOCK_32:
                    K_ptr = (k_q16->as_block_32());
                    V_ptr = (v_q16->as_block_32());
                    break;
                }
                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " HEAD_MAJOR KV cache: no transpose needed"
                                                           << " kv_len=" << effective_kv_len
                                                           << " block_size=" << static_cast<int>(block_size));
            }
            else
            {
                // POSITION_MAJOR: transpose from [kv_len][n_kv_heads][head_dim] to [n_kv_heads][kv_len][head_dim]
                const int total_blocks = effective_kv_len * params_.n_kv_heads * blocks_per_head;
                K_transposed_bytes.resize(total_blocks * block_bytes);
                V_transposed_bytes.resize(total_blocks * block_bytes);

                const uint8_t *K_src = static_cast<const uint8_t *>(k_q16->raw_data());
                const uint8_t *V_src = static_cast<const uint8_t *>(v_q16->raw_data());

                // Transpose: position-major [p][h] -> head-major [h][p]
                for (int h = 0; h < params_.n_kv_heads; ++h)
                {
                    for (int p = 0; p < effective_kv_len; ++p)
                    {
                        // Source index: position-major [p * n_kv_heads + h]
                        const size_t src_block_idx = (static_cast<size_t>(p) * params_.n_kv_heads + h) * blocks_per_head;
                        // Dest index: head-major [h * kv_len + p]
                        const size_t dst_block_idx = (static_cast<size_t>(h) * effective_kv_len + p) * blocks_per_head;

                        // Copy all blocks for this head/position
                        std::memcpy(K_transposed_bytes.data() + dst_block_idx * block_bytes,
                                    K_src + src_block_idx * block_bytes,
                                    blocks_per_head * block_bytes);
                        std::memcpy(V_transposed_bytes.data() + dst_block_idx * block_bytes,
                                    V_src + src_block_idx * block_bytes,
                                    blocks_per_head * block_bytes);
                    }
                }

                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " TRANSPOSE WORKAROUND: kv_len=" << effective_kv_len
                                                           << " n_kv_heads=" << params_.n_kv_heads
                                                           << " blocks_per_head=" << blocks_per_head
                                                           << " total_blocks=" << total_blocks);

                // Create type-safe pointers to transposed data
                switch (block_size)
                {
                case Q16BlockSize::BLOCK_64:
                    K_ptr = (reinterpret_cast<const Q16_1Block_64 *>(K_transposed_bytes.data()));
                    V_ptr = (reinterpret_cast<const Q16_1Block_64 *>(V_transposed_bytes.data()));
                    break;
                case Q16BlockSize::BLOCK_128:
                    K_ptr = (reinterpret_cast<const Q16_1Block_128 *>(K_transposed_bytes.data()));
                    V_ptr = (reinterpret_cast<const Q16_1Block_128 *>(V_transposed_bytes.data()));
                    break;
                case Q16BlockSize::BLOCK_32:
                    K_ptr = (reinterpret_cast<const Q16_1Block *>(K_transposed_bytes.data()));
                    V_ptr = (reinterpret_cast<const Q16_1Block *>(V_transposed_bytes.data()));
                    break;
                }
            }

            // Set type-safe K/V pointers
            q16_params.K = K_ptr;
            q16_params.V = V_ptr;

            // ================================================================
            // Q TRANSPOSE: Pipeline Q is [seq_len, n_heads, head_dim] (position-major)
            // Q16 kernel expects [n_heads, seq_len, head_dim] (head-major)
            // Need to transpose Q for prefill mode (seq_len > 1)
            // ================================================================
            const Q16BlockSize q_block_size = q_q16->q16_block_size();
            const size_t q_block_bytes = q16_block_size_bytes(q_block_size);
            const size_t q_block_elements = q16_block_size_elements(q_block_size);
            const int q_blocks_per_head = (params_.head_dim + static_cast<int>(q_block_elements) - 1) / static_cast<int>(q_block_elements);

            // Storage for transposed Q (only allocated if needed for prefill)
            std::vector<uint8_t> Q_transposed_bytes;
            Q16BlockPtr Q_ptr;

            if (params_.seq_len > 1)
            {
                // Prefill mode: need to transpose Q from [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim]
                const int total_q_blocks = params_.seq_len * params_.n_heads * q_blocks_per_head;
                Q_transposed_bytes.resize(total_q_blocks * q_block_bytes);

                const uint8_t *Q_src = static_cast<const uint8_t *>(q_q16->raw_data());

                // Transpose: position-major [p][h] -> head-major [h][p]
                for (int h = 0; h < params_.n_heads; ++h)
                {
                    for (int p = 0; p < params_.seq_len; ++p)
                    {
                        // Source index: position-major [p * n_heads + h]
                        const size_t src_block_idx = (static_cast<size_t>(p) * params_.n_heads + h) * q_blocks_per_head;
                        // Dest index: head-major [h * seq_len + p]
                        const size_t dst_block_idx = (static_cast<size_t>(h) * params_.seq_len + p) * q_blocks_per_head;

                        // Copy all blocks for this head/position
                        std::memcpy(Q_transposed_bytes.data() + dst_block_idx * q_block_bytes,
                                    Q_src + src_block_idx * q_block_bytes,
                                    q_blocks_per_head * q_block_bytes);
                    }
                }

                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " Q TRANSPOSE (prefill): seq_len=" << params_.seq_len
                                                           << " n_heads=" << params_.n_heads
                                                           << " blocks_per_head=" << q_blocks_per_head
                                                           << " total_blocks=" << total_q_blocks);

                // Create type-safe pointer to transposed Q data
                switch (q_block_size)
                {
                case Q16BlockSize::BLOCK_64:
                    Q_ptr = (reinterpret_cast<const Q16_1Block_64 *>(Q_transposed_bytes.data()));
                    break;
                case Q16BlockSize::BLOCK_128:
                    Q_ptr = (reinterpret_cast<const Q16_1Block_128 *>(Q_transposed_bytes.data()));
                    break;
                case Q16BlockSize::BLOCK_32:
                    Q_ptr = (reinterpret_cast<const Q16_1Block *>(Q_transposed_bytes.data()));
                    break;
                }
            }
            else
            {
                // Decode mode (seq_len=1): no transpose needed - Q is just [n_heads, head_dim]
                // which is equivalent to [n_heads, 1, head_dim]
                switch (q_block_size)
                {
                case Q16BlockSize::BLOCK_64:
                    Q_ptr = (q_q16->as_block_64());
                    break;
                case Q16BlockSize::BLOCK_128:
                    Q_ptr = (q_q16->as_block_128());
                    break;
                case Q16BlockSize::BLOCK_32:
                    Q_ptr = (q_q16->as_block_32());
                    break;
                }
                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " Q NO TRANSPOSE (decode): seq_len=" << params_.seq_len);
            }

            // Set Q pointer (transposed for prefill, direct for decode)
            q16_params.Q = Q_ptr;

            LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                       << " Q/K/V block_size=" << static_cast<int>(block_size)
                                                       << " Q.block_size=" << static_cast<int>(q16_params.Q.block_size)
                                                       << " head_dim=" << params_.head_dim);

            // Wo weights - get VNNI packed weights via KernelFactory
            // Check if Wo implements IINT8Unpackable (quantized formats)
            if (dynamic_cast<IINT8Unpackable *>(params_.Wo) != nullptr)
            {
                // Cast to TensorBase for KernelFactory
                auto *Wo_base = requireTensorBase(params_.Wo, "Wo");
                // Get VNNI-packed weights from cache (or pack on first call)
                auto *packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(Wo_base);
                q16_params.Wo_packed = packed;
                q16_params.Wo_fp32 = nullptr;
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER: Using VNNI-packed Wo weights from "
                          << params_.Wo->dtype_name() << " tensor");
            }
            else if (auto *wo_fp32 = dynamic_cast<FP32Tensor *>(params_.Wo))
            {
                // FP32 Wo weights (no packing needed)
                q16_params.Wo_packed = nullptr;
                q16_params.Wo_fp32 = wo_fp32->data();
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER: Using FP32 Wo weights");
            }
            else
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER: Unsupported Wo weight type: "
                          << (params_.Wo ? params_.Wo->dtype_name() : "null"));
                return false;
            }

            // Residual fusion: output tensor IS the residual (in-place read-modify-write)
            // Q16_INTEGER always uses fused residual path (validated above)
            // Residual uses 32-element Q16_1Block for compatibility with GEMM pipeline
            if (auto *out_q16 = dynamic_cast<Q16_1Tensor *>(params_.output))
            {
                // residual_in = residual_out = output tensor (in-place accumulation)
                // Use as_block_32() for typed access to 32-element blocks
                q16_params.residual_in = out_q16->as_block_32();
                q16_params.residual_out = out_q16->mutable_as_block_32();
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER: Residual fusion enabled (in-place Q16_1Block)");
            }
            else
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER with fuse_residual_add requires Q16_1 output, got "
                          << (params_.output ? params_.output->dtype_name() : "null"));
                return false;
            }

            // Dimensions
            q16_params.seq_len_q = params_.seq_len;
            q16_params.kv_len = effective_kv_len;
            q16_params.n_heads = params_.n_heads;
            q16_params.n_kv_heads = params_.n_kv_heads;
            q16_params.head_dim = params_.head_dim;
            q16_params.d_model = params_.d_model;

            // KV cache layout: HEAD_MAJOR uses sparse allocation with stride = max_seq_len
            // Dense/transposed data uses stride = kv_len (packed)
            if (kv_is_head_major && params_.kv_cache)
            {
                q16_params.kv_head_stride = params_.kv_cache->max_seq_len();
                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " HEAD_MAJOR sparse cache: kv_head_stride="
                                                           << q16_params.kv_head_stride
                                                           << " (max_seq_len), kv_len=" << effective_kv_len);
            }
            else
            {
                q16_params.kv_head_stride = 0; // Use default (kv_len)
            }

            // Debug metadata
            q16_params.layer_idx = params_.layer_idx;

            // Attention config
            q16_params.scale = 1.0f / std::sqrt(static_cast<float>(params_.head_dim));
            q16_params.causal = params_.causal;
            q16_params.position_offset = position_offset;

            // Snapshot buffers - FAIL FAST if snapshot capture is enabled but buffer is wrong type
            q16_params.scores_snapshot = nullptr;
            q16_params.context_snapshot = nullptr;
            q16_params.wo_output_snapshot = nullptr;
            q16_params.attention_residual_snapshot = nullptr;
#ifdef ENABLE_PIPELINE_SNAPSHOTS
            if (params_.context_snapshot)
            {
                auto *ctx_fp32 = dynamic_cast<FP32Tensor *>(params_.context_snapshot);
                LLAMINAR_ASSERT_CAST(ctx_fp32, "FP32Tensor",
                                     "context_snapshot (got " +
                                         std::string(params_.context_snapshot->dtype_name()) + ")");
                q16_params.context_snapshot = ctx_fp32->mutable_data();
                LOG_TRACE("[FusedAttentionWoStage] Q16_INTEGER: context_snapshot buffer set, ptr="
                          << q16_params.context_snapshot);
            }
            if (params_.attention_output_snapshot)
            {
                auto *out_fp32 = dynamic_cast<FP32Tensor *>(params_.attention_output_snapshot);
                LLAMINAR_ASSERT_CAST(out_fp32, "FP32Tensor",
                                     "attention_output_snapshot (got " +
                                         std::string(params_.attention_output_snapshot->dtype_name()) + ")");
                q16_params.wo_output_snapshot = out_fp32->mutable_data();
                LOG_TRACE("[FusedAttentionWoStage] Q16_INTEGER: wo_output_snapshot (attention_output) buffer set, ptr="
                          << q16_params.wo_output_snapshot);
            }
            if (params_.attention_residual_snapshot)
            {
                auto *res_fp32 = dynamic_cast<FP32Tensor *>(params_.attention_residual_snapshot);
                LLAMINAR_ASSERT_CAST(res_fp32, "FP32Tensor",
                                     "attention_residual_snapshot (got " +
                                         std::string(params_.attention_residual_snapshot->dtype_name()) + ")");
                q16_params.attention_residual_snapshot = res_fp32->mutable_data();
                LOG_TRACE("[FusedAttentionWoStage] Q16_INTEGER: attention_residual_snapshot buffer set, ptr="
                          << q16_params.attention_residual_snapshot);
            }
#endif

            // Wire K_head_scales for HybridQ16 K precision fix
            // This enables per-head K scale lookup in the attention kernel
            q16_params.k_head_scales = params_.K_head_scales;
            if (params_.K_head_scales)
            {
                LOG_DEBUG("[FusedAttentionWoStage] Layer " << params_.layer_idx
                                                           << " K_head_scales provided (K precision fix active)"
                                                           << " scales[0]=" << params_.K_head_scales[0]);
                // Debug: print all K scales for first layer
                if (params_.layer_idx == 0)
                {
                    const int n_k_scales = params_.kv_len * params_.n_kv_heads;
                    std::stringstream ss;
                    ss << "[FusedAttentionWoStage] Layer 0 ALL K_head_scales (" << n_k_scales << " total): ";
                    for (int i = 0; i < std::min(n_k_scales, 20); ++i)
                    {
                        ss << params_.K_head_scales[i] << " ";
                    }
                    LOG_DEBUG(ss.str());
                }
            }

            success = q16_kernel_->compute(q16_params);
        }
        else
        {
            // Q8_1 kernel (JIT/TILED/REFERENCE)
            // Cast ITensor* to TensorBase* for kernel interface
            auto *Q_base = requireTensorBase(params_.Q, "Q");
            auto *K_base = requireTensorBase(params_.K, "K");
            auto *V_base = requireTensorBase(params_.V, "V");
            auto *Wo_base = requireTensorBase(params_.Wo, "Wo");
            auto *output_base = asTensorBase(params_.output, "output");
            auto *context_snapshot_base = asTensorBase(params_.context_snapshot, "context_snapshot");

            success = kernel_->compute(
                Q_base,
                K_base,
                V_base,
                Wo_base,
                output_base,
                params_.seq_len,
                effective_kv_len,
                params_.causal,
                position_offset,
                context_snapshot_base);
        }

        if (!success)
        {
            LOG_ERROR("[FusedAttentionWoStage] Kernel compute() failed");
            return false;
        }

        LOG_DEBUG("[FusedAttentionWoStage] Execute complete");
        return true;
    }

    size_t FusedAttentionWoStage::estimatedFlops() const
    {
        // Attention FLOPs + Wo projection FLOPs
        // Attention: 2 * batch * n_heads * seq_len * kv_len * head_dim (QK^T)
        //          + 4 * batch * n_heads * seq_len * kv_len (softmax)
        //          + 2 * batch * n_heads * seq_len * kv_len * head_dim (scores @ V)
        // Wo: 2 * batch * seq_len * d_model * (n_heads * head_dim)
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        const size_t wo_flops = 2ULL * params_.batch_size * params_.seq_len *
                                params_.d_model * (params_.n_heads * params_.head_dim);
        return qk_flops + softmax_flops + sv_flops + wo_flops;
    }

    size_t FusedAttentionWoStage::estimatedMemoryBytes() const
    {
        // Fused kernel eliminates attention context intermediate
        // Main memory: Q, K, V, Wo weights, output
        // Scratch: internal scores/state (managed by kernel)
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool FusedAttentionWoStage::supportsBackend(ComputeBackendType backend) const
    {
        // Currently CPU-only (JIT backend uses AVX-512 VNNI)
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo FusedAttentionWoStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // =====================================================================
        // INPUT TENSORS - Q, K, V as they arrive at the attention kernel
        // =====================================================================

        // Q tensor: [seq_len][n_heads * head_dim]
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            info.addInput("Q", params_.Q, q_rows, q_cols);

            // Log Q tensor stats for debugging
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] Q tensor: type="
                      << static_cast<int>(params_.Q->native_type())
                      << " rows=" << q_rows << " cols=" << q_cols);
        }

        // K tensor: [kv_len][n_kv_heads * head_dim] - from KV cache
        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            info.addInput("K", params_.K, k_rows, k_cols);

            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] K tensor: type="
                      << static_cast<int>(params_.K->native_type())
                      << " rows=" << k_rows << " cols=" << k_cols);
        }

        // V tensor: [kv_len][n_kv_heads * head_dim] - from KV cache
        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            info.addInput("V", params_.V, v_rows, v_cols);

            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] V tensor: type="
                      << static_cast<int>(params_.V->native_type())
                      << " rows=" << v_rows << " cols=" << v_cols);
        }

        // =====================================================================
        // OUTPUT TENSORS - Context and attention snapshots
        // =====================================================================

        // Context snapshot: pre-Wo attention output (for debugging/parity testing)
        // This is ATTENTION_CONTEXT in pipeline terminology
        // Shape: [batch_size * seq_len, n_heads * head_dim] to match buffer allocation
        if (params_.context_snapshot)
        {
            const float *ctx_data = getSafeFp32Data(params_.context_snapshot);
            const size_t total_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] context_snapshot tensor type: "
                      << static_cast<int>(params_.context_snapshot->native_type())
                      << " ctx_data=" << (ctx_data ? "valid" : "NULL")
                      << " batch_size=" << params_.batch_size
                      << " seq_len=" << params_.seq_len
                      << " total_tokens=" << total_tokens
                      << " n_heads*head_dim=" << params_.n_heads * params_.head_dim);
            if (ctx_data)
            {
                info.addOutput("context",
                               ctx_data,
                               total_tokens,
                               params_.n_heads * params_.head_dim);
            }
        }
        else
        {
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] context_snapshot is NULL");
        }

        // Attention output snapshot: Wo projection result (before residual add)
        // This is ATTENTION_OUTPUT in pipeline terminology
        // Shape: [batch_size * seq_len, d_model] to match buffer allocation
        if (params_.attention_output_snapshot)
        {
            const float *out_snap_data = getSafeFp32Data(params_.attention_output_snapshot);
            const size_t total_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] attention_output_snapshot tensor type: "
                      << static_cast<int>(params_.attention_output_snapshot->native_type())
                      << " data=" << (out_snap_data ? "valid" : "NULL")
                      << " batch_size=" << params_.batch_size
                      << " seq_len=" << params_.seq_len
                      << " total_tokens=" << total_tokens
                      << " d_model=" << params_.d_model);
            if (out_snap_data)
            {
                info.addOutput("attention_output",
                               out_snap_data,
                               total_tokens,
                               params_.d_model);
            }
        }

        // Attention residual snapshot: after residual add (final attention block output)
        // This is ATTENTION_RESIDUAL in pipeline terminology
        // Shape: [batch_size * seq_len, d_model] to match buffer allocation
        if (params_.attention_residual_snapshot)
        {
            const float *res_snap_data = getSafeFp32Data(params_.attention_residual_snapshot);
            const size_t total_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] attention_residual_snapshot tensor type: "
                      << static_cast<int>(params_.attention_residual_snapshot->native_type())
                      << " data=" << (res_snap_data ? "valid" : "NULL")
                      << " batch_size=" << params_.batch_size
                      << " seq_len=" << params_.seq_len
                      << " total_tokens=" << total_tokens
                      << " d_model=" << params_.d_model);
            if (res_snap_data)
            {
                info.addOutput("attention_residual",
                               res_snap_data,
                               total_tokens,
                               params_.d_model);
            }
        }

        // Output: fused attention+Wo projection result (primary output tensor)
        // Note: For HybridQ16 with fuse_residual_add, this IS the residual (Q16_1)
        // but we can't expose raw data here - use snapshots instead
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                info.addOutput("output",
                               out_data,
                               params_.seq_len,
                               params_.d_model);
            }
        }

        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("backend", static_cast<int>(params_.backend));
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        return info;
    }

    StageBufferRequirements FusedAttentionWoStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q - [seq_len][n_heads * head_dim] with Q_SEQ_HEAD_DIM layout
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type, TensorLayout::Q_SEQ_HEAD_DIM);
        }

        // Input: K - [kv_len][n_kv_heads * head_dim] with KV_POS_HEAD_DIM layout
        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type, TensorLayout::KV_POS_HEAD_DIM);
        }

        // Input: V - [kv_len][n_kv_heads * head_dim] with KV_POS_HEAD_DIM layout
        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type, TensorLayout::KV_POS_HEAD_DIM);
        }

        // Input: Wo weight (no specific layout constraint)
        if (params_.Wo)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.Wo->native_type());
            reqs.addInput("Wo", params_.Wo->shape(), buf_type);
        }

        // Output: projected attention [seq_len][d_model] - generic 2D layout
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.d_model);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type, TensorLayout::ROW_MAJOR_2D);
        }

        return reqs;
    }

    // =========================================================================
    // Layout Expectation (Phase 3: Declarative Layout Validation)
    // =========================================================================

    verification::LayoutExpectation FusedAttentionWoStage::getLayoutExpectation() const
    {
        // Provide model dimensions for automatic layout validation
        // GraphExecutor uses this with getBufferRequirements().withLayout() declarations
        return verification::LayoutExpectation::forAttention(
            params_.head_dim,
            params_.n_heads,
            params_.n_kv_heads);
    }

} // namespace llaminar2
