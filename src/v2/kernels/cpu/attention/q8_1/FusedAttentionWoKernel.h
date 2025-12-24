/**
 * @file FusedAttentionWoKernel.h
 * @brief Pipeline-compatible wrapper for fused attention + Wo projection
 *
 * This kernel fuses attention computation with the Wo output projection to:
 * 1. Eliminate the context quantization round-trip (FP32 → Q8_1 → FP32)
 * 2. Improve cache locality (context stays in registers through projection)
 * 3. Reduce memory bandwidth (single pass over V and Wo)
 *
 * The kernel supports three backends:
 * - **Reference**: Pure C++ implementation (for correctness testing)
 * - **Tiled**: Cache-blocked implementation with L2/L3 aware tiling
 * - **JIT**: AVX-512 VNNI optimized code generated at runtime
 *
 * @author David Sanftenberg
 * @date December 2025
 */
#pragma once

#include "tensors/Tensors.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoRef.h"
#include "kernels/cpu/attention/q8_1/FusedAttentionWoTiled.h"
#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "kernels/KernelFactory.h"
#include "utils/DebugEnv.h"
#include "utils/CPUFeatures.h"
#include "utils/Logger.h"
#include <memory>
#include <cmath>
#include <array>
#include <mutex>
#include "execution/RuntimeConfig.h" // For FusedAttentionBackend enum

namespace llaminar2
{

    /**
     * @brief Pipeline-compatible fused attention + Wo projection kernel
     *
     * Usage in Qwen2Pipeline::attention_block:
     * ```cpp
     * // Replace separate attention + Wo GEMM:
     * // compute_attention(..., attn_output);
     * // project_row_parallel(attn_output, Wo, attn_proj, ...);
     *
     * // With fused kernel:
     * fused_attn_wo_->compute(Q, K, V, Wo, attn_proj, ...);
     * ```
     *
     * Input/Output precisions:
     * - Q, K, V: Q8_1 tensor blocks
     * - Wo: Q8_1 (preferred), FP32, FP16, or BF16
     * - Output: FP32
     */
    class FusedAttentionWoKernel
    {
    public:
        /**
         * @brief Configuration for the fused kernel
         */
        struct Config
        {
            int num_heads = 0;    ///< Number of query heads
            int num_kv_heads = 0; ///< Number of KV heads (GQA support)
            int head_dim = 64;    ///< Dimension per head
            int d_model = 0;      ///< Model dimension (num_heads * head_dim)
            FusedAttentionBackend backend = FusedAttentionBackend::JIT;
        };

        explicit FusedAttentionWoKernel(const Config &config)
            : config_(config), scale_(1.0f / std::sqrt(static_cast<float>(config.head_dim)))
        {
            LOG_DEBUG("FusedAttentionWoKernel created: heads=" << config.num_heads
                                                               << "/" << config.num_kv_heads << ", head_dim=" << config.head_dim
                                                               << ", d_model=" << config.d_model
                                                               << ", backend=" << static_cast<int>(config.backend));
        }

        /**
         * @brief Compute fused attention + Wo projection
         *
         * @param Q Query tensor [seq_len_q, num_heads * head_dim] (Q8_1)
         * @param K Key tensor [seq_len_kv, num_kv_heads * head_dim] (Q8_1)
         * @param V Value tensor [seq_len_kv, num_kv_heads * head_dim] (Q8_1)
         * @param Wo Output projection weight [d_model, num_heads * head_dim]
         * @param output Output tensor [seq_len_q, d_model] (FP32)
         * @param seq_len_q Query sequence length
         * @param seq_len_kv Key/Value sequence length
         * @param causal Whether to apply causal masking
         * @param position_offset Position offset for causal mask (decode mode)
         * @return true on success
         */
        bool compute(
            TensorBase *Q,
            TensorBase *K,
            TensorBase *V,
            TensorBase *Wo,
            TensorBase *output,
            int seq_len_q,
            int seq_len_kv,
            bool causal = true,
            int position_offset = 0)
        {
            const auto &env = debugEnv();

            // Validate Q8_1 input tensors
            auto *Q_q8 = dynamic_cast<Q8_1Tensor *>(Q);
            auto *K_q8 = dynamic_cast<Q8_1Tensor *>(K);
            auto *V_q8 = dynamic_cast<Q8_1Tensor *>(V);

            if (!Q_q8 || !K_q8 || !V_q8)
            {
                LOG_ERROR("FusedAttentionWoKernel requires Q8_1 input tensors");
                return false;
            }

            // Determine Wo weight type
            llaminar::v2::kernels::microkernels::WoWeightType wo_type;
            const void *wo_data = nullptr;

            if (auto *wo_q8 = dynamic_cast<Q8_1Tensor *>(Wo))
            {
                // For the JIT backend we always assume / require the packed VNNI path.
                // Reference/Tiled backends require raw Q8_1 blocks.
                if (config_.backend == FusedAttentionBackend::JIT)
                {
                    wo_type = llaminar::v2::kernels::microkernels::WoWeightType::Q8_1_VNNI_PACKED;
                    wo_data = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(wo_q8);
                }
                else
                {
                    wo_type = llaminar::v2::kernels::microkernels::WoWeightType::Q8_1;
                    wo_data = wo_q8->q8_1_blocks();
                }
            }
            else if (auto *wo_fp32 = dynamic_cast<FP32Tensor *>(Wo))
            {
                wo_type = llaminar::v2::kernels::microkernels::WoWeightType::FP32;
                wo_data = wo_fp32->data();
            }
            else if (auto *wo_fp16 = dynamic_cast<FP16Tensor *>(Wo))
            {
                wo_type = llaminar::v2::kernels::microkernels::WoWeightType::FP16;
                wo_data = wo_fp16->data();
            }
            else if (auto *wo_bf16 = dynamic_cast<BF16Tensor *>(Wo))
            {
                wo_type = llaminar::v2::kernels::microkernels::WoWeightType::BF16;
                wo_data = wo_bf16->data();
            }
            else
            {
                LOG_ERROR("FusedAttentionWoKernel: Unsupported Wo weight type");
                return false;
            }

            // Validate output tensor
            auto *out_fp32 = dynamic_cast<FP32Tensor *>(output);
            if (!out_fp32)
            {
                LOG_ERROR("FusedAttentionWoKernel requires FP32 output tensor");
                return false;
            }

            // Build params structure
            // Q8_1Block is now unified via microkernels::Q8_1Block = llaminar2::Q8_1Block
            llaminar::v2::kernels::FusedAttentionWoParams params;
            params.Q = Q_q8->q8_1_blocks();
            params.K = K_q8->q8_1_blocks();
            params.V = V_q8->q8_1_blocks();
            params.Wo = wo_data;
            params.wo_type = wo_type;
            params.output = out_fp32->mutable_data();
            params.batch_size = 1;
            params.kv_seq_lens = nullptr;
            params.position_offsets = nullptr;
            params.seq_len = seq_len_q;
            params.kv_seq_len = seq_len_kv;
            params.num_heads = config_.num_heads;
            params.num_kv_heads = config_.num_kv_heads;
            params.head_dim = config_.head_dim;
            params.d_model = config_.d_model;
            params.scale = scale_;
            params.causal = causal;
            params.position_offset = position_offset;

            // Dispatch to appropriate backend
            switch (config_.backend)
            {
            case FusedAttentionBackend::REFERENCE:
                return llaminar::v2::kernels::FusedAttentionWoRef::execute(params);

            case FusedAttentionBackend::TILED:
                return llaminar::v2::kernels::FusedAttentionWoTiled::execute(params);

            case FusedAttentionBackend::JIT:
                return execute_jit(params);

            default:
                LOG_ERROR("Unknown FusedAttentionWoKernel backend");
                return false;
            }
        }

        /**
         * @brief Compute with KV cache support (asymmetric Q/KV lengths)
         *
         * For decode mode: Q has length 1, K/V have full cached context.
         */
        bool compute_with_kv_cache(
            TensorBase *Q,
            TensorBase *K_cache, // Full KV cache
            TensorBase *V_cache, // Full KV cache
            TensorBase *Wo,
            TensorBase *output,
            int seq_len_q,    // Typically 1 for decode
            int kv_cache_len, // Total cached tokens
            bool causal = true,
            int position_offset = 0)
        {
            // position_offset = kv_cache_len - 1 for single token decode
            return compute(Q, K_cache, V_cache, Wo, output,
                           seq_len_q, kv_cache_len, causal, position_offset);
        }

        /**
         * @brief Get the configured backend
         */
        FusedAttentionBackend backend() const { return config_.backend; }

        /**
         * @brief Get configuration
         */
        const Config &config() const { return config_; }

    private:
        Config config_;
        float scale_;

        struct JitKernelSlots
        {
            // Decode: cache per (work_size, tile_width) because the JIT config hash includes both.
            // This lets us pick KV4 vs KV8 without recompiling or thrashing a single slot.
            std::array<std::array<std::unique_ptr<llaminar::v2::kernels::jit::JitFusedAttentionWo>, 2>, 3> decode{};
            std::unique_ptr<llaminar::v2::kernels::jit::JitFusedAttentionWo> prefill;
        };

        mutable std::mutex jit_slots_mutex_;
        mutable std::array<JitKernelSlots, 5> jit_slots_{}; // indexed by static_cast<int>(WoFormat)

        /**
         * @brief Execute using JIT kernel
         */
        bool execute_jit(const llaminar::v2::kernels::FusedAttentionWoParams &params)
        {
            using namespace llaminar::v2::kernels::jit;

            auto work_size_index = [](WorkSizeClass w) -> size_t
            {
                switch (w)
                {
                case WorkSizeClass::SMALL:
                    return 0;
                case WorkSizeClass::LARGE:
                    return 1;
                case WorkSizeClass::XL:
                    return 2;
                default:
                    return 0;
                }
            };
            auto tile_width_index = [](JitAttentionConfig::Fa2TileWidth t) -> size_t
            {
                return (t == JitAttentionConfig::Fa2TileWidth::KV8) ? 1 : 0;
            };

            // This JIT kernel relies on AVX-512 + VNNI.
            // If unavailable, fall back to the tiled kernel.
            if (!cpu_supports_avx512_vnni())
            {
                static bool logged = false;
                if (!logged)
                {
                    LOG_WARN("FusedAttentionWoKernel: AVX512-VNNI not available, falling back to TILED backend");
                    logged = true;
                }
                return llaminar::v2::kernels::FusedAttentionWoTiled::execute(params);
            }

            // Map WoWeightType to WoFormat
            WoFormat wo_format;
            switch (params.wo_type)
            {
            case llaminar::v2::kernels::microkernels::WoWeightType::FP32:
                wo_format = WoFormat::FP32;
                break;
            case llaminar::v2::kernels::microkernels::WoWeightType::FP16:
                wo_format = WoFormat::FP16;
                break;
            case llaminar::v2::kernels::microkernels::WoWeightType::BF16:
                wo_format = WoFormat::BF16;
                break;
            case llaminar::v2::kernels::microkernels::WoWeightType::Q8_1:
                wo_format = WoFormat::Q8_1;
                break;
            case llaminar::v2::kernels::microkernels::WoWeightType::Q8_1_VNNI_PACKED:
                wo_format = WoFormat::Q8_1_VNNI_PACKED;
                break;
            default:
                LOG_ERROR("JIT kernel does not support Wo type: " << static_cast<int>(params.wo_type));
                return false;
            }

            // Create JIT config
            JitAttentionConfig jit_config;
            jit_config.head_dim = params.head_dim;
            jit_config.num_heads = params.num_heads;
            jit_config.num_kv_heads = params.num_kv_heads;
            jit_config.wo_format = wo_format;
            jit_config.batch_size = params.batch_size;
            jit_config.causal = params.causal;
            jit_config.use_fa2_tiling = true; // Always use FA2 path

            // Explicitly specialize by mode (decode vs prefill) independent of batch_size.
            // In this pipeline kernel, batch_size is typically 1 for both decode and prefill.
            jit_config.mode = (params.seq_len <= 1)
                                  ? llaminar::v2::kernels::jit::AttentionMode::DECODE
                                  : llaminar::v2::kernels::jit::AttentionMode::PREFILL;

            // Work-size bucketing: compile two decode kernels (SMALL vs LARGE) and swap.
            // Prefill currently uses a single generic bucket.
            if (jit_config.mode == llaminar::v2::kernels::jit::AttentionMode::DECODE)
            {
                // Thresholds tuned from microbench sweeps (Dec 2024):
                //   kv=512:  LARGE/t8 wins
                //   kv=2048: SMALL/t8 wins
                //   kv=4096: LARGE/t4 wins
                //   kv=8192: SMALL/t4 wins (slight edge)
                // Pattern: alternates LARGE→SMALL→LARGE→SMALL as KV grows.
                // Simplified: short contexts (≤1024) use LARGE; mid (1024-4096) use SMALL
                // except 4096 which prefers LARGE; long (>4096) use SMALL again.
                if (params.kv_seq_len > 8192)
                {
                    jit_config.work_size = WorkSizeClass::XL;
                }
                else if (params.kv_seq_len > 4096)
                {
                    // kv 4097-8192: SMALL wins at 8192
                    jit_config.work_size = WorkSizeClass::SMALL;
                }
                else if (params.kv_seq_len > 2048)
                {
                    // kv 2049-4096: LARGE wins at 4096
                    jit_config.work_size = WorkSizeClass::LARGE;
                }
                else if (params.kv_seq_len > 1024)
                {
                    // kv 1025-2048: SMALL wins at 2048
                    jit_config.work_size = WorkSizeClass::SMALL;
                }
                else
                {
                    // kv <= 1024 (incl 512): LARGE wins
                    jit_config.work_size = WorkSizeClass::LARGE;
                }

                // Decode-only lever: KV8 helps up to ~4k KV; KV4 wins at 8k+.
                // Sweep results (Dec 2024):
                //   kv=512:  t8 wins
                //   kv=2048: t8 wins
                //   kv=4096: t8 wins
                //   kv=8192: t4 wins
                // Use t8 for kv <= 4096, t4 for kv > 4096.
                jit_config.fa2_tile_width =
                    (params.kv_seq_len <= 4096)
                        ? JitAttentionConfig::Fa2TileWidth::KV8
                        : JitAttentionConfig::Fa2TileWidth::KV4;
            }
            else
            {
                jit_config.work_size = WorkSizeClass::SMALL;
                jit_config.fa2_tile_width = JitAttentionConfig::Fa2TileWidth::KV4;
            }

            // Cache the constructed JitFusedAttentionWo object per (wo_format, mode, work_size)
            // so we don't pay the global cache mutex on every token.
            auto &slots = jit_slots_[static_cast<int>(wo_format)];
            std::unique_ptr<JitFusedAttentionWo> *slot_ptr = nullptr;
            if (jit_config.mode == llaminar::v2::kernels::jit::AttentionMode::PREFILL)
            {
                slot_ptr = &slots.prefill;
            }
            else
            {
                slot_ptr = &slots.decode[work_size_index(jit_config.work_size)][tile_width_index(jit_config.fa2_tile_width)];
            }

            {
                std::lock_guard<std::mutex> lock(jit_slots_mutex_);
                if (!(*slot_ptr))
                {
                    *slot_ptr = std::make_unique<JitFusedAttentionWo>(jit_config);
                }
            }

            (*slot_ptr)->compute(
                params.Q,
                params.K,
                params.V,
                params.Wo,
                params.output,
                params.seq_len,
                params.kv_seq_len,
                params.scale,
                params.position_offset);

            return true;
        }
    };

} // namespace llaminar2
