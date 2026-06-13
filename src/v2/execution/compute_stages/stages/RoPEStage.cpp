/**
 * @file RoPEStage.cpp
 * @brief Implementation of RoPEStage
 */

#include "RoPEStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/cpu/primitives/RoPEPrimitives.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include <cstring>

namespace llaminar2
{
    // Helper to convert FP16 scale to FP32 for debug logging
    static inline float debug_fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0)
        {
            if (mant == 0)
                return sign ? -0.0f : 0.0f;
            // Denormal
            float f = mant / 1024.0f;
            f *= (1.0f / 16384.0f); // 2^-14
            return sign ? -f : f;
        }
        if (exp == 31)
        {
            return mant ? NAN : (sign ? -INFINITY : INFINITY);
        }
        float f = 1.0f + mant / 1024.0f;
        f *= std::pow(2.0f, static_cast<float>(exp) - 15);
        return sign ? -f : f;
    }

    // =============================================================================
    // RoPEStage Implementation (Type-Safe via KernelFactory)
    // =============================================================================

    RoPEStage::RoPEStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    ITensorRoPE *RoPEStage::getOrCreateStageKernel(TensorBase *Q_base)
    {
        if (!Q_base)
            return nullptr;

        const int tensor_type = static_cast<int>(Q_base->native_type());
        if (!owned_kernel_ || cached_kernel_tensor_type_ != tensor_type)
        {
            try
            {
                owned_kernel_ = llaminar::v2::kernels::KernelFactory::createRoPE(
                    Q_base,
                    llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id),
                    params_.device_id.ordinal);
                cached_kernel_ = owned_kernel_.get();
                cached_kernel_tensor_type_ = tensor_type;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[RoPEStage] Failed to create stage-owned RoPE kernel for type "
                          << Q_base->dtype_name() << ": " << e.what());
                owned_kernel_.reset();
                cached_kernel_ = nullptr;
                cached_kernel_tensor_type_ = -1;
                return nullptr;
            }
        }

        return cached_kernel_;
    }

    bool RoPEStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "RoPEStage"))
        {
            return false;
        }

        if (!ensureRequiredPointers("RoPEStage", {
                                                     {"Q", params_.Q},
                                                 }))
        {
            return false;
        }

        if (params_.device_id.is_gpu() && !gpuStream())
        {
            LOG_ERROR("[RoPEStage] GPU RoPE requires an explicit non-null stage stream");
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU operations
        auto *Q_base = requireTensorBasePtr(params_.Q, "Q");
        if (!Q_base)
        {
            LOG_ERROR("[RoPEStage] GPU tensors not yet supported");
            return false;
        }

        // Get seq_len: use explicit param if set (for pre-allocated buffers), else from tensor
        // This is critical for decode where buffer is [max_seq_len, dim] but we only process 1 token
        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(Q_base->rows());

        // Detect Hybrid mode: Q8_1 input with FP32 output buffers
        const bool hybrid_mode = (params_.Q_out != nullptr) &&
                                 (Q_base->native_type() == TensorType::Q8_1) &&
                                 (params_.Q_out->native_type() == TensorType::FP32);

        // Detect HybridQ16 mode: Q8_1 Q input with Q16_1 Q output
        const bool hybrid_q16_mode = (params_.Q_out != nullptr) &&
                                     (Q_base->native_type() == TensorType::Q8_1) &&
                                     (params_.Q_out->native_type() == TensorType::Q16_1);

        // Detect K precision fix mode: K input is Q16_1 (from GEMM, not Q8_1)
        // This is the K precision fix for HybridQ16 where GEMM outputs K as Q16_1
        const bool k_is_q16_1 = (params_.K != nullptr) &&
                                (params_.K->native_type() == TensorType::Q16_1);

        LOG_DEBUG("[RoPEStage] Execute: seq_len=" << seq_len
                                                  << " n_heads=" << params_.n_heads
                                                  << " n_kv_heads=" << params_.n_kv_heads
                                                  << " head_dim=" << params_.head_dim
                                                  << " pos_offset=" << params_.pos_offset
                                                  << " partial_rotary_factor=" << params_.partial_rotary_factor
                                                  << " Q_type=" << Q_base->dtype_name()
                                                  << " K_type=" << (params_.K ? params_.K->dtype_name() : "nullptr")
                                                  << " hybrid_mode=" << (hybrid_mode ? "true" : "false")
                                                  << " hybrid_q16_mode=" << (hybrid_q16_mode ? "true" : "false")
                                                  << " k_is_q16_1=" << (k_is_q16_1 ? "true" : "false"));

        // Compute the rotary prefix length, but keep head_dim as the physical
        // per-head stride when invoking kernels. Partial RoPE rotates only the
        // first rotary_dim elements; the rest of each full-width head is passed
        // through unchanged.
        const int rotary_dim = static_cast<int>(
            static_cast<float>(params_.head_dim) * params_.partial_rotary_factor);

        // Get or create the stage-owned kernel so graph-capture metadata is
        // scoped to this RoPE stage instead of shared across graph nodes.
        auto *kernel = getOrCreateStageKernel(Q_base);

        if (!kernel)
        {
            LOG_ERROR("[RoPEStage] Failed to create RoPE kernel for type "
                      << Q_base->dtype_name());
            return false;
        }

        // Thread GPU stream for graph capture
        bindStageStream(kernel);

        // Use provided position_ids if available (for batched execution with per-token positions).
        // Otherwise, pass nullptr to GPU kernels to keep the contiguous position path:
        // positions are computed on device from pos_offset + seq_idx, and pos_offset
        // is pre-uploaded by updateDynamicParams() before graph capture/replay.
        const int *position_ids_ptr = params_.position_ids;

        // CPU paths need explicit position_ids whenever pos_offset != 0.
        // GPU kernels have a contiguous pos_offset/device-param path, and
        // manufacturing a host position_ids array would force an H2D upload.
        if (!params_.device_id.is_gpu() && !position_ids_ptr && seq_len > 0 && params_.pos_offset != 0)
        {
            position_ids_cache_.resize(static_cast<size_t>(seq_len));
            for (int i = 0; i < seq_len; ++i)
            {
                position_ids_cache_[static_cast<size_t>(i)] = params_.pos_offset + i;
            }
            position_ids_ptr = position_ids_cache_.data();
        }

        if (gpuStream() != nullptr && position_ids_ptr != nullptr)
        {
            // Explicit non-contiguous/batched position ids must use a stable
            // host pointer. Contiguous GPU positions intentionally stay null.
            if (seq_len > 0)
            {
                if (position_ids_ptr != position_ids_cache_.data())
                {
                    position_ids_cache_.resize(static_cast<size_t>(seq_len));
                    std::memcpy(position_ids_cache_.data(), position_ids_ptr,
                                static_cast<size_t>(seq_len) * sizeof(int));
                }
                position_ids_ptr = position_ids_cache_.data();
            }
        }
        // NOTE: When position_ids_ptr is nullptr, the kernel uses pos_offset
        // to compute contiguous positions. Do NOT generate a host array here.

        const int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;

        // Cast K and output tensors for kernel calls
        auto *K_base = asTensorBasePtr(params_.K, "K");
        auto *Q_out_base = asTensorBasePtr(params_.Q_out, "Q_out");
        auto *K_out_base = asTensorBasePtr(params_.K_out, "K_out");

        // Hybrid mode: use apply_q8_1_to_fp32() for Q8_1 → FP32 with no requantization
        if (hybrid_mode)
        {
            return kernel->apply_q8_1_to_fp32(
                Q_base,
                K_base,
                Q_out_base,
                K_out_base,
                position_ids_ptr,
                seq_len,
                params_.n_heads,
                n_kv_heads,
                params_.head_dim,
                params_.theta_base,
                params_.mpi_ctx,
                params_.device_id.toKernelDeviceIndex());
        }

        // HybridQ16 mode with K precision fix: Q=Q8_1→Q16_1, K=Q16_1→Q16_1 (dynamic scale)
        // This is the new path for the K precision fix where GEMM outputs K as Q16_1
        if (hybrid_q16_mode && k_is_q16_1)
        {
            const Q16BlockSize block_size = optimal_q16_block_size(params_.head_dim);
            LOG_DEBUG("[RoPEStage] Using HybridQ16 K precision fix mode: Q8_1→Q16_1 (fixed), Q16_1→Q16_1 (dynamic)");
            LOG_DEBUG("[RoPEStage] block_size=" << static_cast<int>(block_size)
                                                << " kv_cache_scale_k=" << params_.kv_cache_scale_k);

            // Debug: Check K_in state from GEMM (TRACE level only - fp32_data() is expensive)
            if (Logger::getInstance().shouldLog(LogLevel::TRACE))
            {
                auto *k16_in = dynamic_cast<Q16_1Tensor *>(K_base);
                if (k16_in)
                {
                    LOG_TRACE("[RoPEStage] K_in block_size=" << static_cast<int>(k16_in->block_size())
                                                             << " shape=[" << k16_in->shape()[0] << "," << k16_in->shape()[1] << "]");
                    const float *k_fp32 = k16_in->fp32_data();
                    if (k_fp32)
                    {
                        float max_val = 0.0f;
                        int nonzero_count = 0;
                        for (int i = 0; i < std::min(128, static_cast<int>(k16_in->numel())); ++i)
                        {
                            if (k_fp32[i] != 0.0f)
                                nonzero_count++;
                            max_val = std::max(max_val, std::fabs(k_fp32[i]));
                        }
                        LOG_TRACE("[RoPEStage] K_in[0:128] nonzero=" << nonzero_count
                                                                     << " max_abs=" << max_val << " fp32[0]=" << k_fp32[0]);
                    }
                    // Debug: Check actual block d values
                    const auto *k_typed = k16_in->typed_data();
                    if (k_typed)
                    {
                        const int blocks_per_head = params_.head_dim / 64; // Q16_1Block_64
                        float max_d = 0.0f;
                        for (int b = 0; b < std::min(blocks_per_head, 4); ++b)
                        {
                            max_d = std::max(max_d, std::fabs(k_typed[b].d));
                        }
                        LOG_TRACE("[RoPEStage] K_in block d values: d[0]=" << k_typed[0].d
                                                                           << " d[1]=" << (blocks_per_head > 1 ? k_typed[1].d : 0.0f)
                                                                           << " max_d=" << max_d);
                    }
                }
            }

            bool success = kernel->apply_mixed_q8_k16_to_q16(
                Q_base,                // Q8_1 Q input
                K_base,                // Q16_1 K input (from GEMM!)
                Q_out_base,            // Q16_1 Q output
                K_out_base,            // Q16_1 K output
                params_.K_head_scales, // Output: per-head K scales [seq_len * n_kv_heads]
                block_size,
                position_ids_ptr,
                seq_len,
                params_.n_heads,
                n_kv_heads,
                params_.head_dim,
                params_.theta_base,
                params_.kv_cache_scale_k, // Fixed scale for Q output
                params_.mpi_ctx,
                params_.device_id.toKernelDeviceIndex());

            if (success)
            {
                auto *q16_out = dynamic_cast<Q16_1Tensor *>(Q_out_base);
                auto *k16_out = dynamic_cast<Q16_1Tensor *>(K_out_base);
                if (q16_out && q16_out->typed_data())
                {
                    const auto &blk0 = q16_out->typed_data()[0];
                    LOG_DEBUG("[RoPEStage] K-fix Q_out[0].d=" << blk0.d << " qs[0]=" << blk0.qs[0]);
                }
                if (k16_out && k16_out->typed_data())
                {
                    const auto &blk0 = k16_out->typed_data()[0];
                    LOG_DEBUG("[RoPEStage] K-fix K_out[0].d=" << blk0.d << " qs[0]=" << blk0.qs[0]);
                }
            }
            return success;
        }

        // HybridQ16 mode (standard): use apply_q8_1_to_q16_fixed_scale() for Q8_1 → Q16_1 with fixed scale
        // This ensures Q and K have the same scale as V (kv_cache_scale_k / 32767), enabling integer attention
        if (hybrid_q16_mode)
        {
            // Use optimal block size matching the attention kernel (1 block per head for head_dim=64/128)
            const Q16BlockSize block_size = optimal_q16_block_size(params_.head_dim);
            LOG_DEBUG("[RoPEStage] Using HybridQ16 mode: Q8_1 → Q16_1 with fixed scale "
                      << params_.kv_cache_scale_k
                      << ", block_size=" << static_cast<int>(block_size));

            // Debug: check INPUT Q8_1 data
            auto *q8_in = dynamic_cast<Q8_1Tensor *>(Q_base);
            if (q8_in && q8_in->typed_data())
            {
                const auto &blk0 = q8_in->typed_data()[0];
                LOG_DEBUG("[RoPEStage] Q8_1 INPUT blk0.d=" << debug_fp16_to_fp32(blk0.d)
                                                           << " qs[0]=" << static_cast<int>(blk0.qs[0])
                                                           << " qs[1]=" << static_cast<int>(blk0.qs[1]));
            }

            bool success = kernel->apply_q8_1_to_q16_fixed_scale(
                Q_base,
                K_base,
                Q_out_base,
                K_out_base,
                block_size, // Match attention kernel's block size for head_dim
                position_ids_ptr,
                seq_len,
                params_.n_heads,
                n_kv_heads,
                params_.head_dim,
                params_.theta_base,
                params_.kv_cache_scale_k, // Fixed scale for integer attention
                params_.mpi_ctx,
                params_.device_id.toKernelDeviceIndex());

            // Debug: verify RoPE Q16_1 output
            if (success)
            {
                auto *q16_out = dynamic_cast<Q16_1Tensor *>(Q_out_base);
                if (q16_out && q16_out->typed_data())
                {
                    const auto &blk0 = q16_out->typed_data()[0];
                    LOG_DEBUG("[RoPEStage] HybridQ16 Q_out[0].d=" << blk0.d
                                                                  << " qs[0]=" << blk0.qs[0]
                                                                  << " Q_out ptr=" << static_cast<const void *>(Q_out_base)
                                                                  << " typed_data ptr=" << static_cast<const void *>(q16_out->typed_data()));
                }
            }
            return success;
        }

        // Standard path: Apply RoPE via kernel's apply_tensor method (in-place)
        //
        // When position_ids is nullptr (contiguous positions), the kernel computes
        // positions on-the-fly on GPU (pos = pos_offset + seq_idx), avoiding a
        // synchronous hipMemcpy that would drain the entire GPU pipeline at every
        // layer during prefill. The contiguous path also fuses Q+K into a single
        // kernel launch for better launch efficiency.
        //
        // When skip_k is true (RoPE-on-read mode), K is stored pre-RoPE in the
        // KV cache and RoPE will be fused into the attention dequant path.
        auto *K_for_rope = (params_.skip_k) ? nullptr : K_base;

        return kernel->apply_tensor(
            Q_base,
            K_for_rope,
            position_ids_ptr,
            seq_len,
            params_.n_heads,
            n_kv_heads,
            params_.head_dim,
            params_.theta_base,
            params_.mpi_ctx,
            params_.device_id.toKernelDeviceIndex(),
            params_.pos_offset,
            rotary_dim);
    }

    size_t RoPEStage::estimatedFlops() const
    {
        if (!params_.Q)
            return 0;

        const int seq_len = static_cast<int>(params_.Q->rows());
        // Per position per head: head_dim/2 rotations, each ~10 FLOPs (sin, cos, 4 muls, 2 adds)
        size_t flops = static_cast<size_t>(10) * seq_len * params_.n_heads * (params_.head_dim / 2);
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            flops += static_cast<size_t>(10) * seq_len * n_kv_heads * (params_.head_dim / 2);
        }
        return flops;
    }

    size_t RoPEStage::estimatedMemoryBytes() const
    {
        if (!params_.Q)
            return 0;

        const int seq_len = static_cast<int>(params_.Q->rows());
        size_t bytes = static_cast<size_t>(2) * seq_len * params_.n_heads *
                       params_.head_dim * sizeof(float); // Q read + write
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            bytes += static_cast<size_t>(2) * seq_len * n_kv_heads * params_.head_dim * sizeof(float);
        }
        return bytes;
    }

    bool RoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    bool RoPEStage::prepareGraphLaunch(IDeviceContext *ctx, void *stream)
    {
        (void)ctx;
        if (!params_.device_id.is_gpu())
            return true;
        if (!stream)
        {
            LOG_ERROR("[RoPEStage] GPU graph launch preparation requires an explicit non-null stream");
            return false;
        }
        if (!params_.Q)
        {
            LOG_ERROR("[RoPEStage] Cannot prepare graph launch without Q tensor");
            return false;
        }

        auto *Q_base = requireTensorBasePtr(params_.Q, "Q");
        if (!Q_base)
            return false;

        auto *kernel = getOrCreateStageKernel(Q_base);
        if (!kernel)
        {
            LOG_ERROR("[RoPEStage] Failed to prepare RoPE kernel before graph launch");
            return false;
        }

        /*
         * Graph capture must record only RoPE compute kernels.  Position
         * metadata is mutable between launches, so upload it to the
         * workspace-owned device buffer on the exact stream the capture
         * controller is about to use.  This hook also runs before graph replay,
         * keeping scalar contiguous positions and explicit request-batch
         * position rows fresh without rebuilding the graph.
         */
        setGPUStream(stream);
        kernel->setGPUStream(stream);
        if (params_.position_ids && params_.seq_len > 0)
        {
            kernel->setDynamicPositionIds(params_.position_ids, params_.seq_len);
        }
        else
        {
            kernel->setDynamicPosOffset(params_.pos_offset);
        }

        return true;
    }

    StageDumpInfo RoPEStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (!params_.Q)
            return info;

        // Use explicit seq_len if provided, otherwise derive from tensor
        const int seq_len = (params_.seq_len > 0) ? params_.seq_len : static_cast<int>(params_.Q->rows());

        // Detect Hybrid mode variants with separate output buffers.
        // Hybrid:      Q8_1 input -> FP32 output buffers
        // HybridQ16:   Q8_1 input -> Q16_1 output buffers
        const bool hybrid_mode = (params_.Q_out != nullptr) &&
                                 (params_.Q->native_type() == TensorType::Q8_1) &&
                                 (params_.Q_out->native_type() == TensorType::FP32);

        const bool hybrid_q16_mode = (params_.Q_out != nullptr) &&
                                     (params_.Q->native_type() == TensorType::Q8_1) &&
                                     (params_.Q_out->native_type() == TensorType::Q16_1);

        const bool separate_output_mode = hybrid_mode || hybrid_q16_mode;

        // OPTIMIZATION: For FP32 in-place RoPE (standard CUDA path), use tensor-based API
        // to avoid GPU->host sync. The sync is deferred to ensureOutputsOnHost().
        const bool is_fp32_inplace = !separate_output_mode &&
                                     params_.Q->native_type() == TensorType::FP32;

        if (is_fp32_inplace)
        {
            // Fast path: tensor-based addInput/addOutput (no GPU sync)
            info.addInput("Q", params_.Q, seq_len, params_.n_heads * params_.head_dim);
            info.addOutput("Q", params_.Q, seq_len, params_.n_heads * params_.head_dim);

            if (params_.K)
            {
                int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
                info.addInput("K", params_.K, seq_len, n_kv_heads * params_.head_dim);
                info.addOutput("K", params_.K, seq_len, n_kv_heads * params_.head_dim);
            }
        }
        else
        {
            // Legacy path: Q8_1/Hybrid modes need data conversion, use getSafeFp32Data()
            // This path is NOT used for standard CUDA FP32 inference

            // Q input
            const float *q_input_data = getSafeFp32Data(params_.Q);
            if (q_input_data)
            {
                info.addInput("Q", q_input_data,
                              seq_len, params_.n_heads * params_.head_dim);
            }

            // Q output - use Q_out in Hybrid/HybridQ16 modes, otherwise same as input (in-place)
            if (separate_output_mode && params_.Q_out)
            {
                const float *q_out_data = getSafeFp32Data(params_.Q_out);
                if (q_out_data)
                {
                    // HybridQ16 RoPE outputs Q in head-major layout [n_heads, seq, dim]
                    // for the attention kernel. But snapshots expect seq-major [seq, n_heads * dim]
                    // to match FP32 reference. Convert layout for snapshot comparison.
                    if (hybrid_q16_mode)
                    {
                        const int d_model = params_.n_heads * params_.head_dim;
                        // Cache the converted buffer in the stage (lazy allocation)
                        if (q_transposed_cache_.size() != static_cast<size_t>(seq_len * d_model))
                        {
                            q_transposed_cache_.resize(seq_len * d_model);
                        }
                        // Convert [n_heads, seq, dim] -> [seq, n_heads * dim]
                        for (int h = 0; h < params_.n_heads; ++h)
                        {
                            for (int s = 0; s < seq_len; ++s)
                            {
                                for (int d = 0; d < params_.head_dim; ++d)
                                {
                                    // Source: head-major [h][s][d]
                                    const int src_idx = h * seq_len * params_.head_dim + s * params_.head_dim + d;
                                    // Dest: seq-major [s][h*dim + d]
                                    const int dst_idx = s * d_model + h * params_.head_dim + d;
                                    q_transposed_cache_[dst_idx] = q_out_data[src_idx];
                                }
                            }
                        }
                        info.addOutput("Q", q_transposed_cache_.data(),
                                       seq_len, d_model);
                    }
                    else
                    {
                        info.addOutput("Q", q_out_data,
                                       seq_len, params_.n_heads * params_.head_dim);
                    }
                }
            }
            else if (q_input_data)
            {
                info.addOutput("Q", q_input_data,
                               seq_len, params_.n_heads * params_.head_dim);
            }

            // K tensor (optional)
            if (params_.K)
            {
                int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;

                // K input
                const float *k_input_data = getSafeFp32Data(params_.K);
                if (k_input_data)
                {
                    info.addInput("K", k_input_data,
                                  seq_len, n_kv_heads * params_.head_dim);
                }

                // K output - use K_out in Hybrid/HybridQ16 modes, otherwise same as input (in-place)
                if (separate_output_mode && params_.K_out)
                {
                    const float *k_out_data = getSafeFp32Data(params_.K_out);
                    if (k_out_data)
                    {
                        info.addOutput("K", k_out_data,
                                       seq_len, n_kv_heads * params_.head_dim);
                    }
                }
                else if (k_input_data)
                {
                    info.addOutput("K", k_input_data,
                                   seq_len, n_kv_heads * params_.head_dim);
                }
            }
        }

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("pos_offset", params_.pos_offset);
        info.addScalar("theta_base", params_.theta_base);

        return info;
    }

    StageBufferRequirements RoPEStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.Q)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t seq_len = params_.Q->rows();
        const size_t q_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());

        // Q is INOUT (in-place operation)
        reqs.addInout("Q", {seq_len, q_dim}, buf_type);

        // K is optional INOUT (in-place operation)
        if (params_.K)
        {
            const int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            const size_t k_dim = static_cast<size_t>(n_kv_heads * params_.head_dim);
            reqs.addInout("K", {seq_len, k_dim}, buf_type);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *RoPEStage::getKernelAsWorkspaceConsumer()
    {
        // Create kernel if not already cached (or dtype variant changed)
        if (!params_.Q)
        {
            LOG_WARN("[RoPEStage::getKernelAsWorkspaceConsumer] Q tensor not set");
            return nullptr;
        }

        auto *Q_base = dynamic_cast<TensorBase *>(params_.Q);
        if (!Q_base)
        {
            LOG_WARN("[RoPEStage::getKernelAsWorkspaceConsumer] Q is not TensorBase");
            return nullptr;
        }

        auto *kernel = getOrCreateStageKernel(Q_base);

        if (!kernel)
        {
            LOG_WARN("[RoPEStage::getKernelAsWorkspaceConsumer] Failed to create RoPE kernel");
            return nullptr;
        }

        // Cast to IWorkspaceConsumer (CUDA/ROCm RoPE kernels implement both ITensorRoPE and IWorkspaceConsumer)
        return dynamic_cast<IWorkspaceConsumer *>(kernel);
    }

    StageBufferContract RoPEStage::bufferContract() const
    {
        if (!params_.q_buffer_id || !params_.k_buffer_id)
            return {};

        auto contract = StageBufferContract::build();

        // Hybrid mode: separate input → output buffers
        if (params_.Q_out && params_.q_out_buffer_id)
        {
            contract.addInput(*params_.q_buffer_id);
            contract.addOutput(*params_.q_out_buffer_id);
        }
        else
        {
            // Standard in-place mode
            contract.addInOut(*params_.q_buffer_id);
        }

        if (params_.K_out && params_.k_out_buffer_id)
        {
            contract.addInput(*params_.k_buffer_id);
            contract.addOutput(*params_.k_out_buffer_id);
        }
        else
        {
            contract.addInOut(*params_.k_buffer_id);
        }

        return contract;
    }

} // namespace llaminar2
