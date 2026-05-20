/**
 * @file GDNRecurrenceStage.cpp
 * @brief Implementation of delta rule recurrence for GDN linear attention
 *
 * Pure glue: extracts raw pointers from tensors, validates them, and
 * delegates all computation to the ITensorGatedDeltaNet kernel.
 *
 * When Q, K, V all point to the same merged QKV buffer (interleaved layout
 * [seq_len, q_dim + k_dim + v_dim]), this stage deinterleaves them into
 * separate contiguous arrays before passing to the kernel.
 *
 * GPU path: Uses ensureOnDevice() / allocateOnDevice() / gpu_data_ptr() to
 * keep data on-device. Merged QKV deinterleave is done on-device via the
 * kernel's deinterleave_qkv_device() method. No H2D/D2H copies in the hot path.
 *
 * CPU path: Uses data() / mutable_data() host pointers with CPU-side deinterleave.
 *
 * All preprocessing (L2 normalization, query scaling, gate computation)
 * is handled by the kernel, keeping this stage device-agnostic.
 */

#include "GDNRecurrenceStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

#include <cstring>
#include <vector>

namespace llaminar2
{

    namespace
    {
        /**
         * Deinterleave a merged QKV buffer into separate contiguous Q, K, V arrays.
         *
         * Merged layout per row t (length = q_src_dim + k_src_dim + v_dim):
         *   [ Q (nkh * d_k) | K (nkh * d_k) | V (n_v_heads_local * d_v) ]
         *
         * Output layout:
         *   q_dst, k_dst : [T, n_v_heads_local * d_k]
         *   v_dst        : [T, n_v_heads_local * d_v]  (straight copy)
         *
         * Head mapping: for each local V-head j in [0, n_v_heads_local),
         *   k_idx = (j + global_v_offset) % nkh
         *   q_dst[t, j] = q_src[t, k_idx]
         *   k_dst[t, j] = k_src[t, k_idx]
         *
         * This single modular formula correctly covers all three regimes:
         *   - nkh > n_v_heads_local  (selection, e.g. high-degree TP)
         *   - nkh == n_v_heads_local (identity or rotation)
         *   - nkh < n_v_heads_local  (expansion / modular GQA repeat)
         * including non-integer ratios (e.g. 27B under TP=2: nkh=16, n_v_local=24).
         *
         * Fast path: when global_v_offset == 0 AND nkh == n_v_heads_local, the
         * Q and K regions become a contiguous copy of the source buffer — emitted
         * as a single memcpy per token.
         */
        void deinterleaveMergedQKV(
            const float *qkv, int T, int qkv_stride,
            int nkh, int n_v_heads_local, int d_k, int d_v,
            int global_v_offset,
            float *q_dst_buf, float *k_dst_buf, float *v_dst_buf)
        {
            const int q_src_dim = nkh * d_k;
            const int k_src_dim = nkh * d_k;
            const int v_dim = n_v_heads_local * d_v;
            const int q_dst_dim = n_v_heads_local * d_k;
            const int k_dst_dim = n_v_heads_local * d_k;

            const bool identity_fast_path =
                (global_v_offset == 0) && (nkh == n_v_heads_local);

            for (int t = 0; t < T; ++t)
            {
                const float *row = qkv + static_cast<size_t>(t) * qkv_stride;
                const float *q_src = row;
                const float *k_src = row + q_src_dim;
                const float *v_src = row + q_src_dim + k_src_dim;
                float *q_dst = q_dst_buf + static_cast<size_t>(t) * q_dst_dim;
                float *k_dst = k_dst_buf + static_cast<size_t>(t) * k_dst_dim;
                float *v_dst = v_dst_buf + static_cast<size_t>(t) * v_dim;

                if (identity_fast_path)
                {
                    std::memcpy(q_dst, q_src, q_dst_dim * sizeof(float));
                    std::memcpy(k_dst, k_src, k_dst_dim * sizeof(float));
                }
                else
                {
                    for (int j = 0; j < n_v_heads_local; ++j)
                    {
                        const int k_idx = (j + global_v_offset) % nkh;
                        std::memcpy(q_dst + j * d_k,
                                    q_src + k_idx * d_k,
                                    d_k * sizeof(float));
                        std::memcpy(k_dst + j * d_k,
                                    k_src + k_idx * d_k,
                                    d_k * sizeof(float));
                    }
                }

                // V is already n_v_heads_local wide — straight copy.
                std::memcpy(v_dst, v_src, v_dim * sizeof(float));
            }
        }
    } // namespace

    GDNRecurrenceStage::GDNRecurrenceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    // =========================================================================
    // Main execute
    // =========================================================================

    bool GDNRecurrenceStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNRecurrenceStage"))
            return false;

        if (!ensureRequiredPointers("GDNRecurrenceStage",
                                    {{"Q", params_.Q},
                                     {"K", params_.K},
                                     {"V", params_.V},
                                     {"alpha", params_.alpha},
                                     {"beta", params_.beta},
                                     {"A_log", params_.A_log},
                                     {"dt_bias", params_.dt_bias},
                                     {"output", params_.output}}))
            return false;

        if (!params_.kernel)
        {
            LOG_ERROR("[GDNRecurrenceStage] kernel (ITensorGatedDeltaNet) not set");
            return false;
        }

        // Bind stage stream to kernel before execution
        params_.kernel->setGPUStream(gpuStream());

        auto *q_base = requireTensorBasePtr(params_.Q, "Q");
        auto *k_base = requireTensorBasePtr(params_.K, "K");
        auto *v_base = requireTensorBasePtr(params_.V, "V");
        auto *alpha_base = requireTensorBasePtr(params_.alpha, "alpha");
        auto *beta_base = requireTensorBasePtr(params_.beta, "beta");
        auto *alog_base = requireTensorBasePtr(params_.A_log, "A_log");
        auto *dtbias_base = requireTensorBasePtr(params_.dt_bias, "dt_bias");
        auto *out_base = requireTensorBasePtr(params_.output, "output");

        if (!q_base || !k_base || !v_base || !alpha_base || !beta_base ||
            !alog_base || !dtbias_base || !out_base)
            return false;

        if (!params_.recurrence_state)
        {
            LOG_ERROR("[GDNRecurrenceStage] recurrence_state is null");
            return false;
        }

        // =====================================================================
        // GPU path: keep data on-device, pass device pointers to kernel
        // =====================================================================
        if (device().is_gpu())
        {
            // Coherence is handled by the executor via bufferContract():
            //   - Arena inputs (QKV, alpha, beta) via prepareForRead
            //   - Model weights (A_log, dt_bias) via contract.weight_tensors
            //   - Output (ATTN_OUTPUT) via prepareForWrite + markWritten

            // Get device pointers
            const float *d_alpha = static_cast<const float *>(alpha_base->gpu_data_ptr());
            const float *d_beta = static_cast<const float *>(beta_base->gpu_data_ptr());
            const float *d_alog = static_cast<const float *>(alog_base->gpu_data_ptr());
            const float *d_dtbias = static_cast<const float *>(dtbias_base->gpu_data_ptr());
            float *d_output = static_cast<float *>(const_cast<TensorBase *>(out_base)->gpu_data_ptr());

            // Resolve Q, K, V device pointers — handle merged QKV case
            const float *d_q = nullptr;
            const float *d_k = nullptr;
            const float *d_v = nullptr;

            // Check if merged: when Q, K, V all point to the same tensor
            const bool merged_qkv = (params_.Q == params_.K && params_.K == params_.V);
            const int nkh = (params_.n_k_heads > 0) ? params_.n_k_heads : params_.n_heads;

            if (merged_qkv)
            {
                // Merged QKV buffer on device
                const float *d_merged = static_cast<const float *>(q_base->gpu_data_ptr());

                const int q_src_dim = nkh * params_.d_k;
                const int k_src_dim = nkh * params_.d_k;
                const int v_dim = params_.n_heads * params_.d_v;

                if (params_.seq_len == 1 && nkh == params_.n_heads && params_.global_v_head_offset == 0)
                {
                    // Decode + identity + offset=0: Q, K, V are contiguous sub-regions
                    // in a single row — just use offset pointers (zero-copy)
                    d_q = d_merged;
                    d_k = d_merged + q_src_dim;
                    d_v = d_merged + q_src_dim + k_src_dim;
                }
                else
                {
                    // Use GPU deinterleave kernel for all other cases
                    float *dq_mut = nullptr, *dk_mut = nullptr, *dv_mut = nullptr;
                    if (!params_.kernel->deinterleave_qkv_device(
                            d_merged, dq_mut, dk_mut, dv_mut,
                            params_.seq_len, nkh, params_.n_heads,
                            params_.d_k, params_.d_v, params_.global_v_head_offset))
                    {
                        LOG_ERROR("[GDNRecurrenceStage] GPU deinterleave_qkv_device failed");
                        return false;
                    }
                    d_q = dq_mut;
                    d_k = dk_mut;
                    d_v = dv_mut;
                }

                LOG_DEBUG("[GDNRecurrenceStage] GPU merged QKV: "
                          << params_.seq_len << "x" << (q_src_dim + k_src_dim + v_dim)
                          << " nkh=" << nkh << " n_heads=" << params_.n_heads
                          << " offset=" << params_.global_v_head_offset
                          << (params_.seq_len == 1 ? " (decode offset)" : " (kernel deinterleave)"));
            }
            else
            {
                // Separate Q, K, V tensors — already on device via executor coherence
                d_q = static_cast<const float *>(q_base->gpu_data_ptr());
                d_k = static_cast<const float *>(k_base->gpu_data_ptr());
                d_v = static_cast<const float *>(v_base->gpu_data_ptr());
            }

            bool ok;
            if (params_.seq_len == 1)
            {
                LOG_DEBUG("[GDNRecurrenceStage] GPU launch pointers layer=" << params_.layer_idx
                         << " Q=" << static_cast<const void *>(d_q)
                         << " K=" << static_cast<const void *>(d_k)
                         << " V=" << static_cast<const void *>(d_v)
                         << " alpha=" << static_cast<const void *>(d_alpha)
                         << " beta=" << static_cast<const void *>(d_beta)
                         << " A_log=" << static_cast<const void *>(d_alog)
                         << " dt_bias=" << static_cast<const void *>(d_dtbias)
                         << " output=" << static_cast<void *>(d_output)
                         << " state=" << static_cast<void *>(params_.recurrence_state)
                         << " heads=" << params_.n_heads
                         << " d_k=" << params_.d_k
                         << " d_v=" << params_.d_v
                         << " seq=" << params_.seq_len);
                ok = params_.kernel->recurrent_step(
                    d_q, d_k, d_v,
                    d_alpha, d_beta,
                    d_alog, d_dtbias,
                    d_output, params_.recurrence_state,
                    params_.n_heads, params_.d_k, params_.d_v,
                    params_.use_qk_l2norm);
            }
            else
            {
                LOG_DEBUG("[GDNRecurrenceStage] GPU launch pointers layer=" << params_.layer_idx
                         << " Q=" << static_cast<const void *>(d_q)
                         << " K=" << static_cast<const void *>(d_k)
                         << " V=" << static_cast<const void *>(d_v)
                         << " alpha=" << static_cast<const void *>(d_alpha)
                         << " beta=" << static_cast<const void *>(d_beta)
                         << " A_log=" << static_cast<const void *>(d_alog)
                         << " dt_bias=" << static_cast<const void *>(d_dtbias)
                         << " output=" << static_cast<void *>(d_output)
                         << " state=" << static_cast<void *>(params_.recurrence_state)
                         << " heads=" << params_.n_heads
                         << " d_k=" << params_.d_k
                         << " d_v=" << params_.d_v
                         << " seq=" << params_.seq_len);
                ok = params_.kernel->chunk_forward(
                    d_q, d_k, d_v,
                    d_alpha, d_beta,
                    d_alog, d_dtbias,
                    d_output, params_.recurrence_state,
                    params_.seq_len, params_.n_heads, params_.d_k, params_.d_v,
                    params_.chunk_size, params_.use_qk_l2norm);
            }

            if (!ok)
            {
                LOG_ERROR("[GDNRecurrenceStage] GPU kernel failed");
                return false;
            }

            LOG_DEBUG("[GDNRecurrenceStage] GPU layer=" << params_.layer_idx
                                                        << " seq_len=" << params_.seq_len
                                                        << " n_heads=" << params_.n_heads
                                                        << " d_k=" << params_.d_k
                                                        << " d_v=" << params_.d_v
                                                        << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));

            return true;
        }

        // =====================================================================
        // CPU path: use host pointers, CPU-side deinterleave
        // =====================================================================
        const float *q_data = q_base->data();
        const float *k_data = k_base->data();
        const float *v_data = v_base->data();
        const float *alpha_data = alpha_base->data();
        const float *beta_data = beta_base->data();
        const float *alog_data = alog_base->data();
        const float *dtbias_data = dtbias_base->data();
        float *output_data = out_base->mutable_data();

        // When Q, K, V all point to the same merged QKV buffer, deinterleave them.
        // Merged layout: [seq_len, q_dim + k_dim + v_dim] per row.
        // q_dim = k_dim = n_k_heads * d_k, v_dim = n_heads * d_v (n_heads = n_v_heads)
        //
        // Three modes depending on n_k_heads vs n_heads (n_v_heads_local):
        //   1) n_k < n_v_local: Expansion — modular GQA repeat (single-device, repeat_factor > 1)
        //   2) n_k == n_v_local: Identity deinterleave (TP where n_k is replicated and equals n_v_local)
        //   3) n_k > n_v_local: Selection — pick the correct K-head subset for each V-head (high-degree TP)
        //
        // The kernel expects separate contiguous [seq_len, n_v_local * dim] arrays.
        const bool merged_qkv = (q_data == k_data && k_data == v_data);

        // Effective key head count for QKV split (may be full count if Q/K replicated for TP)
        const int nkh = (params_.n_k_heads > 0) ? params_.n_k_heads : params_.n_heads;

        if (merged_qkv)
        {
            // Dimensions in the merged QKV buffer
            const int q_src_dim = nkh * params_.d_k;
            const int k_src_dim = nkh * params_.d_k;
            const int v_dim = params_.n_heads * params_.d_v;
            const int qkv_stride = q_src_dim + k_src_dim + v_dim;

            // Dimensions after deinterleave (what the kernel expects: n_v_local heads)
            const int q_dst_dim = params_.n_heads * params_.d_k;
            const int k_dst_dim = params_.n_heads * params_.d_k;
            const int T = params_.seq_len;

            // Grow-only reusable scratch (no allocation after first call at max seq_len)
            const size_t q_size = static_cast<size_t>(T) * q_dst_dim;
            const size_t k_size = static_cast<size_t>(T) * k_dst_dim;
            const size_t v_size = static_cast<size_t>(T) * v_dim;
            if (q_deinterleave_.size() < q_size)
                q_deinterleave_.resize(q_size);
            if (k_deinterleave_.size() < k_size)
                k_deinterleave_.resize(k_size);
            if (v_deinterleave_.size() < v_size)
                v_deinterleave_.resize(v_size);

            const float *qkv = q_data; // merged buffer

            deinterleaveMergedQKV(
                qkv, T, qkv_stride,
                nkh, params_.n_heads, params_.d_k, params_.d_v,
                params_.global_v_head_offset,
                q_deinterleave_.data(),
                k_deinterleave_.data(),
                v_deinterleave_.data());

            q_data = q_deinterleave_.data();
            k_data = k_deinterleave_.data();
            v_data = v_deinterleave_.data();

            LOG_DEBUG("[GDNRecurrenceStage] Deinterleaved merged QKV: "
                      << T << "x" << qkv_stride << " -> Q(" << T << "x" << q_dst_dim
                      << "), K(" << T << "x" << k_dst_dim << "), V(" << T << "x" << v_dim << ")"
                      << " nkh=" << nkh << " n_heads=" << params_.n_heads
                      << " global_v_offset=" << params_.global_v_head_offset);
        }

        bool ok;
        if (params_.seq_len == 1)
        {
            ok = params_.kernel->recurrent_step(
                q_data, k_data, v_data,
                alpha_data, beta_data,
                alog_data, dtbias_data,
                output_data, params_.recurrence_state,
                params_.n_heads, params_.d_k, params_.d_v,
                params_.use_qk_l2norm);
        }
        else
        {
            ok = params_.kernel->chunk_forward(
                q_data, k_data, v_data,
                alpha_data, beta_data,
                alog_data, dtbias_data,
                output_data, params_.recurrence_state,
                params_.seq_len, params_.n_heads, params_.d_k, params_.d_v,
                params_.chunk_size, params_.use_qk_l2norm);
        }

        if (!ok)
        {
            LOG_ERROR("[GDNRecurrenceStage] Kernel failed");
            return false;
        }

        LOG_DEBUG("[GDNRecurrenceStage] layer=" << params_.layer_idx
                                                << " seq_len=" << params_.seq_len
                                                << " n_heads=" << params_.n_heads
                                                << " d_k=" << params_.d_k
                                                << " d_v=" << params_.d_v
                                                << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));

        return ok;
    }

    // =========================================================================
    // Estimation and metadata
    // =========================================================================

    size_t GDNRecurrenceStage::estimatedFlops() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t H = static_cast<size_t>(params_.n_heads);
        const size_t dk = static_cast<size_t>(params_.d_k);
        const size_t dv = static_cast<size_t>(params_.d_v);
        // Per timestep per head: decay(dk*dv) + kv_mem(dk*dv) + delta(dv) + rank1(dk*dv) + output(dk*dv)
        // ≈ 4*dk*dv + dv per step per head
        return S * H * (4 * dk * dv + dv);
    }

    size_t GDNRecurrenceStage::estimatedMemoryBytes() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t H = static_cast<size_t>(params_.n_heads);
        const size_t dk = static_cast<size_t>(params_.d_k);
        const size_t dv = static_cast<size_t>(params_.d_v);
        // State: H*dk*dv, Q: S*H*dk, K: S*H*dk, V: S*H*dv, output: S*H*dv
        return (H * dk * dv + S * H * (2 * dk + 2 * dv)) * sizeof(float);
    }

    bool GDNRecurrenceStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    bool GDNRecurrenceStage::isGraphCapturable() const
    {
#if defined(ENABLE_PIPELINE_SNAPSHOTS) || !defined(HAVE_ROCM)
        return params_.seq_len == 1;
#else
        // Decode: always capturable (single kernel launch, stable pointers)
        if (params_.seq_len == 1)
            return true;

        // Prefill: capturable only on ROCm when GPU state is pre-allocated.
        // chunk_forward() will skip its lazy allocateState() if state is ready.
        if (!params_.device_id.is_rocm() || !params_.kernel)
            return false;

        const int required_state_size = params_.n_heads * params_.d_k * params_.d_v;
        return params_.kernel->isGPUStateReady(required_state_size);
#endif
    }

    StageDumpInfo GDNRecurrenceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Use actual seq_len dimensions, not the buffer capacity
        const size_t rows = static_cast<size_t>(params_.seq_len);
        const size_t cols = static_cast<size_t>(params_.n_heads) * params_.d_v;
        if (params_.output)
            info.addOutput("output", params_.output, rows, cols);

        return info;
    }

    StageBufferRequirements GDNRecurrenceStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNRecurrenceStage::bufferContract() const
    {
        StageBufferContract contract;
        // Arena-managed activation inputs
        if (params_.qkv_buffer_id)
            contract.addInput(*params_.qkv_buffer_id);
        if (params_.alpha_buffer_id)
            contract.addInput(*params_.alpha_buffer_id);
        if (params_.beta_buffer_id)
            contract.addInput(*params_.beta_buffer_id);
        // Arena-managed output
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        // Model weights (not arena-managed)
        if (params_.A_log)
            contract.addWeight(const_cast<ITensor *>(params_.A_log));
        if (params_.dt_bias)
            contract.addWeight(const_cast<ITensor *>(params_.dt_bias));
        return contract;
    }

} // namespace llaminar2
