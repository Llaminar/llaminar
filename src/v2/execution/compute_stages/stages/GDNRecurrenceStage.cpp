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

        // Extract raw pointers — all computation delegated to kernel
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
        // When n_k_heads < n_heads, Q and K are repeat_interleaved to n_heads.
        // The kernel expects separate contiguous [seq_len, dim] arrays.
        const bool merged_qkv = (q_data == k_data && k_data == v_data);

        // Effective key head count for QKV split
        const int nkh = (params_.n_k_heads > 0) ? params_.n_k_heads : params_.n_heads;
        const int repeat_factor = params_.n_heads / nkh;

        if (merged_qkv)
        {
            // Dimensions in the merged QKV buffer (before repeat_interleave)
            const int q_src_dim = nkh * params_.d_k;
            const int k_src_dim = nkh * params_.d_k;
            const int v_dim = params_.n_heads * params_.d_v;
            const int qkv_stride = q_src_dim + k_src_dim + v_dim;

            // Dimensions after repeat_interleave (what the kernel expects)
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

            if (repeat_factor <= 1)
            {
                // No repeat needed: n_k_heads == n_v_heads, simple deinterleave
                for (int t = 0; t < T; ++t)
                {
                    const float *row = qkv + static_cast<size_t>(t) * qkv_stride;
                    std::memcpy(q_deinterleave_.data() + static_cast<size_t>(t) * q_dst_dim,
                                row, q_dst_dim * sizeof(float));
                    std::memcpy(k_deinterleave_.data() + static_cast<size_t>(t) * k_dst_dim,
                                row + q_src_dim, k_dst_dim * sizeof(float));
                    std::memcpy(v_deinterleave_.data() + static_cast<size_t>(t) * v_dim,
                                row + q_src_dim + k_src_dim, v_dim * sizeof(float));
                }
            }
            else
            {
                // Deinterleave + modular GQA expansion of Q/K from n_k_heads to n_v_heads
                // Modular pattern: V-head h maps to K-head (h % n_k_heads)
                //   Q[t, h, d_k] -> Q[t, r*nkh+h, d_k] for r in [0, repeat_factor)
                // This produces: V-heads [0..nkh-1] use K-heads [0..nkh-1],
                //                 V-heads [nkh..2*nkh-1] also use K-heads [0..nkh-1], etc.
                for (int t = 0; t < T; ++t)
                {
                    const float *row = qkv + static_cast<size_t>(t) * qkv_stride;
                    float *q_dst = q_deinterleave_.data() + static_cast<size_t>(t) * q_dst_dim;
                    float *k_dst = k_deinterleave_.data() + static_cast<size_t>(t) * k_dst_dim;
                    const float *q_src = row;
                    const float *k_src = row + q_src_dim;

                    for (int h = 0; h < nkh; ++h)
                    {
                        for (int r = 0; r < repeat_factor; ++r)
                        {
                            std::memcpy(q_dst + (r * nkh + h) * params_.d_k,
                                        q_src + h * params_.d_k,
                                        params_.d_k * sizeof(float));
                            std::memcpy(k_dst + (r * nkh + h) * params_.d_k,
                                        k_src + h * params_.d_k,
                                        params_.d_k * sizeof(float));
                        }
                    }

                    // V: straight copy (already n_v_heads wide)
                    std::memcpy(v_deinterleave_.data() + static_cast<size_t>(t) * v_dim,
                                row + q_src_dim + k_src_dim, v_dim * sizeof(float));
                }
            }

            q_data = q_deinterleave_.data();
            k_data = k_deinterleave_.data();
            v_data = v_deinterleave_.data();

            LOG_DEBUG("[GDNRecurrenceStage] Deinterleaved merged QKV: "
                      << T << "x" << qkv_stride << " -> Q(" << T << "x" << q_dst_dim
                      << "), K(" << T << "x" << k_dst_dim << "), V(" << T << "x" << v_dim << ")"
                      << (repeat_factor > 1 ? " [repeat_interleave x" + std::to_string(repeat_factor) + "]" : ""));
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
        return backend == ComputeBackendType::CPU;
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
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
