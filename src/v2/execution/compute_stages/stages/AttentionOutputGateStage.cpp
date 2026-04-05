/**
 * @file AttentionOutputGateStage.cpp
 * @brief Implementation of sigmoid-gated attention output stage
 */

#include "AttentionOutputGateStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    AttentionOutputGateStage::AttentionOutputGateStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool AttentionOutputGateStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "AttentionOutputGateStage"))
            return false;

        if (!ensureRequiredPointers("AttentionOutputGateStage",
                                    {{"input", params_.input},
                                     {"gate", params_.gate},
                                     {"output", params_.output}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *output_base = requireTensorBasePtr(params_.output, "output");

        if (!input_base || !gate_base || !output_base)
        {
            LOG_ERROR("[AttentionOutputGateStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());

        const size_t cols = input_base->shape().size() > 1 ? input_base->shape()[1] : input_base->numel();
        const size_t total = static_cast<size_t>(seq_len) * cols;

        const float *input_data = input_base->data();
        const float *gate_data = gate_base->data();
        float *output_data = output_base->mutable_data();

        if (!input_data || !gate_data || !output_data)
        {
            LOG_ERROR("[AttentionOutputGateStage] Null data pointer");
            return false;
        }

        // output[i] = sigmoid(gate[i]) * input[i]
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const size_t row_off = static_cast<size_t>(t) * cols;
                const float *gate_row = gate_data + row_off;
                const float *inp_row = input_data + row_off;
                float *out_row = output_data + row_off;

#if defined(__AVX512F__)
                // AVX-512 fast sigmoid: sig(x) = 1/(1+exp(-x))
                // Using rational polynomial approximation for exp(-x)
                const __m512 vone = _mm512_set1_ps(1.0f);
                const __m512 vneg = _mm512_set1_ps(-1.0f);
                // Clamp range for exp stability
                const __m512 vmin = _mm512_set1_ps(-88.0f);
                const __m512 vmax = _mm512_set1_ps(88.0f);
                // exp coefficients (degree-6 minimax on [-88,0])
                const __m512 vlog2e = _mm512_set1_ps(1.4426950408889634f);
                const __m512 vc0 = _mm512_set1_ps(1.0f);
                const __m512 vc1 = _mm512_set1_ps(0.6931471805599453f);
                const __m512 vc2 = _mm512_set1_ps(0.2402265069591007f);
                const __m512 vc3 = _mm512_set1_ps(0.0555041086648216f);
                const __m512 vc4 = _mm512_set1_ps(0.0096181291076285f);
                const __m512 vc5 = _mm512_set1_ps(0.0013333558146428f);

                size_t j = 0;
                const size_t vec_end = cols & ~static_cast<size_t>(15);
                for (; j < vec_end; j += 16)
                {
                    // neg_gate = clamp(-gate, -88, 88)
                    __m512 vg = _mm512_loadu_ps(gate_row + j);
                    __m512 neg_g = _mm512_max_ps(vmin, _mm512_min_ps(vmax, _mm512_mul_ps(vg, vneg)));

                    // exp(neg_g) via range reduction + polynomial
                    // exp(x) = 2^(x*log2e) = 2^n * 2^f where n=round(x*log2e),
                    // f = x*log2e - n ∈ [-0.5, 0.5]. Polynomial approximates 2^f.
                    __m512 neg_g_scaled = _mm512_mul_ps(neg_g, vlog2e);
                    __m512 vn = _mm512_roundscale_ps(neg_g_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m512 vf = _mm512_sub_ps(neg_g_scaled, vn);

                    // Polynomial: p = c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
                    __m512 vp = _mm512_fmadd_ps(vc5, vf, vc4);
                    vp = _mm512_fmadd_ps(vp, vf, vc3);
                    vp = _mm512_fmadd_ps(vp, vf, vc2);
                    vp = _mm512_fmadd_ps(vp, vf, vc1);
                    vp = _mm512_fmadd_ps(vp, vf, vc0);

                    // Scale by 2^n
                    __m512i vi_n = _mm512_cvtps_epi32(vn);
                    vi_n = _mm512_add_epi32(vi_n, _mm512_set1_epi32(127));
                    vi_n = _mm512_slli_epi32(vi_n, 23);
                    __m512 v2n = _mm512_castsi512_ps(vi_n);
                    __m512 vexp = _mm512_mul_ps(vp, v2n);

                    // sigmoid = 1 / (1 + exp(-x))
                    __m512 vsig = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
                    __m512 vinp = _mm512_loadu_ps(inp_row + j);
                    _mm512_storeu_ps(out_row + j, _mm512_mul_ps(vsig, vinp));
                }
                for (; j < cols; ++j)
                {
                    const float sig = 1.0f / (1.0f + std::exp(-gate_row[j]));
                    out_row[j] = sig * inp_row[j];
                }
#else
                for (size_t j = 0; j < cols; ++j)
                {
                    const float sig = 1.0f / (1.0f + std::exp(-gate_row[j]));
                    out_row[j] = sig * inp_row[j];
                }
#endif
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        LOG_DEBUG("[AttentionOutputGateStage] Executed: seq_len=" << seq_len
                                                                  << " cols=" << cols
                                                                  << " total=" << total);
        return true;
    }

    size_t AttentionOutputGateStage::estimatedFlops() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // sigmoid (4 ops) + multiply (1 op) per element
        return input_base->numel() * 5;
    }

    size_t AttentionOutputGateStage::estimatedMemoryBytes() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // Read input + gate, write output (each float)
        return input_base->numel() * 3 * sizeof(float);
    }

    bool AttentionOutputGateStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo AttentionOutputGateStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        auto *gate_base = dynamic_cast<TensorBase *>(params_.gate);
        auto *output_base = dynamic_cast<TensorBase *>(params_.output);

        if (input_base)
            info.inputs.push_back({"input", input_base});
        if (gate_base)
            info.inputs.push_back({"gate", gate_base});
        if (output_base)
            info.outputs.push_back({"output", output_base});

        return info;
    }

    StageBufferRequirements AttentionOutputGateStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract AttentionOutputGateStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.gate_buffer_id)
            contract.addInput(*params_.gate_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
