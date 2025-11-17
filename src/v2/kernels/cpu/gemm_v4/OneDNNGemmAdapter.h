#pragma once

#include "kernels/cpu/gemm_v4/OneDNNGemm.h"
#include "tensors/Tensors.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    namespace gemm_v4
    {
        struct WeightPack
        {
            std::vector<int8_t> data;
            std::vector<float> col_scales;
            int rows = 0;
            int cols = 0;

            size_t element_count() const { return data.size(); }
        };

        inline WeightPack pack_weights_to_int8(const TensorBase &tensor, int K, int N)
        {
            const auto &shape = tensor.shape();
            if (shape.size() != 2)
            {
                throw std::runtime_error("Weight tensor must be 2D");
            }
            if (shape[0] != static_cast<size_t>(N) || shape[1] != static_cast<size_t>(K))
            {
                throw std::runtime_error("Weight tensor shape does not match provided dimensions");
            }

            WeightPack pack;
            pack.rows = K;
            pack.cols = N;
            pack.data.resize(static_cast<size_t>(K) * static_cast<size_t>(N), 0);
            pack.col_scales.resize(static_cast<size_t>(N), 1.0f);

            std::vector<int8_t> row_major(static_cast<size_t>(N) * static_cast<size_t>(K));
            std::vector<float> col_scale_buffer(static_cast<size_t>(K), 1.0f);
            if (!tensor.to_int8_perchannel(row_major.data(), col_scale_buffer.data(), pack.col_scales.data()))
            {
                throw std::runtime_error("TensorBase::to_int8_perchannel failed for weight tensor");
            }

            for (int n = 0; n < N; ++n)
            {
                const int8_t *row_src = row_major.data() + static_cast<size_t>(n) * static_cast<size_t>(K);
                for (int k = 0; k < K; ++k)
                {
                    pack.data[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)] = row_src[static_cast<size_t>(k)];
                }
            }

            return pack;
        }

        inline bool onednn_gemm_from_packed(const ActivationPack &activation,
                                            const WeightPack &weights,
                                            IActivationTensor &output_tensor,
                                            int M,
                                            int N,
                                            int K,
                                            const float *bias = nullptr)
        {
            if (activation.rows != M || activation.cols != K)
            {
                throw std::runtime_error("Activation pack dimensions mismatch");
            }
            if (weights.rows != K || weights.cols != N)
            {
                throw std::runtime_error("Weight pack dimensions mismatch");
            }
            if (activation.row_scales.size() != static_cast<size_t>(M) ||
                weights.col_scales.size() != static_cast<size_t>(N))
            {
                throw std::runtime_error("Scale buffers do not match matrix dimensions");
            }

            static thread_local std::vector<int32_t> accum_buffer;
            const size_t accum_elems = static_cast<size_t>(M) * static_cast<size_t>(N);
            if (accum_buffer.size() < accum_elems)
            {
                accum_buffer.resize(accum_elems);
            }

            if (!run_onednn_int8_matmul(activation.data.data(),
                                        weights.data.data(),
                                        accum_buffer.data(),
                                        M,
                                        N,
                                        K))
            {
                return false;
            }

            return output_tensor.from_int32_with_scales(
                accum_buffer.data(),
                M,
                N,
                activation.row_scales.data(),
                weights.col_scales.data(),
                bias);
        }

        inline bool onednn_gemm_adapter(int M,
                                        int N,
                                        int K,
                                        const IActivationTensor &A,
                                        const TensorBase &B,
                                        IActivationTensor &output_tensor,
                                        const float *bias = nullptr)
        {
            auto activation = A.to_int8_activation_pack(M, K);
            auto weights = pack_weights_to_int8(B, K, N);
            return onednn_gemm_from_packed(activation, weights, output_tensor, M, N, K, bias);
        }
    } // namespace gemm_v4
} // namespace llaminar2
