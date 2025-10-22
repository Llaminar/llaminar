/**
 * @file MPILinearOperator_v2.cpp
 * @brief Simplified MPI-aware linear projection with FP32/BF16 activation support
 *
 * This implementation removes all legacy cruft and focuses on:
 * 1. Clean FP32/BF16 activation dispatch
 * 2. Proper GEMM backend selection via ITensorGemm interface
 * 3. Accumulate in FP32, output in input format
 *
 * @author David Sanftenberg
 * @date October 2025
 */

#include "MPILinearOperator_v2.h"
#include "../Logger.h"
#include "../PerformanceTracer.h"
#include "../DebugUtils.h"
#include "../ITensorGemm.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/BF16Tensor.h"
#include "../utils/BFloat16.h"
#include "attention/AttentionStageContracts.h"
#include <algorithm>
#include <cstring>
#include <cblas.h>

namespace llaminar
{

    MPILinearOperator_v2::MPILinearOperator_v2(MPI_Comm comm)
        : MPIOperatorBase(comm, false)
    {
        LOG_DEBUG("MPILinearOperator_v2 initialized on rank " << getRank() << "/" << getSize());
    }

    bool MPILinearOperator_v2::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                       std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_TRACE_SCOPE_CAT("mpi_linear_v2_execute", "linear");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPILinearOperator_v2 validation failed on rank " << getRank());
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto bias = (inputs.size() >= 3) ? inputs[2] : nullptr;
        auto output = outputs[0];

        // Validate tensors
        ASSERT_TENSOR_NOT_NULL(input, "Linear input");
        ASSERT_TENSOR_NOT_NULL(weight, "Linear weight");
        ASSERT_TENSOR_NOT_NULL(output, "Linear output");

        // Dispatch based on activation type
        auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input);
        if (bf16_input)
        {
            LOG_DEBUG("MPILinearOperator_v2: Using BF16 activation path");
            return executeBF16(input, weight, bias, output);
        }
        else
        {
            LOG_DEBUG("MPILinearOperator_v2: Using FP32 activation path");
            return executeFP32(input, weight, bias, output);
        }
    }

    bool MPILinearOperator_v2::executeFP32(const std::shared_ptr<TensorBase> &input,
                                           const std::shared_ptr<TensorBase> &weight,
                                           const std::shared_ptr<TensorBase> &bias,
                                           std::shared_ptr<TensorBase> &output)
    {
        PERF_TRACE_SCOPE_CAT("fp32_path", "linear");

        // Extract dimensions
        const size_t seq_len = input->shape()[0];
        const size_t in_dim = input->shape()[1];
        const size_t out_dim = weight->shape()[0];

        // Calculate local distribution
        auto [local_out_dim, out_offset] = getRowDistribution(out_dim);

        LOG_DEBUG("FP32: Rank " << getRank() << "/" << getSize() << ": out_dim=" << out_dim
                                << " -> local_out_dim=" << local_out_dim << " out_offset=" << out_offset);

        // Get or cache weight (quantized weights use global reference, FP32 uses distributed copy)
        std::shared_ptr<TensorBase> local_weight = getOrCacheWeight(weight, out_dim, local_out_dim, in_dim);

        // Distribute bias if provided (with caching)
        std::shared_ptr<TensorBase> local_bias;
        if (bias)
        {
            CacheKey bias_key{bias.get(), out_dim};
            auto bias_it = bias_cache_.find(bias_key);

            if (bias_it != bias_cache_.end())
            {
                PERF_TRACE_SCOPE_CAT("bias_cache_hit", "linear");
                local_bias = bias_it->second;
            }
            else
            {
                PERF_TRACE_SCOPE_CAT("bias_cache_miss", "linear");
                local_bias = TensorFactory::create_simple({local_out_dim});
                distributeBias(bias, local_bias, out_dim);
                bias_cache_[bias_key] = local_bias;
            }
        }

        // Allocate local output buffer
        auto local_output = TensorFactory::create_simple({static_cast<int>(seq_len), local_out_dim});

        // Perform GEMM: local_output = input @ local_weight^T
        {
            PERF_TRACE_SCOPE_CAT("gemm", "linear");

            bool success = false;

            // Try ITensorGemm interface first (preferred for quantized weights)
            ITensorGemm *gemm = local_weight->createGemmRaw();
            LOG_DEBUG("createGemmRaw() returned: " << (gemm ? gemm->name() : "nullptr")
                                                   << " for weight type=" << static_cast<int>(local_weight->native_type()));
            if (gemm)
            {
                LOG_DEBUG("Using ITensorGemm: " << gemm->name());

                // For quantized weights: local_weight is global reference, use row_offset
                // For FP32 weights: local_weight is already distributed subset, no offset needed
                int row_offset = (local_weight->native_type() == TensorDataType::QUANTIZED) ? out_offset : 0;

                success = gemm->multiply(
                    input->data(),
                    local_output->data(),
                    static_cast<int>(seq_len),
                    static_cast<int>(local_out_dim),
                    static_cast<int>(in_dim),
                    /*transpose_B=*/true,
                    /*alpha=*/1.0f,
                    /*beta=*/0.0f,
                    /*row_offset=*/row_offset,
                    /*row_count=*/static_cast<int>(local_out_dim));
                delete gemm;
            }
            else
            {
                // Fallback to BLAS for FP32 weights
                if (gemm)
                    delete gemm;

                LOG_DEBUG("Using OpenBLAS for FP32 weight");
                const float *weight_data = local_weight->data();
                if (!weight_data)
                {
                    LOG_ERROR("Failed to get weight data for BLAS");
                    return false;
                }

                // C = A @ B^T where A=[seq_len, in_dim], B=[local_out_dim, in_dim]
                // cblas_sgemm: C = alpha*op(A)*op(B) + beta*C
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            static_cast<int>(seq_len),                              // m
                            static_cast<int>(local_out_dim),                        // n
                            static_cast<int>(in_dim),                               // k
                            1.0f,                                                   // alpha
                            input->data(), static_cast<int>(in_dim),                // A, lda
                            weight_data, static_cast<int>(in_dim),                  // B, ldb
                            0.0f,                                                   // beta
                            local_output->data(), static_cast<int>(local_out_dim)); // C, ldc
                success = true;
            }

            if (!success)
            {
                LOG_ERROR("FP32 GEMM failed on rank " << getRank());
                return false;
            }
        }

        // Add bias if provided
        if (local_bias)
        {
            PERF_TRACE_SCOPE_CAT("add_bias", "linear");
            addBias(local_output->data(), local_bias->data(), seq_len, local_out_dim);
        }

        // Gather outputs from all ranks
        {
            PERF_TRACE_SCOPE_CAT("gather", "mpi_collective");
            gatherOutput(local_output->data(), output->data(), seq_len, local_out_dim, out_dim);
        }

        // Validate output
        ASSERT_TENSOR_NOT_NAN(output, "Linear FP32 output");

        return true;
    }

    bool MPILinearOperator_v2::executeBF16(const std::shared_ptr<TensorBase> &input,
                                           const std::shared_ptr<TensorBase> &weight,
                                           const std::shared_ptr<TensorBase> &bias,
                                           std::shared_ptr<TensorBase> &output)
    {
        PERF_TRACE_SCOPE_CAT("bf16_path", "linear");

        auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input);
        if (!bf16_input)
        {
            LOG_ERROR("Input is not BF16Tensor in BF16 path");
            return false;
        }

        // Extract dimensions
        const size_t seq_len = input->shape()[0];
        const size_t in_dim = input->shape()[1];
        const size_t out_dim = weight->shape()[0];

        // Calculate local distribution
        auto [local_out_dim, out_offset] = getRowDistribution(out_dim);

        // Get or cache weight (quantized weights use global reference, FP32 uses distributed copy)
        std::shared_ptr<TensorBase> local_weight = getOrCacheWeight(weight, out_dim, local_out_dim, in_dim);

        // Distribute bias if provided (with caching)
        std::shared_ptr<TensorBase> local_bias;
        if (bias)
        {
            CacheKey bias_key{bias.get(), out_dim};
            auto bias_it = bias_cache_.find(bias_key);

            if (bias_it != bias_cache_.end())
            {
                PERF_TRACE_SCOPE_CAT("bias_cache_hit", "linear");
                local_bias = bias_it->second;
            }
            else
            {
                PERF_TRACE_SCOPE_CAT("bias_cache_miss", "linear");
                local_bias = TensorFactory::create_simple({local_out_dim});
                distributeBias(bias, local_bias, out_dim);
                bias_cache_[bias_key] = local_bias;
            }
        }

        // Allocate FP32 local output buffer (accumulate in FP32)
        auto local_output_fp32 = TensorFactory::create_simple({static_cast<int>(seq_len), local_out_dim});

        // Get BF16 activation data
        const bfloat16 *input_bf16 = bf16_input->bf16_data();
        if (!input_bf16)
        {
            LOG_ERROR("BF16Tensor::bf16_data() returned null");
            return false;
        }

        // Perform GEMM with BF16 activations
        {
            PERF_TRACE_SCOPE_CAT("gemm_bf16", "linear");

            // Try tensor-specific GEMM with BF16 support
            ITensorGemm *gemm = local_weight->createGemmRaw();
            bool gemm_success = false;

            if (gemm && gemm->supports_bf16())
            {
                // Use native BF16 activation path (e.g., IQ4_NLQuantizedGemm::multiply_bf16)
                int row_offset = (local_weight->native_type() == TensorDataType::QUANTIZED) ? out_offset : 0;

                gemm_success = gemm->multiply_bf16(
                    reinterpret_cast<const uint16_t *>(input_bf16),
                    local_output_fp32->data(),
                    static_cast<int>(seq_len),
                    static_cast<int>(local_out_dim),
                    static_cast<int>(in_dim),
                    /*transpose_B=*/true,
                    /*alpha=*/1.0f,
                    /*beta=*/0.0f,
                    /*row_offset=*/row_offset,
                    /*row_count=*/static_cast<int>(local_out_dim));
                delete gemm;
            }

            // Fallback: Expand BF16→FP32 then use standard GEMM
            if (!gemm_success)
            {
                LOG_DEBUG("BF16 GEMM unavailable, expanding to FP32");
                PERF_TRACE_SCOPE_CAT("bf16_expand_fallback", "linear");

                // Expand BF16→FP32
                std::vector<float> input_fp32(seq_len * in_dim);
#pragma omp parallel for
                for (size_t i = 0; i < seq_len * in_dim; ++i)
                {
                    input_fp32[i] = static_cast<float>(input_bf16[i]);
                }

                // Try ITensorGemm with expanded FP32 input
                ITensorGemm *gemm_fp32 = local_weight->createGemmRaw();
                if (gemm_fp32 && gemm_fp32->supports(static_cast<int>(seq_len),
                                                     static_cast<int>(local_out_dim),
                                                     static_cast<int>(in_dim)))
                {
                    int row_offset = (local_weight->native_type() == TensorDataType::QUANTIZED) ? out_offset : 0;

                    gemm_success = gemm_fp32->multiply(
                        input_fp32.data(),
                        local_output_fp32->data(),
                        static_cast<int>(seq_len),
                        static_cast<int>(local_out_dim),
                        static_cast<int>(in_dim),
                        /*transpose_B=*/true,
                        /*alpha=*/1.0f,
                        /*beta=*/0.0f,
                        /*row_offset=*/row_offset,
                        /*row_count=*/static_cast<int>(local_out_dim));
                    delete gemm_fp32;
                }
                else
                {
                    // Final fallback: BLAS
                    if (gemm_fp32)
                        delete gemm_fp32;

                    const float *weight_data = local_weight->data();
                    if (weight_data)
                    {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    static_cast<int>(seq_len),
                                    static_cast<int>(local_out_dim),
                                    static_cast<int>(in_dim),
                                    1.0f,
                                    input_fp32.data(), static_cast<int>(in_dim),
                                    weight_data, static_cast<int>(in_dim),
                                    0.0f,
                                    local_output_fp32->data(), static_cast<int>(local_out_dim));
                        gemm_success = true;
                    }
                }
            }

            if (!gemm_success)
            {
                LOG_ERROR("BF16 GEMM failed on rank " << getRank());
                return false;
            }
        }

        // Add bias if provided (in FP32)
        if (local_bias)
        {
            PERF_TRACE_SCOPE_CAT("add_bias", "linear");
            addBias(local_output_fp32->data(), local_bias->data(), seq_len, local_out_dim);
        }

        // Gather FP32 outputs from all ranks
        auto global_output_fp32 = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});
        {
            PERF_TRACE_SCOPE_CAT("gather", "mpi_collective");
            gatherOutput(local_output_fp32->data(), global_output_fp32->data(),
                         seq_len, local_out_dim, out_dim);
        }

        // Convert FP32 output back to BF16
        {
            PERF_TRACE_SCOPE_CAT("fp32_to_bf16", "linear");
            auto bf16_output = std::dynamic_pointer_cast<BF16Tensor>(output);
            if (!bf16_output)
            {
                LOG_ERROR("Output is not BF16Tensor in BF16 path");
                return false;
            }

            bfloat16 *output_bf16 = bf16_output->bf16_data(); // Non-const accessor already exists
            const float *output_fp32 = global_output_fp32->data();

#pragma omp parallel for
            for (size_t i = 0; i < seq_len * out_dim; ++i)
            {
                output_bf16[i] = static_cast<bfloat16>(output_fp32[i]);
            }
        }

        // Validate output (check FP32 version before conversion)
        ASSERT_TENSOR_NOT_NAN(global_output_fp32, "Linear BF16 output (FP32)");

        return true;
    }

    std::shared_ptr<TensorBase> MPILinearOperator_v2::getOrCacheWeight(
        const std::shared_ptr<TensorBase> &global_weight,
        size_t output_size,
        int local_output_size,
        size_t input_size)
    {

        CacheKey weight_key{global_weight.get(), output_size};
        auto weight_it = weight_cache_.find(weight_key);

        if (weight_it != weight_cache_.end())
        {
            PERF_TRACE_SCOPE_CAT("weight_cache_hit", "linear");
            return weight_it->second;
        }

        PERF_TRACE_SCOPE_CAT("weight_cache_miss", "linear");

        // For quantized weights: Cache reference to global weight (streaming dequant handles distribution)
        if (global_weight->native_type() == TensorDataType::QUANTIZED)
        {
            LOG_DEBUG("Caching global quantized weight reference (streaming dequant path)");
            weight_cache_[weight_key] = global_weight;
            return global_weight;
        }

        // For FP32 weights: Distribute and cache local rows
        LOG_DEBUG("Distributing FP32 weight rows to rank " << getRank());
        auto local_weight = TensorFactory::create_simple({local_output_size, static_cast<int>(input_size)});
        distributeWeightFP32(global_weight, local_weight, output_size);
        weight_cache_[weight_key] = local_weight;
        return local_weight;
    }

    void MPILinearOperator_v2::distributeWeightFP32(const std::shared_ptr<TensorBase> &global_weight,
                                                    std::shared_ptr<TensorBase> &local_weight,
                                                    size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("distribute_weight_fp32", "linear");

        const size_t input_size = global_weight->shape()[1];
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // FP32 weights - direct copy of local rows
        const float *global_data = global_weight->data();
        float *local_data = local_weight->data();
        const size_t row_size = input_size * sizeof(float);

        for (size_t i = 0; i < static_cast<size_t>(local_output_size); ++i)
        {
            const size_t global_row = output_offset + i;
            std::memcpy(local_data + i * input_size,
                        global_data + global_row * input_size,
                        row_size);
        }
    }

    void MPILinearOperator_v2::distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                                              std::shared_ptr<TensorBase> &local_bias,
                                              size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("distribute_bias", "linear");

        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        const float *global_data = global_bias->data();
        float *local_data = local_bias->data();

        std::memcpy(local_data,
                    global_data + output_offset,
                    local_output_size * sizeof(float));
    }

    void MPILinearOperator_v2::gatherOutput(const float *local_output, float *global_output,
                                            size_t seq_len, size_t local_output_size,
                                            size_t global_output_size)
    {
        PERF_TRACE_SCOPE_CAT("gather_output", "mpi_collective");

        // Calculate receive counts and displacements for each rank
        std::vector<int> recvcounts(getSize());
        std::vector<int> displs(getSize());

        for (int rank = 0; rank < getSize(); ++rank)
        {
            auto [rank_local_size, rank_offset] = getRowDistribution(global_output_size, rank);
            recvcounts[rank] = static_cast<int>(seq_len * rank_local_size);
            displs[rank] = static_cast<int>(seq_len * rank_offset);
        }

        // Gather all outputs
        MPI_Allgatherv(
            local_output,                                  // sendbuf
            static_cast<int>(seq_len * local_output_size), // sendcount
            MPI_FLOAT,                                     // sendtype
            global_output,                                 // recvbuf
            recvcounts.data(),                             // recvcounts
            displs.data(),                                 // displs
            MPI_FLOAT,                                     // recvtype
            getComm()                                      // comm
        );
    }

    void MPILinearOperator_v2::addBias(float *output, const float *bias,
                                       size_t seq_len, size_t output_size)
    {
#pragma omp parallel for
        for (size_t i = 0; i < seq_len; ++i)
        {
            float *row = output + i * output_size;
            for (size_t j = 0; j < output_size; ++j)
            {
                row[j] += bias[j];
            }
        }
    }

    bool MPILinearOperator_v2::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                        const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Check input/output count
        if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
        {
            LOG_ERROR("MPILinearOperator_v2: Expected 2-3 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPILinearOperator_v2: Null tensor provided");
            return false;
        }

        // Check dimensions
        if (input->shape().size() != 2 || weight->shape().size() != 2 || output->shape().size() != 2)
        {
            LOG_ERROR("MPILinearOperator_v2: All tensors must be 2D");
            return false;
        }

        const size_t seq_len = input->shape()[0];
        const size_t in_dim = input->shape()[1];
        const size_t out_dim = weight->shape()[0];
        const size_t weight_in_dim = weight->shape()[1];

        // Validate dimensions
        if (weight_in_dim != in_dim)
        {
            LOG_ERROR("MPILinearOperator_v2: Weight input dimension mismatch - "
                      "weight["
                      << out_dim << "," << weight_in_dim << "] vs "
                                                            "input["
                      << seq_len << "," << in_dim << "]");
            return false;
        }

        if (output->shape()[0] != seq_len || output->shape()[1] != out_dim)
        {
            LOG_ERROR("MPILinearOperator_v2: Output shape mismatch - "
                      "expected ["
                      << seq_len << "," << out_dim << "], got ["
                      << output->shape()[0] << "," << output->shape()[1] << "]");
            return false;
        }

        // Validate activation/output type matching
        auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input);
        auto bf16_output = std::dynamic_pointer_cast<BF16Tensor>(output);

        if (bf16_input && !bf16_output)
        {
            LOG_ERROR("MPILinearOperator_v2: BF16 input requires BF16 output");
            return false;
        }

        if (!bf16_input && bf16_output)
        {
            LOG_ERROR("MPILinearOperator_v2: FP32 input requires FP32 output");
            return false;
        }

        // Validate optional bias
        if (inputs.size() == 3 && inputs[2])
        {
            auto bias = inputs[2];
            if (bias->shape().size() != 1 || bias->shape()[0] != out_dim)
            {
                LOG_ERROR("MPILinearOperator_v2: Bias shape mismatch - "
                          "expected ["
                          << out_dim << "], got [" << bias->shape()[0] << "]");
                return false;
            }
        }

        return true;
    }

} // namespace llaminar
