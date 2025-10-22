/**
 * @file MPILinearOperator.cpp
 * @brief MPI-aware column-partitioned linear projection kernel.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Activations tensor [seq_len, in_dim] (row-major, replicated on all ranks).
 *  - inputs[1]: Global weight tensor [out_dim, in_dim] (row-major; will be column-partition distributed).
 *  - inputs[2] (optional): Global bias [out_dim].
 * Outputs:
 *  - outputs[0]: Global output tensor [seq_len, out_dim] (row-major; assembled via Allgatherv over column partitions).
 * Weight Convention:
 *  - Weights are stored as [out_features, in_features] matching PyTorch nn.Linear and GGUF format.
 *  - Applied as output = input @ weight^T (transpose during matmul).
 *  - See docs/WEIGHT_MATRIX_CONVENTIONS.md for detailed rationale.
 * Distribution Strategy:
 *  - Weight columns are block-distributed across ranks; activations are replicated.
 *  - Local GEMM: [seq_len, in_dim] @ [in_dim, out_dim_local]^T -> [seq_len, out_dim_local].
 *  - Optional local bias add then global gather.
 * Numerical Expectations:
 *  - Deterministic for identical OpenMP scheduling when OMP_NUM_THREADS=1.
 *  - Accumulation in float; differences vs single-process reference bounded by floating reduction order on final gather.
 * Error Modes:
 *  - Shape mismatches, null tensors, NaN detection trigger logged error and return false.
 *  - MPI distribution inconsistencies (size mismatch) abort execution.
 * Threading:
 *  - OpenMP parallelism inside adaptiveMatMul; environment may cap threads.
 *  - No concurrent mutation of shared state beyond local temporaries.
 * MPI:
 *  - Uses rank/size from MPIKernelBase; collective communications must remain ordered.
 * @author David Sanftenberg
 */
#include "MPILinearOperator.h"
#include "../Logger.h"
#include "../PerformanceTracer.h"
#include "../DebugUtils.h"
#include "../AdaptiveMatmul.h"
#include "../utils/DebugEnv.h"
#include "QuantSlabCache.h"
#include "../tensors/TensorFactory.h"
#include "../tensors/BF16Tensor.h"
#include "../tensors/Q8_0Tensor.h" // For Q8_0 streaming decode
#include "attention/AttentionStageContracts.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <omp.h>

namespace llaminar
{

    MPILinearOperator::MPILinearOperator(MPI_Comm comm) : MPIOperatorBase(comm, false)
    {
        LOG_DEBUG("MPILinearOperator initialized on rank " << getRank() << " of " << getSize());
    }

    bool MPILinearOperator::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                    std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_TRACE_SCOPE_CAT("mpi_linear_execute", "linear");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPILinearOperator validation failed on rank " << getRank());
            return false;
        }

        auto input = inputs[0];
        auto global_weight = inputs[1];
        auto global_output = outputs[0];

        // === COMPREHENSIVE TENSOR VALIDATION ===
        ASSERT_TENSOR_VALID(input, "Linear input");
        // Skip data() validation for Q8_0 weights (use native_type check instead)
        if (global_weight->native_type() != TensorDataType::QUANTIZED)
        {
            ASSERT_TENSOR_VALID(global_weight, "Linear weight");
        }
        else
        {
            // For quantized tensors, just check non-null
            ASSERT_TENSOR_NOT_NULL(global_weight, "Linear weight (Q8_0)");
        }
        ASSERT_TENSOR_VALID(global_output, "Linear output");

        // Check for NaN in inputs before computation (skip direct NaN scan for quantized weights: raw bytes not FP32)
        ASSERT_TENSOR_NOT_NAN(input, "Linear input");
        // Skip NaN check for Q8_0Tensor (doesn't support data() - raw bytes not FP32)
        if (global_weight->native_type() != TensorDataType::QUANTIZED && global_weight->data())
        {
            ASSERT_TENSOR_NOT_NAN(global_weight, "Linear weight");
        }

        // Log detailed tensor information
        // COMMENTED OUT - May call data() on Q8_0
        // TensorLogger::logMatMulOperation(input, global_weight, global_output, "MPILinearOperator");

        // Extract dimensions
        size_t seq_len = input->shape()[0];
        size_t input_size = input->shape()[1];
        // Weight is [out_dim, in_dim] per new convention
        size_t output_size = global_weight->shape()[0];
        size_t weight_in_dim = global_weight->shape()[1];

        // Early return for empty tensors (e.g., rank with no tokens)
        if (seq_len == 0 || input_size == 0 || output_size == 0)
        {
            LOG_DEBUG("MPILinearOperator: Empty tensor detected (seq_len=" << seq_len << ", input_size=" << input_size << ", output_size=" << output_size << "), skipping computation on rank " << getRank());
            return true; // Success - nothing to compute
        }

        // Validate dimension compatibility
        if (weight_in_dim != input_size)
        {
            LOG_ERROR("MPILinear: Weight input dimension mismatch - weight[" << global_weight->shape()[0] << ", " << global_weight->shape()[1] << "] vs input[" << seq_len << ", " << input_size << "]");
            return false;
        }

        // Calculate local distribution
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Create local output tensor (needed for all paths)
        auto local_output = createLocalTensor({seq_len, static_cast<size_t>(local_output_size)});

        // Handle optional bias with caching
        std::shared_ptr<TensorBase> local_bias = nullptr;
        if (inputs.size() >= 3 && inputs[2])
        {
            const float *bias_key = inputs[2]->data();
            auto bias_cache_it = bias_cache_.find(bias_key);

            if (bias_cache_it != bias_cache_.end())
            {
                // Cache hit: reuse previously distributed bias
                PERF_TRACE_SCOPE_CAT("bias_cache_hit", "linear_kernel");
                local_bias = bias_cache_it->second;
            }
            else
            {
                // Cache miss: distribute and cache the bias
                PERF_TRACE_SCOPE_CAT("bias_cache_miss", "linear_kernel");
                local_bias = createLocalTensor({static_cast<size_t>(local_output_size)});
                {
                    PERF_TRACE_SCOPE_CAT("distribute_bias", "linear_kernel");
                    distributeBias(inputs[2], local_bias, output_size);
                }
                bias_cache_[bias_key] = local_bias;
            }
        }

        // === Q8_0 STREAMING DECODE PATH (Week 2) ===
        // Check if weight is Q8_0Tensor FIRST - bypass weight distribution for streaming decode
        auto q8_weight = std::dynamic_pointer_cast<Q8_0Tensor>(global_weight);
        if (q8_weight)
        {
            PERF_TRACE_SCOPE_CAT("q8_0_decode_and_matmul", "linear_kernel");

            LOG_DEBUG("MPILinear: Using Q8_0 streaming decode path on rank " << getRank()
                                                                             << " (local_output_size=" << local_output_size << ", input_size=" << input_size << ")");

            // Allocate temporary buffer for local weight slice (released after forward)
            std::vector<float> decoded_weight(local_output_size * input_size);

            // Decode local rows [output_offset, output_offset + local_output_size)
            {
                PERF_TRACE_SCOPE_CAT("q8_0_decode_rows", "q8_0");
                auto decode_start = std::chrono::high_resolution_clock::now();

                // Decode row-by-row (could parallelize with OpenMP if needed)
                for (size_t local_row = 0; local_row < static_cast<size_t>(local_output_size); local_row++)
                {
                    size_t global_row = output_offset + local_row;
                    q8_weight->decodeRow(global_row, decoded_weight.data() + local_row * input_size);
                }

                auto decode_end = std::chrono::high_resolution_clock::now();
                double decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count() / 1000.0;
                LOG_DEBUG("MPILinear: Q8_0 decode " << local_output_size << " rows in " << decode_ms << "ms on rank " << getRank());
            }

            // Perform matmul with decoded weights
            {
                PERF_TRACE_SCOPE_CAT("q8_0_matmul", "linear_kernel");
                auto matmul_start = std::chrono::high_resolution_clock::now();

                // Matrix multiplication: output = input @ weight^T
                // Weight is [local_out_dim, in_dim], so we transpose it during matmul
                bool matmul_success = adaptiveMatMul(
                    input->data(), decoded_weight.data(), local_output->data(),
                    static_cast<int>(seq_len), static_cast<int>(local_output_size), static_cast<int>(input_size),
                    /*is_prefill*/ false,
                    /*distributed_partition*/ true,
                    /*transpose_A*/ false,
                    /*transpose_B*/ true, // Weight is [local_out, in], need transpose
                    /*alpha*/ 1.0f,
                    /*beta*/ 0.0f);

                auto matmul_end = std::chrono::high_resolution_clock::now();
                double matmul_ms = std::chrono::duration_cast<std::chrono::microseconds>(matmul_end - matmul_start).count() / 1000.0;
                LOG_DEBUG("MPILinear: Q8_0 matmul " << matmul_ms << "ms on rank " << getRank());

                if (!matmul_success)
                {
                    LOG_ERROR("Q8_0 adaptive matmul failed on rank " << getRank());
                    return false;
                }
            }

            // Add bias if provided
            if (local_bias)
            {
                PERF_TRACE_SCOPE_CAT("add_bias", "linear_kernel");
                addBiasLocal(local_output->data(), local_bias->data(), seq_len, local_output_size);
            }

            // Gather results from all processes
            {
                PERF_TRACE_SCOPE_CAT("gather_output", "mpi_collective");
                gatherOutput(local_output, global_output, seq_len, output_size);
            }

            // Validate output
            ASSERT_TENSOR_NOT_NAN(global_output, "Linear output after Q8_0 computation");
            TensorLogger::logTensorStats(global_output, "Linear Q8_0_output", "MPILinearOperator_Q8_0");

            LOG_DEBUG("MPILinear: Q8_0 path complete on rank " << getRank());
            return true;
        }

        // === NON-Q8_0 PATH: Distribute weights (regular FP32 or quantized slab) ===
        // Check weight cache before distributing
        // For Q8_0Tensor, use tensor pointer as key (data() not supported)
        // For regular tensors, use data pointer as key
        const float *weight_key = nullptr;
        if (global_weight->native_type() != TensorDataType::QUANTIZED)
        {
            weight_key = global_weight->data();
        }
        else
        {
            // Use tensor pointer as key for quantized tensors
            weight_key = reinterpret_cast<const float *>(global_weight.get());
        }
        std::shared_ptr<TensorBase> local_weight;

        auto weight_cache_it = weight_cache_.find(weight_key);
        if (weight_cache_it != weight_cache_.end())
        {
            // Cache hit: reuse previously distributed weight
            PERF_TRACE_SCOPE_CAT("weight_cache_hit", "linear_kernel");
            local_weight = weight_cache_it->second;
        }
        else
        {
            // Cache miss: distribute and cache the weight
            PERF_TRACE_SCOPE_CAT("weight_cache_miss", "linear_kernel");
            local_weight = createLocalTensor({static_cast<size_t>(local_output_size), input_size});
            {
                PERF_TRACE_SCOPE_CAT("distribute_weight", "linear_kernel");
                distributeWeight(global_weight, local_weight, output_size);
            }
            weight_cache_[weight_key] = local_weight;
        }

        // Perform local computation using COSMA
        // Matrix multiplication: local_output = input * local_weight
        // Use adaptive matrix multiplication for optimal performance
        const float *input_data = input->data();
        const float *weight_data = local_weight->data();

        // === Quantized slab path ===
        bool use_slab = debugEnv().quant.slab_enable && debugEnv().quant_slab.enable;
        const auto *quant_tensor = dynamic_cast<QuantizedTensor *>(global_weight.get());
        if (use_slab && quant_tensor)
        {
            // Decode slab for this rank's local output slice (rows = local_output_size, cols = input_size logically after transpose)
            // Weight shape is [out_dim, in_dim]; our local rows span [output_offset, output_offset + local_output_size)
            size_t output_offset;
            size_t local_output_size_tmp;
            std::tie(local_output_size_tmp, output_offset) = getRowDistribution(output_size);
            (void)local_output_size_tmp;          // already have local_output_size
            size_t col_start = output_offset;     // columns in transposed view correspond to output rows pre-transpose
            size_t col_count = local_output_size; // span for this rank
            QuantSlab slab;
            bool reused = QuantSlabCache::instance().getOrDecode(*quant_tensor, col_start, col_count, slab, /*reuse_allowed*/ true);
            if (debugEnv().quant.slab_stats && getRank() == 0)
            {
                LOG_INFO("[QuantSlab] rank=" << getRank() << " reused=" << (reused ? 1 : 0) << " k=" << slab.k << " n=" << slab.n);
            }

            // Try direct BF16 GEMM path first (requires LLAMINAR_QUANT_BF16_GEMM=1)
            // Slab stored as (k x n_local) row-major; B is already in correct orientation (transpose_B=false).
            bool ok = adaptiveMatMulBF16(input_data, slab.data.data(), local_output->data(),
                                         (int)seq_len, (int)local_output_size, (int)input_size,
                                         /*alpha*/ 1.0f, /*beta*/ 0.0f,
                                         /*is_prefill*/ false,
                                         /*distributed_partition*/ true,
                                         /*transpose_B*/ false);

            // Fallback to BF16→FP32 expansion if BF16 GEMM disabled or failed
            if (!ok)
            {
                if (debugEnv().quant.slab_stats && getRank() == 0)
                {
                    LOG_DEBUG("[QuantSlab] BF16 GEMM unavailable, falling back to FP32 expansion");
                }

                // Expand BF16 slab to FP32 then let existing adaptive path handle backend selection.
                std::vector<float> slab_fp32(slab.k * slab.n);
#ifdef _OPENMP
                size_t total = slab_fp32.size();
#pragma omp parallel for if (total > 32768) schedule(static)
                for (size_t idx = 0; idx < total; ++idx)
                    slab_fp32[idx] = (float)slab.data[idx];
#else
                for (size_t idx = 0; idx < slab_fp32.size(); ++idx)
                    slab_fp32[idx] = (float)slab.data[idx];
#endif
                // Perform matmul immediately using expanded weights
                ok = adaptiveMatMul(input_data, slab_fp32.data(), local_output->data(),
                                    (int)seq_len, (int)local_output_size, (int)input_size,
                                    /*is_prefill*/ false,
                                    /*distributed_partition*/ true,
                                    /*transpose_A*/ false,
                                    /*transpose_B*/ false,
                                    /*alpha*/ 1.0f,
                                    /*beta*/ 0.0f);
                if (!ok)
                {
                    LOG_ERROR("BF16 expanded adaptive matmul failed on rank " << getRank());
                }
            }
            return ok;
        }
        float *output_data = local_output->data();

        int seq_len_int = static_cast<int>(seq_len);
        int d_model = static_cast<int>(input_size);
        int d_out = static_cast<int>(local_output_size);

        // Optional lightweight diagnostics (now minimal): only if LLAMINAR_LINEAR_DIAG is set.
        bool linear_diag = debugEnv().linear.diag;
        if (linear_diag && getRank() == 0)
        {
            LOG_INFO("[LinearDiag] rank=" << getRank() << " seq_len=" << seq_len_int
                                          << " d_model=" << d_model << " d_out_local=" << d_out);
        }

        // Use adaptive matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        bool matmul_success;
        {
            PERF_TRACE_SCOPE_CAT("linear_matmul", "linear_kernel");
            // Matrix multiplication: output = input @ weight^T
            // Weight is [local_out_dim, in_dim], so we transpose it during matmul
            // Use new TensorBase* overload for potential fused quantized GEMM
            matmul_success = adaptiveMatMul(input_data, local_weight.get(), output_data,
                                            seq_len_int, d_out, d_model,
                                            /*is_prefill*/ false,
                                            /*distributed_partition*/ true,
                                            /*transpose_A*/ false,
                                            /*transpose_B*/ true,
                                            /*alpha*/ 1.0f,
                                            /*beta*/ 0.0f);
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (!matmul_success)
        {
            LOG_ERROR("Adaptive matrix multiplication failed on rank " << getRank());
            return false;
        }

        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_DEBUG("MPILinear matmul: " << ms << "ms for " << seq_len_int << "x" << d_model << " * " << d_model << "x" << d_out);
        // (Legacy deep diagnostic & auto parity overwrite removed.)

        // Add bias if provided
        if (local_bias)
        {
            PERF_TRACE_SCOPE_CAT("add_bias", "linear_kernel");
            addBiasLocal(local_output->data(), local_bias->data(), seq_len, local_output_size);
        }

        // Gather results from all processes
        {
            PERF_TRACE_SCOPE_CAT("gather_output", "mpi_collective");
            gatherOutput(local_output, global_output, seq_len, output_size);
        }

        if (linear_diag && getRank() == 0)
        {
            LOG_INFO("[LinearDiag] rank=" << getRank() << " gathered_global_shape=[" << seq_len << "," << output_size << "]");
        }

        // === POST-COMPUTATION VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(global_output, "Linear output after computation");

        // Log output tensor statistics
        TensorLogger::logTensorStats(global_output, "Linear final_output", "MPILinearOperator_COMPLETE");

        LOG_DEBUG("MPILinearOperator executed successfully on rank " << getRank());
        return true;
    }

    bool MPILinearOperator::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Count check
        if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
        {
            LOG_ERROR("MPILinearOperator: Expected 2-3 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        // Extract dimensions from inputs for contract specification
        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPILinearOperator: Null tensor provided");
            return false;
        }

        // Early dimension check before building contracts
        if (input->shape().size() != 2 || weight->shape().size() != 2)
        {
            LOG_ERROR("MPILinearOperator: Input and weight must be 2D tensors");
            return false;
        }

        const int seq_len = input->shape()[0];
        const int in_dim = input->shape()[1];
        const int out_dim = weight->shape()[0];
        const int weight_in_dim = weight->shape()[1];

        // Build input contract
        StageContract input_contract("MPILinearOperator::validate");
        input_contract.inputs = {
            TensorContract("input",
                           ShapeSpec({seq_len, in_dim}, {"seq_len", "in_dim"}),
                           TensorLayout::RowMajor,
                           TensorSemantic::Activation),
            TensorContract("weight",
                           ShapeSpec({out_dim, in_dim}, {"out_dim", "in_dim"}),
                           TensorLayout::RowMajor,
                           TensorSemantic::Weight)};

        // Add optional bias contract
        if (inputs.size() == 3)
        {
            input_contract.inputs.push_back(
                TensorContract("bias",
                               ShapeSpec({out_dim}, {"out_dim"}),
                               TensorLayout::RowMajor,
                               TensorSemantic::Weight,
                               true,  // optional
                               false) // no broadcast
            );
        }

        // Build output contract
        input_contract.outputs = {
            TensorContract("output",
                           ShapeSpec({seq_len, out_dim}, {"seq_len", "out_dim"}),
                           TensorLayout::RowMajor,
                           TensorSemantic::Activation)};

        // Validate using contracts
        try
        {
            input_contract.validate_inputs(inputs);
            input_contract.validate_outputs(outputs);
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("MPILinearOperator contract violation: " << e.what());
            return false;
        }
    }

    void MPILinearOperator::distributeWeight(const std::shared_ptr<TensorBase> &global_weight,
                                             std::shared_ptr<TensorBase> &local_weight,
                                             size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("distribute_weight_internal", "linear_kernel");

        // New convention: weight is [output_size, input_size]
        size_t input_size = global_weight->shape()[1];
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Check if weight is quantized - need to decode first
        if (global_weight->native_type() == TensorDataType::QUANTIZED)
        {
            std::cerr << "[DISTRIBUTE_Q8_0] Weight is Q8_0, decoding to FP32 first" << std::endl;

            // Cast to Q8_0Tensor to access decodeRow
            auto q8_weight = std::dynamic_pointer_cast<Q8_0Tensor>(global_weight);
            if (!q8_weight)
            {
                LOG_ERROR("Weight is quantized but not Q8_0Tensor!");
                return;
            }

            // Decode Q8_0 to FP32 buffer
            std::vector<float> decoded_weight(output_size * input_size);
            for (size_t row = 0; row < output_size; ++row)
            {
                q8_weight->decodeRow(row, decoded_weight.data() + row * input_size);
            }

            // Now extract local portion from decoded buffer
            float *local_data = local_weight->data();
            const size_t elements_to_copy = local_output_size * input_size;
            const float *src_start = decoded_weight.data() + output_offset * input_size;

            std::memcpy(local_data, src_start, elements_to_copy * sizeof(float));

            LOG_DEBUG("Distributed Q8_0 weight matrix: local size [" << local_output_size << ", " << input_size
                                                                     << "], offset " << output_offset << " on rank " << getRank());
            return;
        }

        // Extract local portion of weight matrix
        // Global weight: [output_size, input_size] - row-major storage
        // Local weight: [local_output_size, input_size] - subset of rows
        const float *global_data = global_weight->data();
        float *local_data = local_weight->data();

        // With weight as [output_size, input_size], each row is one output feature.
        // We partition output features across ranks, so each rank gets a contiguous block of rows.
        // Row offset in global: output_offset
        // Number of rows to copy: local_output_size
        // Each row has input_size elements (contiguous in memory).
        const size_t elements_to_copy = local_output_size * input_size;
        const float *src_start = global_data + output_offset * input_size;

        // Single contiguous memcpy is optimal for row-major slicing
        {
            PERF_TRACE_SCOPE_CAT("weight_memcpy", "linear_kernel");
            std::memcpy(local_data, src_start, elements_to_copy * sizeof(float));
        }

        LOG_DEBUG("Distributed weight matrix: local size [" << local_output_size << ", " << input_size
                                                            << "], offset " << output_offset << " on rank " << getRank());
    }

    void MPILinearOperator::distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                                           std::shared_ptr<TensorBase> &local_bias,
                                           size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("distribute_bias_internal", "linear_kernel");

        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Extract local portion of bias vector
        const float *global_data = global_bias->data();
        float *local_data = local_bias->data();

        {
            PERF_TRACE_SCOPE_CAT("bias_memcpy", "linear_kernel");
            std::memcpy(local_data, global_data + output_offset, local_output_size * sizeof(float));
        }

        LOG_DEBUG("Distributed bias vector: local size " << local_output_size
                                                         << ", offset " << output_offset << " on rank " << getRank());
    }

    void MPILinearOperator::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                         std::shared_ptr<TensorBase> &global_output,
                                         size_t seq_len,
                                         size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("gather_output_internal", "linear_kernel");

        // OPTIMIZED: Single MPI_Allgatherv instead of per-row loop
        // Strategy: Pack all local data, single gather, unpack into strided global output

        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Compute recv_counts and recv_offsets for packed data (all ranks contribute seq_len * local_size elements)
        std::vector<int> recv_counts(getSize());
        std::vector<int> recv_offsets(getSize());
        std::vector<size_t> rank_local_sizes(getSize());
        std::vector<size_t> rank_offsets(getSize());

        {
            PERF_TRACE_SCOPE_CAT("gather_setup_metadata", "linear_kernel");
            size_t packed_offset = 0;
            for (int rank = 0; rank < getSize(); ++rank)
            {
                auto [rank_local_size, rank_col_offset] = getRowDistribution(output_size, rank);
                rank_local_sizes[rank] = rank_local_size;
                rank_offsets[rank] = rank_col_offset;

                // Each rank sends seq_len * rank_local_size elements (contiguous in local buffer)
                recv_counts[rank] = static_cast<int>(seq_len * rank_local_size);
                recv_offsets[rank] = static_cast<int>(packed_offset);
                packed_offset += seq_len * rank_local_size;
            }
        }

        // Allocate packed buffer for gathered data
        std::vector<float> packed_global(seq_len * output_size);

        // Single Allgatherv: Gather all packed local data
        {
            PERF_TRACE_SCOPE_CAT("allgatherv_single_collective", "mpi_collective");
            const float *local_data = local_output->data(); // Already contiguous [seq_len, local_output_size]
            int send_count = static_cast<int>(seq_len * local_output_size);

            checkMPIError(MPI_Allgatherv(local_data, send_count, MPI_FLOAT,
                                         packed_global.data(), recv_counts.data(), recv_offsets.data(), MPI_FLOAT,
                                         getComm()),
                          "MPI_Allgatherv in gatherOutput (optimized)");
        }

        // Unpack: Interleave columns from each rank into global output
        {
            PERF_TRACE_SCOPE_CAT("unpack_interleaved_columns", "linear_kernel");
            float *global_data = global_output->data();

            // Optimize for common case: 2 ranks (can unroll inner loop)
            if (getSize() == 2)
            {
                // Two-rank fast path: unroll the rank loop
                size_t size0 = rank_local_sizes[0];
                size_t size1 = rank_local_sizes[1];
                size_t offset0 = rank_offsets[0];
                size_t offset1 = rank_offsets[1];
                int recv_off0 = recv_offsets[0];
                int recv_off1 = recv_offsets[1];

#pragma omp parallel for schedule(static)
                for (size_t row = 0; row < seq_len; ++row)
                {
                    float *global_row = global_data + row * output_size;
                    const float *src0 = packed_global.data() + recv_off0 + row * size0;
                    const float *src1 = packed_global.data() + recv_off1 + row * size1;

                    std::memcpy(global_row + offset0, src0, size0 * sizeof(float));
                    std::memcpy(global_row + offset1, src1, size1 * sizeof(float));
                }
            }
            else
            {
// General multi-rank path
#pragma omp parallel for schedule(static)
                for (size_t row = 0; row < seq_len; ++row)
                {
                    float *global_row = global_data + row * output_size;

                    // Copy column blocks from each rank into the global row
                    for (int rank = 0; rank < getSize(); ++rank)
                    {
                        size_t rank_local_size = rank_local_sizes[rank];
                        size_t rank_col_offset = rank_offsets[rank];

                        // Source: packed_global at [recv_offsets[rank] + row * rank_local_size]
                        const float *src = packed_global.data() + recv_offsets[rank] + row * rank_local_size;

                        // Destination: global_row at column offset rank_col_offset
                        float *dst = global_row + rank_col_offset;

                        // Copy rank's column block for this row
                        std::memcpy(dst, src, rank_local_size * sizeof(float));
                    }
                }
            }
        }

        LOG_DEBUG("Gathered output (optimized): [" << seq_len << ", " << output_size << "] on rank " << getRank());
    }

    void MPILinearOperator::addBiasLocal(float *output, const float *bias,
                                         size_t seq_len, size_t local_output_size)
    {
        PERF_TRACE_SCOPE_CAT("add_bias_internal", "linear_kernel");

        // Add bias to each sequence position: output[i, j] += bias[j]
        auto omp_start = std::chrono::high_resolution_clock::now();
        {
            PERF_TRACE_SCOPE_CAT("bias_omp_parallel_loop", "linear_kernel");
#pragma omp parallel for
            for (size_t i = 0; i < seq_len; ++i)
            {
#pragma omp simd
                for (size_t j = 0; j < local_output_size; ++j)
                {
                    output[i * local_output_size + j] += bias[j];
                }
            }
        }
        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("OpenMP bias addition: " << omp_time << "ms, threads: " << omp_get_max_threads() << ", rank: " << getRank());

        LOG_DEBUG("Local bias addition completed: [" << seq_len << ", " << local_output_size
                                                     << "] on rank " << getRank());
    }

    std::shared_ptr<TensorBase> MPILinearOperator::createLocalTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());

        // Phase 5: BF16 activation storage
        const auto &env = debugEnv();
        if (env.quant.output_bf16)
        {
            return TensorFactory::create_bf16(int_shape);
        }

        // Default: FP32 storage
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar