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
        ASSERT_TENSOR_VALID(global_weight, "Linear weight");
        ASSERT_TENSOR_VALID(global_output, "Linear output");

        // Check for NaN in inputs before computation
        ASSERT_TENSOR_NOT_NAN(input, "Linear input");
        ASSERT_TENSOR_NOT_NAN(global_weight, "Linear weight");

        // Log detailed tensor information
        TensorLogger::logMatMulOperation(input, global_weight, global_output, "MPILinearOperator");

        // Extract dimensions
        size_t seq_len = input->shape()[0];
        size_t input_size = input->shape()[1];
        // Weight is [out_dim, in_dim] per new convention
        size_t output_size = global_weight->shape()[0];
        size_t weight_in_dim = global_weight->shape()[1];

        // Validate dimension compatibility
        if (weight_in_dim != input_size)
        {
            LOG_ERROR("MPILinear: Weight input dimension mismatch - weight[" << global_weight->shape()[0] << ", " << global_weight->shape()[1] << "] vs input[" << seq_len << ", " << input_size << "]");
            return false;
        }

        // Calculate local distribution
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Check weight cache before distributing
        const float *weight_key = global_weight->data();
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

        // Create local output tensor
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

        // Ensure input is available on all processes (broadcast if needed)
        // For simplicity, assuming input is already replicated across processes

        // Perform local computation using COSMA
        // Matrix multiplication: local_output = input * local_weight
        // Use adaptive matrix multiplication for optimal performance
        const float *input_data = input->data();
        const float *weight_data = local_weight->data();
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
            matmul_success = adaptiveMatMul(input_data, weight_data, output_data,
                                            seq_len_int, d_out, d_model,
                                            /*is_prefill*/ false,
                                            /*distributed_partition*/ true,
                                            /*transpose_A*/ false,
                                            /*transpose_B*/ true, // CRITICAL: Transpose weight per PyTorch/GGUF convention
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
        // Use TensorFactory to create a modern tensor
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar