/**
 * @file MPIRoPEKernel.cpp
 * @brief Applies Rotary Positional Embeddings (RoPE) to query/key tensors in a distributed context.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Query tensor Q [seq_len, n_heads, head_dim] (row-major contiguous; replicated across ranks).
 *  - inputs[1]: Key tensor K   [seq_len, n_kv_heads, head_dim] (replicated; may be subset of heads for GQA).
 *  - inputs[2] (optional): Precomputed cos/sin cache tensor [max_rot, head_dim]. If absent, kernel may compute on demand.
 * Outputs:
 *  - outputs[0]: In-place or new tensor for rotated Q (same shape as Q).
 *  - outputs[1]: In-place or new tensor for rotated K (same shape as K).
 * Behavior:
 *  - Even-index dims treated as real, odd as imaginary components for complex rotation pairs.
 *  - Angle index derived from absolute sequence position (supports prefill + decode continuity if provided offset env).
 * Numerical Properties:
 *  - Pure elementwise fused sin/cos application; deterministic.
 *  - Maximum relative error vs reference (double precision build) expected < 2e-7 for float32 inputs.
 * Error Modes:
 *  - Dimension mismatch between Q and K head_dim.
 *  - head_dim must be even (strictly enforced); violation returns false with LOG_ERROR.
 *  - Null tensor pointers.
 * Distribution:
 *  - Currently replicated application; future optimization may shard sequence dimension with identical rotation factors.
 * Threading:
 *  - OpenMP parallel for over (seq_len * total_heads) groups; no data races (distinct index ranges).
 * @note Any caching of sin/cos tables must remain thread-safe (either precomputed or guarded).
 * @author David Sanftenberg
 */
#include "MPIRoPEKernel.h"
#include "../debug_utils.h"
#include "../performance_timer.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

namespace llaminar
{

    MPIRoPEKernel::MPIRoPEKernel(int max_seq_len, int head_dim, float theta, DistributionStrategy strategy)
        : MPIKernelBase(), max_seq_len_(max_seq_len), head_dim_(head_dim), theta_(theta),
          strategy_(strategy), num_threads_(omp_get_max_threads())
    {
        if (head_dim_ % 2 != 0)
        {
            LOG_ERROR("MPIRoPEKernel: head_dim must be even, got " << head_dim_);
            throw std::invalid_argument("head_dim must be even for RoPE");
        }

        precomputeFrequencyTables();

        LOG_DEBUG("MPIRoPEKernel initialized on rank " << getRank() << "/" << getSize()
                                                       << " with max_seq_len: " << max_seq_len_ << ", head_dim: " << head_dim_
                                                       << ", theta: " << theta_ << ", strategy: "
                                                       << (strategy == DistributionStrategy::SEQUENCE_WISE ? "SEQUENCE_WISE" : "HEAD_WISE")
                                                       << ", OpenMP threads: " << num_threads_);
    }

    bool MPIRoPEKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_SCOPED_TIMER("MPIRoPEKernel::execute");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIRoPEKernel: Validation failed");
            return false;
        }

        auto input_tensor = inputs[0];
        auto position_ids_tensor = inputs[1];
        auto output_tensor = outputs[0];

        const auto &input_shape = input_tensor->shape();
        int seq_len = input_shape[0];
        int n_heads = input_shape[1];
        int actual_head_dim = input_shape[2];

        if (actual_head_dim != head_dim_)
        {
            LOG_ERROR("MPIRoPEKernel: Head dimension mismatch - expected: " << head_dim_
                                                                            << ", got: " << actual_head_dim);
            return false;
        }

        if (seq_len > max_seq_len_)
        {
            LOG_WARN("MPIRoPEKernel: Sequence length " << seq_len
                                                       << " exceeds max_seq_len " << max_seq_len_ << ", updating tables");
            updateMaxSeqLen(seq_len);
        }

        // Configure OpenMP threading based on problem size
        size_t total_elements = seq_len * n_heads * head_dim_;
        configureOpenMPThreading(total_elements);

        const float *input_data = input_tensor->data();
        const float *position_ids_data = position_ids_tensor->data();
        float *output_data = output_tensor->data();

        // --- Diagnostics: capture pre-rotation slice for last position if enabled ---
        const auto &env = debugEnv();
        bool diag = env.attention.internal_diff && getRank() == 0;
        if (getRank() == 0)
        {
            static int gate_prints = 0;
            if (gate_prints < 4)
            {
                LOG_INFO("[RoPEKernelGate] internal_diff=" << env.attention.internal_diff << " layer_token_diff=" << env.pipeline.layer_token_diff << " replay_compare=" << env.pipeline.layer_replay_compare << " seq_len=" << seq_len);
                ++gate_prints;
            }
        }
        // Determine whether this is an incremental single-token decode (seq_len==1 with prior history encoded in position_ids)
        bool incr = (seq_len == 1 && position_ids_data[0] > 0);
        int preview = std::min(head_dim_ * n_heads, 8);
        std::array<float, 8> before{};
        before.fill(0.f);
        int last_pos_index = seq_len - 1;
        if (diag && preview > 0)
        {
            size_t row_offset = static_cast<size_t>(last_pos_index) * n_heads * head_dim_;
            for (int i = 0; i < preview; ++i)
                before[i] = input_data[row_offset + i];
            static int gate_notes = 0;
            if (!env.pipeline.layer_replay_compare && gate_notes < 1)
            {
                LOG_INFO("[RoPEDiag] note=rope_kernel_replay_compare_flag_off");
                ++gate_notes;
            }
        }

        // Convert position_ids to int array for efficiency
        std::vector<int> position_ids(seq_len);
        for (int i = 0; i < seq_len; ++i)
        {
            position_ids[i] = static_cast<int>(position_ids_data[i]);
        }

        // === ROPE INPUT VALIDATION ===
        ASSERT_TENSOR_VALID(input_tensor, "RoPE input");
        ASSERT_TENSOR_VALID(position_ids_tensor, "RoPE position_ids");
        ASSERT_TENSOR_NOT_NAN(input_tensor, "RoPE input has NaN");
        TensorLogger::logTensorStats(input_tensor, "rope_input");

        auto start_time = std::chrono::high_resolution_clock::now();

        // Execute based on distribution strategy
        try
        {
            switch (strategy_)
            {
            case DistributionStrategy::SEQUENCE_WISE:
                executeSequenceWise(input_data, position_ids.data(), output_data, seq_len, n_heads, head_dim_);
                break;
            case DistributionStrategy::HEAD_WISE:
                executeHeadWise(input_data, position_ids.data(), output_data, seq_len, n_heads, head_dim_);
                break;
            default:
                LOG_ERROR("MPIRoPEKernel: Unknown distribution strategy");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("MPIRoPEKernel: Execution failed: " << e.what());
            return false;
        }

        if (diag && preview > 0)
        {
            std::array<float, 8> after{};
            after.fill(0.f);
            size_t row_offset = static_cast<size_t>(last_pos_index) * n_heads * head_dim_;
            for (int i = 0; i < preview; ++i)
                after[i] = output_data[row_offset + i];
            // compute relative movement
            long double l2_b = 0, l2_a = 0, l2_delta = 0;
            for (int i = 0; i < preview; ++i)
            {
                long double b = before[i], a = after[i];
                l2_b += b * b;
                l2_a += a * a;
                long double d = a - b;
                l2_delta += d * d;
            }
            double rel_move = (l2_b > 0) ? std::sqrt((double)l2_delta / (double)l2_b) : 0.0;
            // derive first angle used
            int pos_last = static_cast<int>(position_ids_data[last_pos_index]);
            int pairs = head_dim_ / 2;
            float theta0 = (pairs > 0) ? (1.f / std::pow(theta_, (2.f * 0) / head_dim_)) : 0.f;
            float angle0 = theta0 * pos_last;
            float cs = std::cos(angle0), sn = std::sin(angle0);
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6);
            oss << "[RoPEDiagKernel] mode=" << (incr ? "INC" : "REPLAY")
                << " seq_len=" << seq_len
                << " pos_last=" << pos_last
                << " n_heads=" << n_heads
                << " head_dim=" << head_dim_
                << " theta=" << theta_
                << " theta0=" << theta0
                << " angle0=" << angle0
                << " cos0=" << cs << " sin0=" << sn
                << " before=";
            for (int i = 0; i < preview; ++i)
            {
                oss << before[i];
                if (i + 1 < preview)
                    oss << ",";
            }
            oss << " after=";
            for (int i = 0; i < preview; ++i)
            {
                oss << after[i];
                if (i + 1 < preview)
                    oss << ",";
            }
            oss << " rel_move=" << rel_move;
            LOG_INFO(oss.str());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // === ROPE OUTPUT VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(output_tensor, "RoPE output has NaN");
        TensorLogger::logTensorStats(output_tensor, "rope_output");

        LOG_DEBUG("MPIRoPEKernel executed: " << seq_len << "x" << n_heads << "x" << head_dim_
                                             << " in " << std::fixed << std::setprecision(2) << execution_time
                                             << "ms on rank " << getRank() << " (threads: " << omp_get_max_threads() << ")");

        return true;
    }

    bool MPIRoPEKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                 const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Check input count
        if (inputs.size() != 2)
        {
            LOG_ERROR("MPIRoPEKernel: Expected 2 inputs (input, position_ids), got " << inputs.size());
            return false;
        }

        // Check output count
        if (outputs.size() != 1)
        {
            LOG_ERROR("MPIRoPEKernel: Expected 1 output (rotated), got " << outputs.size());
            return false;
        }

        // Validate input tensor
        auto input_tensor = inputs[0];
        auto position_ids_tensor = inputs[1];

        if (!input_tensor || !position_ids_tensor)
        {
            LOG_ERROR("MPIRoPEKernel: Input tensors are null");
            return false;
        }

        const auto &input_shape = input_tensor->shape();
        const auto &position_shape = position_ids_tensor->shape();

        // Check input tensor dimensions
        if (input_shape.size() != 3)
        {
            LOG_ERROR("MPIRoPEKernel: Input tensor must be 3D [seq_len, n_heads, head_dim], got "
                      << input_shape.size() << "D");
            return false;
        }

        // Check position_ids dimensions
        if (position_shape.size() != 1)
        {
            LOG_ERROR("MPIRoPEKernel: Position IDs must be 1D [seq_len], got "
                      << position_shape.size() << "D");
            return false;
        }

        // Check sequence length consistency
        if (input_shape[0] != position_shape[0])
        {
            LOG_ERROR("MPIRoPEKernel: Sequence length mismatch - input: "
                      << input_shape[0] << ", position_ids: " << position_shape[0]);
            return false;
        }

        // Check head dimension
        if (input_shape[2] % 2 != 0)
        {
            LOG_ERROR("MPIRoPEKernel: Head dimension must be even, got " << input_shape[2]);
            return false;
        }

        // Validate output tensor
        auto output_tensor = outputs[0];
        if (!output_tensor)
        {
            LOG_ERROR("MPIRoPEKernel: Output tensor is null");
            return false;
        }

        const auto &output_shape = output_tensor->shape();
        if (output_shape.size() != 3)
        {
            LOG_ERROR("MPIRoPEKernel: Output tensor must be 3D, got " << output_shape.size() << "D");
            return false;
        }

        // Check output shape matches input
        if (output_shape[0] != input_shape[0] ||
            output_shape[1] != input_shape[1] ||
            output_shape[2] != input_shape[2])
        {
            LOG_ERROR("MPIRoPEKernel: Output shape mismatch - expected: ["
                      << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2]
                      << "], got: [" << output_shape[0] << ", " << output_shape[1]
                      << ", " << output_shape[2] << "]");
            return false;
        }

        return true;
    }

    void MPIRoPEKernel::setDistributionStrategy(DistributionStrategy strategy)
    {
        strategy_ = strategy;
        LOG_DEBUG("MPIRoPEKernel: Distribution strategy changed to "
                  << (strategy == DistributionStrategy::SEQUENCE_WISE ? "SEQUENCE_WISE" : "HEAD_WISE"));
    }

    void MPIRoPEKernel::updateMaxSeqLen(int max_seq_len)
    {
        max_seq_len_ = max_seq_len;
        precomputeFrequencyTables();
        LOG_DEBUG("MPIRoPEKernel: Updated max_seq_len to " << max_seq_len_
                                                           << ", recomputed frequency tables");
    }

    void MPIRoPEKernel::precomputeFrequencyTables()
    {
        int freq_dims = head_dim_ / 2;
        cos_table_.resize(max_seq_len_ * freq_dims);
        sin_table_.resize(max_seq_len_ * freq_dims);

        // Precompute for all positions and frequency dimensions
        for (int pos = 0; pos < max_seq_len_; ++pos)
        {
            for (int dim = 0; dim < freq_dims; ++dim)
            {
                float freq = 1.0f / std::pow(theta_, static_cast<float>(2 * dim) / head_dim_);
                float angle = pos * freq;

                int idx = pos * freq_dims + dim;
                cos_table_[idx] = std::cos(angle);
                sin_table_[idx] = std::sin(angle);
            }
        }

        LOG_DEBUG("MPIRoPEKernel: Precomputed frequency tables for max_seq_len: "
                  << max_seq_len_ << ", freq_dims: " << freq_dims);
    }

    void MPIRoPEKernel::configureOpenMPThreading(size_t tensor_size)
    {
        // Small operations: single-threaded to avoid overhead
        if (tensor_size < 8192)
        {
            omp_set_num_threads(1);
            LOG_DEBUG("MPIRoPEKernel: Using single thread for small tensor (" << tensor_size << " elements)");
            return;
        }

        // Medium to large operations: use available threads
        int max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        LOG_DEBUG("MPIRoPEKernel: Using " << max_threads << " threads for tensor (" << tensor_size << " elements)");
    }

    void MPIRoPEKernel::distributeMPIWork(size_t total_elements, size_t &start_idx, size_t &end_idx) const
    {
        int rank = getRank();
        int size = getSize();

        size_t elements_per_rank = total_elements / size;
        size_t remainder = total_elements % size;

        start_idx = rank * elements_per_rank;
        end_idx = start_idx + elements_per_rank;

        // Distribute remainder elements to first ranks
        if (rank < remainder)
        {
            start_idx += rank;
            end_idx += rank + 1;
        }
        else
        {
            start_idx += remainder;
            end_idx += remainder;
        }

        LOG_DEBUG("MPIRoPEKernel: Rank " << rank << " processing elements ["
                                         << start_idx << ", " << end_idx << ") of " << total_elements);
    }

    void MPIRoPEKernel::executeSequenceWise(const float *input_data, const int *position_ids, float *output_data,
                                            int seq_len, int n_heads, int head_dim)
    {
        // Distribute sequence positions across MPI ranks
        size_t total_positions = seq_len;
        size_t start_pos, end_pos;
        distributeMPIWork(total_positions, start_pos, end_pos);

        // Process assigned sequence positions with OpenMP parallelization
        auto omp_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(static)
        for (size_t pos = start_pos; pos < end_pos; ++pos)
        {
            for (int head = 0; head < n_heads; ++head)
            {
                size_t offset = pos * n_heads * head_dim + head * head_dim;
                applyRotaryEmbedding(&input_data[offset], &output_data[offset],
                                     position_ids[pos], head_dim);
            }
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPIRoPEKernel sequence-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void MPIRoPEKernel::executeHeadWise(const float *input_data, const int *position_ids, float *output_data,
                                        int seq_len, int n_heads, int head_dim)
    {
        // Distribute attention heads across MPI ranks
        size_t total_heads = n_heads;
        size_t start_head, end_head;
        distributeMPIWork(total_heads, start_head, end_head);

        // Process assigned heads across all sequence positions
        auto omp_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(static)
        for (int pos = 0; pos < seq_len; ++pos)
        {
            for (size_t head = start_head; head < end_head; ++head)
            {
                size_t offset = pos * n_heads * head_dim + head * head_dim;
                applyRotaryEmbedding(&input_data[offset], &output_data[offset],
                                     position_ids[pos], head_dim);
            }
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPIRoPEKernel head-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void MPIRoPEKernel::applyRotaryEmbedding(const float *input_ptr, float *output_ptr, int position, int head_dim)
    {
// Apply rotary embedding to pairs of dimensions
#pragma omp simd aligned(input_ptr, output_ptr : 32)
        for (int i = 0; i < head_dim; i += 2)
        {
            float x1 = input_ptr[i];
            float x2 = input_ptr[i + 1];

            float cos_val = getCos(position, i);
            float sin_val = getSin(position, i);

            // Apply rotation: [cos, -sin; sin, cos] * [x1; x2]
            output_ptr[i] = x1 * cos_val - x2 * sin_val;
            output_ptr[i + 1] = x1 * sin_val + x2 * cos_val;
        }
    }

    inline float MPIRoPEKernel::getCos(int position, int dim) const
    {
        if (position >= max_seq_len_ || dim >= head_dim_)
        {
            LOG_ERROR("MPIRoPEKernel: Position " << position << " or dim " << dim << " out of bounds");
            return 1.0f; // Safe fallback
        }

        int freq_dim = dim / 2;
        int freq_dims = head_dim_ / 2;
        return cos_table_[position * freq_dims + freq_dim];
    }

    inline float MPIRoPEKernel::getSin(int position, int dim) const
    {
        if (position >= max_seq_len_ || dim >= head_dim_)
        {
            LOG_ERROR("MPIRoPEKernel: Position " << position << " or dim " << dim << " out of bounds");
            return 0.0f; // Safe fallback
        }

        int freq_dim = dim / 2;
        int freq_dims = head_dim_ / 2;
        return sin_table_[position * freq_dims + freq_dim];
    }

} // namespace llaminar