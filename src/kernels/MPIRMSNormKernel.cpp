/**
 * @file MPIRMSNormKernel.cpp
 * @brief Root Mean Square Layer Normalization (RMSNorm) applied row-wise to activation matrix.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Activation tensor X [seq_len, hidden_dim].
 *  - inputs[1]: Scale (gamma) tensor [hidden_dim].
 * Outputs:
 *  - outputs[0]: Normalized tensor Y [seq_len, hidden_dim].
 * Formula:
 *  - rms = sqrt( mean( X_i^2 ) + eps ) over hidden_dim
 *  - Y_i = (X_i / rms) * gamma_i
 * Numerical Expectations:
 *  - Deterministic; relative diff vs high precision reference < 1e-7 typical.
 * Error Modes:
 *  - Hidden dim mismatch, zero or negative epsilon, null tensors.
 *  - NaN propagation if input contains NaN (not masked by kernel).
 * Distribution:
 *  - Replicated across ranks; each rank processes full sequence (potential future sharding along sequence dimension).
 * Threading:
 *  - Parallel over rows; per-row reduction + scale broadcast; uses local temporaries only.
 * Performance Notes:
 *  - Compute-bound for large hidden_dim; consider vectorized reduction and reciprocal sqrt improvements.
 * Future Extensions:
 *  - FP16/BF16 mixed-precision normalization; fused residual + RMSNorm variant.
 * @todo Investigate epsilon tuning for quantized activations path.
 * @author David Sanftenberg
 */
#include "MPIRMSNormKernel.h"
#include "../logger.h"
#include "../debug_utils.h"
#include "../utils/debug_env.h"
#include "common/rmsnorm_core.h" // centralized RMSNorm primitives
#include "common/rmsnorm_t5.h"   // T5-style RMSNorm matching HuggingFace
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include "../utils/perf_counters.h"

namespace llaminar
{

    namespace
    {
        bool env_flag_enabled(const char *value)
        {
            if (!value)
            {
                return false;
            }
            std::string token;
            token.reserve(std::strlen(value));
            for (const char *p = value; *p; ++p)
            {
                unsigned char ch = static_cast<unsigned char>(*p);
                if (!std::isspace(ch))
                {
                    token.push_back(static_cast<char>(std::tolower(ch)));
                }
            }
            if (token.empty())
            {
                return true;
            }
            return !(token == "0" || token == "false" || token == "off" || token == "no");
        }
    }

    MPIRMSNormKernel::MPIRMSNormKernel(DistributionStrategy strategy)
        : strategy_(strategy), epsilon_(1e-6f)
    {
        LOG_DEBUG("MPIRMSNormKernel initialized with epsilon=" << epsilon_ << " on rank " << getRank());
    }

    bool MPIRMSNormKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                   std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIRMSNormKernel validation failed on rank " << getRank());
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto global_input = inputs[0];
        auto weight = inputs[1];
        auto global_output = outputs[0];

        // === COMPREHENSIVE TENSOR VALIDATION ===
        ASSERT_TENSOR_VALID(global_input, "RMSNorm input");
        ASSERT_TENSOR_VALID(weight, "RMSNorm weight");
        ASSERT_TENSOR_VALID(global_output, "RMSNorm output");

        // Check for NaN in inputs before computation
        ASSERT_TENSOR_NOT_NAN(global_input, "RMSNorm input");
        ASSERT_TENSOR_NOT_NAN(weight, "RMSNorm weight");

        // Log detailed tensor information (basic always; extended only if verbose)
        if (debugEnv().rmsnorm.verbose)
            TensorLogger::logNormalizationOperation(global_input, weight, global_output, "MPIRMSNormKernel");

        std::vector<int> trace_rows;
        if (!debugEnv().rmsnorm.trace_rows_spec.empty())
        {
            trace_rows = parseRowSpecification(debugEnv().rmsnorm.trace_rows_spec.c_str(), static_cast<size_t>(global_input->shape()[0]));
            LOG_INFO("[MPIRMSNormKernel] trace_rows request='" << debugEnv().rmsnorm.trace_rows_spec << "' parsed_count=" << trace_rows.size());
        }
        const bool validate_reference = debugEnv().rmsnorm.validate_ref;
        if (validate_reference)
        {
            LOG_INFO("[MPIRMSNormKernel] reference validation enabled");
        }

        // Additional epsilon logging
        LOG_INFO("[MPIRMSNormKernel] Using epsilon=" << epsilon_);

        // Extract dimensions
        size_t global_seq_len = static_cast<size_t>(global_input->shape()[0]);
        size_t hidden_size = static_cast<size_t>(global_input->shape()[1]); // local view (full if not sharded)
        size_t global_hidden_dim_reported = hidden_size;                    // full hidden size when sharded

        // Shard-aware feature-slice detection (head/hidden sharding along feature dimension)
        bool feature_sharded = false;
        size_t feature_local_offset = 0, feature_local_dim = hidden_size;
        if (auto *ss = dynamic_cast<ShardedSimpleTensor *>(global_input.get()))
        {
            const ShardSpec &spec = ss->shard_spec();
            // Updated detection: rely on spec.global_dim as canonical full hidden size; if local_dim < global_dim treat as sharded slice.
            if (spec.is_sharded() && (spec.axis == ShardSpec::Axis::Heads || spec.axis == ShardSpec::Axis::Hidden))
            {
                if ((size_t)spec.local_dim < (size_t)spec.global_dim)
                {
                    feature_sharded = true;
                    feature_local_offset = spec.local_offset;
                    feature_local_dim = spec.local_dim;
                    // Track full hidden separately; do not overwrite local buffer dimension.
                    global_hidden_dim_reported = spec.global_dim;
                }
            }
        }

        // Calculate local row distribution (sequence-wise) always
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        // Local tensor views / buffers
        std::shared_ptr<TensorBase> local_input;
        std::shared_ptr<TensorBase> local_output;
        if (feature_sharded)
        {
            // For feature-sharded inputs we keep ALL rows locally (each rank owns full sequence rows for its feature slice)
            // Override any row distribution result.
            local_seq_len = global_seq_len;
            seq_offset = 0;
            local_input = global_input;   // already sized [seq_len, local_feature_dim]
            local_output = global_output; // same layout for output
        }
        else
        {
            // Legacy row distribution path over full hidden size
            local_input = createLocalTensor({static_cast<size_t>(local_seq_len), hidden_size});
            local_output = createLocalTensor({static_cast<size_t>(local_seq_len), hidden_size});
            distributeInput(global_input, local_input, global_seq_len, hidden_size);
        }

        // Optional gamma diagnostics / override
        const bool dump_gamma = debugEnv().rmsnorm.dump_gamma;
        const bool force_unit_gamma = debugEnv().rmsnorm.force_unit_gamma;
        const bool track_gamma_checksum = debugEnv().rmsnorm.gamma_checksum;
        const float *gamma_ptr = weight->data();
        std::vector<float> unit_gamma;
        if (force_unit_gamma)
        {
            unit_gamma.assign(global_hidden_dim_reported, 1.0f);
            gamma_ptr = unit_gamma.data();
            if (getRank() == 0)
            {
                LOG_WARN("[MPIRMSNormKernel] FORCING UNIT GAMMA (debug override) hidden_size=" << global_hidden_dim_reported);
            }
        }
        if (dump_gamma && getRank() == 0)
        {
            // Dump first handful + min/max/mean for gamma
            double g_min = std::numeric_limits<double>::infinity();
            double g_max = -std::numeric_limits<double>::infinity();
            long double g_sum = 0.0L;
            for (size_t i = 0; i < global_hidden_dim_reported; ++i)
            {
                double v = static_cast<double>(weight->data()[i]);
                g_min = std::min(g_min, v);
                g_max = std::max(g_max, v);
                g_sum += v;
            }
            double g_mean = static_cast<double>(g_sum / static_cast<long double>(global_hidden_dim_reported));
            std::ostringstream goss;
            goss << "[MPIRMSNormKernel][GammaDump] size=" << global_hidden_dim_reported
                 << " min=" << g_min << " max=" << g_max << " mean=" << g_mean << " first32=";
            size_t dump_n = std::min<size_t>(32, global_hidden_dim_reported);
            for (size_t i = 0; i < dump_n; ++i)
            {
                goss << weight->data()[i];
                if (i + 1 < dump_n)
                    goss << ',';
            }
            LOG_INFO(goss.str());
        }

        static uint64_t last_checksum = 0;
        static bool have_last = false;
        if (track_gamma_checksum && getRank() == 0)
        {
            auto checksum = [](const float *ptr, size_t n) -> uint64_t
            {
                const uint64_t FNV_OFFSET = 1469598103934665603ull;
                const uint64_t FNV_PRIME = 1099511628211ull;
                uint64_t h = FNV_OFFSET;
                const uint8_t *bytes = reinterpret_cast<const uint8_t *>(ptr);
                size_t len_bytes = n * sizeof(float);
                for (size_t i = 0; i < len_bytes; ++i)
                {
                    h ^= bytes[i];
                    h *= FNV_PRIME;
                }
                return h;
            };
            uint64_t cs = checksum(weight->data(), hidden_size);
            if (have_last && cs != last_checksum)
            {
                LOG_WARN("[MPIRMSNormKernel][GammaChecksum] CHANGED checksum_old=0x" << std::hex << last_checksum << " checksum_new=0x" << cs << std::dec << " (gamma modified between invocations?)");
            }
            else if (!have_last)
            {
                LOG_INFO("[MPIRMSNormKernel][GammaChecksum] initial checksum=0x" << std::hex << cs << std::dec);
                have_last = true;
            }
            last_checksum = cs;
        }

        // Compute (shard-aware) RMSNorm
        if (feature_sharded)
        {
            // Need per-row global sumsq across feature shards: compute local then Allreduce
            std::vector<double> local_row_sumsq(local_seq_len, 0.0);
            size_t feat_dim = feature_local_dim;
            const float *in_ptr = local_input->data();
            // Use core primitive for per-row sumsq
            {
                kernels::RMSNormExecOptions opts; // default heuristics
                kernels::rmsnorm_compute_row_sumsq(in_ptr, local_seq_len, feat_dim, local_row_sumsq.data(), opts);
            }
            std::vector<double> global_row_sumsq(local_row_sumsq.size(), 0.0);
            checkMPIError(PerfAllreduce(local_row_sumsq.data(), global_row_sumsq.data(), (int)local_row_sumsq.size(), MPI_DOUBLE, MPI_SUM, getComm()), "RMSNorm shard row sumsq");
            // Determine global hidden size from shard spec (if available)
            size_t global_hidden = hidden_size;
            if (auto *ss = dynamic_cast<ShardedSimpleTensor *>(global_input.get()))
                global_hidden = ss->shard_spec().global_dim;
            std::vector<float> inv_scale(local_seq_len, 0.f);
            for (size_t r = 0; r < local_seq_len; ++r)
            {
                double denom = (double)global_hidden;
                double inv = 0.0;
                if (denom > 0.0)
                    inv = 1.0 / std::sqrt(global_row_sumsq[r] / denom + (double)epsilon_);
                inv_scale[r] = (float)inv;
                if (r == 0 && getRank() == 0 && debugEnv().rmsnorm.validate_ref)
                {
                    LOG_INFO("[MPIRMSNormKernel][Diag] r=0 global_row_sumsq=" << global_row_sumsq[r]
                                                                              << " denom=" << denom << " inv=" << inv);
                }
            }
            // Apply scaling to local slice
            float *out_ptr = local_output->data();
            bool weight_sharded = false;
            if (auto *wss = dynamic_cast<ShardedSimpleTensor *>(weight.get()))
            {
                const ShardSpec &wspec = wss->shard_spec();
                weight_sharded = wspec.is_sharded() && (size_t)wspec.local_dim < (size_t)wspec.global_dim;
            }
            // Use core apply primitive with appropriate GammaMode semantics
            {
                kernels::RMSNormExecOptions opts; // default heuristics
                kernels::GammaMode mode = weight_sharded ? kernels::GammaMode::SHARDED : kernels::GammaMode::REPLICATED;
                kernels::rmsnorm_apply(in_ptr, gamma_ptr, inv_scale.data(), local_seq_len, feat_dim, out_ptr, mode, feature_local_offset, opts);
            }
            // If we used a temporary row-sliced buffer, need to scatter back into global_output
            if (local_output != global_output)
            {
                float *gout = global_output->data();
                for (size_t r = 0; r < local_seq_len; ++r)
                {
                    size_t gr = seq_offset + r;
                    float *dst = gout + gr * feat_dim;
                    const float *src = out_ptr + r * feat_dim;
                    std::memcpy(dst, src, sizeof(float) * feat_dim);
                }
            }
        }
        else
        {
            computeDistributedRMSNorm(local_input->data(), gamma_ptr,
                                      local_output->data(), local_seq_len,
                                      hidden_size, global_seq_len);
            gatherOutput(local_output, global_output, global_seq_len, hidden_size);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // === POST-COMPUTATION VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(global_output, "RMSNorm output after computation");

        // Log output tensor statistics
        TensorLogger::logTensorStats(global_output, "RMSNorm final_output", "MPIRMSNormKernel_COMPLETE");

        if (!trace_rows.empty())
        {
            logTensorRowPreview(global_input, "RMSNorm input", trace_rows, 8, "RMSNORM_TRACE");
            logTensorRowPreview(global_output, "RMSNorm output", trace_rows, 8, "RMSNORM_TRACE");
        }

        // Heavy reference validation (disabled in Release unless explicitly enabled)
#ifdef LLAMINAR_ENABLE_RMSNORM_REFERENCE
        if (validate_reference && getRank() == 0)
        {
            runReferenceValidation(global_input, weight, global_output, trace_rows);
        }
#endif // LLAMINAR_ENABLE_RMSNORM_REFERENCE

        if (debugEnv().rmsnorm.verbose)
        {
            LOG_DEBUG("[MPIRMSNormKernel][verbose] rank=" << getRank() << " exec_us=" << duration.count());
        }
        else
        {
            LOG_TRACE("MPIRMSNormKernel done rank=" << getRank());
        }

        return true;
    }

    bool MPIRMSNormKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                    const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Basic validation
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("MPIRMSNormKernel: Expected 2 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPIRMSNormKernel: Null tensor provided");
            return false;
        }

        // Check input is 2D [seq_len, hidden_size]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIRMSNormKernel: Input must be 2D [seq_len, hidden_size], got "
                      << input->shape().size() << "D");
            return false;
        }

        // Check weight is 1D [hidden_size]
        if (weight->shape().size() != 1)
        {
            LOG_ERROR("MPIRMSNormKernel: Weight must be 1D [hidden_size], got "
                      << weight->shape().size() << "D");
            return false;
        }

        // Check output is 2D [seq_len, hidden_size]
        if (output->shape().size() != 2)
        {
            LOG_ERROR("MPIRMSNormKernel: Output must be 2D [seq_len, hidden_size], got "
                      << output->shape().size() << "D");
            return false;
        }

        // Check dimension compatibility (allow sharded input + replicated weight scenario)
        if (input->shape()[1] != weight->shape()[0])
        {
            bool allow_replicated_weight = false;
            if (auto *sharded_in = dynamic_cast<ShardedSimpleTensor *>(input.get()))
            {
                const ShardSpec &spec = sharded_in->shard_spec();
                if (spec.is_sharded() && (spec.axis == ShardSpec::Axis::Hidden || spec.axis == ShardSpec::Axis::Heads))
                {
                    // Local slice dim vs global dim (weight may be replicated full vector)
                    if ((size_t)spec.local_dim == (size_t)input->shape()[1] && (size_t)spec.global_dim == (size_t)weight->shape()[0])
                    {
                        allow_replicated_weight = true;
                        LOG_DEBUG("MPIRMSNormKernel: Accepting replicated weight (global_hidden=" << spec.global_dim
                                                                                                  << ") with local feature slice dim=" << spec.local_dim << " on rank " << getRank());
                    }
                }
            }
            if (!allow_replicated_weight)
            {
                LOG_ERROR("MPIRMSNormKernel: Dimension mismatch between input hidden_size ("
                          << input->shape()[1] << ") and weight (" << weight->shape()[0] << ")");
                return false;
            }
        }

        // Check input and output have same shape
        if (input->shape()[0] != output->shape()[0] || input->shape()[1] != output->shape()[1])
        {
            LOG_ERROR("MPIRMSNormKernel: Input and output shape mismatch");
            return false;
        }

        return true;
    }

    void MPIRMSNormKernel::distributeInput(const std::shared_ptr<TensorBase> &global_input,
                                           std::shared_ptr<TensorBase> &local_input,
                                           size_t global_seq_len, size_t hidden_size)
    {
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        const float *global_data = global_input->data();
        float *local_data = local_input->data();

        // Copy the local portion of the input
        for (size_t i = 0; i < local_seq_len; ++i)
        {
            size_t global_row = seq_offset + i;
            std::memcpy(local_data + i * hidden_size,
                        global_data + global_row * hidden_size,
                        hidden_size * sizeof(float));
        }

        LOG_DEBUG("Distributed input: local size [" << local_seq_len << ", " << hidden_size
                                                    << "], offset " << seq_offset << " on rank " << getRank());
    }

    void MPIRMSNormKernel::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                        std::shared_ptr<TensorBase> &global_output,
                                        size_t global_seq_len, size_t hidden_size)
    {
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        // Prepare MPI gather parameters
        std::vector<int> recv_counts(getSize());
        std::vector<int> recv_offsets(getSize());

        for (int rank = 0; rank < getSize(); ++rank)
        {
            auto [rank_local_seq_len, rank_seq_offset] = getRowDistribution(global_seq_len, rank);
            recv_counts[rank] = static_cast<int>(rank_local_seq_len * hidden_size);
            recv_offsets[rank] = static_cast<int>(rank_seq_offset * hidden_size);
        }

        checkMPIError(PerfAllgatherv(local_output->data(),
                                     static_cast<int>(local_seq_len * hidden_size), MPI_FLOAT,
                                     global_output->data(),
                                     recv_counts.data(), recv_offsets.data(), MPI_FLOAT,
                                     getComm()),
                      "MPI_Allgatherv in gatherOutput");

        LOG_DEBUG("Gathered output: [" << global_seq_len << ", " << hidden_size << "] on rank " << getRank());
    }

    void MPIRMSNormKernel::computeDistributedRMSNorm(const float *local_input, const float *weight,
                                                     float *local_output, size_t local_seq_len,
                                                     size_t hidden_size, size_t global_seq_len)
    {
        // Instrumentation: compute basic stats of local input before normalization (parallelized)
        float local_min = std::numeric_limits<float>::infinity();
        float local_max = -std::numeric_limits<float>::infinity();
        double local_sum = 0.0;
        size_t local_count = local_seq_len * hidden_size;
#pragma omp parallel for reduction(min : local_min) reduction(max : local_max) reduction(+ : local_sum) schedule(static)
        for (long long i = 0; i < (long long)local_count; ++i)
        {
            float v = local_input[i];
            if (v < local_min)
                local_min = v;
            if (v > local_max)
                local_max = v;
            local_sum += v;
        }
        double local_mean = local_count ? (local_sum / local_count) : 0.0;

        // Use T5-style RMSNorm computation (matches HuggingFace Transformers exactly)
        // This replaces the old multi-step computation with a direct implementation
        // Formula: output = weight * input / sqrt(mean(input^2) + eps)
        bool use_parallel = local_seq_len > 1;

        // Use double precision accumulation to potentially reduce numerical errors
        // PyTorch T5LayerNorm uses float32 by default, but we'll try double
        kernels::rmsnorm_t5_forward_double_acc(
            local_input, weight, local_output,
            local_seq_len, hidden_size, epsilon_, use_parallel);

        // Compute diagnostic statistics for logging (per-row variance)
        std::vector<double> row_sumsq(local_seq_len, 0.0);
        for (size_t r = 0; r < local_seq_len; ++r)
        {
            const float *row = local_input + r * hidden_size;
            double sum_sq = 0.0;
            for (size_t c = 0; c < hidden_size; ++c)
            {
                double val = static_cast<double>(row[c]);
                sum_sq += val * val;
            }
            row_sumsq[r] = sum_sq;
        }

        double local_sum_sq_total = 0.0;
        for (size_t r = 0; r < local_seq_len; ++r)
            local_sum_sq_total += row_sumsq[r];

        // Aggregate statistics for logging (global averages are informational only)
        double global_sum_sq = 0.0;
        checkMPIError(PerfAllreduce(&local_sum_sq_total, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, getComm()),
                      "MPI_Allreduce for RMS statistics");

        // Compute representative RMS for logging (using first row)
        double first_row_sumsq = row_sumsq.empty() ? 0.0 : row_sumsq[0];
        double rms_global = std::sqrt(hidden_size ? (first_row_sumsq / (double)hidden_size + (double)epsilon_) : (double)epsilon_);

        // Gather global stats for debugging (min, max, mean)
        float global_min = 0.0f, global_max = 0.0f;
        double global_mean_sum = 0.0;
        double global_mean = 0.0;
        checkMPIError(PerfAllreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, getComm()), "MPI_Allreduce min in RMSNorm");
        checkMPIError(PerfAllreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, getComm()), "MPI_Allreduce max in RMSNorm");
        checkMPIError(PerfAllreduce(&local_sum, &global_mean_sum, 1, MPI_DOUBLE, MPI_SUM, getComm()), "MPI_Allreduce mean sum in RMSNorm");
        global_mean = (global_seq_len * hidden_size) ? (global_mean_sum / (double)(global_seq_len * hidden_size)) : 0.0;

        // Weight stats (local - identical across ranks expected). Just compute once and broadcast rank 0's view.
        float w_min = std::numeric_limits<float>::infinity();
        float w_max = -std::numeric_limits<float>::infinity();
        double w_sum = 0.0;
        for (size_t j = 0; j < hidden_size; ++j)
        {
            float w = weight[j];
            if (w < w_min)
                w_min = w;
            if (w > w_max)
                w_max = w;
            w_sum += w;
        }
        double w_sum_global = 0.0;
        float w_min_global = 0.0f, w_max_global = 0.0f;
        checkMPIError(PerfAllreduce(&w_min, &w_min_global, 1, MPI_FLOAT, MPI_MIN, getComm()), "MPI_Allreduce weight min");
        checkMPIError(PerfAllreduce(&w_max, &w_max_global, 1, MPI_FLOAT, MPI_MAX, getComm()), "MPI_Allreduce weight max");
        checkMPIError(PerfAllreduce(&w_sum, &w_sum_global, 1, MPI_DOUBLE, MPI_SUM, getComm()), "MPI_Allreduce weight sum");
        double w_mean_global = (double)w_sum_global / (double)(hidden_size * getSize());

        if (getRank() == 0)
        {
            LOG_INFO("[MPIRMSNormKernel] Pre-Norm stats: min=" << global_min << " max=" << global_max << " mean=" << global_mean
                                                               << " rms_first_row=" << rms_global << " mode=T5");
            LOG_INFO("[MPIRMSNormKernel] Weight stats: min=" << w_min_global << " max=" << w_max_global << " mean=" << w_mean_global);
        }

        // Post-norm local stats to detect zeroing (parallel)
        float out_local_min = std::numeric_limits<float>::infinity();
        float out_local_max = -std::numeric_limits<float>::infinity();
        double out_local_sum = 0.0;
#pragma omp parallel for reduction(min : out_local_min) reduction(max : out_local_max) reduction(+ : out_local_sum) schedule(static)
        for (long long i = 0; i < (long long)local_count; ++i)
        {
            float v = local_output[i];
            if (v < out_local_min)
                out_local_min = v;
            if (v > out_local_max)
                out_local_max = v;
            out_local_sum += v;
        }
        float out_global_min = 0.0f, out_global_max = 0.0f;
        double out_global_sum = 0.0;
        checkMPIError(PerfAllreduce(&out_local_min, &out_global_min, 1, MPI_FLOAT, MPI_MIN, getComm()), "MPI_Allreduce out min");
        checkMPIError(PerfAllreduce(&out_local_max, &out_global_max, 1, MPI_FLOAT, MPI_MAX, getComm()), "MPI_Allreduce out max");
        checkMPIError(PerfAllreduce(&out_local_sum, &out_global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm()), "MPI_Allreduce out sum");
        double out_global_mean = (global_seq_len * hidden_size) ? (out_global_sum / (double)(global_seq_len * hidden_size)) : 0.0;
        if (getRank() == 0)
        {
            LOG_INFO("[MPIRMSNormKernel] Post-Norm stats: min=" << out_global_min << " max=" << out_global_max << " mean=" << out_global_mean << " mode=T5");
        }

        LOG_DEBUG("Computed distributed RMS normalization: rms_metric=" << rms_global
                                                                        << ", local_seq_len=" << local_seq_len << " on rank " << getRank()
                                                                        << " mode=T5");
    }

    std::shared_ptr<TensorBase> MPIRMSNormKernel::createLocalTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());
        // Use TensorFactory to create a modern tensor
        return TensorFactory::create_simple(int_shape);
    }

#ifdef LLAMINAR_ENABLE_RMSNORM_REFERENCE
    void MPIRMSNormKernel::runReferenceValidation(const std::shared_ptr<TensorBase> &global_input,
                                                  const std::shared_ptr<TensorBase> &weight,
                                                  const std::shared_ptr<TensorBase> &global_output,
                                                  const std::vector<int> &trace_rows)
    {
        size_t seq_len = static_cast<size_t>(global_input->shape()[0]);
        size_t hidden_size = static_cast<size_t>(global_input->shape()[1]);
        if (seq_len == 0 || hidden_size == 0)
        {
            LOG_WARN("[MPIRMSNormKernel][validate] Skipping reference check due to empty dimensions");
            return;
        }

        const float *input_ptr = global_input->data();
        const float *weight_ptr = weight->data();
        const float *actual_ptr = global_output->data();

        std::vector<float> reference(seq_len * hidden_size, 0.0f);
        std::vector<double> row_sum_sq(seq_len, 0.0);
        std::vector<float> row_inv(seq_len, 0.0f);
        // Parallel reference computation
        double diff_sq = 0.0; // reduction
        double ref_sq = 0.0;  // reduction
        double max_abs = 0.0; // updated under critical section
        size_t worst_index = 0;

        const size_t total_elements = seq_len * hidden_size;
        // Only parallelize if reasonably large to amortize overhead
        bool do_parallel = total_elements >= 8192 && seq_len > 1;

#pragma omp parallel for schedule(static) reduction(+ : diff_sq, ref_sq) if (do_parallel)
        for (long long row_ll = 0; row_ll < (long long)seq_len; ++row_ll)
        {
            size_t row = (size_t)row_ll;
            size_t base = row * hidden_size;
            double sum_sq = 0.0;
#pragma omp simd reduction(+ : sum_sq)
            for (long long col = 0; col < (long long)hidden_size; ++col)
            {
                double value = (double)input_ptr[base + (size_t)col];
                sum_sq += value * value;
            }
            row_sum_sq[row] = sum_sq;
            double inv = hidden_size ? 1.0 / std::sqrt(sum_sq / (double)hidden_size + (double)epsilon_) : 0.0;
            row_inv[row] = (float)inv;

            double row_worst_abs = 0.0;
            size_t row_worst_index = base;
            for (size_t col = 0; col < hidden_size; ++col)
            {
                float ref_val = (float)((double)input_ptr[base + col] * inv * (double)weight_ptr[col]);
                reference[base + col] = ref_val;
                double diff = (double)actual_ptr[base + col] - (double)ref_val;
                double abs_diff = std::fabs(diff);
                if (abs_diff > row_worst_abs)
                {
                    row_worst_abs = abs_diff;
                    row_worst_index = base + col;
                }
                diff_sq += diff * diff;
                ref_sq += (double)ref_val * (double)ref_val;
            }
#pragma omp critical
            {
                if (row_worst_abs > max_abs)
                {
                    max_abs = row_worst_abs;
                    worst_index = row_worst_index;
                }
            }
        }

        // If we skipped parallelism (small problem), worst_index/max_abs already set inside loop logic; for small path we replicate sequential logic
        if (!do_parallel)
        {
            // Sequential fallback (mirrors old logic) - already executed in loop above since if(do_parallel)==false parallel region degenerates to serial
        }

        double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq) / std::sqrt(ref_sq) : 0.0;
        size_t worst_row = hidden_size ? worst_index / hidden_size : 0;
        size_t worst_col = hidden_size ? worst_index % hidden_size : 0;

        LOG_INFO("[MPIRMSNormKernel][validate] reference diff max_abs=" << max_abs
                                                                        << " rel_l2=" << rel_l2
                                                                        << " worst_row=" << worst_row
                                                                        << " worst_col=" << worst_col
                                                                        << " actual=" << actual_ptr[worst_index]
                                                                        << " ref=" << reference[worst_index]);

        if (hidden_size > 0)
        {
            size_t start_col = (worst_col > 3) ? worst_col - 3 : 0;
            size_t end_col = std::min(hidden_size, start_col + static_cast<size_t>(8));
            std::ostringstream actual_preview;
            std::ostringstream ref_preview;
            std::ostringstream diff_preview;
            actual_preview << "[";
            ref_preview << "[";
            diff_preview << "[";
            for (size_t col = start_col; col < end_col; ++col)
            {
                if (col > start_col)
                {
                    actual_preview << ", ";
                    ref_preview << ", ";
                    diff_preview << ", ";
                }
                size_t idx = worst_row * hidden_size + col;
                actual_preview << actual_ptr[idx];
                ref_preview << reference[idx];
                diff_preview << (actual_ptr[idx] - reference[idx]);
            }
            if (end_col < hidden_size)
            {
                actual_preview << ", ...";
                ref_preview << ", ...";
                diff_preview << ", ...";
            }
            actual_preview << "]";
            ref_preview << "]";
            diff_preview << "]";
            LOG_INFO("[MPIRMSNormKernel][validate] worst_row_preview actual=" << actual_preview.str()
                                                                              << " ref=" << ref_preview.str()
                                                                              << " diff=" << diff_preview.str());
        }

        if (!trace_rows.empty())
        {
            for (int row : trace_rows)
            {
                if (row < 0 || static_cast<size_t>(row) >= seq_len)
                {
                    continue;
                }
                size_t base = static_cast<size_t>(row) * hidden_size;
                size_t cols_to_show = std::min<size_t>(static_cast<size_t>(8), hidden_size);
                std::ostringstream actual_preview;
                std::ostringstream ref_preview;
                std::ostringstream diff_preview;
                actual_preview << "[";
                ref_preview << "[";
                diff_preview << "[";
                for (size_t col = 0; col < cols_to_show; ++col)
                {
                    if (col)
                    {
                        actual_preview << ", ";
                        ref_preview << ", ";
                        diff_preview << ", ";
                    }
                    size_t idx = base + col;
                    actual_preview << actual_ptr[idx];
                    ref_preview << reference[idx];
                    diff_preview << (actual_ptr[idx] - reference[idx]);
                }
                if (cols_to_show < hidden_size)
                {
                    actual_preview << ", ...";
                    ref_preview << ", ...";
                    diff_preview << ", ...";
                }
                actual_preview << "]";
                ref_preview << "]";
                diff_preview << "]";
                LOG_INFO("[RMSNORM_TRACE] row=" << row
                                                << " sum_sq=" << row_sum_sq[static_cast<size_t>(row)]
                                                << " inv_scale=" << row_inv[static_cast<size_t>(row)]
                                                << " actual=" << actual_preview.str()
                                                << " ref=" << ref_preview.str()
                                                << " diff=" << diff_preview.str());
            }
        }
    }
#endif // LLAMINAR_ENABLE_RMSNORM_REFERENCE

} // namespace llaminar