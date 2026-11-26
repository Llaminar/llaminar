/**
 * @file ROCmSamplingSpeculativeKernels.hip
 * @brief ROCm speculative verification, processed-logit sampling/softmax, and verify summary kernels.
 */
#include "ROCmSamplingCommon.hip.inc"

__global__ void rocm_speculative_verify_distribution_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int draft_token,
    unsigned long long accept_seed,
    unsigned long long accept_offset,
    unsigned long long residual_seed,
    unsigned long long residual_offset,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        llaminar2::sampling_math::uniform01(accept_seed, accept_offset),
        llaminar2::sampling_math::uniform01(residual_seed, residual_offset),
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);
}

__global__ void rocm_speculative_verify_distribution_threshold_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int draft_token,
    float accept_threshold,
    float residual_threshold,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        accept_threshold,
        residual_threshold,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);
}

__global__ void rocm_speculative_verify_distribution_thresholds_batch_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int distribution_stride,
    llaminar2::sampling_math::SpeculativeBatchHostParameters parameters,
    int row_count,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count)
        return;

    const int offset = row * distribution_stride;
    llaminar2::sampling_math::speculative_verify_with_thresholds(
        target_token_ids + offset,
        target_probs + offset,
        draft_token_ids + offset,
        draft_probs + offset,
        k,
        parameters.draft_tokens[row],
        parameters.thresholds.accept[row],
        parameters.thresholds.residual[row],
        out_token + row,
        out_accepted + row,
        out_accept_probability ? out_accept_probability + row : nullptr,
        out_accept_threshold ? out_accept_threshold + row : nullptr);
}

/**
 * @brief Batched verifier variant whose draft tokens are already on device.
 *
 * ROCm mirrors the CUDA entry point so the orchestrator can use one
 * backend-neutral device-token verifier contract. Thresholds may be explicit,
 * seeded from a compatibility scalar position, or derived from the mutable
 * device-resident publication mailbox. Sampled draft tokens remain in
 * arena-owned device scratch produced by the compact draft sampler.
 */
__global__ void rocm_speculative_verify_distribution_thresholds_batch_device_tokens_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    const int *__restrict__ draft_token_ids,
    const float *__restrict__ draft_probs,
    int k,
    int distribution_stride,
    const int *__restrict__ sampled_draft_tokens,
    const float *__restrict__ sampled_draft_probabilities,
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    int row_count,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int inverse_sample_vocab_size,
    unsigned long long threshold_seed,
    int threshold_first_logical_position,
    int thresholds_from_seed,
    const int *__restrict__ threshold_base_position,
    int threshold_position_offset,
    const int *__restrict__ generation_control,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold,
    const uint64_t *__restrict__ threshold_seed_device)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    using namespace llaminar2::sampling_math;
    // Request admission publishes this bank before replay. Each draw purpose
    // reads the same seed; no seed value participates in capture identity.
    if (threshold_seed_device)
    {
        const uint64_t seed = *threshold_seed_device;
        if (seed == 0)
        {
            if (row == 0)
                fail_device_generation_control(const_cast<int *>(generation_control),
                    DeviceGenerationError::InvalidRequestSeed);
            return;
        }
        inverse_sample_seed = seed;
        threshold_seed = seed;
    }
    const int live_depth = generation_control
        ? generation_control[kDeviceGenerationControlCurrentDraftDepth]
        : row_count;
    if (generation_control &&
        (generation_control[kDeviceGenerationControlOk] == 0 ||
         live_depth <= 0 || live_depth > row_count ||
         generation_control[kDeviceGenerationControlActiveVerifierRowCount] != live_depth + 1))
        return;
    if (row >= live_depth + (generation_control ? 1 : 0))
        return;

    // The live bonus belongs to target row depth, never to captured capacity.
    // It has no draft or acceptance decision. Inactive rows remain untouched.
    if (generation_control && row == live_depth)
    {
        const float threshold = mtp_spec_threshold_from_seed(
            threshold_seed, *threshold_base_position + threshold_position_offset + row,
            0 /* MTPSpecStochasticDrawPurpose::Sample */);
        out_token[row] = sample_distribution_with_threshold(
            target_token_ids + row * distribution_stride,
            target_probs + row * distribution_stride, k, threshold);
        return;
    }

    float accept_threshold = 0.0f;
    float residual_threshold = 0.0f;

    const int threshold_logical_position =
        threshold_base_position
            ? *threshold_base_position + threshold_position_offset + row
            : threshold_first_logical_position + row;
    const int inverse_sample_logical_position =
        threshold_base_position
            ? *threshold_base_position + threshold_position_offset + row
            : inverse_sample_first_logical_position + row;

    if (thresholds_from_seed)
    {
        accept_threshold =
            llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                threshold_seed,
                threshold_logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */);
        residual_threshold =
            llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                threshold_seed,
                threshold_logical_position,
                2 /* MTPSpecStochasticDrawPurpose::Residual */);
    }
    else
    {
        /*
         * Explicit host thresholds are packed into bounded value-owned kernel
         * parameters by the diagnostic wrapper. Seeded production rows never
         * index this bounded payload, so their runtime depth is unrestricted.
         */
        accept_threshold = thresholds.accept[row];
        residual_threshold = thresholds.residual[row];
    }

    const int offset = row * distribution_stride;
    if (!draft_token_ids && !draft_probs)
    {
        if (inverse_sample_vocab_size > 0)
        {
            llaminar2::sampling_math::speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
                target_token_ids + offset,
                target_probs + offset,
                k,
                inverse_sample_vocab_size,
                sampled_draft_tokens[row],
                accept_threshold,
                inverse_sample_seed,
                inverse_sample_logical_position,
                out_token + row,
                out_accepted + row,
                out_accept_probability ? out_accept_probability + row : nullptr,
                out_accept_threshold ? out_accept_threshold + row : nullptr);
        }
        else
        {
            llaminar2::sampling_math::speculative_verify_with_thresholds_one_hot_draft(
                target_token_ids + offset,
                target_probs + offset,
                k,
                sampled_draft_tokens[row],
                accept_threshold,
                residual_threshold,
                out_token + row,
                out_accepted + row,
                out_accept_probability ? out_accept_probability + row : nullptr,
                out_accept_threshold ? out_accept_threshold + row : nullptr);
        }
    }
    else
    {
        llaminar2::sampling_math::speculative_verify_with_thresholds_and_draft_probability(
            target_token_ids + offset,
            target_probs + offset,
            draft_token_ids + offset,
            draft_probs + offset,
            k,
            sampled_draft_tokens[row],
            sampled_draft_probabilities ? sampled_draft_probabilities[row] : 0.0f,
            sampled_draft_probabilities != nullptr,
            accept_threshold,
            residual_threshold,
            out_token + row,
            out_accepted + row,
            out_accept_probability ? out_accept_probability + row : nullptr,
            out_accept_threshold ? out_accept_threshold + row : nullptr);
    }
}

constexpr int PROCESSED_LOGIT_VERIFY_THREADS = 256;

__device__ __forceinline__ bool processedLogitActive(float value)
{
    return isfinite(value);
}

__device__ __forceinline__ bool processedLogitBetter(
    float candidate_value,
    int candidate_index,
    float current_value,
    int current_index)
{
    if (candidate_index < 0)
        return false;
    if (current_index < 0)
        return true;
    return candidate_value > current_value ||
           (candidate_value == current_value &&
            candidate_index < current_index);
}

template <int THREADS>
__device__ void rocm_processed_logit_row_stats_block(
    const float *__restrict__ logits,
    int vocab_size,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs,
    float *__restrict__ out_max,
    float *__restrict__ out_sum,
    int *__restrict__ out_argmax)
{
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float value = logits[token];
        if (processedLogitActive(value) &&
            processedLogitBetter(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    scratch_vals[tid] = local_max;
    scratch_idxs[tid] = local_idx;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    const float max_logit = scratch_vals[0];
    const int argmax_token = scratch_idxs[0];
    float local_sum = 0.0f;
    if (argmax_token >= 0)
    {
        for (int token = tid; token < vocab_size; token += THREADS)
        {
            const float value = logits[token];
            if (processedLogitActive(value))
                local_sum += expf(value - max_logit);
        }
    }

    scratch_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            scratch_vals[tid] += scratch_vals[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
    {
        *out_max = max_logit;
        *out_sum = scratch_vals[0];
        *out_argmax = argmax_token;
    }
    __syncthreads();
}

__device__ __forceinline__ float rocm_processed_logit_probability(
    const float *__restrict__ logits,
    int vocab_size,
    float max_logit,
    float exp_sum,
    int token)
{
    if (token < 0 || token >= vocab_size || !(exp_sum > 0.0f))
        return 0.0f;
    const float value = logits[token];
    if (!processedLogitActive(value))
        return 0.0f;
    return expf(value - max_logit) / exp_sum;
}

__device__ __forceinline__ float rocm_temperature_scaled_logit(
    const float *__restrict__ logits,
    int token,
    float inv_temperature)
{
    const float value = logits[token];
    return isfinite(value) ? value * inv_temperature : -FLT_MAX;
}

/**
 * @brief Build draft proposal probabilities and sample from them.
 *
 * ROCm mirrors CUDA's vLLM-style draft-side primitive: only temperature
 * shapes the proposal distribution. Target processing owns top-k/top-p,
 * penalties, and residual correction.
 */
template <int THREADS>
__device__ void rocm_softmax_sample_temperature_logits_row_block(
    const float *__restrict__ logits,
    int vocab_size,
    float inv_temperature,
    float threshold,
    float *__restrict__ out_probabilities,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs,
    int *__restrict__ selected_token,
    float *__restrict__ selected_probability)
{
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float value =
            rocm_temperature_scaled_logit(logits, token, inv_temperature);
        if (processedLogitBetter(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    scratch_vals[tid] = local_max;
    scratch_idxs[tid] = local_idx;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    const float max_logit = scratch_vals[0];
    const bool has_active = scratch_idxs[0] >= 0;
    float local_sum = 0.0f;
    if (has_active)
    {
        for (int token = tid; token < vocab_size; token += THREADS)
        {
            const float value =
                rocm_temperature_scaled_logit(logits, token, inv_temperature);
            if (isfinite(value))
                local_sum += expf(value - max_logit);
        }
    }

    scratch_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            scratch_vals[tid] += scratch_vals[tid + stride];
        __syncthreads();
    }

    const float exp_sum = scratch_vals[0];
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);
    float chunk_mass = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        float probability = 0.0f;
        if (has_active && exp_sum > 0.0f)
        {
            const float value =
                rocm_temperature_scaled_logit(logits, token, inv_temperature);
            if (isfinite(value))
                probability = expf(value - max_logit) / exp_sum;
        }
        out_probabilities[token] = probability;
        chunk_mass += probability;
    }
    scratch_vals[tid] = chunk_mass;
    __syncthreads();

    if (tid == 0)
    {
        float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold);
        float prefix = 0.0f;
        int selected = has_active ? scratch_idxs[0] : 0;
        float selected_prob = has_active ? out_probabilities[selected] : 0.0f;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + scratch_vals[i])
            {
                prefix += scratch_vals[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                const float probability = out_probabilities[token];
                if (!(probability > 0.0f))
                    continue;
                prefix += probability;
                if (target_mass <= prefix)
                {
                    selected = token;
                    selected_prob = probability;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
        *selected_probability = selected_prob;
    }
    __syncthreads();
}

/**
 * @brief Store temperature-scaled draft logits and sample from their softmax.
 *
 * ROCm mirrors CUDA's lower-memory vLLM-shaped draft primitive: the verifier
 * receives logits/logsumexp inputs instead of a materialized proposal
 * probability matrix.
 */
template <int THREADS>
__device__ void rocm_scale_sample_temperature_logits_row_block(
    const float *__restrict__ logits,
    int vocab_size,
    float inv_temperature,
    float threshold,
    float *__restrict__ out_logits,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs,
    int *__restrict__ selected_token,
    float *__restrict__ selected_probability)
{
    const int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float value =
            rocm_temperature_scaled_logit(logits, token, inv_temperature);
        if (processedLogitBetter(value, token, local_max, local_idx))
        {
            local_max = value;
            local_idx = token;
        }
    }

    scratch_vals[tid] = local_max;
    scratch_idxs[tid] = local_idx;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    const float max_logit = scratch_vals[0];
    const bool has_active = scratch_idxs[0] >= 0;
    float local_sum = 0.0f;
    if (has_active)
    {
        for (int token = tid; token < vocab_size; token += THREADS)
        {
            const float value =
                rocm_temperature_scaled_logit(logits, token, inv_temperature);
            if (isfinite(value))
                local_sum += expf(value - max_logit);
        }
    }

    scratch_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            scratch_vals[tid] += scratch_vals[tid + stride];
        __syncthreads();
    }

    const float exp_sum = scratch_vals[0];
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);
    float chunk_mass = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        const float value =
            rocm_temperature_scaled_logit(logits, token, inv_temperature);
        out_logits[token] = value;
        if (has_active && exp_sum > 0.0f && isfinite(value))
            chunk_mass += expf(value - max_logit) / exp_sum;
    }
    scratch_vals[tid] = chunk_mass;
    __syncthreads();

    if (tid == 0)
    {
        const float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold);
        float prefix = 0.0f;
        int selected = has_active ? scratch_idxs[0] : 0;
        float selected_prob = 0.0f;
        if (has_active && exp_sum > 0.0f)
            selected_prob = expf(out_logits[selected] - max_logit) / exp_sum;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + scratch_vals[i])
            {
                prefix += scratch_vals[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                const float value = out_logits[token];
                if (!isfinite(value) || !(exp_sum > 0.0f))
                    continue;
                const float probability = expf(value - max_logit) / exp_sum;
                if (!(probability > 0.0f))
                    continue;
                prefix += probability;
                if (target_mass <= prefix)
                {
                    selected = token;
                    selected_prob = probability;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
        *selected_probability = selected_prob;
    }
    __syncthreads();
}

template <int THREADS>
__device__ int rocm_sample_processed_logit_residual_block(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int vocab_size,
    float target_max,
    float target_sum,
    int target_argmax,
    float draft_max,
    float draft_sum,
    float threshold,
    float *__restrict__ chunk_sums,
    int *__restrict__ selected_token)
{
    const int tid = threadIdx.x;
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);

    float local_residual = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        const float p = rocm_processed_logit_probability(
            target_logits, vocab_size, target_max, target_sum, token);
        if (p > 0.0f)
        {
            const float q = rocm_processed_logit_probability(
                draft_logits, vocab_size, draft_max, draft_sum, token);
            local_residual += fmaxf(0.0f, p - q);
        }
    }
    chunk_sums[tid] = local_residual;
    __syncthreads();

    if (tid == 0)
    {
        float total = 0.0f;
        for (int i = 0; i < THREADS; ++i)
            total += chunk_sums[i];
        const bool use_residual = total > 0.0f;
        if (!use_residual)
        {
            total = target_sum;
            for (int i = 0; i < THREADS; ++i)
                chunk_sums[i] = 0.0f;
        }

        if (!use_residual)
        {
            for (int i = 0; i < THREADS; ++i)
            {
                const int b = i * chunk;
                const int e = min(vocab_size, b + chunk);
                float sum = 0.0f;
                for (int token = b; token < e; ++token)
                {
                    const float value = target_logits[token];
                    if (processedLogitActive(value))
                        sum += expf(value - target_max);
                }
                chunk_sums[i] = sum;
            }
        }

        float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold) * total;
        float prefix = 0.0f;
        int selected = target_argmax;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + chunk_sums[i])
            {
                prefix += chunk_sums[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                float weight = 0.0f;
                if (use_residual)
                {
                    const float p = rocm_processed_logit_probability(
                        target_logits, vocab_size, target_max, target_sum, token);
                    if (p > 0.0f)
                    {
                        const float q = rocm_processed_logit_probability(
                            draft_logits, vocab_size, draft_max, draft_sum, token);
                        weight = fmaxf(0.0f, p - q);
                    }
                }
                else
                {
                    const float value = target_logits[token];
                    if (processedLogitActive(value))
                        weight = expf(value - target_max);
                }
                if (!(weight > 0.0f))
                    continue;
                prefix += weight;
                if (target_mass <= prefix)
                {
                    selected = token;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
    }
    __syncthreads();
    return *selected_token;
}

template <int THREADS>
__device__ int rocm_sample_processed_logit_row_block(
    const float *__restrict__ logits,
    int vocab_size,
    float max_logit,
    float exp_sum,
    int argmax_token,
    float threshold,
    float *__restrict__ chunk_sums,
    int *__restrict__ selected_token)
{
    const int tid = threadIdx.x;
    const int chunk = (vocab_size + THREADS - 1) / THREADS;
    const int begin = tid * chunk;
    const int end = min(vocab_size, begin + chunk);

    float local_sum = 0.0f;
    for (int token = begin; token < end; ++token)
    {
        const float value = logits[token];
        if (processedLogitActive(value))
            local_sum += expf(value - max_logit);
    }
    chunk_sums[tid] = local_sum;
    __syncthreads();

    if (tid == 0)
    {
        const float total = exp_sum;
        float target_mass =
            llaminar2::sampling_math::clamp_unit_threshold(threshold) * total;
        float prefix = 0.0f;
        int selected = argmax_token;
        for (int i = 0; i < THREADS; ++i)
        {
            if (target_mass > prefix + chunk_sums[i])
            {
                prefix += chunk_sums[i];
                continue;
            }
            const int b = i * chunk;
            const int e = min(vocab_size, b + chunk);
            for (int token = b; token < e; ++token)
            {
                const float value = logits[token];
                if (!processedLogitActive(value))
                    continue;
                prefix += expf(value - max_logit);
                if (target_mass <= prefix)
                {
                    selected = token;
                    i = THREADS;
                    break;
                }
            }
        }
        *selected_token = selected;
    }
    __syncthreads();
    return *selected_token;
}

template <int THREADS>
__device__ int rocm_sample_recovered_probability_row_block(
    const float *__restrict__ target_probs,
    const float *__restrict__ draft_probs,
    const float *__restrict__ inverse_rejection_samples,
    int vocab_size,
    int draft_token,
    bool no_draft_probabilities,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs)
{
    const int tid = threadIdx.x;
    float local_best = -1.0f;
    int local_token = -1;

    for (int token = tid; token < vocab_size; token += THREADS)
    {
        float probability = target_probs[token];
        if (!isfinite(probability) || !(probability > 0.0f))
            probability = 0.0f;

        if (no_draft_probabilities)
        {
            if (token == draft_token)
                probability = 0.0f;
        }
        else
        {
            float draft_probability = draft_probs[token];
            if (!isfinite(draft_probability) || !(draft_probability > 0.0f))
                draft_probability = 0.0f;
            probability = fmaxf(0.0f, probability - draft_probability);
        }

        const float inverse_sample = inverse_rejection_samples[token];
        const float value =
            (inverse_sample > 0.0f && isfinite(inverse_sample))
                ? probability * inverse_sample
                : 0.0f;
        if (processedLogitBetter(value, token, local_best, local_token))
        {
            local_best = value;
            local_token = token;
        }
    }

    scratch_vals[tid] = local_best;
    scratch_idxs[tid] = local_token;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    return scratch_idxs[0] >= 0 ? scratch_idxs[0] : 0;
}

/**
 * @brief Sample one processed full-logit row on an explicit stream.
 *
 * This is used for the vLLM-style all-accepted bonus token without creating a
 * compact top-k/top-p table.
 */
__global__ void rocm_sample_processed_logits_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float threshold,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float max_logit;
    __shared__ float exp_sum;
    __shared__ int argmax_token;
    __shared__ int selected_token;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &max_logit,
        &exp_sum,
        &argmax_token);

    rocm_sample_processed_logit_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        max_logit,
        exp_sum,
        argmax_token,
        threshold,
        scratch_vals,
        &selected_token);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
        {
            *out_probability = rocm_processed_logit_probability(
                logits,
                vocab_size,
                max_logit,
                exp_sum,
                selected_token);
        }
    }
}

/**
 * @brief Lazily sample a processed bonus row only if the verifier needs it.
 *
 * ROCm mirrors the CUDA guarded bonus contract: rejected or stopped batches
 * write `-1` and skip the full-vocabulary processed-logit scan, while
 * all-accepted batches sample exactly like the eager bonus sampler.
 */
__global__ void rocm_sample_processed_logits_if_speculative_batch_needs_bonus_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float threshold,
    unsigned long long threshold_seed,
    const int *__restrict__ threshold_position,
    int threshold_position_offset,
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    int row_count,
    int first_token,
    const int *__restrict__ first_token_device,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    __shared__ int should_sample_bonus;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float max_logit;
    __shared__ float exp_sum;
    __shared__ int argmax_token;
    __shared__ int selected_token;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    if (threadIdx.x == 0)
    {
        int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
            stop_token0,
            stop_token1,
            stop_token2,
            stop_token3,
            stop_token4,
            stop_token5,
            stop_token6,
            stop_token7};
        const int sampled_first_token =
            first_token_device ? *first_token_device : first_token;
        should_sample_bonus =
            llaminar2::sampling_math::speculative_batch_needs_bonus_ready_token(
                sampled_first_token,
                verify_tokens,
                verify_accepted,
                row_count,
                stop_tokens,
                stop_token_count)
                ? 1
                : 0;
        if (!should_sample_bonus)
        {
            *out_token = -1;
            if (out_probability)
                *out_probability = 0.0f;
        }
    }
    __syncthreads();
    if (!should_sample_bonus)
        return;

    const float effective_threshold =
        threshold_position
            ? llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                  threshold_seed,
                  *threshold_position + threshold_position_offset,
                  0 /* MTPSpecStochasticDrawPurpose::Sample */)
            : threshold;

    rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &max_logit,
        &exp_sum,
        &argmax_token);

    rocm_sample_processed_logit_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        max_logit,
        exp_sum,
        argmax_token,
        effective_threshold,
        scratch_vals,
        &selected_token);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
        {
            *out_probability = rocm_processed_logit_probability(
                logits,
                vocab_size,
                max_logit,
                exp_sum,
                selected_token);
        }
    }
}

__global__ void rocm_softmax_sample_temperature_logits_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float temperature,
    float threshold,
    float *__restrict__ out_probabilities,
    int out_row_stride,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    (void)out_row_stride;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int selected_token;
    __shared__ float selected_probability;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const float safe_temperature =
        (isfinite(temperature) && temperature > 0.0f) ? temperature : 1.0f;
    rocm_softmax_sample_temperature_logits_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        1.0f / safe_temperature,
        threshold,
        out_probabilities,
        scratch_vals,
        scratch_idxs,
        &selected_token,
        &selected_probability);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
            *out_probability = selected_probability;
    }
}

__global__ void rocm_scale_sample_temperature_logits_f32_kernel(
    const float *__restrict__ logits,
    int vocab_size,
    int row_stride,
    float temperature,
    float threshold,
    float *__restrict__ out_logits,
    int out_row_stride,
    int *__restrict__ out_token,
    float *__restrict__ out_probability)
{
    (void)row_stride;
    (void)out_row_stride;
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int selected_token;
    __shared__ float selected_probability;

    if (blockIdx.x != 0 || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const float safe_temperature =
        (isfinite(temperature) && temperature > 0.0f) ? temperature : 1.0f;
    rocm_scale_sample_temperature_logits_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        logits,
        vocab_size,
        1.0f / safe_temperature,
        threshold,
        out_logits,
        scratch_vals,
        scratch_idxs,
        &selected_token,
        &selected_probability);

    if (threadIdx.x == 0)
    {
        *out_token = selected_token;
        if (out_probability)
            *out_probability = selected_probability;
    }
}

__global__ void rocm_softmax_processed_logits_f32_kernel(
    const float *__restrict__ logits,
    int row_count,
    int vocab_size,
    int row_stride,
    float *__restrict__ out_probabilities,
    int out_row_stride)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float max_logit;
    __shared__ float exp_sum;
    __shared__ int argmax_token;

    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int tid = threadIdx.x;
    const float *row_logits =
        logits + static_cast<size_t>(blockIdx.x) * row_stride;
    float *row_out =
        out_probabilities + static_cast<size_t>(blockIdx.x) * out_row_stride;

    rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        row_logits,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &max_logit,
        &exp_sum,
        &argmax_token);

    for (int token = tid; token < vocab_size; token += blockDim.x)
    {
        row_out[token] =
            rocm_processed_logit_probability(
                row_logits,
                vocab_size,
                max_logit,
                exp_sum,
            token);
    }
}

/**
 * @brief Materialize vLLM-style inverse-exponential rejection samples.
 *
 * ROCm mirrors the CUDA implementation. Values are keyed by seed, logical
 * output position, and token id so replayed stochastic verification remains
 * deterministic without host-generated full-vocab random matrices.
 */
__global__ void rocm_fill_inverse_exponential_samples_f32_kernel(
    float *__restrict__ out_samples,
    int row_count,
    int vocab_size,
    int row_stride,
    unsigned long long seed,
    int first_logical_position)
{
    constexpr unsigned long long kInverseSampleDomain =
        0xA0761D6478BD642FULL;

    const int row = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || token >= vocab_size)
        return;

    const int signed_position = first_logical_position + row;
    const unsigned long long logical_position =
        static_cast<unsigned long long>(signed_position > 0 ? signed_position : 0);
    const unsigned long long offset =
        logical_position * static_cast<unsigned long long>(vocab_size) +
        static_cast<unsigned long long>(token);
    const float uniform =
        llaminar2::sampling_math::uniform01(seed ^ kInverseSampleDomain, offset);
    out_samples[static_cast<size_t>(row) * row_stride + token] =
        llaminar2::sampling_math::inverse_exponential_from_uniform(uniform);
}

/**
 * @brief Verify stochastic MTP rows directly from processed full logits.
 *
 * ROCm mirrors CUDA's reference kernel: block-level row stats recover the
 * sampled draft token probabilities, and residual sampling is paid only for
 * rejected rows.
 */
__global__ void rocm_speculative_verify_processed_logits_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold,
    const float *__restrict__ draft_token_probabilities)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float target_max;
    __shared__ float target_sum;
    __shared__ float draft_max;
    __shared__ float draft_sum;
    __shared__ int target_argmax;
    __shared__ int draft_argmax;
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_logits + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        draft_logits + static_cast<size_t>(row) * draft_row_stride;
    rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        target_row,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &target_max,
        &target_sum,
        &target_argmax);
    const bool has_draft_token_probability =
        draft_token_probabilities != nullptr;
    if (!has_draft_token_probability)
    {
        rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
            draft_row,
            vocab_size,
            scratch_vals,
            scratch_idxs,
            &draft_max,
            &draft_sum,
            &draft_argmax);
    }

    const float accept_threshold = thresholds.accept[row];
    const float residual_threshold = thresholds.residual[row];

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
            const int draft_token = sampled_draft_tokens[row];
            const float p = rocm_processed_logit_probability(
                target_row, vocab_size, target_max, target_sum, draft_token);
            const float q =
                has_draft_token_probability
                    ? draft_token_probabilities[row]
                    : rocm_processed_logit_probability(
                          draft_row, vocab_size, draft_max, draft_sum, draft_token);
        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold < accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        if (has_draft_token_probability)
        {
            rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
                draft_row,
                vocab_size,
                scratch_vals,
                scratch_idxs,
                &draft_max,
                &draft_sum,
                &draft_argmax);
        }
        rocm_sample_processed_logit_residual_block<PROCESSED_LOGIT_VERIFY_THREADS>(
            target_row,
            draft_row,
            vocab_size,
            target_max,
            target_sum,
            target_argmax,
            draft_max,
            draft_sum,
            residual_threshold,
            scratch_vals,
            &row_token);
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

/**
 * @brief Verify rows from processed target logits and draft proposal probabilities.
 *
 * ROCm mirrors the CUDA lower-memory verifier: target probabilities are
 * computed from processed logits in the row block, draft probabilities are the
 * temperature-only proposal rows captured during MTP drafting, and recovered
 * token noise is generated on the fly instead of materializing a full inverse
 * random matrix.
 */
template <int THREADS>
__device__ int rocm_sample_recovered_processed_target_draft_probability_row_block(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_probs,
    int vocab_size,
    float target_max,
    float target_sum,
    int draft_token,
    unsigned long long inverse_sample_seed,
    int logical_position,
    bool no_draft_probabilities,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs)
{
    constexpr unsigned long long kInverseSampleDomain =
        0xA0761D6478BD642FULL;

    const int tid = threadIdx.x;
    const unsigned long long safe_position =
        static_cast<unsigned long long>(logical_position > 0 ? logical_position : 0);
    float local_best = -1.0f;
    int local_token = -1;

    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float p = rocm_processed_logit_probability(
            target_logits, vocab_size, target_max, target_sum, token);
        float q = no_draft_probabilities
                      ? (token == draft_token ? 1.0f : 0.0f)
                      : draft_probs[token];
        if (!isfinite(q) || !(q > 0.0f))
            q = 0.0f;
        const float probability = fmaxf(0.0f, p - q);

        const unsigned long long offset =
            safe_position * static_cast<unsigned long long>(vocab_size) +
            static_cast<unsigned long long>(token);
        const float uniform =
            llaminar2::sampling_math::uniform01(
                inverse_sample_seed ^ kInverseSampleDomain,
                offset);
        const float inverse_sample =
            llaminar2::sampling_math::inverse_exponential_from_uniform(uniform);
        const float value =
            (inverse_sample > 0.0f && isfinite(inverse_sample))
                ? probability * inverse_sample
                : 0.0f;
        if (processedLogitBetter(value, token, local_best, local_token))
        {
            local_best = value;
            local_token = token;
        }
    }

    scratch_vals[tid] = local_best;
    scratch_idxs[tid] = local_token;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    return scratch_idxs[0] >= 0 ? scratch_idxs[0] : 0;
}

template <int THREADS>
__device__ int rocm_sample_recovered_processed_target_draft_logit_row_block(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int vocab_size,
    float target_max,
    float target_sum,
    float draft_max,
    float draft_sum,
    int draft_token,
    unsigned long long inverse_sample_seed,
    int logical_position,
    float *__restrict__ scratch_vals,
    int *__restrict__ scratch_idxs)
{
    constexpr unsigned long long kInverseSampleDomain =
        0xA0761D6478BD642FULL;

    const int tid = threadIdx.x;
    const unsigned long long safe_position =
        static_cast<unsigned long long>(logical_position > 0 ? logical_position : 0);
    float local_best = -1.0f;
    int local_token = -1;

    for (int token = tid; token < vocab_size; token += THREADS)
    {
        const float p = rocm_processed_logit_probability(
            target_logits, vocab_size, target_max, target_sum, token);
        const float q = rocm_processed_logit_probability(
            draft_logits, vocab_size, draft_max, draft_sum, token);
        const float probability = fmaxf(0.0f, p - q);

        const unsigned long long offset =
            safe_position * static_cast<unsigned long long>(vocab_size) +
            static_cast<unsigned long long>(token);
        const float uniform =
            llaminar2::sampling_math::uniform01(
                inverse_sample_seed ^ kInverseSampleDomain,
                offset);
        const float inverse_sample =
            llaminar2::sampling_math::inverse_exponential_from_uniform(uniform);
        const float value =
            (inverse_sample > 0.0f && isfinite(inverse_sample))
                ? probability * inverse_sample
                : 0.0f;
        if (processedLogitBetter(value, token, local_best, local_token))
        {
            local_best = value;
            local_token = token;
        }
    }

    scratch_vals[tid] = local_best;
    scratch_idxs[tid] = local_token;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride &&
            processedLogitBetter(
                scratch_vals[tid + stride],
                scratch_idxs[tid + stride],
                scratch_vals[tid],
                scratch_idxs[tid]))
        {
            scratch_vals[tid] = scratch_vals[tid + stride];
            scratch_idxs[tid] = scratch_idxs[tid + stride];
        }
        __syncthreads();
    }

    return scratch_idxs[0] >= 0 ? scratch_idxs[0] : 0;
}

__global__ void rocm_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_probabilities,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int thresholds_from_seed,
    const int *__restrict__ threshold_base_position,
    int threshold_position_offset,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold,
    int no_draft_probabilities)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float target_max;
    __shared__ float target_sum;
    __shared__ int target_argmax;
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_logits + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        no_draft_probabilities
            ? nullptr
            : draft_probabilities + static_cast<size_t>(row) * draft_row_stride;

    rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        target_row,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &target_max,
        &target_sum,
        &target_argmax);

    float accept_threshold = 0.0f;

    const int logical_position =
        threshold_base_position
            ? *threshold_base_position + threshold_position_offset + row
            : inverse_sample_first_logical_position + row;
    if (thresholds_from_seed)
    {
        accept_threshold =
            llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                inverse_sample_seed,
                logical_position,
                1 /* MTPSpecStochasticDrawPurpose::Accept */);
    }
    else
    {
        accept_threshold = thresholds.accept[row];
    }

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
        const int draft_token = sampled_draft_tokens[row];
        const float p = rocm_processed_logit_probability(
            target_row, vocab_size, target_max, target_sum, draft_token);
        float q = no_draft_probabilities ? 1.0f : 0.0f;
        if (draft_token >= 0 && draft_token < vocab_size)
        {
            if (!no_draft_probabilities)
            {
                q = draft_row[draft_token];
                if (!isfinite(q) || !(q > 0.0f))
                    q = 0.0f;
            }
        }
        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold <= accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        const int draft_token = sampled_draft_tokens[row];
        const int recovered_token =
            rocm_sample_recovered_processed_target_draft_probability_row_block<
                PROCESSED_LOGIT_VERIFY_THREADS>(
                target_row,
                draft_row,
                vocab_size,
                target_max,
                target_sum,
                draft_token,
                inverse_sample_seed,
                logical_position,
                no_draft_probabilities != 0,
                scratch_vals,
                scratch_idxs);
        if (tid == 0)
            row_token = recovered_token;
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

__global__ void rocm_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_logits,
    const float *__restrict__ draft_logits,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    const float *__restrict__ sampled_draft_probabilities,
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ float target_max;
    __shared__ float target_sum;
    __shared__ float draft_max;
    __shared__ float draft_sum;
    __shared__ int target_argmax;
    __shared__ int draft_argmax;
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_logits + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        draft_logits + static_cast<size_t>(row) * draft_row_stride;
    const bool has_sampled_draft_probability =
        sampled_draft_probabilities != nullptr;

    rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
        target_row,
        vocab_size,
        scratch_vals,
        scratch_idxs,
        &target_max,
        &target_sum,
        &target_argmax);

    if (!has_sampled_draft_probability)
    {
        rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
            draft_row,
            vocab_size,
            scratch_vals,
            scratch_idxs,
            &draft_max,
            &draft_sum,
            &draft_argmax);
    }

    const float accept_threshold = thresholds.accept[row];

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
        const int draft_token = sampled_draft_tokens[row];
        const float p = rocm_processed_logit_probability(
            target_row, vocab_size, target_max, target_sum, draft_token);
        float q = 0.0f;
        if (has_sampled_draft_probability)
        {
            q = sampled_draft_probabilities[row];
            if (!isfinite(q) || !(q > 0.0f))
                q = 0.0f;
        }
        else
        {
            q = rocm_processed_logit_probability(
                draft_row, vocab_size, draft_max, draft_sum, draft_token);
        }
        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold <= accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        if (has_sampled_draft_probability)
        {
            rocm_processed_logit_row_stats_block<PROCESSED_LOGIT_VERIFY_THREADS>(
                draft_row,
                vocab_size,
                scratch_vals,
                scratch_idxs,
                &draft_max,
                &draft_sum,
                &draft_argmax);
        }
        const int draft_token = sampled_draft_tokens[row];
        const int recovered_token =
            rocm_sample_recovered_processed_target_draft_logit_row_block<
                PROCESSED_LOGIT_VERIFY_THREADS>(
                target_row,
                draft_row,
                vocab_size,
                target_max,
                target_sum,
                draft_max,
                draft_sum,
                draft_token,
                inverse_sample_seed,
                inverse_sample_first_logical_position + row,
                scratch_vals,
                scratch_idxs);
        if (tid == 0)
            row_token = recovered_token;
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

/**
 * @brief Verify stochastic MTP rows from full target/draft probability rows.
 *
 * ROCm mirrors the CUDA vLLM-style primitive: accept with p/q for the sampled
 * draft token and use a parallel recovered-token reduction only on rejection.
 */
__global__ void rocm_speculative_verify_probabilities_thresholds_batch_device_tokens_kernel(
    const float *__restrict__ target_probabilities,
    const float *__restrict__ draft_probabilities,
    const float *__restrict__ inverse_rejection_samples,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    int inverse_sample_row_stride,
    const int *__restrict__ sampled_draft_tokens,
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters thresholds,
    int no_draft_probabilities,
    int *__restrict__ out_token,
    int *__restrict__ out_accepted,
    float *__restrict__ out_accept_probability,
    float *__restrict__ out_accept_threshold)
{
    __shared__ float scratch_vals[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int scratch_idxs[PROCESSED_LOGIT_VERIFY_THREADS];
    __shared__ int row_token;
    __shared__ int row_accepted;

    const int tid = threadIdx.x;
    if (blockIdx.x >= row_count || blockDim.x != PROCESSED_LOGIT_VERIFY_THREADS)
        return;

    const int row = blockIdx.x;
    const float *target_row =
        target_probabilities + static_cast<size_t>(row) * target_row_stride;
    const float *draft_row =
        no_draft_probabilities
            ? nullptr
            : draft_probabilities + static_cast<size_t>(row) * draft_row_stride;
    const float *inverse_row =
        inverse_rejection_samples +
        static_cast<size_t>(row) * inverse_sample_row_stride;

    const float accept_threshold = thresholds.accept[row];

    if (tid == 0)
    {
        row_token = -1;
        row_accepted = 0;
        const int draft_token = sampled_draft_tokens[row];
        float p = 0.0f;
        float q = no_draft_probabilities ? 1.0f : 0.0f;
        if (draft_token >= 0 && draft_token < vocab_size)
        {
            p = target_row[draft_token];
            if (!isfinite(p) || !(p > 0.0f))
                p = 0.0f;
            if (!no_draft_probabilities)
            {
                q = draft_row[draft_token];
                if (!isfinite(q) || !(q > 0.0f))
                    q = 0.0f;
            }
        }

        const float accept_probability =
            llaminar2::sampling_math::speculative_accept_probability(p, q);
        const float threshold =
            llaminar2::sampling_math::clamp_unit_threshold(accept_threshold);
        if (out_accept_probability)
            out_accept_probability[row] = accept_probability;
        if (out_accept_threshold)
            out_accept_threshold[row] = threshold;
        if (threshold <= accept_probability)
        {
            row_token = draft_token;
            row_accepted = 1;
        }
    }
    __syncthreads();

    if (!row_accepted)
    {
        const int draft_token = sampled_draft_tokens[row];
        const int recovered_token =
            rocm_sample_recovered_probability_row_block<PROCESSED_LOGIT_VERIFY_THREADS>(
                target_row,
                draft_row,
                inverse_row,
                vocab_size,
                draft_token,
                no_draft_probabilities != 0,
                scratch_vals,
                scratch_idxs);
        if (tid == 0)
            row_token = recovered_token;
    }
    __syncthreads();

    if (tid == 0)
    {
        out_token[row] = row_token;
        out_accepted[row] = row_accepted;
    }
}

/**
 * @brief Reduce row-wise stochastic verifier results into one speculative commit plan.
 *
 * This mirrors the CUDA kernel and calls shared SamplingMath so ROCm and CUDA
 * keep the same stop/reject/all-accepted semantics.
 */
__global__ void rocm_summarize_speculative_verify_batch_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    int row_count,
    int first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    const int *__restrict__ bonus_token,
    int has_bonus_token,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    int leading_committed_output_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7};
    const int ready_token =
        has_bonus_token && bonus_token ? *bonus_token : -1;
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(row_count + 1)
            ? static_cast<int>(*max_state_commit_rows)
            : row_count + 1;
    llaminar2::sampling_math::summarize_speculative_verify_batch_at_commit_boundary(
        first_token,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        stop_token_count,
        ready_token,
        has_bonus_token,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        /*greedy_draft_tokens=*/nullptr,
        leading_committed_output_count);
}

/**
 * @brief Device-first-token variant of the speculative batch reducer.
 *
 * The first sampled main-model token is already resident in arena scratch.
 * This kernel reads that scalar on the GPU and then delegates to shared
 * SamplingMath, keeping ROCm semantics matched with CUDA while avoiding a
 * pre-verifier D2H synchronization.
 */
__global__ void rocm_summarize_speculative_verify_batch_device_first_token_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    int row_count,
    const int *__restrict__ first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    const int *__restrict__ bonus_token,
    int has_bonus_token,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    int leading_committed_output_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7};
    const int ready_token =
        has_bonus_token && bonus_token ? *bonus_token : -1;
    const int sampled_first_token = first_token ? *first_token : -1;
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(row_count + 1)
            ? static_cast<int>(*max_state_commit_rows)
            : row_count + 1;
    llaminar2::sampling_math::summarize_speculative_verify_batch_at_commit_boundary(
        sampled_first_token,
        verify_tokens,
        verify_accepted,
        row_count,
        stop_tokens,
        stop_token_count,
        ready_token,
        has_bonus_token,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        /*greedy_draft_tokens=*/nullptr,
        leading_committed_output_count);
}

/**
 * @brief Reduce a generation transaction using only resident mutable controls.
 *
 * This is the HIP twin of the CUDA production reducer.  The request-admitted
 * stop row and the generation-controller words are read directly on the exact
 * verifier stream, so graph replay cannot observe a stale host carry bit or
 * commit budget.  Exactly one of the explicit acceptance row and the
 * serial-equivalent draft row is supplied by the launcher.
 */
__global__ void
rocm_summarize_speculative_verify_batch_device_generation_controls_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ verify_accepted,
    const int *__restrict__ greedy_draft_tokens,
    int row_count,
    const int *__restrict__ first_token,
    const int *__restrict__ stop_tokens,
    const int *__restrict__ bonus_token,
    int has_bonus_token,
    const int *__restrict__ generation_control,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::summarize_speculative_verify_batch_device_generation_controls(
        first_token ? *first_token : -1,
        verify_tokens, verify_accepted, greedy_draft_tokens, row_count,
        stop_tokens, llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens,
        bonus_token, has_bonus_token, generation_control,
        out_tokens, out_token_capacity, out_meta);
}

/**
 * @brief Fused seeded sampling and serial-equivalent speculative reduction.
 *
 * One wavefront owns the complete MTP transaction. Each active lane executes
 * the canonical scalar probability scan for one compact target row, preserving
 * serial decode's floating-point order. Lane zero reduces the sampled rows
 * after a workgroup barrier. The single-workgroup topology avoids atomics,
 * temporary storage, and a second launch while covering depths one through
 * fifteen plus the terminal bonus row.
 */
template <bool RetainFirstTransactionDiagnostic>
__global__ __launch_bounds__(64, 1) void
rocm_sample_and_summarize_serial_equivalent_speculative_batch_kernel(
    const int *__restrict__ target_token_ids,
    const float *__restrict__ target_probs,
    int target_row_stride,
    int top_k,
    int row_count,
    unsigned long long threshold_seed,
    const int *__restrict__ threshold_position,
    int threshold_position_offset,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ stop_tokens,
    const int *__restrict__ generation_control,
    int *__restrict__ sampled_target_tokens,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    llaminar2::sampling_math::MTPFirstTransactionDiagnosticRecord
        *__restrict__ first_transaction_diagnostic,
    const uint64_t *__restrict__ threshold_seed_device)
{
    constexpr int kMaxSampleRows = 16;
    __shared__ int sampled_rows[kMaxSampleRows];
    __shared__ int active_comparison_rows;

    // The immutable address names request data, including across graph resets.
    // All lanes consume the same admitted value before the row-zero guard.
    const uint64_t seed = threshold_seed_device ? *threshold_seed_device : threshold_seed;
    const int sample_row = static_cast<int>(threadIdx.x);
    if (sample_row == 0)
    {
        const int controller_depth =
            generation_control[
                llaminar2::sampling_math::
                    kDeviceGenerationControlCurrentDraftDepth];
        const int controller_verifier_rows =
            generation_control[
                llaminar2::sampling_math::
                    kDeviceGenerationControlActiveVerifierRowCount];
        const bool valid_depth =
            generation_control[
                llaminar2::sampling_math::kDeviceGenerationControlOk] != 0 &&
            controller_depth >= 1 && controller_depth <= row_count &&
            controller_verifier_rows == controller_depth + 1;
        active_comparison_rows = valid_depth && seed != 0 ? controller_depth : 0;
        if (!valid_depth || seed == 0)
        {
            llaminar2::sampling_math::fail_device_generation_control(
                const_cast<int *>(generation_control),
                seed == 0 ? llaminar2::sampling_math::DeviceGenerationError::InvalidRequestSeed
                          : llaminar2::sampling_math::DeviceGenerationError::InvalidDepthSelector);
            llaminar2::sampling_math::
                initialize_invalid_speculative_batch_outcome(
                    out_tokens,
                    out_token_capacity,
                    out_meta);
            if constexpr (RetainFirstTransactionDiagnostic)
            {
                if (first_transaction_diagnostic)
                    first_transaction_diagnostic->valid = 0;
            }
        }
    }
    __syncthreads();

    if (active_comparison_rows == 0)
    {
        if (sample_row <= row_count)
            sampled_target_tokens[sample_row] = -1;
        return;
    }

    const int sample_row_count = active_comparison_rows + 1;
    const int transaction_count =
        generation_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlTransactionCount];
    const bool retain_first_transaction =
        RetainFirstTransactionDiagnostic &&
        first_transaction_diagnostic &&
        transaction_count == 0;
    if (sample_row < kMaxSampleRows)
    {
        const bool active = sample_row < sample_row_count;
        float threshold = 0.0f;
        int sampled = -1;
        if (active)
        {
            threshold =
                llaminar2::sampling_math::mtp_spec_threshold_from_seed(
                    seed,
                    *threshold_position + threshold_position_offset +
                        sample_row,
                    0 /* MTPSpecStochasticDrawPurpose::Sample */);
            sampled =
                llaminar2::sampling_math::
                    sample_distribution_with_threshold(
                        target_token_ids +
                            sample_row * target_row_stride,
                        target_probs + sample_row * target_row_stride,
                        top_k,
                        threshold);
            sampled_rows[sample_row] = sampled;
            sampled_target_tokens[sample_row] = sampled;
        }
        else
        {
            sampled_rows[sample_row] = -1;
            // Wavefront scratch capacity is not caller-owned output capacity.
            // Preserve the next request when this captured depth is below 15.
            if (sample_row <= row_count)
                sampled_target_tokens[sample_row] = -1;
        }
        if constexpr (RetainFirstTransactionDiagnostic)
        {
            if (retain_first_transaction)
            {
                llaminar2::sampling_math::
                    record_mtp_first_transaction_sample_row(
                        first_transaction_diagnostic,
                        sample_row,
                        active,
                        active
                            ? target_token_ids +
                                  sample_row * target_row_stride
                            : nullptr,
                        active
                            ? target_probs +
                                  sample_row * target_row_stride
                            : nullptr,
                        top_k,
                        verifier_input_tokens,
                        active_comparison_rows,
                        threshold,
                        sampled);
            }
        }
    }
    __syncthreads();

    if (sample_row != 0)
        return;

    const int commit_budget =
        generation_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlTransactionCommitBudget];
    const int leading_committed_output_count =
        generation_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlNextLeadingCommittedOutputCount];
    llaminar2::sampling_math::
        summarize_speculative_verify_batch_at_commit_boundary(
            verifier_input_tokens[0],
            sampled_rows,
            /*row_accepted=*/nullptr,
            active_comparison_rows,
            stop_tokens,
            llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens,
            sampled_rows[active_comparison_rows],
            /*has_bonus_ready_token=*/1,
            commit_budget,
            out_tokens,
            out_token_capacity,
            out_meta,
            verifier_input_tokens,
            leading_committed_output_count);

    if constexpr (RetainFirstTransactionDiagnostic)
    {
        if (retain_first_transaction)
        {
            llaminar2::sampling_math::
                finalize_mtp_first_transaction_diagnostic(
                    first_transaction_diagnostic,
                    seed,
                    *threshold_position,
                    threshold_position_offset,
                    active_comparison_rows,
                    top_k,
                    transaction_count,
                    commit_budget,
                    leading_committed_output_count,
                    out_tokens,
                    out_token_capacity,
                    out_meta);
        }
    }
}

/**
 * @brief Retain one transaction-zero sidecar boundary on the exact graph stream.
 *
 * The fixed-order lane fold is the HIP counterpart of the CUDA diagnostic.
 * It intentionally exists only in opt-in diagnostic graph topology; ordinary
 * MTP publication pays no branch, launch, transfer, or storage traffic.
 */
__global__ __launch_bounds__(256, 1) void
rocm_retain_mtp_first_transaction_draft_boundary_kernel(
    const uint32_t *__restrict__ data_words,
    int word_count,
    int boundary,
    int draft_slot,
    const int *__restrict__ condition_token,
    const int *__restrict__ position_id,
    const int *__restrict__ generation_control,
    int generation_control_stride,
    llaminar2::sampling_math::MTPFirstTransactionDiagnosticRecord
        *__restrict__ diagnostic)
{
    using namespace llaminar2::sampling_math;
    constexpr int kThreads = 256;
    __shared__ unsigned long long lane_hashes[kThreads];

    if (!generation_control || !diagnostic || generation_control_stride <
            kDeviceGenerationControlCount ||
        generation_control[kDeviceGenerationControlTransactionCount] != 0)
    {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x);
    uint64_t hash = 1469598103934665603ULL ^
                    static_cast<uint64_t>(lane + 1);
    for (int index = lane; index < word_count; index += kThreads)
    {
        hash = append_diagnostic_u32_fnv1a(
            hash,
            static_cast<uint32_t>(index));
        hash = append_diagnostic_u32_fnv1a(hash, data_words[index]);
    }
    lane_hashes[lane] = hash;
    __syncthreads();

    if (lane != 0)
        return;

    if (draft_slot == 0 && boundary == 0)
    {
        diagnostic->valid = 0;
        diagnostic->version = kMTPFirstTransactionDiagnosticVersion;
        diagnostic->draft_diagnostic_depth = 0;
        for (int slot = 0; slot < kSpeculativeBatchMaxRows; ++slot)
        {
            diagnostic->draft_condition_tokens[slot] = -1;
            diagnostic->draft_position_ids[slot] = -1;
            for (int retained_boundary = 0;
                 retained_boundary < kMTPFirstTransactionDraftBoundaryCount;
                 ++retained_boundary)
            {
                diagnostic->draft_boundary_word_counts[slot]
                                                       [retained_boundary] = 0;
                diagnostic->draft_boundary_hashes[slot]
                                                   [retained_boundary] = 0;
            }
        }
    }

    if (boundary == 0)
    {
        diagnostic->draft_condition_tokens[draft_slot] = *condition_token;
        diagnostic->draft_position_ids[draft_slot] = *position_id;
        diagnostic->draft_diagnostic_depth = draft_slot + 1;
    }

    uint64_t folded = 1469598103934665603ULL;
    if (word_count > 0)
    {
        for (int source_lane = 0; source_lane < kThreads; ++source_lane)
        {
            const uint64_t lane_hash = lane_hashes[source_lane];
            folded = append_diagnostic_u32_fnv1a(
                folded,
                static_cast<uint32_t>(lane_hash));
            folded = append_diagnostic_u32_fnv1a(
                folded,
                static_cast<uint32_t>(lane_hash >> 32U));
        }
    }
    else
    {
        folded = 0;
    }
    diagnostic->draft_boundary_word_counts[draft_slot][boundary] =
        static_cast<uint32_t>(word_count);
    diagnostic->draft_boundary_hashes[draft_slot][boundary] = folded;
}

/**
 * @brief Reduce greedy verifier argmax rows into one speculative commit plan.
 *
 * ROCm uses the same tiny graph-capturable reducer as CUDA: compare
 * device-resident verifier tokens against the compact verifier input row and
 * write only fixed-size summary buffers for the host handoff.
 */
__global__ void rocm_summarize_greedy_speculative_verify_batch_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ draft_tokens,
    int compare_row_count,
    int first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    int leading_committed_output_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    int stop_tokens[llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens] = {
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7};
    const int sampled_first_token = draft_tokens ? draft_tokens[0] : first_token;
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(compare_row_count + 1)
            ? static_cast<int>(*max_state_commit_rows)
            : compare_row_count + 1;
    llaminar2::sampling_math::summarize_greedy_speculative_verify_batch_at_commit_boundary(
        sampled_first_token,
        verify_tokens,
        draft_tokens,
        compare_row_count,
        stop_tokens,
        stop_token_count,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        leading_committed_output_count);
}

/**
 * @brief HIP graph-replayable greedy reducer with device-owned controls.
 */
__global__ void
rocm_summarize_greedy_speculative_verify_batch_device_controls_kernel(
    const int *__restrict__ verify_tokens,
    const int *__restrict__ draft_tokens,
    int compare_row_count,
    const int *__restrict__ active_verifier_row_count,
    const int *__restrict__ stop_tokens,
    int *__restrict__ out_tokens,
    int out_token_capacity,
    int *__restrict__ out_meta,
    const uint32_t *__restrict__ max_state_commit_rows,
    const int *__restrict__ next_leading_committed_output_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    const int active_rows = *active_verifier_row_count;
    if (active_rows <= 0 || active_rows > compare_row_count + 1)
    {
        for (int token = 0; token < out_token_capacity; ++token)
            out_tokens[token] = -1;
        for (int field = 0;
             field < llaminar2::sampling_math::kSpeculativeBatchMetaCount;
             ++field)
        {
            out_meta[field] = 0;
        }
        return;
    }

    /* Keep graph geometry physical while request semantics remain logical. */
    const int active_compare_row_count = active_rows - 1;
    const int commit_budget =
        max_state_commit_rows && *max_state_commit_rows <
                                     static_cast<uint32_t>(active_rows)
            ? static_cast<int>(*max_state_commit_rows)
            : active_rows;
    /*
     * The response ledger, not sampling history, owns whether row zero is a
     * carry from the preceding transaction.  Reading the controller scalar
     * directly keeps that lifecycle device-resident and graph-replayable.
     */
    const int leading_committed_output_count =
        *next_leading_committed_output_count;
    llaminar2::sampling_math::summarize_greedy_speculative_verify_batch_at_commit_boundary(
        draft_tokens[0],
        verify_tokens,
        draft_tokens,
        active_compare_row_count,
        stop_tokens,
        llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens,
        commit_budget,
        out_tokens,
        out_token_capacity,
        out_meta,
        leading_committed_output_count);
}


extern "C"
{
bool rocmOps_speculative_verify_distribution_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int draft_token,
    unsigned long long accept_seed,
    unsigned long long accept_offset,
    unsigned long long residual_seed,
    unsigned long long residual_offset,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K || !target_token_ids || !target_probs ||
        !draft_token_ids || !draft_probs || !out_token || !out_accepted || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_speculative_verify_distribution_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        accept_seed,
        accept_offset,
        residual_seed,
        residual_offset,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Distribution FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}


bool rocmOps_sample_processed_logits_f32(
    const float *logits,
    int vocab_size,
    int row_stride,
    float threshold,
    int *out_token,
    float *out_probability,
    int device_idx,
    void *stream)
{
    if (!logits || !out_token || !stream ||
        vocab_size <= 0 || row_stride < vocab_size)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_sample_processed_logits_f32_kernel,
        dim3(1),
        dim3(PROCESSED_LOGIT_VERIFY_THREADS),
        0,
        static_cast<hipStream_t>(stream),
        logits,
        vocab_size,
        row_stride,
        threshold,
        out_token,
        out_probability);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Processed-Logit Sample kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
    const float *logits,
    int vocab_size,
    int row_stride,
    float threshold,
    const int *verify_tokens,
    const int *verify_accepted,
    int row_count,
    int first_token,
    const int *first_token_device,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    int *out_token,
    float *out_probability,
    unsigned long long threshold_seed,
    const int *threshold_position,
    int threshold_position_offset,
    int device_idx,
    void *stream)
{
    if (!logits || !verify_tokens || !verify_accepted || !out_token ||
        !stream || vocab_size <= 0 || row_stride < vocab_size ||
        row_count < 0 ||
        (first_token < 0 && !first_token_device) ||
        stop_token_count < 0 ||
        stop_token_count >
            llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
        (threshold_position && threshold_seed == 0))
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_sample_processed_logits_if_speculative_batch_needs_bonus_f32_kernel,
        dim3(1),
        dim3(PROCESSED_LOGIT_VERIFY_THREADS),
        0,
        static_cast<hipStream_t>(stream),
        logits,
        vocab_size,
        row_stride,
        threshold,
        threshold_seed,
        threshold_position,
        threshold_position_offset,
        verify_tokens,
        verify_accepted,
        row_count,
        first_token,
        first_token_device,
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7,
        stop_token_count,
        out_token,
        out_probability);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Lazy Processed-Logit Bonus Sample kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_softmax_sample_temperature_logits_f32(
    const float *logits,
    int vocab_size,
    int row_stride,
    float temperature,
    float threshold,
    float *out_probabilities,
    int out_row_stride,
    int *out_token,
    float *out_probability,
    int device_idx,
    void *stream)
{
    if (!logits || !out_probabilities || !out_token || !stream ||
        vocab_size <= 0 || row_stride < vocab_size ||
        out_row_stride < vocab_size)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_softmax_sample_temperature_logits_f32_kernel,
        dim3(1),
        dim3(PROCESSED_LOGIT_VERIFY_THREADS),
        0,
        static_cast<hipStream_t>(stream),
        logits,
        vocab_size,
        row_stride,
        temperature,
        threshold,
        out_probabilities,
        out_row_stride,
        out_token,
        out_probability);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Temperature Draft Proposal kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_scale_sample_temperature_logits_f32(
    const float *logits,
    int vocab_size,
    int row_stride,
    float temperature,
    float threshold,
    float *out_logits,
    int out_row_stride,
    int *out_token,
    float *out_probability,
    int device_idx,
    void *stream)
{
    if (!logits || !out_logits || !out_token || !stream ||
        vocab_size <= 0 || row_stride < vocab_size ||
        out_row_stride < vocab_size)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_scale_sample_temperature_logits_f32_kernel,
        dim3(1),
        dim3(PROCESSED_LOGIT_VERIFY_THREADS),
        0,
        static_cast<hipStream_t>(stream),
        logits,
        vocab_size,
        row_stride,
        temperature,
        threshold,
        out_logits,
        out_row_stride,
        out_token,
        out_probability);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Temperature Draft Logit Proposal kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_softmax_processed_logits_f32(
    const float *logits,
    int row_count,
    int vocab_size,
    int row_stride,
    float *out_probabilities,
    int out_row_stride,
    int device_idx,
    void *stream)
{
    if (!logits || !out_probabilities || !stream ||
        row_count <= 0 || vocab_size <= 0 ||
        row_stride < vocab_size || out_row_stride < vocab_size)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_softmax_processed_logits_f32_kernel,
        dim3(row_count),
        dim3(PROCESSED_LOGIT_VERIFY_THREADS),
        0,
        static_cast<hipStream_t>(stream),
        logits,
        row_count,
        vocab_size,
        row_stride,
        out_probabilities,
        out_row_stride);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Processed-Logit Softmax kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_fill_inverse_exponential_samples_f32(
    float *out_samples,
    int row_count,
    int vocab_size,
    int row_stride,
    unsigned long long seed,
    int first_logical_position,
    int device_idx,
    void *stream)
{
    if (!out_samples || !stream ||
        row_count <= 0 ||
        vocab_size <= 0 || row_stride < vocab_size)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    const dim3 grid((vocab_size + threads - 1) / threads, row_count);
    hipLaunchKernelGGL(
        rocm_fill_inverse_exponential_samples_f32_kernel,
        grid,
        dim3(threads),
        0,
        static_cast<hipStream_t>(stream),
        out_samples,
        row_count,
        vocab_size,
        row_stride,
        seed,
        first_logical_position);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm inverse-exponential sample fill launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_speculative_verify_distribution_threshold_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int draft_token,
    float accept_threshold,
    float residual_threshold,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K || !target_token_ids || !target_probs ||
        !draft_token_ids || !draft_probs || !out_token || !out_accepted || !stream)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_speculative_verify_distribution_threshold_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        target_token_ids,
        target_probs,
        draft_token_ids,
        draft_probs,
        k,
        draft_token,
        accept_threshold,
        residual_threshold,
        out_token,
        out_accepted,
        out_accept_probability,
        out_accept_threshold);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Distribution Threshold FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_speculative_verify_distribution_thresholds_batch_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int distribution_stride,
    const int *draft_tokens_host,
    const float *accept_thresholds_host,
    const float *residual_thresholds_host,
    int row_count,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (k <= 0 || k > TOPK_MAX_K ||
        distribution_stride < k ||
        row_count <= 0 ||
        !target_token_ids || !target_probs ||
        !draft_token_ids || !draft_probs ||
        !draft_tokens_host || !accept_thresholds_host ||
        !residual_thresholds_host ||
        !out_token || !out_accepted || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    constexpr int kRowsPerLaunch =
        llaminar2::sampling_math::kSpeculativeBatchMaxRows;
    for (int row_base = 0; row_base < row_count;
         row_base += kRowsPerLaunch)
    {
        const int launch_rows =
            std::min(kRowsPerLaunch, row_count - row_base);
        llaminar2::sampling_math::SpeculativeBatchHostParameters parameters{};
        for (int row = 0; row < launch_rows; ++row)
        {
            parameters.draft_tokens[row] =
                draft_tokens_host[row_base + row];
            parameters.thresholds.accept[row] =
                accept_thresholds_host[row_base + row];
            parameters.thresholds.residual[row] =
                residual_thresholds_host[row_base + row];
        }

        const size_t row_offset =
            static_cast<size_t>(row_base) *
            static_cast<size_t>(distribution_stride);
        hipLaunchKernelGGL(
            rocm_speculative_verify_distribution_thresholds_batch_kernel,
            dim3(1),
            dim3(kRowsPerLaunch),
            0,
            static_cast<hipStream_t>(stream),
            target_token_ids + row_offset,
            target_probs + row_offset,
            draft_token_ids + row_offset,
            draft_probs + row_offset,
            k,
            distribution_stride,
            parameters,
            launch_rows,
            out_token + row_base,
            out_accepted + row_base,
            out_accept_probability
                ? out_accept_probability + row_base
                : nullptr,
            out_accept_threshold
                ? out_accept_threshold + row_base
                : nullptr);

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "ROCm Speculative Verify Batch FP32 kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
    const int *target_token_ids,
    const float *target_probs,
    const int *draft_token_ids,
    const float *draft_probs,
    int k,
    int distribution_stride,
    const int *sampled_draft_tokens,
    const float *sampled_draft_probabilities,
    const float *accept_thresholds_host,
    const float *residual_thresholds_host,
    int row_count,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int inverse_sample_vocab_size,
    unsigned long long threshold_seed,
    int threshold_first_logical_position,
    int thresholds_from_seed,
    const int *threshold_base_position,
    int threshold_position_offset,
    const int *generation_control,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream,
    const uint64_t *threshold_seed_device)
{
    const bool has_draft_distribution =
        draft_token_ids != nullptr && draft_probs != nullptr;
    const bool has_one_hot_draft_distribution =
        draft_token_ids == nullptr && draft_probs == nullptr;
    if (k <= 0 || k > TOPK_MAX_K ||
        distribution_stride < k ||
        row_count <= 0 ||
        !target_token_ids || !target_probs ||
        (!has_draft_distribution && !has_one_hot_draft_distribution) ||
        !sampled_draft_tokens ||
        (threshold_base_position && !thresholds_from_seed) ||
        !out_token || !out_accepted || !stream ||
        (threshold_seed_device &&
         (!thresholds_from_seed || !generation_control || threshold_seed != 0 ||
          inverse_sample_seed != 0 ||
          reinterpret_cast<uintptr_t>(threshold_seed_device) % alignof(uint64_t) != 0)) ||
        (generation_control && (!threshold_base_position || !thresholds_from_seed ||
         !has_one_hot_draft_distribution || inverse_sample_vocab_size <= 0)))
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int kRuntimeVerifierThreads = 128;
    constexpr int kExplicitThresholdRows =
        llaminar2::sampling_math::kSpeculativeBatchMaxRows;
    const bool use_seeded_thresholds = thresholds_from_seed != 0;
    if (!use_seeded_thresholds &&
        (!accept_thresholds_host || !residual_thresholds_host))
    {
        return false;
    }

    const int launch_count = use_seeded_thresholds
                                 ? 1
                                 : (row_count + kExplicitThresholdRows - 1) /
                                       kExplicitThresholdRows;
    for (int launch = 0; launch < launch_count; ++launch)
    {
        const int row_base = use_seeded_thresholds
                                 ? 0
                                 : launch * kExplicitThresholdRows;
        const int launch_rows = use_seeded_thresholds
                                    ? row_count
                                    : std::min(kExplicitThresholdRows,
                                               row_count - row_base);
        llaminar2::sampling_math::SpeculativeBatchThresholdParameters
            thresholds{};
        if (!use_seeded_thresholds)
        {
            for (int row = 0; row < launch_rows; ++row)
            {
                thresholds.accept[row] =
                    accept_thresholds_host[row_base + row];
                thresholds.residual[row] =
                    residual_thresholds_host[row_base + row];
            }
        }

        const int threads = use_seeded_thresholds
                                ? kRuntimeVerifierThreads
                                : kExplicitThresholdRows;
        const int blocks = (launch_rows + (generation_control ? 1 : 0) + threads - 1) / threads;
        const size_t row_offset =
            static_cast<size_t>(row_base) *
            static_cast<size_t>(distribution_stride);
        hipLaunchKernelGGL(
            rocm_speculative_verify_distribution_thresholds_batch_device_tokens_kernel,
            dim3(blocks),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            target_token_ids + row_offset,
            target_probs + row_offset,
            draft_token_ids ? draft_token_ids + row_offset : nullptr,
            draft_probs ? draft_probs + row_offset : nullptr,
            k,
            distribution_stride,
            sampled_draft_tokens + row_base,
            sampled_draft_probabilities
                ? sampled_draft_probabilities + row_base
                : nullptr,
            thresholds,
            launch_rows,
            inverse_sample_seed,
            inverse_sample_first_logical_position + row_base,
            inverse_sample_vocab_size,
            threshold_seed,
            threshold_first_logical_position + row_base,
            thresholds_from_seed,
            threshold_base_position,
            threshold_position_offset + row_base,
            generation_control,
            out_token + row_base,
            out_accepted + row_base,
            out_accept_probability
                ? out_accept_probability + row_base
                : nullptr,
            out_accept_threshold
                ? out_accept_threshold + row_base
                : nullptr, threshold_seed_device);

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "ROCm Speculative Verify Batch Device Tokens FP32 kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
    const float *target_logits,
    const float *draft_logits,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *sampled_draft_tokens,
    const float *accept_thresholds_host,
    const float *residual_thresholds_host,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    const float *draft_token_probabilities,
    int device_idx,
    void *stream)
{
    if (!target_logits || !draft_logits || !sampled_draft_tokens ||
        !out_token || !out_accepted || !stream ||
        row_count <= 0 ||
        vocab_size <= 0 ||
        target_row_stride < vocab_size ||
        draft_row_stride < vocab_size ||
        !accept_thresholds_host || !residual_thresholds_host)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int kRowsPerLaunch =
        llaminar2::sampling_math::kSpeculativeBatchMaxRows;
    for (int row_base = 0; row_base < row_count;
         row_base += kRowsPerLaunch)
    {
        const int launch_rows =
            std::min(kRowsPerLaunch, row_count - row_base);
        const auto thresholds = packROCmSpeculativeBatchThresholds(
            accept_thresholds_host + row_base,
            residual_thresholds_host + row_base,
            launch_rows);
        hipLaunchKernelGGL(
            rocm_speculative_verify_processed_logits_thresholds_batch_device_tokens_kernel,
            dim3(launch_rows),
            dim3(PROCESSED_LOGIT_VERIFY_THREADS),
            0,
            static_cast<hipStream_t>(stream),
            target_logits +
                static_cast<size_t>(row_base) * target_row_stride,
            draft_logits +
                static_cast<size_t>(row_base) * draft_row_stride,
            launch_rows,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            sampled_draft_tokens + row_base,
            thresholds,
            out_token + row_base,
            out_accepted + row_base,
            out_accept_probability
                ? out_accept_probability + row_base
                : nullptr,
            out_accept_threshold
                ? out_accept_threshold + row_base
                : nullptr,
            draft_token_probabilities
                ? draft_token_probabilities + row_base
                : nullptr);

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "ROCm Processed-Logit Stochastic Speculative Verify kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
    const float *target_logits,
    const float *draft_probabilities,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *sampled_draft_tokens,
    const float *accept_thresholds_host,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int thresholds_from_seed,
    const int *threshold_base_position,
    int threshold_position_offset,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int no_draft_probabilities,
    int device_idx,
    void *stream)
{
    if (!target_logits ||
        (!no_draft_probabilities && !draft_probabilities) ||
        !sampled_draft_tokens ||
        !out_token || !out_accepted || !stream ||
        row_count <= 0 ||
        vocab_size <= 0 ||
        target_row_stride < vocab_size ||
        (!no_draft_probabilities && draft_row_stride < vocab_size) ||
        (threshold_base_position && !thresholds_from_seed))
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int kRowsPerLaunch =
        llaminar2::sampling_math::kSpeculativeBatchMaxRows;
    const bool use_seeded_thresholds = thresholds_from_seed != 0;
    if (!use_seeded_thresholds && !accept_thresholds_host)
        return false;

    const int launch_count = use_seeded_thresholds
                                 ? 1
                                 : (row_count + kRowsPerLaunch - 1) /
                                       kRowsPerLaunch;
    for (int launch = 0; launch < launch_count; ++launch)
    {
        const int row_base = use_seeded_thresholds
                                 ? 0
                                 : launch * kRowsPerLaunch;
        const int launch_rows = use_seeded_thresholds
                                    ? row_count
                                    : std::min(kRowsPerLaunch,
                                               row_count - row_base);
        const auto thresholds = packROCmSpeculativeBatchThresholds(
            use_seeded_thresholds
                ? nullptr
                : accept_thresholds_host + row_base,
            /*residual_thresholds=*/nullptr,
            use_seeded_thresholds ? 0 : launch_rows);
        hipLaunchKernelGGL(
            rocm_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_kernel,
            dim3(launch_rows),
            dim3(PROCESSED_LOGIT_VERIFY_THREADS),
            0,
            static_cast<hipStream_t>(stream),
            target_logits +
                static_cast<size_t>(row_base) * target_row_stride,
            no_draft_probabilities
                ? nullptr
                : draft_probabilities +
                      static_cast<size_t>(row_base) * draft_row_stride,
            launch_rows,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            sampled_draft_tokens + row_base,
            thresholds,
            inverse_sample_seed,
            inverse_sample_first_logical_position + row_base,
            thresholds_from_seed,
            threshold_base_position,
            threshold_position_offset + row_base,
            out_token + row_base,
            out_accepted + row_base,
            out_accept_probability
                ? out_accept_probability + row_base
                : nullptr,
            out_accept_threshold
                ? out_accept_threshold + row_base
                : nullptr,
            no_draft_probabilities);

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "ROCm Processed-Target/Draft-Probability Stochastic Verify kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
    const float *target_logits,
    const float *draft_logits,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    const int *sampled_draft_tokens,
    const float *sampled_draft_probabilities,
    const float *accept_thresholds_host,
    unsigned long long inverse_sample_seed,
    int inverse_sample_first_logical_position,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (!target_logits || !draft_logits || !sampled_draft_tokens ||
        !out_token || !out_accepted || !stream ||
        row_count <= 0 ||
        vocab_size <= 0 ||
        target_row_stride < vocab_size ||
        draft_row_stride < vocab_size ||
        !accept_thresholds_host)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int kRowsPerLaunch =
        llaminar2::sampling_math::kSpeculativeBatchMaxRows;
    for (int row_base = 0; row_base < row_count;
         row_base += kRowsPerLaunch)
    {
        const int launch_rows =
            std::min(kRowsPerLaunch, row_count - row_base);
        const auto thresholds = packROCmSpeculativeBatchThresholds(
            accept_thresholds_host + row_base,
            /*residual_thresholds=*/nullptr,
            launch_rows);
        hipLaunchKernelGGL(
            rocm_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_kernel,
            dim3(launch_rows),
            dim3(PROCESSED_LOGIT_VERIFY_THREADS),
            0,
            static_cast<hipStream_t>(stream),
            target_logits +
                static_cast<size_t>(row_base) * target_row_stride,
            draft_logits +
                static_cast<size_t>(row_base) * draft_row_stride,
            launch_rows,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            sampled_draft_tokens + row_base,
            sampled_draft_probabilities
                ? sampled_draft_probabilities + row_base
                : nullptr,
            thresholds,
            inverse_sample_seed,
            inverse_sample_first_logical_position + row_base,
            out_token + row_base,
            out_accepted + row_base,
            out_accept_probability
                ? out_accept_probability + row_base
                : nullptr,
            out_accept_threshold
                ? out_accept_threshold + row_base
                : nullptr);

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "ROCm Processed-Target/Draft-Logit Stochastic Verify kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
    const float *target_probabilities,
    const float *draft_probabilities,
    const float *inverse_rejection_samples,
    int row_count,
    int vocab_size,
    int target_row_stride,
    int draft_row_stride,
    int inverse_sample_row_stride,
    const int *sampled_draft_tokens,
    const float *accept_thresholds_host,
    int no_draft_probabilities,
    int *out_token,
    int *out_accepted,
    float *out_accept_probability,
    float *out_accept_threshold,
    int device_idx,
    void *stream)
{
    if (!target_probabilities || !inverse_rejection_samples ||
        (!no_draft_probabilities && !draft_probabilities) ||
        !sampled_draft_tokens || !out_token || !out_accepted || !stream ||
        row_count <= 0 ||
        vocab_size <= 0 ||
        target_row_stride < vocab_size ||
        (!no_draft_probabilities && draft_row_stride < vocab_size) ||
        inverse_sample_row_stride < vocab_size ||
        !accept_thresholds_host)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int kRowsPerLaunch =
        llaminar2::sampling_math::kSpeculativeBatchMaxRows;
    for (int row_base = 0; row_base < row_count;
         row_base += kRowsPerLaunch)
    {
        const int launch_rows =
            std::min(kRowsPerLaunch, row_count - row_base);
        const auto thresholds = packROCmSpeculativeBatchThresholds(
            accept_thresholds_host + row_base,
            /*residual_thresholds=*/nullptr,
            launch_rows);
        hipLaunchKernelGGL(
            rocm_speculative_verify_probabilities_thresholds_batch_device_tokens_kernel,
            dim3(launch_rows),
            dim3(PROCESSED_LOGIT_VERIFY_THREADS),
            0,
            static_cast<hipStream_t>(stream),
            target_probabilities +
                static_cast<size_t>(row_base) * target_row_stride,
            no_draft_probabilities
                ? nullptr
                : draft_probabilities +
                      static_cast<size_t>(row_base) * draft_row_stride,
            inverse_rejection_samples +
                static_cast<size_t>(row_base) * inverse_sample_row_stride,
            launch_rows,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            inverse_sample_row_stride,
            sampled_draft_tokens + row_base,
            thresholds,
            no_draft_probabilities,
            out_token + row_base,
            out_accepted + row_base,
            out_accept_probability
                ? out_accept_probability + row_base
                : nullptr,
            out_accept_threshold
                ? out_accept_threshold + row_base
                : nullptr);

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "ROCm Full-Probability Stochastic Speculative Verify kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmOps_summarize_speculative_verify_batch(
    const int *verify_tokens,
    const int *verify_accepted,
    int row_count,
    int first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    const int *bonus_token,
    int has_bonus_token,
    int *out_tokens,
    int out_token_capacity,
    int *out_meta,
    const uint32_t *max_state_commit_rows,
    int leading_committed_output_count,
    int device_idx,
    void *stream)
{
    if (row_count < 0 ||
        out_token_capacity < row_count + 1 ||
        stop_token_count < 0 ||
        stop_token_count >
            llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
        !verify_tokens || !verify_accepted ||
        (has_bonus_token && !bonus_token) ||
        !out_tokens || !out_meta || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_summarize_speculative_verify_batch_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        verify_tokens,
        verify_accepted,
        row_count,
        first_token,
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7,
        stop_token_count,
        bonus_token,
        has_bonus_token,
        out_tokens,
        out_token_capacity,
        out_meta,
        max_state_commit_rows,
        leading_committed_output_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Batch Summary kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_summarize_speculative_verify_batch_device_first_token(
    const int *verify_tokens,
    const int *verify_accepted,
    int row_count,
    const int *first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    const int *bonus_token,
    int has_bonus_token,
    int *out_tokens,
    int out_token_capacity,
    int *out_meta,
    const uint32_t *max_state_commit_rows,
    int leading_committed_output_count,
    int device_idx,
    void *stream)
{
    if (row_count < 0 ||
        out_token_capacity < row_count + 1 ||
        stop_token_count < 0 ||
        stop_token_count >
            llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
        !verify_tokens || !verify_accepted || !first_token ||
        (has_bonus_token && !bonus_token) ||
        !out_tokens || !out_meta || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_summarize_speculative_verify_batch_device_first_token_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        verify_tokens,
        verify_accepted,
        row_count,
        first_token,
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7,
        stop_token_count,
        bonus_token,
        has_bonus_token,
        out_tokens,
        out_token_capacity,
        out_meta,
        max_state_commit_rows,
        leading_committed_output_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Speculative Verify Batch Device-First Summary kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_summarize_speculative_verify_batch_device_generation_controls(
    const int *verify_tokens,
    const int *verify_accepted,
    const int *greedy_draft_tokens,
    int row_count,
    const int *first_token,
    const int *stop_tokens,
    const int *bonus_token,
    int has_bonus_token,
    const int *generation_control,
    int *out_tokens,
    int out_token_capacity,
    int *out_meta,
    int device_idx,
    void *stream)
{
    const bool has_acceptance_rows = verify_accepted != nullptr;
    const bool has_greedy_rows = greedy_draft_tokens != nullptr;
    if (!verify_tokens || has_acceptance_rows == has_greedy_rows ||
        row_count < 0 || !first_token || !stop_tokens ||
        (has_bonus_token && !bonus_token) || !generation_control ||
        out_token_capacity < row_count + 1 || !out_tokens || !out_meta ||
        !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_summarize_speculative_verify_batch_device_generation_controls_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        verify_tokens,
        verify_accepted,
        greedy_draft_tokens,
        row_count,
        first_token,
        stop_tokens,
        bonus_token,
        has_bonus_token,
        generation_control,
        out_tokens,
        out_token_capacity,
        out_meta);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm device-generation speculative summary launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool
rocmOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
    const int *target_token_ids,
    const float *target_probs,
    int target_row_stride,
    int top_k,
    int row_count,
    unsigned long long threshold_seed,
    const int *threshold_position,
    int threshold_position_offset,
    const int *verifier_input_tokens,
    const int *stop_tokens,
    const int *generation_control,
    int *sampled_target_tokens,
    int *out_tokens,
    int out_token_capacity,
    int *out_meta,
    void *first_transaction_diagnostic,
    int device_idx,
    void *stream,
    const uint64_t *threshold_seed_device)
{
    constexpr int kMaxComparisonRows = 15;
    if (!target_token_ids || !target_probs ||
        target_row_stride <= 0 || top_k <= 0 ||
        top_k > target_row_stride || row_count <= 0 ||
        row_count > kMaxComparisonRows ||
        ((threshold_seed != 0) + (threshold_seed_device != nullptr) != 1) ||
        (threshold_seed_device &&
         reinterpret_cast<uintptr_t>(threshold_seed_device) % alignof(uint64_t) != 0) ||
        !threshold_position || !verifier_input_tokens || !stop_tokens ||
        !generation_control || !sampled_target_tokens || !out_tokens ||
        out_token_capacity < row_count + 1 || !out_meta || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    using DiagnosticRecord =
        llaminar2::sampling_math::MTPFirstTransactionDiagnosticRecord;
    auto *diagnostic =
        static_cast<DiagnosticRecord *>(first_transaction_diagnostic);
    if (diagnostic)
    {
        hipLaunchKernelGGL(
            (rocm_sample_and_summarize_serial_equivalent_speculative_batch_kernel<
                true>),
            dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream),
            target_token_ids,
            target_probs,
            target_row_stride,
            top_k,
            row_count,
            threshold_seed,
            threshold_position,
            threshold_position_offset,
            verifier_input_tokens,
            stop_tokens,
            generation_control,
            sampled_target_tokens,
            out_tokens,
            out_token_capacity,
            out_meta,
            diagnostic, threshold_seed_device);
    }
    else
    {
        hipLaunchKernelGGL(
            (rocm_sample_and_summarize_serial_equivalent_speculative_batch_kernel<
                false>),
            dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream),
            target_token_ids,
            target_probs,
            target_row_stride,
            top_k,
            row_count,
            threshold_seed,
            threshold_position,
            threshold_position_offset,
            verifier_input_tokens,
            stop_tokens,
            generation_control,
            sampled_target_tokens,
            out_tokens,
            out_token_capacity,
            out_meta,
            nullptr, threshold_seed_device);
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm fused serial-equivalent speculative outcome launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_retain_mtp_first_transaction_draft_boundary(
    const uint32_t *data_words,
    int word_count,
    int boundary,
    int draft_slot,
    const int *condition_token,
    const int *position_id,
    const int *generation_control,
    int generation_control_stride,
    void *diagnostic_record,
    int device_idx,
    void *stream)
{
    using namespace llaminar2::sampling_math;
    if (word_count < 0 || (word_count > 0 && !data_words) ||
        boundary < 0 ||
        boundary >= kMTPFirstTransactionDraftBoundaryCount ||
        draft_slot < 0 || draft_slot >= kSpeculativeBatchMaxRows ||
        !condition_token || !position_id || !generation_control ||
        generation_control_stride < kDeviceGenerationControlCount ||
        !diagnostic_record || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_retain_mtp_first_transaction_draft_boundary_kernel,
        dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream),
        data_words,
        word_count,
        boundary,
        draft_slot,
        condition_token,
        position_id,
        generation_control,
        generation_control_stride,
        static_cast<MTPFirstTransactionDiagnosticRecord *>(
            diagnostic_record));

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm first-transaction MTP draft-boundary diagnostic launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_summarize_greedy_speculative_verify_batch(
    const int *verify_tokens,
    const int *draft_tokens,
    int compare_row_count,
    int first_token,
    int stop_token0,
    int stop_token1,
    int stop_token2,
    int stop_token3,
    int stop_token4,
    int stop_token5,
    int stop_token6,
    int stop_token7,
    int stop_token_count,
    int *out_tokens,
    int out_token_capacity,
    int *out_meta,
    const uint32_t *max_state_commit_rows,
    int leading_committed_output_count,
    int device_idx,
    void *stream)
{
    if (compare_row_count < 0 ||
        out_token_capacity < compare_row_count + 1 ||
        stop_token_count < 0 ||
        stop_token_count >
            llaminar2::sampling_math::kSpeculativeBatchMaxStopTokens ||
        !verify_tokens || !draft_tokens ||
        !out_tokens || !out_meta || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    hipLaunchKernelGGL(
        rocm_summarize_greedy_speculative_verify_batch_kernel,
        dim3(1), dim3(1), 0, static_cast<hipStream_t>(stream),
        verify_tokens,
        draft_tokens,
        compare_row_count,
        first_token,
        stop_token0,
        stop_token1,
        stop_token2,
        stop_token3,
        stop_token4,
        stop_token5,
        stop_token6,
        stop_token7,
        stop_token_count,
        out_tokens,
        out_token_capacity,
        out_meta,
        max_state_commit_rows,
        leading_committed_output_count);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Greedy Speculative Verify Batch Summary kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmOps_summarize_greedy_speculative_verify_batch_device_controls(
    const int *verify_tokens,
    const int *draft_tokens,
    int compare_row_count,
    const int *active_verifier_row_count,
    const int *stop_tokens,
    int *out_tokens,
    int out_token_capacity,
    int *out_meta,
    const uint32_t *max_state_commit_rows,
    const int *next_leading_committed_output_count,
    int device_idx,
    void *stream)
{
    if (compare_row_count < 0 ||
        out_token_capacity < compare_row_count + 1 ||
        !verify_tokens || !draft_tokens || !active_verifier_row_count ||
        !stop_tokens || !next_leading_committed_output_count ||
        !out_tokens || !out_meta || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipLaunchKernelGGL(
        rocm_summarize_greedy_speculative_verify_batch_device_controls_kernel,
        dim3(1),
        dim3(1),
        0,
        static_cast<hipStream_t>(stream),
        verify_tokens,
        draft_tokens,
        compare_row_count,
        active_verifier_row_count,
        stop_tokens,
        out_tokens,
        out_token_capacity,
        out_meta,
        max_state_commit_rows,
        next_leading_committed_output_count);

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(
            stderr,
            "ROCm device-control greedy speculative summary launch failed: %s\n",
            hipGetErrorString(err));
        return false;
    }
    return true;
}


} // extern "C"
