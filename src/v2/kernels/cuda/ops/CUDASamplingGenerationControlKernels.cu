/**
 * @file CUDASamplingGenerationControlKernels.cu
 * @brief CUDA device generation controller, commit boundary, MTP penalty history, and verifier geometry kernels.
 */
#include "CUDASamplingCommon.cuh"

/**
 * @brief Advance one shared decode-round boundary from compact request rows.
 *
 * Request summaries are produced sequentially on the verifier stream and all
 * read the same pre-transaction budget. The largest committed-state count is
 * the number of lockstep serial decode rounds represented by the batch.
 */
__global__ void cuda_advance_speculative_commit_boundary_kernel(
    const int *__restrict__ meta,
    int request_count,
    int meta_stride,
    uint32_t *__restrict__ decode_rounds_committed,
    uint32_t *__restrict__ decode_rounds_until_maintenance,
    uint32_t *__restrict__ maintenance_due,
    uint32_t *__restrict__ decode_boundary_advanced)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (!meta || request_count <= 0 ||
        meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
        !decode_rounds_committed ||
        !decode_rounds_until_maintenance ||
        !maintenance_due || !decode_boundary_advanced ||
        *maintenance_due != 0u || *decode_boundary_advanced != 0u)
    {
        if (maintenance_due)
            *maintenance_due = 2u;
        return;
    }

    uint32_t committed_rounds = 0u;
    for (int request = 0; request < request_count; ++request)
    {
        const int *request_meta =
            meta + static_cast<size_t>(request) * meta_stride;
        if (request_meta[llaminar2::sampling_math::kSpecBatchMetaOk] == 0)
        {
            *maintenance_due = 2u;
            return;
        }
        const int request_commits =
            llaminar2::sampling_math::speculative_new_commit_count(
                request_meta[llaminar2::sampling_math::kSpecBatchMetaOutputCount],
                request_meta[llaminar2::sampling_math::kSpecBatchMetaLeadingCommittedOutputCount]);
        if (request_commits < 0)
        {
            *maintenance_due = 2u;
            return;
        }
        committed_rounds = max(
            committed_rounds,
            static_cast<uint32_t>(request_commits));
    }

    const uint32_t remaining = *decode_rounds_until_maintenance;
    if (committed_rounds > remaining)
    {
        *maintenance_due = 2u;
        return;
    }
    *decode_rounds_committed += committed_rounds;
    *decode_rounds_until_maintenance = remaining - committed_rounds;
    if (committed_rounds == remaining)
        *maintenance_due = 1u;
    *decode_boundary_advanced = 1u;
}

/**
 * @brief Publish one ordinary serial decode boundary into the shared clock.
 *
 * A pre-existing marker means grouped MTP already published this exact edge;
 * the kernel then validates the pair and returns without advancing again.
 */
__global__ void cuda_publish_serial_decode_commit_boundary_kernel(
    uint32_t *__restrict__ decode_rounds_committed,
    uint32_t *__restrict__ decode_rounds_until_maintenance,
    uint32_t *__restrict__ maintenance_due,
    uint32_t *__restrict__ decode_boundary_advanced)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (!decode_rounds_committed ||
        !decode_rounds_until_maintenance || !maintenance_due ||
        !decode_boundary_advanced)
    {
        if (maintenance_due)
            *maintenance_due = 2u;
        return;
    }

    const uint32_t due = *maintenance_due;
    const uint32_t advanced = *decode_boundary_advanced;
    const uint32_t remaining = *decode_rounds_until_maintenance;
    if (advanced > 1u || due > 1u)
    {
        *maintenance_due = 2u;
        return;
    }
    if (advanced == 1u)
    {
        const bool valid_existing_edge =
            (due == 0u && remaining > 0u) ||
            (due == 1u && remaining == 0u);
        if (!valid_existing_edge)
            *maintenance_due = 2u;
        return;
    }
    if (due != 0u || remaining == 0u ||
        *decode_rounds_committed == 0xffffffffu)
    {
        *maintenance_due = 2u;
        return;
    }

    ++(*decode_rounds_committed);
    *decode_rounds_until_maintenance = remaining - 1u;
    if (remaining == 1u)
        *maintenance_due = 1u;
    *decode_boundary_advanced = 1u;
}

/** Retire only a non-due published edge after the conditional child. */
__global__ void cuda_acknowledge_decode_commit_boundary_kernel(
    uint32_t *__restrict__ decode_rounds_until_maintenance,
    uint32_t *__restrict__ maintenance_due,
    uint32_t *__restrict__ decode_boundary_advanced)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (!decode_rounds_until_maintenance || !maintenance_due ||
        !decode_boundary_advanced)
    {
        if (maintenance_due)
            *maintenance_due = 2u;
        return;
    }
    const uint32_t due = *maintenance_due;
    const uint32_t advanced = *decode_boundary_advanced;
    const uint32_t remaining = *decode_rounds_until_maintenance;
    if (due == 0u && remaining > 0u && advanced <= 1u)
    {
        *decode_boundary_advanced = 0u;
        return;
    }
    if (due == 1u && remaining == 0u && advanced == 1u)
        return;
    *maintenance_due = 2u;
}

/** Bind one persistent MoE scheduler ticket to the active request lifecycle. */
__global__ void cuda_initialize_device_moe_rebalance_dispatch_ticket_kernel(
    uint64_t session_epoch,
    uint64_t workspace_generation,
    uint32_t participant_id,
    uint32_t participant_count,
    llaminar2::DeviceMoERebalanceDispatchTicket *__restrict__ ticket)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || !ticket)
        return;

    llaminar2::DeviceMoERebalanceDispatchTicket initialized{};
    initialized.magic =
        llaminar2::DeviceMoERebalanceDispatchTicket::kMagic;
    initialized.abi_version =
        llaminar2::DeviceMoERebalanceDispatchTicket::kABIVersion;
    initialized.session_epoch_low = static_cast<uint32_t>(session_epoch);
    initialized.session_epoch_high =
        static_cast<uint32_t>(session_epoch >> 32u);
    initialized.workspace_generation_low =
        static_cast<uint32_t>(workspace_generation);
    initialized.workspace_generation_high =
        static_cast<uint32_t>(workspace_generation >> 32u);
    initialized.participant_id = participant_id;
    initialized.participant_count = participant_count;
    *ticket = initialized;
}

/** Pack only the cadence decision consumed by HIP's rank scheduler. */
__global__ void cuda_publish_device_moe_rebalance_dispatch_ticket_kernel(
    const uint32_t *__restrict__ controller_magic,
    const uint32_t *__restrict__ controller_version,
    const uint32_t *__restrict__ controller_error,
    const uint32_t *__restrict__ decode_rounds_committed,
    const uint32_t *__restrict__ decode_rounds_until_maintenance,
    const uint32_t *__restrict__ maintenance_due,
    const uint32_t *__restrict__ decode_boundary_advanced,
    llaminar2::DeviceMoERebalanceDispatchTicket *__restrict__ ticket)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || !ticket)
        return;

    const bool pointers_valid = controller_magic && controller_version &&
                                controller_error && decode_rounds_committed &&
                                decode_rounds_until_maintenance &&
                                maintenance_due && decode_boundary_advanced;
    const uint32_t version =
        pointers_valid ? *controller_version : 0u;
    const uint32_t error = pointers_valid ? *controller_error : 0xffffffffu;
    const uint32_t committed =
        pointers_valid ? *decode_rounds_committed : 0u;
    const uint32_t remaining =
        pointers_valid ? *decode_rounds_until_maintenance : 0u;
    const uint32_t due = pointers_valid ? *maintenance_due : 2u;
    const uint32_t advanced =
        pointers_valid ? *decode_boundary_advanced : 2u;
    /* Keep the ticket ABI backend-symmetric even though CUDA normally consumes
     * this state through a native conditional graph.  A non-due boundary may
     * already have been acknowledged when a diagnostic ticket is published. */
    const bool boundary_valid =
        pointers_valid &&
        *controller_magic == 0x4d4f4552u &&
        version == llaminar2::moe_rebalance_abi::kVersion && error == 0u &&
        due <= 1u && advanced <= 1u &&
        ((due != 0u && advanced == 1u && remaining == 0u) ||
         (due == 0u && remaining > 0u));

    ticket->healthy = boundary_valid ? 1u : 0u;
    ticket->controller_version = version;
    ticket->decode_rounds_committed = committed;
    ticket->decode_rounds_until_maintenance = remaining;
    ticket->maintenance_due = due <= 1u ? due : 0u;
    ticket->decode_boundary_advanced = advanced <= 1u ? advanced : 0u;
    ticket->error_code = error;
    __threadfence_system();
}

/**
 * @brief Publish one request's immutable grouped-greedy controls on device.
 */
__global__ void cuda_configure_mtp_greedy_penalty_policy_kernel(
    llaminar2::MTPGreedyPenaltyPolicy *__restrict__ controls,
    float presence_penalty,
    float frequency_penalty,
    int first_token_already_in_history, float repetition_penalty)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    controls->presence_penalty = presence_penalty;
    controls->frequency_penalty = frequency_penalty;
    controls->repetition_penalty = repetition_penalty;
    controls->first_token_already_in_history =
        first_token_already_in_history != 0 ? 1 : 0;
    controls->enabled =
        presence_penalty != 0.0f || frequency_penalty != 0.0f || repetition_penalty != 1.0f ? 1 : 0;
}

/**
 * @brief Apply durable-history penalties when no verifier prefix is present.
 *
 * This is the latency-critical M=1 specialization. Adjacent threads own
 * adjacent vocabulary tokens and reuse one computed penalty across every row.
 * Keeping verifier-prefix control flow out of this kernel preserves the small
 * register footprint and launch latency of ordinary serial decode.
 */
__global__ void cuda_apply_mtp_durable_penalties_f32_rows_kernel(
    float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ active_rows_device)
{
    const int active_rows = active_rows_device ? *active_rows_device : rows;
    if (active_rows <= 0 || active_rows > rows)
        return;

    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    for (int token = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         token < cols;
         token += static_cast<int>(blockDim.x * gridDim.x))
    {
        const int history = generated_token_counts[token];
        const int count = llaminar2::sampling_math::TokenPenaltyHistory::generatedCount(history);
        if (count <= 0 && (request_policy.repetition_penalty == 1.0f ||
            !llaminar2::sampling_math::TokenPenaltyHistory::inPrompt(history)))
            continue;

        float penalty = 0.0f;
        if (count > 0 && request_policy.presence_penalty != 0.0f)
            penalty += request_policy.presence_penalty;
        if (request_policy.frequency_penalty != 0.0f)
        {
            penalty += request_policy.frequency_penalty *
                       static_cast<float>(count);
        }
        for (int row = 0; row < active_rows; ++row)
        {
            data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)] = llaminar2::sampling_math::penalizedLogit(
                data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)], request_policy.repetition_penalty, penalty);
        }
    }
}

/**
 * @brief Apply device-owned serial-decode penalties to verifier logit rows.
 *
 * Threads traverse adjacent vocabulary tokens, yielding coalesced histogram
 * reads and in-place logit writes. A token that occurs anywhere in the short
 * verifier branch is deliberately left for sparse row owners in block zero.
 * All other tokens have the same durable-history count in every row, so one
 * thread can apply their penalties to all rows after scanning the branch
 * exactly once.
 *
 * This ownership split reduces the verifier work from O(rows^2 * vocabulary)
 * token comparisons to O(rows * vocabulary), while retaining one writer for
 * every output element. The sparse owners and dense owners touch disjoint
 * token addresses, so they can share one kernel launch without a grid-wide
 * barrier. The operation needs no atomics or temporary storage.
 */
__global__ void cuda_apply_mtp_verifier_penalties_f32_rows_kernel(
    float *__restrict__ data,
    int rows,
    int cols,
    int row_stride,
    const int *__restrict__ verifier_input_tokens,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ active_rows_device)
{
    const int active_rows = active_rows_device ? *active_rows_device : rows;
    if (active_rows <= 0 || active_rows > rows)
        return;

    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    const int prefix_begin =
        request_policy.first_token_already_in_history != 0 ? 1 : 0;

    // A verifier launch reserves block zero for sparse branch-token ownership.
    // It executes concurrently with all dense blocks, and then exits because
    // dense blocks deliberately skip every address this block can modify.
    if (blockIdx.x == 0)
    {
        for (int row = static_cast<int>(threadIdx.x);
             row < active_rows;
             row += static_cast<int>(blockDim.x))
        {
            for (int candidate_index = prefix_begin;
                 candidate_index < active_rows;
                 ++candidate_index)
            {
                const int token = verifier_input_tokens[candidate_index];
                if (token < 0 || token >= cols)
                    continue;

                bool already_processed = false;
                for (int earlier = prefix_begin;
                     earlier < candidate_index;
                     ++earlier)
                {
                    already_processed |=
                        verifier_input_tokens[earlier] == token;
                }
                if (already_processed)
                    continue;

                const int history = generated_token_counts[token];
        int count = llaminar2::sampling_math::TokenPenaltyHistory::generatedCount(history);
                for (int visible = prefix_begin; visible <= row; ++visible)
                    count += verifier_input_tokens[visible] == token ? 1 : 0;
                if (count <= 0 && (request_policy.repetition_penalty == 1.0f ||
            !llaminar2::sampling_math::TokenPenaltyHistory::inPrompt(history)))
                    continue;

                float penalty = 0.0f;
                if (count > 0 && request_policy.presence_penalty != 0.0f)
                    penalty += request_policy.presence_penalty;
                if (request_policy.frequency_penalty != 0.0f)
                {
                    penalty += request_policy.frequency_penalty *
                               static_cast<float>(count);
                }
                data[static_cast<size_t>(row) *
                         static_cast<size_t>(row_stride) +
                     static_cast<size_t>(token)] = llaminar2::sampling_math::penalizedLogit(
                data[static_cast<size_t>(row) *
                         static_cast<size_t>(row_stride) +
                     static_cast<size_t>(token)], request_policy.repetition_penalty, penalty);
            }
        }
        return;
    }

    const int dense_block = static_cast<int>(blockIdx.x) - 1;
    const int dense_blocks = static_cast<int>(gridDim.x) - 1;
    for (int token = dense_block * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
         token < cols;
         token += static_cast<int>(blockDim.x) * dense_blocks)
    {
        bool verifier_branch_token = false;
        for (int history_index = prefix_begin;
             history_index < active_rows;
             ++history_index)
        {
            verifier_branch_token |=
                verifier_input_tokens[history_index] == token;
        }
        if (verifier_branch_token)
            continue;

        const int history = generated_token_counts[token];
        const int count = llaminar2::sampling_math::TokenPenaltyHistory::generatedCount(history);
        if (count <= 0 && (request_policy.repetition_penalty == 1.0f ||
            !llaminar2::sampling_math::TokenPenaltyHistory::inPrompt(history)))
            continue;

        float penalty = 0.0f;
        if (count > 0 && request_policy.presence_penalty != 0.0f)
            penalty += request_policy.presence_penalty;
        if (request_policy.frequency_penalty != 0.0f)
        {
            penalty += request_policy.frequency_penalty *
                       static_cast<float>(count);
        }
        for (int row = 0; row < active_rows; ++row)
        {
            data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)] = llaminar2::sampling_math::penalizedLogit(
                data[static_cast<size_t>(row) * static_cast<size_t>(row_stride) +
                 static_cast<size_t>(token)], request_policy.repetition_penalty, penalty);
        }
    }

}

/**
 * @brief Return one token from the scalar MTP proposal branch.
 *
 * The first condition token lives in the full-sidecar capture slot while all
 * later branch tokens live in contiguous draft sample slots.  Treating those
 * two persistent owners as one logical row avoids staging a temporary verifier
 * prefix merely to score the next proposal.
 */
__device__ __forceinline__ int cuda_mtp_branch_token_at(
    int branch_index,
    int include_first_condition,
    const int *__restrict__ first_condition_token,
    const int *__restrict__ prior_draft_tokens)
{
    if (include_first_condition != 0 && branch_index == 0)
        return first_condition_token[0];
    return prior_draft_tokens[
        branch_index - (include_first_condition != 0 ? 1 : 0)];
}

/**
 * @brief Apply durable plus speculative-branch penalties to one proposal row.
 *
 * Block zero owns every vocabulary address mentioned by the short branch and
 * deduplicates repeated token ids.  All remaining blocks own the dense durable
 * histogram addresses and skip branch tokens.  The two ownership sets are
 * disjoint, so they may execute concurrently without atomics or a grid barrier.
 */
__global__ void cuda_apply_mtp_branch_penalties_f32_row_kernel(
    float *__restrict__ data,
    int cols,
    const int *__restrict__ first_condition_token,
    const int *__restrict__ prior_draft_tokens,
    int prior_draft_count,
    const int *__restrict__ generated_token_counts,
    const llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy)
{
    const llaminar2::MTPGreedyPenaltyPolicy request_policy = *policy;
    if (request_policy.enabled == 0)
        return;

    const int include_first_condition =
        request_policy.first_token_already_in_history == 0 ? 1 : 0;
    const int branch_count = include_first_condition + prior_draft_count;

    if (blockIdx.x == 0)
    {
        for (int branch_index = static_cast<int>(threadIdx.x);
             branch_index < branch_count;
             branch_index += static_cast<int>(blockDim.x))
        {
            const int token = cuda_mtp_branch_token_at(
                branch_index,
                include_first_condition,
                first_condition_token,
                prior_draft_tokens);
            if (token < 0 || token >= cols)
                continue;

            bool already_processed = false;
            for (int earlier = 0; earlier < branch_index; ++earlier)
            {
                already_processed |=
                    cuda_mtp_branch_token_at(
                        earlier,
                        include_first_condition,
                        first_condition_token,
                        prior_draft_tokens) == token;
            }
            if (already_processed)
                continue;

            const int history = generated_token_counts[token];
        int count = llaminar2::sampling_math::TokenPenaltyHistory::generatedCount(history);
            for (int visible = 0; visible < branch_count; ++visible)
            {
                count += cuda_mtp_branch_token_at(
                             visible,
                             include_first_condition,
                             first_condition_token,
                             prior_draft_tokens) == token
                             ? 1
                             : 0;
            }
            if (count <= 0 && (request_policy.repetition_penalty == 1.0f ||
            !llaminar2::sampling_math::TokenPenaltyHistory::inPrompt(history)))
                continue;

            float penalty = 0.0f;
            if (count > 0 && request_policy.presence_penalty != 0.0f)
                penalty += request_policy.presence_penalty;
            if (request_policy.frequency_penalty != 0.0f)
            {
                penalty += request_policy.frequency_penalty *
                           static_cast<float>(count);
            }
            data[token] = llaminar2::sampling_math::penalizedLogit(
                data[token], request_policy.repetition_penalty, penalty);
        }
        return;
    }

    const int dense_block = static_cast<int>(blockIdx.x) - 1;
    const int dense_blocks = static_cast<int>(gridDim.x) - 1;
    for (int token = dense_block * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
         token < cols;
         token += dense_blocks * static_cast<int>(blockDim.x))
    {
        bool branch_owned = false;
        for (int branch_index = 0;
             branch_index < branch_count;
             ++branch_index)
        {
            branch_owned |=
                cuda_mtp_branch_token_at(
                    branch_index,
                    include_first_condition,
                    first_condition_token,
                    prior_draft_tokens) == token;
        }
        if (branch_owned)
            continue;

        const int history = generated_token_counts[token];
        const int count = llaminar2::sampling_math::TokenPenaltyHistory::generatedCount(history);
        if (count <= 0 && (request_policy.repetition_penalty == 1.0f ||
            !llaminar2::sampling_math::TokenPenaltyHistory::inPrompt(history)))
            continue;

        float penalty = 0.0f;
        if (count > 0 && request_policy.presence_penalty != 0.0f)
            penalty += request_policy.presence_penalty;
        if (request_policy.frequency_penalty != 0.0f)
        {
            penalty += request_policy.frequency_penalty *
                       static_cast<float>(count);
        }
        data[token] = llaminar2::sampling_math::penalizedLogit(
                data[token], request_policy.repetition_penalty, penalty);
    }
}

/**
 * @brief Advance the generated-token histogram from the compact outcome.
 *
 * A single thread owns the short row, so duplicate tokens are updated in
 * deterministic program order without atomics.  Pending-condition row zero was
 * emitted by the previous transaction and is deliberately skipped.
 */
__global__ void cuda_commit_mtp_greedy_penalty_history_kernel(
    const int *__restrict__ output_tokens,
    const int *__restrict__ output_meta,
    llaminar2::MTPGreedyPenaltyPolicy *__restrict__ policy,
    const int *__restrict__ accepted_state_counts,
    const int *__restrict__ stopped_flags,
    int output_token_capacity,
    int vocab_size,
    int *__restrict__ generated_token_counts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    if (output_meta[llaminar2::sampling_math::kSpecBatchMetaOk] == 0)
        return;

    if (policy->enabled == 0)
    {
        policy->first_token_already_in_history = 0;
        return;
    }

    int output_count =
        output_meta[llaminar2::sampling_math::kSpecBatchMetaOutputCount];
    if (output_count < 0)
        output_count = 0;
    if (output_count > output_token_capacity)
        output_count = output_token_capacity;
    const int accepted_state_count = accepted_state_counts[0];
    if (accepted_state_count < 0 || accepted_state_count > output_count)
        return;
    const int begin =
        policy->first_token_already_in_history != 0 ? 1 : 0;
    for (int i = begin; i < output_count; ++i)
    {
        const int token = output_tokens[i];
        if (token >= 0 && token < vocab_size)
            ++generated_token_counts[token];
    }

    policy->first_token_already_in_history =
        stopped_flags[0] == 0 && output_count > accepted_state_count ? 1 : 0;
}

/** Commit one forced policy token with the same serial history order as sampling. */
__global__ void cuda_commit_generation_token_history_kernel(
    const int *__restrict__ token,
    int vocab_size,
    int *__restrict__ generated_token_counts,
    llaminar2::MTPGreedyPenaltyPolicy *__restrict__ direct_output_policy)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    const int value = token[0];
    if (value >= 0 && value < vocab_size)
    {
        ++generated_token_counts[value];
        // Only the direct MTP API supplies this original policy owner. Its
        // nonterminal token is consumed by the committed main row before the
        // next fresh target sampler; a grouped carried condition never enters
        // this transition. Preserve every admitted policy magnitude unchanged.
        if (direct_output_policy)
            direct_output_policy->first_token_already_in_history = 0;
    }
}

/**
 * @brief Publish one ordinary sample per independent resident request.
 *
 * One lane owns each response/continuation transition; the block spans requests,
 * not vocabulary elements. CUDA and HIP use the identical shared transition,
 * including failure atomicity and sampler-free forward-only publication.
 */
__global__ void cuda_publish_ordinary_generation_sample_kernel(
    llaminar2::sampling_math::OrdinaryGenerationPublication publication)
{
    using namespace llaminar2::sampling_math;
    const size_t request = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (request >= static_cast<size_t>(publication.request_count))
        return;
    (void)publish_ordinary_generation_request(publication, request);
}

/**
 * @brief Initialize one persistent device-generation controller per request.
 *
 * The response payload is intentionally left untouched.  Its authoritative
 * length starts at zero, so clearing a potentially long context-sized token row
 * would add request-admission bandwidth without changing any observable state.
 */
__global__ void cuda_initialize_device_generation_kernel(
    int request_count,
    int max_new_tokens,
    llaminar2::sampling_math::DeviceGenerationPolicy depth_policy,
    llaminar2::sampling_math::DeviceGenerationLeadingRowDisposition
        initial_leading_row_disposition,
    int response_token_stride,
    int control_stride,
    int *__restrict__ control,
    llaminar2::sampling_math::DeviceGenerationInitialization initialization)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    if (initialization.prior_tickets)
        initialization.prior_tickets += request_index;
    llaminar2::sampling_math::initialize_device_generation_control(
        max_new_tokens,
        response_token_stride,
        depth_policy,
        request_control,
        initial_leading_row_disposition, initialization);
}

/**
 * @brief Initialize one isolated HIP/CUDA host-scheduler ticket per request.
 *
 * Ticket identity is written directly from immutable admission scalars.  The
 * host never uploads this record and therefore cannot become an alternate
 * generation-state producer.
 */
__global__ void cuda_initialize_device_generation_dispatch_ticket_kernel(
    uint64_t session_epoch,
    uint64_t workspace_generation,
    int *__restrict__ control,
    int control_stride,
    int request_count,
    llaminar2::sampling_math::DeviceGenerationDispatchTicket
        *__restrict__ tickets)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    auto *const ticket = tickets + request_index;
    if (llaminar2::sampling_math::
            initialize_device_generation_dispatch_ticket(
                session_epoch,
                workspace_generation,
                ticket))
    {
        llaminar2::sampling_math::publish_device_generation_dispatch_ticket(
            request_control,
            /*maintenance_due=*/nullptr,
            ticket);
    }
}

/**
 * @brief Pack the narrow host-scheduling decision from device authority.
 */
__global__ void cuda_publish_device_generation_dispatch_ticket_kernel(
    int *__restrict__ control,
    int control_stride,
    int request_count,
    const uint32_t *__restrict__ maintenance_due,
    llaminar2::sampling_math::DeviceGenerationDispatchTicket
        *__restrict__ tickets)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    llaminar2::sampling_math::publish_device_generation_dispatch_ticket(
        request_control,
        maintenance_due,
        tickets + request_index);
}

/**
 * @brief Acknowledge an ordinary prior boundary and publish the next budget.
 *
 * A speculative outcome leaves `decode_boundary_advanced` set so the due
 * maintenance transaction can consume that exact edge without advancing the
 * clock twice. When the outcome did not make maintenance due, the next verifier
 * admission is the structural consumer of that acknowledgement. A due,
 * poisoned, or malformed boundary is rejected before any new transaction can
 * observe a budget.
 */
__global__ void cuda_prepare_device_generation_transaction_budget_kernel(
    int *__restrict__ control,
    int control_stride,
    int request_count,
    int verifier_row_capacity,
    const uint32_t *__restrict__ maintenance_rows_remaining,
    const uint32_t *__restrict__ maintenance_due,
    uint32_t *__restrict__ decode_boundary_advanced)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    const bool maintenance_boundary_bound =
        maintenance_rows_remaining || maintenance_due ||
        decode_boundary_advanced;
    const bool maintenance_boundary_complete =
        maintenance_rows_remaining && maintenance_due &&
        decode_boundary_advanced;
    const uint32_t due =
        maintenance_boundary_complete ? *maintenance_due : 0u;
    const uint32_t boundary_advanced =
        maintenance_boundary_complete ? *decode_boundary_advanced : 0u;
    const bool maintenance_boundary_valid =
        !maintenance_boundary_bound ||
        (maintenance_boundary_complete && due == 0u &&
         boundary_advanced <= 1u);
    const int maintenance_budget =
        maintenance_boundary_valid && maintenance_rows_remaining
            ? static_cast<int>(*maintenance_rows_remaining)
            : (maintenance_boundary_valid ? verifier_row_capacity : 0);
    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    llaminar2::sampling_math::prepare_device_generation_transaction_budget(
        verifier_row_capacity,
        maintenance_budget,
        request_control);

    /*
     * Exactly one lane retires the shared acknowledgement. Other lanes may
     * observe either one or zero, both of which are valid ordinary states; no
     * request depends on the value after validating it.
     */
    if (request_index == 0 && maintenance_boundary_valid &&
        maintenance_boundary_complete && boundary_advanced == 1u)
    {
        *decode_boundary_advanced = 0u;
    }
}

/**
 * @brief Commit one compact outcome and derive its accepted publication row.
 *
 * One thread owns one request transaction.  Combining response-ledger commit
 * with publication derivation removes a launch from every MTP transaction and
 * makes the serial-visible response/state boundary indivisible.
 */
__global__ void
cuda_commit_device_generation_and_derive_speculative_publication_metadata_kernel(
    const int32_t *__restrict__ compact_tokens,
    int output_token_stride,
    int *__restrict__ compact_meta,
    int meta_stride,
    const int *__restrict__ base_cached_tokens,
    int request_count,
    int padded_state_rows_per_request,
    int32_t *__restrict__ response_tokens,
    int response_token_stride,
    int *__restrict__ control,
    int control_stride,
    int *__restrict__ out_restore_rows,
    int *__restrict__ out_target_cached_tokens,
    int *__restrict__ out_accepted_state_counts,
    int *__restrict__ out_ok,
    int32_t *__restrict__ out_next_condition_tokens,
    int *__restrict__ out_all_drafts_accepted_flags,
    int *__restrict__ out_stopped_flags,
    int32_t *__restrict__ out_next_sidecar_condition_tokens,
    int32_t *__restrict__ out_next_sidecar_position_ids,
    int32_t *__restrict__ out_next_verifier_condition_tokens,
    const int32_t *__restrict__ verifier_input_tokens,
    int verifier_input_token_stride,
    llaminar2::sampling_math::MTPCommittedVerifierIdentityRecord *
        __restrict__ out_committed_verifier_identity,
    llaminar2::sampling_math::MTPDepthTransactionTrace *depth_transaction_trace)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    int32_t *request_response_tokens =
        response_tokens + static_cast<size_t>(request_index) *
                              static_cast<size_t>(response_token_stride);
    int *request_control =
        control + static_cast<size_t>(request_index) *
                      static_cast<size_t>(control_stride);
    const int base_cached_tokens_for_request =
        base_cached_tokens ? base_cached_tokens[request_index] : -1;
    const bool has_verifier_identity_binding =
        verifier_input_tokens != nullptr ||
        verifier_input_token_stride != 0 ||
        out_committed_verifier_identity != nullptr;
    const bool has_complete_verifier_identity_binding =
        verifier_input_tokens != nullptr &&
        verifier_input_token_stride > 0 &&
        out_committed_verifier_identity != nullptr;
    const bool controller_is_live =
        request_control[
            llaminar2::sampling_math::kDeviceGenerationControlOk] != 0 &&
        request_control[
            llaminar2::sampling_math::
                kDeviceGenerationControlRequestComplete] == 0;
    const int32_t *request_verifier_input_tokens =
        has_complete_verifier_identity_binding
            ? verifier_input_tokens +
                  static_cast<size_t>(request_index) *
                      static_cast<size_t>(verifier_input_token_stride)
            : nullptr;
    if ((has_verifier_identity_binding &&
         !has_complete_verifier_identity_binding) ||
        (has_complete_verifier_identity_binding && controller_is_live &&
         !llaminar2::sampling_math::
              valid_committed_verifier_identity_input(
                  request_verifier_input_tokens,
                  verifier_input_token_stride,
                  request_control)))
    {
        if (out_ok)
            out_ok[request_index] = 0;
        llaminar2::sampling_math::fail_device_generation_control(
            request_control,
            llaminar2::sampling_math::DeviceGenerationError::
                InvalidVerifierTransactionIdentity);
        return;
    }

    const bool committed = llaminar2::sampling_math::
        commit_device_generation_and_derive_speculative_publication_metadata(
        compact_tokens,
        output_token_stride,
        compact_meta,
        meta_stride,
        request_index,
        padded_state_rows_per_request,
        base_cached_tokens_for_request,
        request_response_tokens,
        response_token_stride,
        request_control,
        out_restore_rows ? out_restore_rows + request_index : nullptr,
        out_target_cached_tokens
            ? out_target_cached_tokens + request_index
            : nullptr,
        out_accepted_state_counts
            ? out_accepted_state_counts + request_index
            : nullptr,
        out_ok ? out_ok + request_index : nullptr,
        out_next_condition_tokens
            ? out_next_condition_tokens + request_index
            : nullptr,
        out_all_drafts_accepted_flags
            ? out_all_drafts_accepted_flags + request_index
            : nullptr,
        out_stopped_flags ? out_stopped_flags + request_index : nullptr,
        depth_transaction_trace);

    if (committed && out_ok && out_ok[request_index] != 0 &&
        has_complete_verifier_identity_binding &&
        !llaminar2::sampling_math::publish_committed_verifier_identity(
            request_verifier_input_tokens,
            verifier_input_token_stride,
            request_control,
            out_committed_verifier_identity + request_index))
    {
        out_ok[request_index] = 0;
        llaminar2::sampling_math::fail_device_generation_control(
            request_control,
            llaminar2::sampling_math::DeviceGenerationError::
                InvalidVerifierTransactionIdentity);
        return;
    }

    /*
     * These mirrors are the direct inputs of the next loop iteration's full
     * sidecar. Keeping them in the response/state commit kernel guarantees that
     * the controller, logical mailbox, and sidecar mailbox cross one atomic
     * device publication boundary.
     */
    if (out_next_sidecar_condition_tokens &&
        out_next_sidecar_position_ids)
    {
        out_next_sidecar_condition_tokens[request_index] =
            out_next_condition_tokens[request_index];
        out_next_sidecar_position_ids[request_index] =
            out_target_cached_tokens[request_index];
    }
    if (out_next_verifier_condition_tokens)
    {
        out_next_verifier_condition_tokens[request_index] =
            out_next_condition_tokens[request_index];
    }
}

/**
 * @brief Derive publication rows/counts from compact speculative metadata.
 *
 * Each request is independent, so one CUDA thread maps one compact metadata row
 * to the state row and cache count that later publication stages consume. The
 * helper is deliberately shared with CPU tests to keep the tricky
 * "accepted draft prefix" versus "committed verifier state rows" distinction in
 * one place.
 */
__global__ void cuda_derive_speculative_publication_metadata_kernel(
    const int *__restrict__ meta,
    int meta_stride,
    const int *__restrict__ base_cached_tokens,
    int request_count,
    int padded_state_rows_per_request,
    int max_state_commit_rows,
    int *__restrict__ out_restore_rows,
    int *__restrict__ out_target_cached_tokens,
    int *__restrict__ out_accepted_state_counts,
    int *__restrict__ out_ok,
    int32_t *__restrict__ out_next_condition_tokens,
    const int32_t *__restrict__ output_tokens,
    int output_token_stride,
    int *__restrict__ out_all_drafts_accepted_flags,
    int *__restrict__ out_stopped_flags,
    int32_t *__restrict__ out_next_verifier_condition_tokens)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    const int base_cached_tokens_for_request =
        base_cached_tokens ? base_cached_tokens[request_index] : -1;
    llaminar2::sampling_math::derive_speculative_publication_metadata(
        meta,
        meta_stride,
        request_index,
        padded_state_rows_per_request,
        base_cached_tokens_for_request,
        max_state_commit_rows,
        out_restore_rows ? out_restore_rows + request_index : nullptr,
        out_target_cached_tokens ? out_target_cached_tokens + request_index : nullptr,
        out_accepted_state_counts ? out_accepted_state_counts + request_index : nullptr,
        out_ok ? out_ok + request_index : nullptr,
        output_tokens,
        output_token_stride,
        out_next_condition_tokens ? out_next_condition_tokens + request_index : nullptr,
        out_all_drafts_accepted_flags ? out_all_drafts_accepted_flags + request_index : nullptr,
        out_stopped_flags ? out_stopped_flags + request_index : nullptr);
    if (out_next_verifier_condition_tokens)
    {
        out_next_verifier_condition_tokens[request_index] =
            out_next_condition_tokens[request_index];
    }
}

/**
 * @brief Derive shifted MTP KV counts from canonical primary publication.
 *
 * One CUDA thread handles one request. The shared SamplingMath helper owns the
 * depth-dependent off-by-one rule.  The primary rows already carry the exact
 * device-owned transaction boundary, so this kernel cannot diverge by
 * re-reading compact metadata with a host scalar.
 */
__global__ void
cuda_derive_shifted_speculative_publication_metadata_from_primary_kernel(
    const int *__restrict__ base_cached_tokens,
    const int *__restrict__ main_target_cached_tokens,
    const int *__restrict__ main_publication_ok,
    int request_count,
    int mtp_depth,
    int *__restrict__ out_target_cached_tokens,
    int *__restrict__ out_accepted_state_counts,
    int *__restrict__ out_ok, llaminar2::MTPShiftedCacheOrigin origin)
{
    const int request_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (request_index >= request_count)
        return;

    llaminar2::sampling_math::
        derive_shifted_speculative_publication_metadata_from_primary(
        base_cached_tokens[request_index],
        main_target_cached_tokens[request_index],
        main_publication_ok[request_index],
        mtp_depth,
        out_target_cached_tokens ? out_target_cached_tokens + request_index : nullptr,
        out_accepted_state_counts ? out_accepted_state_counts + request_index : nullptr,
        out_ok ? out_ok + request_index : nullptr,origin);
}

/**
 * @brief Prepare shifted-MTP suffix tokens and absolute positions from compact metadata.
 *
 * A single CUDA thread is enough for the four-row MTP verifier bound.  Keeping
 * this as a kernel still matters because the accepted-state count stays
 * device-resident; the CPU never decides how many suffix rows are publishable.
 * The same lane expands the request's canonical verifier-base position into the
 * contiguous position row consumed by the captured KV-only sidecar. Fusing
 * these stores avoids a second launch and keeps both dynamic graph inputs under
 * one stream-ordered device owner.
 */
__global__ void cuda_prepare_speculative_shifted_kv_tokens_kernel(
    const int *__restrict__ meta,
    int meta_stride,
    const int32_t *__restrict__ output_tokens,
    int output_token_stride,
    int request_index,
    int first_output_token_index,
    int row_count,
    int32_t filler_token,
    int32_t *__restrict__ out_tokens,
    const int32_t *__restrict__ base_positions,
    int position_offset,
    int32_t *__restrict__ out_position_ids)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    llaminar2::sampling_math::prepare_speculative_shifted_kv_tokens(
        meta,
        meta_stride,
        output_tokens,
        output_token_stride,
        request_index,
        first_output_token_index,
        row_count,
        filler_token,
        out_tokens);

    const int32_t first_position =
        base_positions[request_index] + position_offset;
    for (int row = 0; row < row_count; ++row)
        out_position_ids[row] = first_position + row;
}

/**
 * @brief Gather strided MTP draft slots and derive grouped sidecar positions.
 *
 * Request-batched drafting lays proposal tokens out request-major so the final
 * verifier can consume each request row contiguously. At an intermediate depth,
 * adjacent requests are therefore separated by the configured draft depth. One
 * thread per request gathers that token and adds the current depth to the
 * device-resident logical position. The output arrays are stable arena buffers
 * captured by the next MTP sidecar graph.
 */
__global__ void cuda_prepare_mtp_batched_sidecar_inputs_kernel(
    const int32_t *__restrict__ condition_tokens,
    int condition_token_stride,
    const int32_t *__restrict__ base_positions,
    int position_offset,
    int request_count,
    int32_t *__restrict__ out_condition_tokens,
    int32_t *__restrict__ out_position_ids)
{
    const int request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request >= request_count)
        return;

    out_condition_tokens[request] =
        condition_tokens[request * condition_token_stride];
    out_position_ids[request] = base_positions[request] + position_offset;
}

/**
 * @brief Publish all dynamic grouped-verifier geometry from device-owned rows.
 *
 * Each request owns one device-resident next-position scalar in its KV cache.
 * Grouped verification needs one absolute position per physical graph row. A
 * flattened thread index covers the request-major matrix and derives both
 * coordinates without any host-visible cursor:
 *
 * `position[request, token] = live_kv_count[request] + token`.
 *
 * Padded columns are populated as well. The first @p request_count threads also
 * publish each request's valid width. Rectangular graphs need no row-index
 * input; ragged graphs count their already-published physical verifier rows.
 * Thus position and recurrent-state masks become visible in one ordered launch,
 * with no host length mirror or second metadata transfer.
 */
__global__ void cuda_prepare_mtp_verifier_geometry_kernel(
    const int32_t *__restrict__ base_positions,
    const int32_t *__restrict__ valid_graph_rows,
    int valid_graph_row_count,
    int *__restrict__ generation_control,
    int generation_control_stride,
    int request_count,
    int padded_seq_len,
    int32_t *__restrict__ out_position_ids,
    int32_t *__restrict__ out_request_lengths)
{
    const int flat_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rows = request_count * padded_seq_len;
    if (flat_row < total_rows)
    {
        const int request = flat_row / padded_seq_len;
        const int token = flat_row - request * padded_seq_len;
        out_position_ids[flat_row] = base_positions[request] + token;
    }

    if (out_request_lengths && flat_row < request_count)
    {
        int valid_tokens = valid_graph_row_count / request_count;
        if (generation_control)
        {
            int *const control =
                generation_control +
                static_cast<size_t>(flat_row) *
                    static_cast<size_t>(generation_control_stride);
            const int depth =
                control[llaminar2::sampling_math::
                            kDeviceGenerationControlCurrentDraftDepth];
            valid_tokens =
                control[llaminar2::sampling_math::
                            kDeviceGenerationControlActiveVerifierRowCount];
            if (depth <= 0 || valid_tokens != depth + 1 ||
                valid_tokens > padded_seq_len)
            {
                llaminar2::sampling_math::fail_device_generation_control(
                    control,
                    llaminar2::sampling_math::
                        DeviceGenerationError::InvalidDepthSelector);
                valid_tokens = 0;
            }
        }
        else if (valid_graph_rows)
        {
            valid_tokens = 0;
            for (int row = 0; row < valid_graph_row_count; ++row)
            {
                const int physical_row = valid_graph_rows[row];
                valid_tokens +=
                    physical_row / padded_seq_len == flat_row ? 1 : 0;
            }
        }
        out_request_lengths[flat_row] = valid_tokens;
    }
}

/**
 * @brief Fuse one controller-selected verifier row into graph-stable buffers.
 *
 * One warp owns the complete scalar-request transaction. Lane zero validates
 * the controller and publishes its scalar outputs; every lane then writes one
 * or more token/position columns. The physical launch is invariant while the
 * active prefix changes from M=2 through M=16 behind the controller pointer.
 */
__global__ void cuda_prepare_mtp_verifier_controlled_row_kernel(
    const int32_t *__restrict__ first_token,
    const int32_t *__restrict__ draft_tokens,
    const int32_t *__restrict__ base_position,
    int *__restrict__ generation_control_row,
    int padded_seq_len,
    int32_t *__restrict__ out_tokens,
    int32_t *__restrict__ out_position_ids,
    int32_t *__restrict__ out_request_length,
    int32_t *__restrict__ out_base_position_snapshot)
{
    __shared__ int active_rows;
    if (threadIdx.x == 0)
    {
        const bool active_request =
            generation_control_row[
                llaminar2::sampling_math::kDeviceGenerationControlOk] != 0 &&
            generation_control_row[
                llaminar2::sampling_math::
                    kDeviceGenerationControlRequestComplete] == 0;
        const int depth = generation_control_row[
            llaminar2::sampling_math::
                kDeviceGenerationControlCurrentDraftDepth];
        const int verifier_rows = generation_control_row[
            llaminar2::sampling_math::
                kDeviceGenerationControlActiveVerifierRowCount];
        if (!active_request || depth <= 0 || verifier_rows != depth + 1 ||
            verifier_rows > padded_seq_len)
        {
            llaminar2::sampling_math::fail_device_generation_control(
                generation_control_row,
                llaminar2::sampling_math::
                    DeviceGenerationError::InvalidDepthSelector);
            active_rows = 0;
        }
        else
        {
            active_rows = verifier_rows;
        }
        *out_request_length = active_rows;
        *out_base_position_snapshot = *base_position;
    }
    __syncthreads();

    const int base = *base_position;
    for (int row = static_cast<int>(threadIdx.x); row < padded_seq_len;
         row += static_cast<int>(blockDim.x))
    {
        out_position_ids[row] = base + row;
        out_tokens[row] =
            row == 0 ? *first_token
                     : row < active_rows ? draft_tokens[row - 1] : 0;
    }
}

/**
 * @brief Initialize one typed frontier without manufacturing a sampled token.
 *
 * Each request has a unique writer. The shared contract reads its source
 * position before writes, including when that position row is reused in place.
 * Unsampled logits publish a valid position and an explicitly absent condition.
 */
__global__ void cuda_initialize_generation_logical_state_kernel(
    llaminar2::GenerationLogicalStateInitialization initialization)
{
    const unsigned request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request < static_cast<unsigned>(initialization.request_count))
        (void)initialization.initializeRequest(static_cast<int>(request));
}


extern "C"
{
    bool cudaOps_configure_mtp_greedy_penalty_policy(
        llaminar2::MTPGreedyPenaltyPolicy *controls,
        float presence_penalty,
        float frequency_penalty,
        bool first_token_already_in_history,
        int device_idx,
        void *stream, float repetition_penalty)
    {
        if (!controls || !stream)
            return false;

        cudaSetDevice(device_idx);
        cuda_configure_mtp_greedy_penalty_policy_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            controls,
            presence_penalty,
            frequency_penalty,
            first_token_already_in_history ? 1 : 0, repetition_penalty);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP greedy penalty policy launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }


    bool cudaOps_apply_mtp_penalties_f32_rows(
        float *data,
        int rows,
        int cols,
        int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const llaminar2::MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        int device_idx,
        void *stream)
    {
        if (!data || rows <= 0 || cols <= 0 || row_stride < cols ||
            !generated_token_counts || !policy || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int blocks = (cols + threads - 1) / threads;
        blocks = blocks > 1024 ? 1024 : blocks;
        if (verifier_input_tokens)
        {
            cuda_apply_mtp_verifier_penalties_f32_rows_kernel<<<
                dim3(blocks + 1),
                dim3(threads),
                0,
                static_cast<cudaStream_t>(stream)>>>(
                data,
                rows,
                cols,
                row_stride,
                verifier_input_tokens,
                generated_token_counts,
                policy,
                active_rows);
        }
        else
        {
            cuda_apply_mtp_durable_penalties_f32_rows_kernel<<<
                dim3(blocks),
                dim3(threads),
                0,
                static_cast<cudaStream_t>(stream)>>>(
                data,
                rows,
                cols,
                row_stride,
                generated_token_counts,
                policy,
                active_rows);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP penalty row transform launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    bool cudaOps_apply_mtp_branch_penalties_f32_row(
        float *data,
        int cols,
        const int *first_condition_token,
        const int *prior_draft_tokens,
        int prior_draft_count,
        const int *generated_token_counts,
        const llaminar2::MTPGreedyPenaltyPolicy *policy,
        int device_idx,
        void *stream)
    {
        if (!data || cols <= 0 || !first_condition_token ||
            prior_draft_count < 0 ||
            (prior_draft_count > 0 && !prior_draft_tokens) ||
            !generated_token_counts || !policy || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads = 256;
        int dense_blocks = (cols + threads - 1) / threads;
        dense_blocks = dense_blocks > 1024 ? 1024 : dense_blocks;
        cuda_apply_mtp_branch_penalties_f32_row_kernel<<<
            dim3(dense_blocks + 1),
            dim3(threads),
            0,
            static_cast<cudaStream_t>(stream)>>>(
            data,
            cols,
            first_condition_token,
            prior_draft_tokens,
            prior_draft_count,
            generated_token_counts,
            policy);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP branch penalty row transform launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_commit_mtp_greedy_penalty_history(
        const int *output_tokens,
        const int *output_meta,
        llaminar2::MTPGreedyPenaltyPolicy *policy,
        const int *accepted_state_counts,
        const int *stopped_flags,
        int output_token_capacity,
        int vocab_size,
        int *generated_token_counts,
        int device_idx,
        void *stream)
    {
        if (!output_tokens || !output_meta || !policy ||
            !accepted_state_counts || !stopped_flags ||
            output_token_capacity <= 0 || vocab_size <= 0 ||
            !generated_token_counts || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_commit_mtp_greedy_penalty_history_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            output_tokens,
            output_meta,
            policy,
            accepted_state_counts,
            stopped_flags,
            output_token_capacity,
            vocab_size,
            generated_token_counts);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MTP greedy penalty history commit launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_commit_generation_token_history(
        const int *token,
        int vocab_size,
        int *generated_token_counts,
        llaminar2::MTPGreedyPenaltyPolicy *direct_output_policy,
        int device_idx,
        void *stream)
    {
        if (!token || vocab_size <= 0 || !generated_token_counts || !stream)
            return false;
        cudaSetDevice(device_idx);
        cuda_commit_generation_token_history_kernel<<<
            1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            token, vocab_size, generated_token_counts, direct_output_policy);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA generation-token history commit launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }


    bool cudaOps_advance_speculative_commit_boundary(
        const int *meta,
        int request_count,
        int meta_stride,
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream)
    {
        if (!meta || request_count <= 0 ||
            meta_stride <
                llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            !decode_rounds_committed ||
            !decode_rounds_until_maintenance ||
            !maintenance_due || !decode_boundary_advanced || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_advance_speculative_commit_boundary_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            meta,
            request_count,
            meta_stride,
            decode_rounds_committed,
            decode_rounds_until_maintenance,
            maintenance_due,
            decode_boundary_advanced);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA speculative commit-boundary advance launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_publish_serial_decode_commit_boundary(
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream)
    {
        if (!decode_rounds_committed ||
            !decode_rounds_until_maintenance || !maintenance_due ||
            !decode_boundary_advanced || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_publish_serial_decode_commit_boundary_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            decode_rounds_committed,
            decode_rounds_until_maintenance,
            maintenance_due,
            decode_boundary_advanced);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA serial decode commit-boundary publication failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_acknowledge_decode_commit_boundary(
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream)
    {
        if (!decode_rounds_until_maintenance || !maintenance_due ||
            !decode_boundary_advanced || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_acknowledge_decode_commit_boundary_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            decode_rounds_until_maintenance,
            maintenance_due,
            decode_boundary_advanced);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA decode commit-boundary acknowledgement failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_initialize_device_moe_rebalance_dispatch_ticket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        uint32_t participant_id,
        uint32_t participant_count,
        llaminar2::DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream)
    {
        if (session_epoch == 0u || workspace_generation == 0u ||
            participant_count == 0u || participant_id >= participant_count ||
            !ticket || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_initialize_device_moe_rebalance_dispatch_ticket_kernel<<<
            1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            session_epoch,
            workspace_generation,
            participant_id,
            participant_count,
            ticket);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MoE dispatch-ticket initialization failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_publish_device_moe_rebalance_dispatch_ticket(
        const uint32_t *controller_magic,
        const uint32_t *controller_version,
        const uint32_t *controller_error,
        const uint32_t *decode_rounds_committed,
        const uint32_t *decode_rounds_until_maintenance,
        const uint32_t *maintenance_due,
        const uint32_t *decode_boundary_advanced,
        llaminar2::DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream)
    {
        if (!controller_magic || !controller_version || !controller_error ||
            !decode_rounds_committed || !decode_rounds_until_maintenance ||
            !maintenance_due || !decode_boundary_advanced || !ticket ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_publish_device_moe_rebalance_dispatch_ticket_kernel<<<
            1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
            controller_magic,
            controller_version,
            controller_error,
            decode_rounds_committed,
            decode_rounds_until_maintenance,
            maintenance_due,
            decode_boundary_advanced,
            ticket);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA MoE dispatch-ticket publication failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /** @brief Enqueue ordinary publication on the exact caller-owned stream. */
    bool cudaOps_publish_ordinary_generation_sample(
        const llaminar2::sampling_math::OrdinaryGenerationPublication &publication,
        int device_idx, void *stream)
    {
        if (!publication.valid() || !stream || device_idx < 0)
            return false;
        if (cudaSetDevice(device_idx) != cudaSuccess) return false;
        constexpr int threads = 32;
        const int blocks = 1 + (publication.request_count - 1) / threads;
        cuda_publish_ordinary_generation_sample_kernel<<<
            blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(publication);
        return cudaGetLastError() == cudaSuccess;
    }

    bool cudaOps_initialize_device_generation(
        int request_count,
        int max_new_tokens,
        const llaminar2::sampling_math::DeviceGenerationPolicy &depth_policy,
        llaminar2::sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition,
        int response_token_stride,
        int control_stride,
        int *control,
        int device_idx,
        void *stream,
    const llaminar2::sampling_math::DeviceGenerationInitialization &initialization)
    {
        if (request_count <= 0 || !initialization.valid() ||
            !llaminar2::sampling_math::valid_device_generation_admission(
                depth_policy, max_new_tokens, initial_leading_row_disposition) ||
            response_token_stride <= 0 || response_token_stride < max_new_tokens ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            !control || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        // One request owns one scalar controller row. Production generation is
        // normally batch one, so one warp avoids launching three immediately
        // retired warps while retaining grid totality for larger request sets.
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_initialize_device_generation_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            request_count,
            max_new_tokens,
            depth_policy,
            initial_leading_row_disposition,
            response_token_stride,
            control_stride,
            control, initialization);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA device-generation initialization launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_initialize_device_generation_dispatch_tickets(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        int *control,
        int control_stride,
        int request_count,
        llaminar2::sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream)
    {
        if (session_epoch == 0 || workspace_generation == 0 || !control ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !tickets || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_initialize_device_generation_dispatch_ticket_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            session_epoch,
            workspace_generation,
            control,
            control_stride,
            request_count,
            tickets);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA device-generation ticket initialization launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_publish_device_generation_dispatch_tickets(
        int *control,
        int control_stride,
        int request_count,
        const uint32_t *maintenance_due,
        llaminar2::sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream)
    {
        if (!control ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !tickets || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_publish_device_generation_dispatch_ticket_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            control,
            control_stride,
            request_count,
            maintenance_due,
            tickets);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA device-generation ticket publication launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_prepare_device_generation_transaction_budget(
        int *control,
        int control_stride,
        int request_count,
        int verifier_row_capacity,
        const uint32_t *maintenance_rows_remaining,
        const uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream)
    {
        const bool has_maintenance_boundary =
            maintenance_rows_remaining || maintenance_due ||
            decode_boundary_advanced;
        const bool has_complete_maintenance_boundary =
            maintenance_rows_remaining && maintenance_due &&
            decode_boundary_advanced;
        if (!control ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || verifier_row_capacity <= 0 || !stream ||
            (has_maintenance_boundary &&
             !has_complete_maintenance_boundary))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        // Keep the tiny row transition to one physical CUDA warp per block.
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_prepare_device_generation_transaction_budget_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            control,
            control_stride,
            request_count,
            verifier_row_capacity,
            maintenance_rows_remaining,
            maintenance_due,
            decode_boundary_advanced);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA device-generation budget launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool
    cudaOps_commit_device_generation_and_derive_speculative_publication_metadata(
        const int32_t *compact_tokens,
        int output_token_stride,
        int *compact_meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int32_t *response_tokens,
        int response_token_stride,
        int *control,
        int control_stride,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_sidecar_condition_tokens,
        int *out_next_sidecar_position_ids,
        int *out_next_verifier_condition_tokens,
        const int32_t *verifier_input_tokens,
        int verifier_input_token_stride,
        llaminar2::sampling_math::MTPCommittedVerifierIdentityRecord *
            out_committed_verifier_identity,
        llaminar2::sampling_math::MTPDepthTransactionTrace *depth_transaction_trace,
        int device_idx,
        void *stream)
    {
        if (!compact_tokens || output_token_stride <= 0 || !compact_meta ||
            meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            !base_cached_tokens || request_count <= 0 || (depth_transaction_trace && request_count != 1) ||
            padded_state_rows_per_request <= 0 ||
            !response_tokens || response_token_stride <= 0 ||
            !control ||
            control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            !out_restore_rows || !out_target_cached_tokens ||
            !out_accepted_state_counts || !out_ok ||
            ((out_next_sidecar_condition_tokens ||
              out_next_sidecar_position_ids) &&
             (!out_next_sidecar_condition_tokens ||
              !out_next_sidecar_position_ids ||
              !out_next_condition_tokens)) ||
            (out_next_verifier_condition_tokens &&
             !out_next_condition_tokens) ||
            ((verifier_input_tokens ||
              verifier_input_token_stride != 0 ||
              out_committed_verifier_identity) &&
             (!verifier_input_tokens ||
              verifier_input_token_stride <= 0 ||
              !out_committed_verifier_identity)) ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        // The fused transaction is request-independent. One physical warp is
        // the minimum economical CUDA launch for the production batch-one case;
        // larger batches retain grid totality without divergent cooperation.
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_commit_device_generation_and_derive_speculative_publication_metadata_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            compact_tokens,
            output_token_stride,
            compact_meta,
            meta_stride,
            base_cached_tokens,
            request_count,
            padded_state_rows_per_request,
            response_tokens,
            response_token_stride,
            control,
            control_stride,
            out_restore_rows,
            out_target_cached_tokens,
            out_accepted_state_counts,
            out_ok,
            reinterpret_cast<int32_t *>(out_next_condition_tokens),
            out_all_drafts_accepted_flags,
            out_stopped_flags,
            reinterpret_cast<int32_t *>(
                out_next_sidecar_condition_tokens),
            reinterpret_cast<int32_t *>(out_next_sidecar_position_ids),
            reinterpret_cast<int32_t *>(
                out_next_verifier_condition_tokens),
            verifier_input_tokens,
            verifier_input_token_stride,
            out_committed_verifier_identity, depth_transaction_trace);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA fused device-generation publication launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_derive_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int max_state_commit_rows,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        const int32_t *output_tokens,
        int output_token_stride,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_verifier_condition_tokens,
        int device_idx,
        void *stream)
    {
        if (!meta || !base_cached_tokens ||
            !out_restore_rows || !out_target_cached_tokens ||
            !out_accepted_state_counts || !out_ok || !stream ||
            ((out_next_condition_tokens || output_tokens) &&
             (!out_next_condition_tokens || !output_tokens || output_token_stride <= 0)) ||
            (out_next_verifier_condition_tokens &&
             !out_next_condition_tokens) ||
            meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            request_count <= 0 ||
            padded_state_rows_per_request <= 0 ||
            max_state_commit_rows < 0 ||
            max_state_commit_rows > padded_state_rows_per_request)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 128;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_derive_speculative_publication_metadata_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            meta,
            meta_stride,
            base_cached_tokens,
            request_count,
            padded_state_rows_per_request,
            max_state_commit_rows,
            out_restore_rows,
            out_target_cached_tokens,
            out_accepted_state_counts,
            out_ok,
            reinterpret_cast<int32_t *>(out_next_condition_tokens),
            output_tokens,
            output_token_stride,
            out_all_drafts_accepted_flags,
            out_stopped_flags,
            reinterpret_cast<int32_t *>(
                out_next_verifier_condition_tokens));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Publication Metadata kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_derive_shifted_speculative_publication_metadata_from_primary(
        const int *base_cached_tokens,
        const int *main_target_cached_tokens,
        const int *main_publication_ok,
        int request_count,
        int mtp_depth,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int device_idx,
        void *stream, llaminar2::MTPShiftedCacheOrigin origin)
    {
        if (!base_cached_tokens ||
            !main_target_cached_tokens ||
            !main_publication_ok ||
            !out_target_cached_tokens ||
            !out_accepted_state_counts ||
            !out_ok ||
            !stream ||
            request_count <= 0 ||
            llaminar2::mtpShiftedPrefixOffset(origin,mtp_depth)<0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        const int blocks =
            (request_count + threads_per_block - 1) / threads_per_block;
        cuda_derive_shifted_speculative_publication_metadata_from_primary_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            base_cached_tokens,
            main_target_cached_tokens,
            main_publication_ok,
            request_count,
            mtp_depth,
            out_target_cached_tokens,
            out_accepted_state_counts,
            out_ok,origin);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Shifted Speculative Publication Metadata kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_prepare_speculative_shifted_kv_tokens(
        const int *meta,
        int meta_stride,
        const int32_t *output_tokens,
        int output_token_stride,
        int request_index,
        int first_output_token_index,
        int row_count,
        int32_t filler_token,
        int32_t *out_tokens,
        const int32_t *base_positions,
        int position_offset,
        int32_t *out_position_ids,
        int device_idx,
        void *stream)
    {
        if (!meta ||
            !output_tokens ||
            !out_tokens ||
            !base_positions ||
            !out_position_ids ||
            !stream ||
            meta_stride < llaminar2::sampling_math::kSpeculativeBatchMetaCount ||
            output_token_stride <= first_output_token_index ||
            request_index < 0 ||
            first_output_token_index < 0 ||
            row_count <= 0)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        cuda_prepare_speculative_shifted_kv_tokens_kernel<<<
            1,
            1,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            meta,
            meta_stride,
            output_tokens,
            output_token_stride,
            request_index,
            first_output_token_index,
            row_count,
            filler_token,
            out_tokens,
            base_positions,
            position_offset,
            out_position_ids);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Speculative Shifted KV Token Prep kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Enqueue device-resident preparation for one grouped MTP sidecar.
     */
    bool cudaOps_prepare_mtp_batched_sidecar_inputs(
        const int32_t *condition_tokens,
        int condition_token_stride,
        const int32_t *base_positions,
        int position_offset,
        int request_count,
        int32_t *out_condition_tokens,
        int32_t *out_position_ids,
        int device_idx,
        void *stream)
    {
        if (!condition_tokens ||
            condition_token_stride <= 0 ||
            !base_positions ||
            request_count <= 0 ||
            !out_condition_tokens ||
            !out_position_ids ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        cuda_prepare_mtp_batched_sidecar_inputs_kernel<<<
            1,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            condition_tokens,
            condition_token_stride,
            base_positions,
            position_offset,
            request_count,
            out_condition_tokens,
            out_position_ids);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA batched MTP sidecar input preparation failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /** @brief Enqueue the position-only scalar/rectangular compatibility form. */
    bool cudaOps_prepare_mtp_verifier_position_ids(
        const int32_t *base_positions,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int device_idx,
        void *stream)
    {
        if (!base_positions ||
            request_count <= 0 ||
            padded_seq_len <= 0 ||
            !out_position_ids ||
            !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 128;
        const int total_rows = request_count * padded_seq_len;
        const int blocks =
            (total_rows + threads_per_block - 1) / threads_per_block;
        cuda_prepare_mtp_verifier_geometry_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            base_positions,
            /*valid_graph_rows=*/nullptr,
            /*valid_graph_row_count=*/0,
            /*generation_control=*/nullptr,
            /*generation_control_stride=*/0,
            request_count,
            padded_seq_len,
            out_position_ids,
            /*out_request_lengths=*/nullptr);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA grouped MTP verifier position preparation failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Enqueue one device-resident grouped-verifier geometry publication.
     */
    bool cudaOps_prepare_mtp_verifier_geometry(
        const int32_t *base_positions,
        const int32_t *valid_graph_rows,
        int valid_graph_row_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int32_t *out_request_lengths,
        int device_idx,
        void *stream)
    {
        const int total_rows = request_count * padded_seq_len;
        const bool rectangular = valid_graph_rows == nullptr;
        const bool device_controlled = generation_control != nullptr;
        if (!base_positions || request_count <= 0 || padded_seq_len <= 0 ||
            !out_position_ids || !out_request_lengths || !stream ||
            (device_controlled != (generation_control_stride > 0)) ||
            (device_controlled &&
             generation_control_stride <
                 llaminar2::sampling_math::kDeviceGenerationControlCount) ||
            (rectangular &&
             (valid_graph_row_count <= 0 ||
              valid_graph_row_count > total_rows ||
              valid_graph_row_count % request_count != 0)) ||
            (!rectangular &&
             (valid_graph_row_count <= 0 || valid_graph_row_count > total_rows)))
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 128;
        const int blocks =
            (total_rows + threads_per_block - 1) / threads_per_block;
        cuda_prepare_mtp_verifier_geometry_kernel<<<
            blocks,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            base_positions,
            valid_graph_rows,
            valid_graph_row_count,
            generation_control,
            generation_control_stride,
            request_count,
            padded_seq_len,
            out_position_ids,
            out_request_lengths);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "CUDA grouped MTP verifier geometry preparation failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /** @brief Enqueue one fused controller-owned verifier row publication. */
    bool cudaOps_prepare_mtp_verifier_controlled_row(
        const int32_t *first_token,
        const int32_t *draft_tokens,
        const int32_t *base_position,
        int *generation_control_row,
        int generation_control_stride,
        int padded_seq_len,
        int32_t *out_tokens,
        int32_t *out_position_ids,
        int32_t *out_request_length,
        int32_t *out_base_position_snapshot,
        int device_idx,
        void *stream)
    {
        if (!first_token || !draft_tokens || !base_position ||
            !generation_control_row ||
            generation_control_stride <
                llaminar2::sampling_math::kDeviceGenerationControlCount ||
            padded_seq_len <= 1 || !out_tokens || !out_position_ids ||
            !out_request_length || !out_base_position_snapshot || !stream)
        {
            return false;
        }

        cudaSetDevice(device_idx);
        constexpr int threads_per_block = 32;
        cuda_prepare_mtp_verifier_controlled_row_kernel<<<
            1,
            threads_per_block,
            0,
            static_cast<cudaStream_t>(stream)>>>(
            first_token,
            draft_tokens,
            base_position,
            generation_control_row,
            padded_seq_len,
            out_tokens,
            out_position_ids,
            out_request_length,
            out_base_position_snapshot);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(
                stderr,
                "CUDA controlled MTP verifier row preparation failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    /** @brief Enqueue all initial frontier rows, with one exclusive lane per request. */
    bool cudaOps_initialize_generation_logical_state(
        const llaminar2::GenerationLogicalStateInitialization &initialization,
        int device_idx, void *stream)
    {
        if (!stream || !initialization.valid() || cudaSetDevice(device_idx) != cudaSuccess)
            return false;
        // Ceiling division without signed addition overflow; no request is
        // silently lost when a batch spans more than one hardware warp.
        constexpr unsigned threads = 32;
        const unsigned blocks = 1u + (static_cast<unsigned>(initialization.request_count) - 1u) / threads;
        cuda_initialize_generation_logical_state_kernel<<<blocks, threads, 0,
            static_cast<cudaStream_t>(stream)>>>(initialization);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA generation logical-state initialization failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }


} // extern "C"
