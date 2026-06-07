#include "MTPSpecDecodeMetadata.h"

#include "../../backends/IBackend.h"
#include "../local_execution/device/DeviceWorkspaceManager.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        WorkspaceDescriptor int32Buffer(const char *name, int count)
        {
            return WorkspaceDescriptor{
                name,
                static_cast<size_t>(std::max(0, count)) * sizeof(int32_t),
                256,
                true};
        }

        MTPSpecDecodeMetadataBatch metadataFailure(
            const MTPSpecDecodeMetadataShape &shape,
            std::string reason)
        {
            MTPSpecDecodeMetadataBatch batch;
            batch.shape = shape;
            batch.ok = false;
            batch.error = std::move(reason);
            return batch;
        }

        void fillInvalid(std::vector<int32_t> &values, int count)
        {
            values.assign(
                static_cast<size_t>(std::max(0, count)),
                kMTPSpecDecodeInvalidToken);
        }

        void fillZero(std::vector<int32_t> &values, int count)
        {
            values.assign(static_cast<size_t>(std::max(0, count)), 0);
        }
    } // namespace

    WorkspaceRequirements buildMTPSpecDecodeWorkspaceRequirements(
        const MTPSpecDecodeMetadataShape &shape)
    {
        WorkspaceRequirements reqs;
        if (!shape.valid())
            return reqs;

        const int requests = shape.max_requests;
        const int draft_slots = requests * shape.max_draft_tokens;
        const int target_slots = requests * shape.maxTargetQueryLen();
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::TARGET_QUERY_LENS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::VALID_SAMPLED_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::COMMITTED_OUTPUT_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::REJECTED_TOKEN_COUNTS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::TOKEN_INDICES_TO_SAMPLE, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::NEXT_CONDITION_TOKENS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::ALL_DRAFTS_ACCEPTED_FLAGS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::STOPPED_FLAGS, requests));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS, requests + 1));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::STATE_INDICES, target_slots));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS, draft_slots));
        reqs.buffers.push_back(int32Buffer(MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS, target_slots));
        return reqs;
    }

    MTPSpecDecodeMetadataBatch buildMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataShape &shape,
        const std::vector<MTPSpecDecodeRequest> &requests,
        const std::vector<int32_t> &committed_output_counts,
        const std::vector<int32_t> &stopped_flags)
    {
        if (!shape.valid())
            return metadataFailure(shape, "invalid MTP spec-decode metadata shape");
        if (requests.empty())
            return metadataFailure(shape, "MTP spec-decode metadata batch has no requests");
        if (static_cast<int>(requests.size()) > shape.max_requests)
            return metadataFailure(shape, "MTP spec-decode metadata batch exceeds max_requests");
        if (committed_output_counts.size() != requests.size())
            return metadataFailure(shape, "committed output count vector does not match request count");
        if (stopped_flags.size() != requests.size())
            return metadataFailure(shape, "stopped flag vector does not match request count");

        MTPSpecDecodeMetadataBatch batch;
        batch.ok = true;
        batch.shape = shape;
        batch.request_count = static_cast<int>(requests.size());
        batch.transactions.reserve(requests.size());

        const int requests_slots = shape.max_requests;
        const int draft_slots = shape.max_requests * shape.max_draft_tokens;
        const int target_slots = shape.max_requests * shape.maxTargetQueryLen();
        fillZero(batch.draft_counts, requests_slots);
        fillZero(batch.target_query_lens, requests_slots);
        fillZero(batch.valid_sampled_counts, requests_slots);
        fillZero(batch.accepted_draft_prefixes, requests_slots);
        fillZero(batch.committed_output_counts, requests_slots);
        fillZero(batch.rejected_token_counts, requests_slots);
        fillInvalid(batch.token_indices_to_sample, requests_slots);
        fillInvalid(batch.next_condition_tokens, requests_slots);
        fillZero(batch.all_drafts_accepted_flags, requests_slots);
        fillZero(batch.stopped_flags, requests_slots);
        fillZero(batch.query_start_locs, requests_slots + 1);
        fillInvalid(batch.state_indices, target_slots);
        fillInvalid(batch.draft_tokens, draft_slots);
        fillInvalid(batch.sampled_tokens, target_slots);

        int query_cursor = 0;
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            const MTPSpecDecodeRequest &request = requests[request_index];
            if (static_cast<int>(request.draft_tokens.size()) > shape.max_draft_tokens)
            {
                return metadataFailure(shape, "request draft token count exceeds metadata shape");
            }
            if (static_cast<int>(request.sampled_tokens.size()) > shape.maxTargetQueryLen())
            {
                return metadataFailure(shape, "request sampled token count exceeds metadata shape");
            }

            MTPSpecDecodeTransaction tx =
                buildMTPSpecDecodeTransaction(request);
            if (!tx.ok)
            {
                return metadataFailure(
                    shape,
                    std::string("request transaction metadata failed: ") + tx.error);
            }
            const int32_t committed_count =
                committed_output_counts[request_index];
            if (committed_count < 0 ||
                committed_count > tx.valid_sampled_count)
            {
                return metadataFailure(
                    shape,
                    "committed output count is outside the valid sampled prefix");
            }

            const int i = static_cast<int>(request_index);
            const int draft_offset = i * shape.max_draft_tokens;
            const int target_offset = i * shape.maxTargetQueryLen();
            batch.draft_counts[request_index] = tx.draft_count;
            batch.target_query_lens[request_index] = tx.target_query_len;
            batch.valid_sampled_counts[request_index] = tx.valid_sampled_count;
            batch.accepted_draft_prefixes[request_index] = tx.accepted_speculative_prefix;
            batch.committed_output_counts[request_index] =
                committed_count;
            batch.rejected_token_counts[request_index] = tx.rejected_token_count;
            batch.token_indices_to_sample[request_index] = tx.token_index_to_sample;
            batch.next_condition_tokens[request_index] = tx.next_condition_token;
            batch.all_drafts_accepted_flags[request_index] =
                tx.allDraftsAccepted() ? 1 : 0;
            batch.stopped_flags[request_index] =
                stopped_flags[request_index] != 0 ? 1 : 0;
            batch.query_start_locs[request_index] = query_cursor;

            for (int row = 0; row < tx.target_query_len; ++row)
            {
                batch.state_indices[static_cast<size_t>(target_offset + row)] =
                    query_cursor + row;
            }
            query_cursor += tx.target_query_len;

            for (size_t j = 0; j < request.draft_tokens.size(); ++j)
            {
                batch.draft_tokens[static_cast<size_t>(draft_offset) + j] =
                    request.draft_tokens[j];
            }
            for (size_t j = 0; j < request.sampled_tokens.size(); ++j)
            {
                batch.sampled_tokens[static_cast<size_t>(target_offset) + j] =
                    request.sampled_tokens[j];
            }

            batch.transactions.push_back(std::move(tx));
        }

        batch.query_start_locs[requests.size()] = query_cursor;
        batch.total_target_query_tokens = query_cursor;
        return batch;
    }

    bool MTPSpecDecodeMetadataDevicePointers::complete() const
    {
        return draft_counts &&
               target_query_lens &&
               valid_sampled_counts &&
               accepted_draft_prefixes &&
               committed_output_counts &&
               rejected_token_counts &&
               token_indices_to_sample &&
               next_condition_tokens &&
               all_drafts_accepted_flags &&
               stopped_flags &&
               query_start_locs &&
               state_indices &&
               draft_tokens &&
               sampled_tokens;
    }

    MTPSpecDecodeMetadataWorkspaceBinding::MTPSpecDecodeMetadataWorkspaceBinding(
        MTPSpecDecodeMetadataShape shape)
        : shape_(shape)
    {
    }

    void MTPSpecDecodeMetadataWorkspaceBinding::setShape(
        MTPSpecDecodeMetadataShape shape)
    {
        shape_ = shape;
        refreshDevicePointers();
    }

    MTPSpecDecodeMetadataShape MTPSpecDecodeMetadataWorkspaceBinding::effectiveShape(
        int m,
        int n,
        int k) const
    {
        MTPSpecDecodeMetadataShape effective = shape_;
        if (m > 0)
            effective.max_requests = std::max(effective.max_requests, m);
        if (n > 0)
            effective.max_draft_tokens =
                std::max(effective.max_draft_tokens, n);
        if (k > 0)
            effective.max_draft_tokens =
                std::max(effective.max_draft_tokens, k);
        return effective;
    }

    WorkspaceRequirements MTPSpecDecodeMetadataWorkspaceBinding::getWorkspaceRequirements(
        int m,
        int n,
        int k) const
    {
        return buildMTPSpecDecodeWorkspaceRequirements(effectiveShape(m, n, k));
    }

    void MTPSpecDecodeMetadataWorkspaceBinding::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
        refreshDevicePointers();
    }

    bool MTPSpecDecodeMetadataWorkspaceBinding::hasWorkspace() const
    {
        return workspace_ != nullptr && device_pointers_.complete();
    }

    void MTPSpecDecodeMetadataWorkspaceBinding::refreshDevicePointers()
    {
        device_pointers_ = {};
        binding_error_.clear();
        if (!workspace_)
            return;

        const WorkspaceRequirements reqs =
            buildMTPSpecDecodeWorkspaceRequirements(shape_);
        if (reqs.buffers.empty())
        {
            binding_error_ = "invalid MTP spec-decode metadata workspace shape";
            return;
        }

        std::ostringstream error;
        auto bind_int32 = [&](const char *name, int32_t **out) -> bool
        {
            const WorkspaceDescriptor *desc = reqs.find(name);
            if (!desc)
            {
                error << "missing descriptor for " << name;
                return false;
            }
            if (!workspace_->hasBuffer(name))
            {
                error << "workspace missing buffer " << name;
                return false;
            }
            const size_t size = workspace_->getBufferSize(name);
            if (size < desc->size_bytes)
            {
                error << "workspace buffer " << name << " is too small: "
                      << size << " < " << desc->size_bytes;
                return false;
            }
            *out = static_cast<int32_t *>(workspace_->getBuffer(name));
            if (!*out)
            {
                error << "workspace buffer " << name << " returned null";
                return false;
            }
            return true;
        };

        if (!bind_int32(
                MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS,
                &device_pointers_.draft_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::TARGET_QUERY_LENS,
                &device_pointers_.target_query_lens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::VALID_SAMPLED_COUNTS,
                &device_pointers_.valid_sampled_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES,
                &device_pointers_.accepted_draft_prefixes) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::COMMITTED_OUTPUT_COUNTS,
                &device_pointers_.committed_output_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::REJECTED_TOKEN_COUNTS,
                &device_pointers_.rejected_token_counts) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::TOKEN_INDICES_TO_SAMPLE,
                &device_pointers_.token_indices_to_sample) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::NEXT_CONDITION_TOKENS,
                &device_pointers_.next_condition_tokens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::ALL_DRAFTS_ACCEPTED_FLAGS,
                &device_pointers_.all_drafts_accepted_flags) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::STOPPED_FLAGS,
                &device_pointers_.stopped_flags) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS,
                &device_pointers_.query_start_locs) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::STATE_INDICES,
                &device_pointers_.state_indices) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS,
                &device_pointers_.draft_tokens) ||
            !bind_int32(
                MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS,
                &device_pointers_.sampled_tokens))
        {
            binding_error_ = error.str();
            device_pointers_ = {};
        }
    }

    MTPSpecDecodeMetadataUploadResult uploadMTPSpecDecodeMetadataBatch(
        const MTPSpecDecodeMetadataBatch &batch,
        const MTPSpecDecodeMetadataWorkspaceBinding &binding,
        DeviceId device,
        IBackend *backend,
        void *stream)
    {
        MTPSpecDecodeMetadataUploadResult result;
        if (!batch.ok)
        {
            result.error = std::string("cannot upload invalid MTP metadata batch: ") +
                           batch.error;
            return result;
        }
        if (!binding.hasWorkspace())
        {
            result.error = std::string("MTP metadata workspace is not completely bound: ") +
                           binding.bindingError();
            return result;
        }
        if (device.is_gpu())
        {
            if (!stream)
            {
                result.error =
                    "MTP metadata GPU upload requires an explicit non-null stream";
                return result;
            }
            if (!backend)
            {
                result.error = "MTP metadata GPU upload requires a backend";
                return result;
            }
        }

        const auto &ptrs = binding.devicePointers();
        auto upload = [&](int32_t *dst,
                          const std::vector<int32_t> &src,
                          const char *name) -> bool
        {
            if (src.empty())
                return true;
            if (!dst)
            {
                result.error = std::string("MTP metadata destination is null for ") + name;
                return false;
            }

            const size_t bytes = src.size() * sizeof(int32_t);
            bool ok = true;
            if (device.is_gpu())
            {
                ok = backend->hostToDeviceOnStream(
                    dst,
                    src.data(),
                    bytes,
                    device.ordinal,
                    stream);
            }
            else
            {
                std::memcpy(dst, src.data(), bytes);
            }

            if (!ok)
            {
                result.error = std::string("MTP metadata upload failed for ") + name;
                return false;
            }

            result.bytes_uploaded += bytes;
            return true;
        };

        if (!upload(ptrs.draft_counts, batch.draft_counts,
                    MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS) ||
            !upload(ptrs.target_query_lens, batch.target_query_lens,
                    MTPSpecDecodeWorkspaceBuffers::TARGET_QUERY_LENS) ||
            !upload(ptrs.valid_sampled_counts, batch.valid_sampled_counts,
                    MTPSpecDecodeWorkspaceBuffers::VALID_SAMPLED_COUNTS) ||
            !upload(ptrs.accepted_draft_prefixes, batch.accepted_draft_prefixes,
                    MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES) ||
            !upload(ptrs.committed_output_counts, batch.committed_output_counts,
                    MTPSpecDecodeWorkspaceBuffers::COMMITTED_OUTPUT_COUNTS) ||
            !upload(ptrs.rejected_token_counts, batch.rejected_token_counts,
                    MTPSpecDecodeWorkspaceBuffers::REJECTED_TOKEN_COUNTS) ||
            !upload(ptrs.token_indices_to_sample, batch.token_indices_to_sample,
                    MTPSpecDecodeWorkspaceBuffers::TOKEN_INDICES_TO_SAMPLE) ||
            !upload(ptrs.next_condition_tokens, batch.next_condition_tokens,
                    MTPSpecDecodeWorkspaceBuffers::NEXT_CONDITION_TOKENS) ||
            !upload(ptrs.all_drafts_accepted_flags, batch.all_drafts_accepted_flags,
                    MTPSpecDecodeWorkspaceBuffers::ALL_DRAFTS_ACCEPTED_FLAGS) ||
            !upload(ptrs.stopped_flags, batch.stopped_flags,
                    MTPSpecDecodeWorkspaceBuffers::STOPPED_FLAGS) ||
            !upload(ptrs.query_start_locs, batch.query_start_locs,
                    MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS) ||
            !upload(ptrs.state_indices, batch.state_indices,
                    MTPSpecDecodeWorkspaceBuffers::STATE_INDICES) ||
            !upload(ptrs.draft_tokens, batch.draft_tokens,
                    MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS) ||
            !upload(ptrs.sampled_tokens, batch.sampled_tokens,
                    MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS))
        {
            return result;
        }

        result.ok = true;
        return result;
    }

} // namespace llaminar2
