#include "MTPSpecDecodeTransaction.h"

#include <algorithm>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPSpecDecodeTransaction failure(
            const MTPSpecDecodeRequest &request,
            std::string reason)
        {
            MTPSpecDecodeTransaction tx;
            tx.request_id = request.request_id;
            tx.draft_count = static_cast<int>(request.draft_tokens.size());
            tx.target_query_len = tx.draft_count + 1;
            tx.ok = false;
            tx.error = std::move(reason);
            return tx;
        }

        bool tokenInVocab(int32_t token, int vocab_size)
        {
            return token >= 0 && (vocab_size <= 0 || token < vocab_size);
        }

        int contiguousValidPrefix(
            const std::vector<int32_t> &tokens,
            int vocab_size)
        {
            int valid = 0;
            for (int32_t token : tokens)
            {
                if (!tokenInVocab(token, vocab_size))
                    break;
                ++valid;
            }
            return valid;
        }

        bool hasValidTokenAfterPrefix(
            const std::vector<int32_t> &tokens,
            int valid_prefix,
            int vocab_size)
        {
            for (size_t i = static_cast<size_t>(valid_prefix);
                 i < tokens.size();
                 ++i)
            {
                if (tokenInVocab(tokens[i], vocab_size))
                    return true;
            }
            return false;
        }

        int acceptedDraftPrefix(
            const std::vector<int32_t> &draft_tokens,
            const std::vector<int32_t> &sampled_tokens,
            int valid_sampled_count)
        {
            const int limit = std::min<int>(
                static_cast<int>(draft_tokens.size()),
                valid_sampled_count);
            int accepted = 0;
            for (; accepted < limit; ++accepted)
            {
                if (sampled_tokens[static_cast<size_t>(accepted)] !=
                    draft_tokens[static_cast<size_t>(accepted)])
                {
                    break;
                }
            }
            return accepted;
        }
    } // namespace

    MTPSpecDecodeTransaction buildMTPSpecDecodeTransaction(
        const MTPSpecDecodeRequest &request)
    {
        const int draft_count = static_cast<int>(request.draft_tokens.size());
        if (draft_count <= 0)
            return failure(request, "spec decode request has no draft tokens");
        if (request.vocab_size < 0)
            return failure(request, "spec decode request has negative vocab size");
        if (static_cast<int>(request.sampled_tokens.size()) > draft_count + 1)
            return failure(request, "sampled token row is longer than draft_count + 1");
        for (int32_t token : request.draft_tokens)
        {
            if (!tokenInVocab(token, request.vocab_size))
                return failure(request, "draft token is outside the vocabulary");
        }

        MTPSpecDecodeTransaction tx;
        tx.ok = true;
        tx.request_id = request.request_id;
        tx.draft_count = draft_count;
        tx.target_query_len = draft_count + 1;

        if (request.discarded)
        {
            if (!tokenInVocab(request.backup_next_token, request.vocab_size))
                return failure(request, "discarded request has invalid backup token");
            tx.valid_sampled_count = 0;
            tx.accepted_speculative_prefix = 0;
            tx.rejected_token_count = draft_count + 1;
            tx.token_index_to_sample = -1;
            tx.next_condition_token = request.backup_next_token;
            tx.rejected_draft_tokens = request.draft_tokens;
            return tx;
        }

        tx.valid_sampled_count =
            contiguousValidPrefix(request.sampled_tokens, request.vocab_size);
        if (hasValidTokenAfterPrefix(
                request.sampled_tokens,
                tx.valid_sampled_count,
                request.vocab_size))
        {
            return failure(request, "sampled token row has a non-contiguous valid prefix");
        }
        if (tx.valid_sampled_count == 0)
        {
            if (!tokenInVocab(request.backup_next_token, request.vocab_size))
                return failure(request, "request has no valid sampled token and no valid backup token");
            tx.next_condition_token = request.backup_next_token;
        }
        else
        {
            tx.next_condition_token =
                request.sampled_tokens[static_cast<size_t>(tx.valid_sampled_count - 1)];
        }

        tx.accepted_speculative_prefix = acceptedDraftPrefix(
            request.draft_tokens,
            request.sampled_tokens,
            tx.valid_sampled_count);
        tx.rejected_token_count =
            draft_count + 1 - tx.valid_sampled_count;
        tx.token_index_to_sample = tx.valid_sampled_count - 1;

        tx.committed_output_tokens.assign(
            request.sampled_tokens.begin(),
            request.sampled_tokens.begin() + tx.valid_sampled_count);
        tx.rejected_draft_tokens.assign(
            request.draft_tokens.begin() + tx.accepted_speculative_prefix,
            request.draft_tokens.end());
        return tx;
    }

    MTPSpecDecodeRequest buildMTPSpecDecodeRequestFromVerifierOutput(
        int request_id,
        int vocab_size,
        const std::vector<int32_t> &draft_tokens,
        const std::vector<int32_t> &committed_output_tokens,
        std::optional<int32_t> bonus_ready_token,
        bool all_drafts_accepted,
        bool stopped_on_output)
    {
        MTPSpecDecodeRequest request;
        request.request_id = request_id;
        request.vocab_size = vocab_size;
        request.draft_tokens = draft_tokens;
        request.sampled_tokens = committed_output_tokens;
        if (all_drafts_accepted &&
            !stopped_on_output &&
            bonus_ready_token.has_value())
        {
            request.sampled_tokens.push_back(*bonus_ready_token);
        }
        const size_t target_query_len = draft_tokens.size() + 1;
        if (request.sampled_tokens.size() < target_query_len)
        {
            request.sampled_tokens.resize(
                target_query_len,
                kMTPSpecDecodeInvalidToken);
        }
        return request;
    }

    MTPSpecDecodeTransaction buildMTPSpecDecodeTransactionFromVerifierOutput(
        int request_id,
        int vocab_size,
        const std::vector<int32_t> &draft_tokens,
        const std::vector<int32_t> &committed_output_tokens,
        std::optional<int32_t> bonus_ready_token,
        bool all_drafts_accepted,
        bool stopped_on_output)
    {
        return buildMTPSpecDecodeTransaction(
            buildMTPSpecDecodeRequestFromVerifierOutput(
                request_id,
                vocab_size,
                draft_tokens,
                committed_output_tokens,
                bonus_ready_token,
                all_drafts_accepted,
                stopped_on_output));
    }

    MTPSpecDecodeBatchSummary buildMTPSpecDecodeBatchSummary(
        const std::vector<MTPSpecDecodeRequest> &requests)
    {
        MTPSpecDecodeBatchSummary summary;
        summary.ok = true;
        summary.transactions.reserve(requests.size());

        for (const MTPSpecDecodeRequest &request : requests)
        {
            MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);
            if (!tx.ok)
            {
                summary.ok = false;
                summary.error = tx.error;
            }
            summary.total_target_query_tokens += tx.target_query_len;
            summary.total_valid_sampled_tokens += tx.valid_sampled_count;
            summary.total_rejected_tokens += tx.rejected_token_count;
            summary.transactions.push_back(std::move(tx));
        }
        return summary;
    }

} // namespace llaminar2
