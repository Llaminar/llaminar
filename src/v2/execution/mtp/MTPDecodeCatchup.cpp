/**
 * @file MTPDecodeCatchup.cpp
 * @brief Shared decode-equivalent MTP catch-up implementation.
 */

#include "MTPDecodeCatchup.h"

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool tokenIsStop(
            const std::vector<int32_t> &stop_tokens,
            int32_t token)
        {
            return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
                   stop_tokens.end();
        }

        std::string joinTokens(const std::vector<int32_t> &tokens)
        {
            std::string out;
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i > 0)
                    out += ",";
                out += std::to_string(tokens[i]);
            }
            return out;
        }

        MTPDecodeCatchupGreedyEquivalence equivalenceFailure(
            std::string reason)
        {
            MTPDecodeCatchupGreedyEquivalence result;
            result.ok = false;
            result.error = std::move(reason);
            return result;
        }

        MTPDecodeCatchupGreedyEquivalence equivalenceSuccess()
        {
            MTPDecodeCatchupGreedyEquivalence result;
            result.ok = true;
            return result;
        }

        MTPDecodeCatchupGreedyEquivalence compareTokenVector(
            const char *name,
            const std::vector<int32_t> &oracle,
            const std::vector<int32_t> &candidate)
        {
            if (oracle == candidate)
                return equivalenceSuccess();
            std::ostringstream msg;
            msg << name << " mismatch: oracle=[" << joinTokens(oracle)
                << "], candidate=[" << joinTokens(candidate) << "]";
            return equivalenceFailure(msg.str());
        }

    } // namespace

    MTPDecodeCatchupGreedyEquivalence compareMTPDecodeCatchupGreedyResults(
        const MTPDecodeCatchupGreedyResult &oracle,
        const MTPDecodeCatchupGreedyResult &candidate)
    {
        if (!oracle.ok)
            return equivalenceFailure(
                std::string("oracle catch-up failed: ") + oracle.error);
        if (!candidate.ok)
            return equivalenceFailure(
                std::string("candidate catch-up failed: ") + candidate.error);

        if (auto eq = compareTokenVector(
                "accepted tokens",
                oracle.accepted_tokens,
                candidate.accepted_tokens);
            !eq.ok)
        {
            return eq;
        }
        if (auto eq = compareTokenVector(
                "verifier tokens",
                oracle.verifier_tokens,
                candidate.verifier_tokens);
            !eq.ok)
        {
            return eq;
        }
        if (oracle.all_speculative_accepted !=
            candidate.all_speculative_accepted)
        {
            return equivalenceFailure(
                "all-speculative-accepted flag mismatch");
        }
        if (oracle.stopped_on_output != candidate.stopped_on_output)
            return equivalenceFailure("stopped-on-output flag mismatch");
        if (oracle.accepted_speculative_prefix !=
            candidate.accepted_speculative_prefix)
        {
            return equivalenceFailure(
                "accepted speculative prefix mismatch");
        }
        if (oracle.rejected_verified_token !=
            candidate.rejected_verified_token)
        {
            return equivalenceFailure("rejected verified token mismatch");
        }
        if (oracle.ready_token != candidate.ready_token)
        {
            std::ostringstream msg;
            msg << "ready token mismatch: oracle=" << oracle.ready_token
                << ", candidate=" << candidate.ready_token
                << ", accepted=[" << joinTokens(oracle.accepted_tokens)
                << "], verifier=[" << joinTokens(oracle.verifier_tokens)
                << "], oracle_forwards=" << oracle.main_forward_token_count
                << ", candidate_forwards=" << candidate.main_forward_token_count
                << ", candidate_state_commit_count="
                << candidate.target_verifier_state_commit_count;
            if (!candidate.debug_trace.empty())
                msg << ", candidate_debug={" << candidate.debug_trace << "}";
            return equivalenceFailure(msg.str());
        }
        if (oracle.shifted_commit_count != candidate.shifted_commit_count)
            return equivalenceFailure("shifted MTP commit count mismatch");

        return equivalenceSuccess();
    }

    MTPDecodeCatchupGreedyResult buildAllPositionMTPDecodeCatchupGreedyResult(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<int32_t> &sampled_verifier_rows,
        std::optional<int32_t> correction_replay_ready_token)
    {
        MTPDecodeCatchupGreedyResult result;
        result.accepted_tokens.reserve(request.draft_tokens.size());
        result.verifier_tokens.reserve(request.draft_tokens.size());

        auto fail = [&](std::string reason) -> MTPDecodeCatchupGreedyResult
        {
            result.ok = false;
            result.error = std::move(reason);
            return result;
        };

        if (request.draft_tokens.empty())
            return fail("all-position MTP verifier received no draft tokens");
        if (sampled_verifier_rows.size() != request.draft_tokens.size())
        {
            std::ostringstream msg;
            msg << "all-position MTP verifier row count mismatch: rows="
                << sampled_verifier_rows.size()
                << ", draft_tokens=" << request.draft_tokens.size();
            return fail(msg.str());
        }
        if (correction_replay_ready_token.has_value() &&
            *correction_replay_ready_token < 0)
        {
            return fail("all-position MTP verifier received an invalid correction replay ready token");
        }

        result.main_forward_token_count =
            static_cast<int>(sampled_verifier_rows.size());

        const int32_t first_token = request.draft_tokens.front();
        result.accepted_tokens.push_back(first_token);
        result.target_verifier_state_commit_count = 1;

        if (tokenIsStop(request.stop_tokens, first_token))
        {
            result.stopped_on_output = true;
            result.shifted_commit_count =
                static_cast<int>(result.accepted_tokens.size());
            result.ok = true;
            result.debug_trace =
                "first token stopped output; published_state_count=1";
            return result;
        }

        for (int draft_idx = 1;
             draft_idx < static_cast<int>(request.draft_tokens.size());
             ++draft_idx)
        {
            const int row = draft_idx - 1;
            const int32_t verifier_token =
                sampled_verifier_rows[static_cast<size_t>(row)];
            if (verifier_token < 0)
            {
                std::ostringstream msg;
                msg << "all-position MTP verifier sampled an invalid token at row "
                    << row;
                return fail(msg.str());
            }

            const int32_t draft_token =
                request.draft_tokens[static_cast<size_t>(draft_idx)];
            result.verifier_tokens.push_back(verifier_token);

            const bool accepted = verifier_token == draft_token;
            const int32_t output_token = accepted ? draft_token : verifier_token;
            if (accepted)
            {
                ++result.accepted_speculative_prefix;
            }
            else
            {
                result.all_speculative_accepted = false;
                result.rejected_verified_token = verifier_token;
            }

            result.accepted_tokens.push_back(output_token);

            if (tokenIsStop(request.stop_tokens, output_token))
            {
                result.stopped_on_output = true;
                break;
            }
            if (!accepted)
                break;
        }

        result.target_verifier_state_commit_count =
            std::min<int>(
                static_cast<int>(request.draft_tokens.size()),
                result.accepted_speculative_prefix + 1);

        if (!result.stopped_on_output)
        {
            if (result.all_speculative_accepted)
            {
                const int32_t ready =
                    sampled_verifier_rows.back();
                if (ready < 0)
                    return fail("all-position MTP verifier sampled an invalid bonus ready token");
                result.ready_token = ready;
            }
            else if (correction_replay_ready_token.has_value())
            {
                result.ready_token = *correction_replay_ready_token;
                ++result.main_forward_token_count;
            }
        }

        result.shifted_commit_count =
            static_cast<int>(result.accepted_tokens.size());
        result.ok = true;
        std::ostringstream trace;
        trace << "all_position_rows=" << sampled_verifier_rows.size()
              << ", accepted_tokens=[" << joinTokens(result.accepted_tokens)
              << "], verifier_tokens=[" << joinTokens(result.verifier_tokens)
              << "], publish_state_count="
              << result.target_verifier_state_commit_count
              << ", correction_ready="
              << (correction_replay_ready_token.has_value()
                      ? std::to_string(*correction_replay_ready_token)
                      : std::string("none"));
        result.debug_trace = trace.str();
        return result;
    }

    MTPDecodeCatchupGreedyBatchResult buildAllPositionMTPDecodeCatchupGreedyBatchResult(
        const std::vector<MTPDecodeCatchupGreedyRequest> &requests,
        const std::vector<int32_t> &sampled_verifier_rows,
        const std::vector<std::optional<int32_t>> &correction_replay_ready_tokens)
    {
        MTPDecodeCatchupGreedyBatchResult batch;

        auto fail = [&](std::string reason) -> MTPDecodeCatchupGreedyBatchResult
        {
            batch.ok = false;
            batch.error = std::move(reason);
            return batch;
        };

        if (requests.empty())
            return fail("all-position MTP verifier batch has no requests");
        if (!correction_replay_ready_tokens.empty() &&
            correction_replay_ready_tokens.size() != requests.size())
        {
            return fail("all-position MTP verifier batch correction-ready vector does not match request count");
        }

        batch.results.reserve(requests.size());
        size_t compact_row_cursor = 0;
        for (size_t request_index = 0; request_index < requests.size(); ++request_index)
        {
            const MTPDecodeCatchupGreedyRequest &request = requests[request_index];
            const size_t row_count = request.draft_tokens.size();
            if (row_count == 0)
            {
                std::ostringstream msg;
                msg << "request " << request_index << ": no draft tokens";
                return fail(msg.str());
            }
            if (compact_row_cursor + row_count > sampled_verifier_rows.size())
            {
                std::ostringstream msg;
                msg << "request " << request_index
                    << ": compact verifier rows end past sampled-row vector";
                return fail(msg.str());
            }

            std::vector<int32_t> request_rows(
                sampled_verifier_rows.begin() + static_cast<std::ptrdiff_t>(compact_row_cursor),
                sampled_verifier_rows.begin() + static_cast<std::ptrdiff_t>(compact_row_cursor + row_count));
            const std::optional<int32_t> correction_ready =
                correction_replay_ready_tokens.empty()
                    ? std::optional<int32_t>{}
                    : correction_replay_ready_tokens[request_index];

            MTPDecodeCatchupGreedyResult result =
                buildAllPositionMTPDecodeCatchupGreedyResult(
                    request,
                    request_rows,
                    correction_ready);
            if (!result.ok)
            {
                std::ostringstream msg;
                msg << "request " << request_index << ": " << result.error;
                return fail(msg.str());
            }
            batch.results.push_back(std::move(result));
            compact_row_cursor += row_count;
        }

        if (compact_row_cursor != sampled_verifier_rows.size())
        {
            std::ostringstream msg;
            msg << "all-position MTP verifier batch has "
                << (sampled_verifier_rows.size() - compact_row_cursor)
                << " unused sampled rows";
            return fail(msg.str());
        }

        batch.ok = true;
        return batch;
    }

    MTPDecodeCatchupGreedyResult runSharedStepwiseMTPDecodeCatchupGreedy(
        IInferenceRunner &runner,
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDecodeCatchupGreedySampler &sample_after_forward)
    {
        MTPDecodeCatchupGreedyResult result;
        result.accepted_tokens.reserve(request.draft_tokens.size() + 1);
        result.verifier_tokens.reserve(request.draft_tokens.size());
        const std::string implementation =
            request.implementation_name.empty()
                ? std::string("shared_stepwise")
                : request.implementation_name;

        auto fail = [&](std::string reason) -> MTPDecodeCatchupGreedyResult
        {
            result.ok = false;
            result.error = std::move(reason);
            PerfStatsCollector::addCounter(
                "mtp",
                "decode_equivalent_catchup_failures",
                1.0,
                "decode",
                {},
                {{"implementation", implementation},
                 {"reason", result.error}});
            return result;
        };

        if (request.draft_tokens.empty())
            return fail("MTP decode-equivalent catch-up received no draft tokens");
        if (!sample_after_forward)
            return fail("MTP decode-equivalent catch-up received no sampler callback");

        auto forward_one_and_sample = [&](int32_t token) -> int32_t
        {
            int forward_token = static_cast<int>(token);
            bool ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "decode_equivalent_catchup_forward_one",
                    "decode",
                    {},
                    {{"implementation", implementation}});
                ok = runner.forward(&forward_token, 1);
            }
            if (!ok)
                return -1;

            ++result.main_forward_token_count;

            int32_t sampled = -1;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "decode_equivalent_catchup_sample_one",
                    "decode",
                    {},
                    {{"implementation", implementation}});
                sampled = sample_after_forward(token);
            }
            return sampled;
        };

        auto commit_shifted_before_forward = [&](int32_t token, int token_index) -> bool
        {
            bool ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "decode_equivalent_catchup_shifted_commit",
                    "decode",
                    {},
                    {{"implementation", implementation}});
                ok = runner.commitMTPShiftedRowFromCurrentTerminalHidden(
                    token,
                    token_index,
                    request.allow_speculative_discard,
                    request.base_sidecar_position);
            }
            if (ok)
                ++result.shifted_commit_count;
            return ok;
        };

        const int32_t first_token = request.draft_tokens.front();
        result.accepted_tokens.push_back(first_token);

        if (!commit_shifted_before_forward(first_token, 0))
            return fail("MTP decode-equivalent catch-up initial shifted-cache commit failed");

        int32_t verifier_sample = forward_one_and_sample(first_token);
        if (verifier_sample < 0)
            return fail("MTP decode-equivalent catch-up failed to forward/sample first token");

        for (int draft_idx = 1;
             draft_idx < static_cast<int>(request.draft_tokens.size());
             ++draft_idx)
        {
            const int32_t draft_token =
                request.draft_tokens[static_cast<size_t>(draft_idx)];
            result.verifier_tokens.push_back(verifier_sample);
            PerfStatsCollector::addCounter(
                "mtp",
                "greedy_verifier_token",
                1.0,
                "decode",
                {},
                {{"row", std::to_string(draft_idx - 1)},
                 {"draft_token", std::to_string(draft_token)},
                 {"verified_token", std::to_string(verifier_sample)},
                 {"verifier_path", request.verifier_path},
                 {"implementation", implementation}});

            const bool accepted = verifier_sample == draft_token;
            const int32_t output_token = accepted ? draft_token : verifier_sample;
            if (accepted)
            {
                ++result.accepted_speculative_prefix;
            }
            else
            {
                result.all_speculative_accepted = false;
                result.rejected_verified_token = verifier_sample;
            }

            result.accepted_tokens.push_back(output_token);
            const int token_index =
                static_cast<int>(result.accepted_tokens.size()) - 1;
            if (!commit_shifted_before_forward(output_token, token_index))
                return fail("MTP decode-equivalent catch-up shifted-cache commit failed");

            verifier_sample = forward_one_and_sample(output_token);
            if (verifier_sample < 0)
                return fail("MTP decode-equivalent catch-up failed while forwarding accepted output");

            if (tokenIsStop(request.stop_tokens, output_token))
            {
                result.stopped_on_output = true;
                break;
            }
            if (!accepted)
                break;
        }

        if (!result.stopped_on_output)
            result.ready_token = verifier_sample;

        result.ok = true;
        PerfStatsCollector::addCounter(
            "mtp",
            "decode_equivalent_catchup_runs",
            1.0,
            "decode",
            {},
            {{"implementation", implementation},
             {"draft_tokens", joinTokens(request.draft_tokens)},
             {"accepted_tokens", joinTokens(result.accepted_tokens)},
             {"verifier_tokens", joinTokens(result.verifier_tokens)},
             {"accepted_speculative_prefix",
              std::to_string(result.accepted_speculative_prefix)},
             {"all_speculative_accepted",
              result.all_speculative_accepted ? "true" : "false"},
             {"stopped_on_output", result.stopped_on_output ? "true" : "false"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "decode_equivalent_catchup_forward_tokens",
            static_cast<double>(result.main_forward_token_count),
            "decode",
            {},
            {{"implementation", implementation}});
        return result;
    }

} // namespace llaminar2
