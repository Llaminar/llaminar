/**
 * @file MTPDecodeCatchup.cpp
 * @brief Shared decode-equivalent MTP catch-up implementation.
 */

#include "MTPDecodeCatchup.h"

#include "../local_execution/orchestrators/IInferenceRunner.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
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
            return equivalenceFailure(msg.str());
        }
        if (oracle.shifted_commit_count != candidate.shifted_commit_count)
            return equivalenceFailure("shifted MTP commit count mismatch");

        return equivalenceSuccess();
    }

    MTPDecodeCatchupGreedyResult buildMTPDecodeCatchupGreedyResultFromVerifierRows(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<int32_t> &sampled_verifier_tokens)
    {
        MTPDecodeCatchupGreedyResult result;
        result.accepted_tokens.reserve(request.draft_tokens.size());
        result.verifier_tokens.reserve(request.draft_tokens.size());
        result.main_forward_token_count =
            static_cast<int>(request.draft_tokens.size());

        auto fail = [&](std::string reason) -> MTPDecodeCatchupGreedyResult
        {
            result.ok = false;
            result.error = std::move(reason);
            return result;
        };

        if (request.draft_tokens.empty())
            return fail("MTP verifier-row catch-up received no draft tokens");
        if (sampled_verifier_tokens.size() < request.draft_tokens.size())
            return fail("MTP verifier-row catch-up received too few verifier rows");

        result.accepted_tokens.push_back(request.draft_tokens.front());
        result.target_verifier_state_commit_count = 1;

        if (tokenIsStop(request.stop_tokens, request.draft_tokens.front()))
        {
            result.stopped_on_output = true;
            result.shifted_commit_count =
                static_cast<int>(result.accepted_tokens.size());
            result.ok = true;
            return result;
        }

        for (int draft_idx = 1;
             draft_idx < static_cast<int>(request.draft_tokens.size());
             ++draft_idx)
        {
            const int row = draft_idx - 1;
            const int32_t verified_token =
                sampled_verifier_tokens[static_cast<size_t>(row)];
            if (verified_token < 0)
                return fail("MTP verifier-row catch-up encountered an invalid verifier token");

            const int32_t draft_token =
                request.draft_tokens[static_cast<size_t>(draft_idx)];
            result.verifier_tokens.push_back(verified_token);

            const bool accepted = verified_token == draft_token;
            const int32_t output_token = accepted ? draft_token : verified_token;
            if (accepted)
            {
                ++result.accepted_speculative_prefix;
                result.accepted_tokens.push_back(output_token);
                result.target_verifier_state_commit_count =
                    static_cast<int>(result.accepted_tokens.size());
            }
            else
            {
                result.all_speculative_accepted = false;
                result.rejected_verified_token = verified_token;
                result.accepted_tokens.push_back(output_token);
                // The correction token was sampled from this row but has not
                // been forwarded. State is only valid through the accepted
                // verifier-input prefix; a caller that wants decode-equivalent
                // state must replay the correction suffix.
                result.target_verifier_state_commit_count =
                    std::max(1, static_cast<int>(result.accepted_tokens.size()) - 1);
            }

            if (tokenIsStop(request.stop_tokens, output_token))
            {
                result.stopped_on_output = true;
                break;
            }
            if (!accepted)
                break;
        }

        if (result.all_speculative_accepted && !result.stopped_on_output)
        {
            const int terminal_row =
                static_cast<int>(request.draft_tokens.size()) - 1;
            const int32_t ready =
                sampled_verifier_tokens[static_cast<size_t>(terminal_row)];
            if (ready < 0)
                return fail("MTP verifier-row catch-up accepted all drafts without a ready token");
            result.ready_token = ready;
            result.target_verifier_state_commit_count =
                static_cast<int>(request.draft_tokens.size());
        }

        result.shifted_commit_count =
            static_cast<int>(result.accepted_tokens.size());
        result.ok = true;
        return result;
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
                sampled = sample_after_forward();
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
