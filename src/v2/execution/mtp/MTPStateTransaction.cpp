/**
 * @file MTPStateTransaction.cpp
 * @brief Validates MTP decode state transitions and runtime-state snapshots.
 *
 * MTP publication, rollback, and prefix restore all mutate long-lived
 * inference state.  The comparisons in this file therefore favor explicit,
 * diagnostic-rich failures over tolerant fallbacks so request-boundary bugs are
 * reported at the owning state surface: positions, KV caches, GDN state, or
 * terminal hidden/logits.
 */

#include "MTPStateTransaction.h"

#include "tensors/FP16Utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Decode one canonical floating KV segment into FP32 values.
         *
         * Quantized KV formats intentionally fail here until their canonical
         * logical-block codec exposes an equally explicit numerical decoder.
         * A placement-aware proof must never reinterpret unknown bytes or
         * silently downgrade to hashes after those hashes already differed.
         *
         * @param payload Canonical native-precision bytes from a KV probe.
         * @param precision Declared precision of the enclosing cache payload.
         * @param decoded Receives every decoded scalar in logical-block order.
         * @param reason Receives an actionable failure diagnostic.
         * @return True only when the complete payload was decoded.
         */
        bool decodeFloatingKVPayload(
            const std::vector<uint8_t> &payload,
            ActivationPrecision precision,
            std::vector<float> *decoded,
            std::string *reason)
        {
            if (!decoded || !reason)
                return false;
            decoded->clear();
            reason->clear();
            if (payload.empty())
            {
                *reason = "retained KV segment payload is empty";
                return false;
            }

            if (precision == ActivationPrecision::FP32)
            {
                if (payload.size() % sizeof(float) != 0u)
                {
                    *reason = "FP32 KV segment has malformed byte geometry";
                    return false;
                }
                decoded->resize(payload.size() / sizeof(float));
                std::memcpy(
                    decoded->data(),
                    payload.data(),
                    payload.size());
                return true;
            }

            if (precision == ActivationPrecision::FP16 ||
                precision == ActivationPrecision::BF16)
            {
                if (payload.size() % sizeof(uint16_t) != 0u)
                {
                    *reason =
                        std::string(activationPrecisionToString(precision)) +
                        " KV segment has malformed byte geometry";
                    return false;
                }
                decoded->resize(payload.size() / sizeof(uint16_t));
                for (size_t index = 0; index < decoded->size(); ++index)
                {
                    uint16_t word = 0;
                    std::memcpy(
                        &word,
                        payload.data() + index * sizeof(uint16_t),
                        sizeof(word));
                    if (precision == ActivationPrecision::FP16)
                    {
                        (*decoded)[index] = fp16_to_fp32(word);
                    }
                    else
                    {
                        const uint32_t fp32_bits =
                            static_cast<uint32_t>(word) << 16u;
                        std::memcpy(
                            &(*decoded)[index],
                            &fp32_bits,
                            sizeof(fp32_bits));
                    }
                }
                return true;
            }

            *reason =
                std::string("placement-aware KV comparison has no canonical decoder for ") +
                activationPrecisionToString(precision);
            return false;
        }
    } // namespace

    MTPStateValidationResult MTPStateValidationResult::success()
    {
        return {true, {}};
    }

    MTPStateValidationResult MTPStateValidationResult::failure(std::string reason)
    {
        return {false, std::move(reason)};
    }

    int expectedShiftedMTPTokens(int logical_tokens)
    {
        return std::max(0, logical_tokens - 1);
    }

    MTPVisibleStateCommitPlan planMTPVisibleStateCommit(
        int verifier_input_rows,
        int emitted_token_start_index,
        int remaining_output_budget)
    {
        MTPVisibleStateCommitPlan plan{
            .verifier_input_rows = verifier_input_rows,
            .emitted_token_start_index = emitted_token_start_index,
            .remaining_output_budget = remaining_output_budget};

        if (verifier_input_rows <= 0)
        {
            plan.reason = "visible MTP commit planning requires verifier input rows";
            return plan;
        }
        if (emitted_token_start_index < 0 ||
            emitted_token_start_index > verifier_input_rows)
        {
            plan.reason =
                "visible MTP commit planning received an invalid emitted-token start";
            return plan;
        }
        if (remaining_output_budget < 0)
        {
            plan.reason =
                "visible MTP commit planning received a negative output budget";
            return plan;
        }

        plan.max_state_commit_rows = verifier_input_rows;
        const int maximum_newly_emitted_tokens =
            verifier_input_rows - emitted_token_start_index;
        if (remaining_output_budget > 0 &&
            remaining_output_budget <= maximum_newly_emitted_tokens)
        {
            /*
             * Serial decode processes every previously emitted condition row and
             * every newly emitted token except the final visible token.  The
             * latter remains pending so a later decode step can consume it once.
             */
            plan.max_state_commit_rows = std::clamp(
                remaining_output_budget + emitted_token_start_index - 1,
                0,
                verifier_input_rows);
            plan.response_boundary_clipped =
                plan.max_state_commit_rows < verifier_input_rows;
        }

        plan.ok = true;
        return plan;
    }

    MTPShiftedReplayRowPlan planMTPShiftedReplayRow(
        int main_cached_tokens,
        int shifted_cached_tokens)
    {
        if (main_cached_tokens < 0 || shifted_cached_tokens < 0)
        {
            return {
                .ok = false,
                .reason = "main and shifted cached-token counts must be nonnegative"};
        }
        if (shifted_cached_tokens == main_cached_tokens)
        {
            return {
                .ok = true,
                .action = MTPShiftedReplayRowAction::ReuseResidentRow,
                .append_position_offset = -1};
        }
        if (shifted_cached_tokens + 1 == main_cached_tokens)
        {
            return {
                .ok = true,
                .action = MTPShiftedReplayRowAction::AppendRow,
                .append_position_offset = main_cached_tokens};
        }

        std::ostringstream reason;
        reason << "shifted MTP replay state must equal main state or trail by one"
               << ": main=" << main_cached_tokens
               << " shifted=" << shifted_cached_tokens;
        return {
            .ok = false,
            .reason = reason.str()};
    }

    PrefixStateSnapshot makeLogicalMTPVerifierBaseSnapshot(int cached_tokens)
    {
        PrefixStateSnapshot snapshot;
        if (cached_tokens < 0)
        {
            snapshot.valid = false;
            snapshot.logical_checkpoint = true;
            snapshot.provenance = PrefixStateProvenance::LogicalCheckpoint;
            return snapshot;
        }

        snapshot.valid = true;
        snapshot.logical_checkpoint = true;
        snapshot.provenance = PrefixStateProvenance::LogicalCheckpoint;
        snapshot.cached_tokens = cached_tokens;
        snapshot.mtp_cached_tokens = {expectedShiftedMTPTokens(cached_tokens)};
        return snapshot;
    }

    MTPStateValidationResult validateCommittedMTPDecodeState(
        const MTPDecodeStateStamp &state,
        const MTPCommitValidationOptions &options)
    {
        if (!state.valid)
            return MTPStateValidationResult::failure("state is invalid");
        if (state.logical_tokens < 0 ||
            state.main_kv_tokens < 0 ||
            state.shifted_mtp_kv_tokens < 0 ||
            state.position < 0)
        {
            return MTPStateValidationResult::failure("state contains negative token or position counts");
        }
        if (state.main_kv_tokens != state.logical_tokens)
        {
            return MTPStateValidationResult::failure("main KV token count does not match logical token count");
        }
        if (state.position != state.logical_tokens)
        {
            return MTPStateValidationResult::failure("decode position does not match logical token count");
        }
        if (options.require_shifted_mtp_kv &&
            state.shifted_mtp_kv_tokens != expectedShiftedMTPTokens(state.logical_tokens))
        {
            return MTPStateValidationResult::failure("shifted MTP KV token count does not match logical token count");
        }
        if (options.require_decode_equivalent_source && !state.decodeEquivalent())
        {
            return MTPStateValidationResult::failure(
                std::string("state provenance is not decode-equivalent: ") +
                toString(state.provenance));
        }
        if (options.require_terminal_hidden && !state.has_terminal_hidden)
            return MTPStateValidationResult::failure("terminal hidden is missing");
        if (options.require_terminal_logits && !state.has_terminal_logits)
            return MTPStateValidationResult::failure("terminal logits are missing");
        if (options.require_ready_token && !state.has_ready_token)
            return MTPStateValidationResult::failure("ready token is missing");
        return MTPStateValidationResult::success();
    }

    MTPStateValidationResult validateAtomicMTPCommit(
        const MTPDecodeStateStamp &base,
        const MTPDecodeStateStamp &committed,
        int emitted_tokens,
        PrefixStateProvenance verifier_source,
        const MTPCommitValidationOptions &options)
    {
        if (emitted_tokens <= 0)
            return MTPStateValidationResult::failure("atomic MTP commit emitted no tokens");
        MTPCommitValidationOptions base_options = options;
        base_options.require_shifted_mtp_kv =
            options.require_base_shifted_mtp_kv;
        MTPCommitValidationOptions committed_options = options;
        committed_options.require_shifted_mtp_kv =
            options.require_committed_shifted_mtp_kv;

        auto base_result = validateCommittedMTPDecodeState(base, base_options);
        if (!base_result)
            return MTPStateValidationResult::failure("base state failed validation: " + base_result.reason);
        auto committed_result = validateCommittedMTPDecodeState(
            committed,
            committed_options);
        if (!committed_result)
            return MTPStateValidationResult::failure("committed state failed validation: " + committed_result.reason);
        if (options.require_decode_equivalent_source && !isDecodeEquivalent(verifier_source))
        {
            return MTPStateValidationResult::failure(
                std::string("verifier source is not decode-equivalent: ") +
                toString(verifier_source));
        }
        if (committed.logical_tokens != base.logical_tokens + emitted_tokens)
        {
            return MTPStateValidationResult::failure("committed logical token count does not equal base plus emitted tokens");
        }
        if (committed.main_kv_tokens < base.main_kv_tokens ||
            committed.shifted_mtp_kv_tokens < base.shifted_mtp_kv_tokens)
        {
            return MTPStateValidationResult::failure("committed state moved KV token counts backwards");
        }
        return MTPStateValidationResult::success();
    }

    MTPStateValidationResult compareMTPRuntimeStateSnapshots(
        const PrefixRuntimeStateSnapshot &oracle,
        const PrefixRuntimeStateSnapshot &candidate,
        const MTPRuntimeSnapshotComparisonOptions &options)
    {
        MTPStateValidationResult::TerminalPayloadNumericalEvidence
            terminal_hidden_numerical;
        MTPStateValidationResult::TerminalPayloadNumericalEvidence
            terminal_logits_numerical;
        MTPStateValidationResult::MainKVNumericalEvidence
            main_kv_numerical;
        MTPStateValidationResult::GDNStateNumericalEvidence
            gdn_numerical;
        auto mismatch = [&](std::string reason)
        {
            MTPStateValidationResult result =
                MTPStateValidationResult::failure(std::move(reason));
            result.terminal_hidden_numerical = terminal_hidden_numerical;
            result.terminal_logits_numerical = terminal_logits_numerical;
            result.main_kv_numerical = main_kv_numerical;
            result.gdn_numerical = gdn_numerical;
            return result;
        };

        if (oracle.initialized != candidate.initialized)
            return mismatch("initialized flag mismatch");
        if (!oracle.initialized)
            return MTPStateValidationResult::success();
        auto format_values = [](const std::vector<int> &values)
        {
            std::ostringstream message;
            message << "[";
            for (size_t index = 0; index < values.size(); ++index)
            {
                if (index > 0)
                    message << ",";
                message << values[index];
            }
            message << "]";
            return message.str();
        };

        if (oracle.current_position != candidate.current_position)
        {
            std::ostringstream message;
            message << "current position mismatch: oracle="
                    << oracle.current_position
                    << " candidate=" << candidate.current_position;
            return mismatch(message.str());
        }
        if (oracle.positions != candidate.positions)
        {
            return mismatch(
                "per-sequence position vector mismatch: oracle=" +
                format_values(oracle.positions) +
                " candidate=" + format_values(candidate.positions));
        }
        if (oracle.sequence_lengths != candidate.sequence_lengths)
        {
            return mismatch(
                "per-sequence length vector mismatch: oracle=" +
                format_values(oracle.sequence_lengths) +
                " candidate=" + format_values(candidate.sequence_lengths));
        }
        /*
         * Do not short-circuit on aggregate cached-token totals here. The
         * per-cache comparison below reports the first divergent layer,
         * sequence, count, and ring head, which is the actionable ownership
         * boundary when a device publication updates only part of a cache.
         */
        if (oracle.has_hidden != candidate.has_hidden)
            return mismatch("terminal hidden availability mismatch");
        if (oracle.has_logits != candidate.has_logits)
            return mismatch("terminal logits availability mismatch");
        auto compare_terminal_payload =
            [&](const char *label,
                bool oracle_hash_available,
                bool candidate_hash_available,
                size_t oracle_bytes,
                size_t candidate_bytes,
                uint64_t oracle_hash,
                uint64_t candidate_hash,
                const std::vector<float> &oracle_values,
                const std::vector<float> &candidate_values,
                MTPTerminalPayloadComparisonPolicy policy,
                double minimum_cosine,
                MTPStateValidationResult::TerminalPayloadNumericalEvidence
                    *numerical) -> MTPStateValidationResult
        {
            if (oracle_hash_available != candidate_hash_available)
            {
                return mismatch(
                    std::string("terminal ") + label +
                    " hash availability mismatch");
            }
            if (!oracle_hash_available)
                return MTPStateValidationResult::success();
            if (oracle_bytes != candidate_bytes)
            {
                std::ostringstream message;
                message << "terminal " << label << " payload mismatch: bytes="
                        << oracle_bytes << "/" << candidate_bytes << " hash="
                        << oracle_hash << "/" << candidate_hash;
                return mismatch(message.str());
            }
            if (oracle_hash == candidate_hash)
                return MTPStateValidationResult::success();

            const bool placement_aware =
                policy == MTPTerminalPayloadComparisonPolicy::
                              ExactUnlessMoEPlacementChanged;
            if (!placement_aware ||
                oracle.moe_runtime_movement_epoch ==
                    candidate.moe_runtime_movement_epoch)
            {
                std::ostringstream message;
                message << "terminal " << label << " payload mismatch: bytes="
                        << oracle_bytes << "/" << candidate_bytes << " hash="
                        << oracle_hash << "/" << candidate_hash
                        << " moe_movement_epoch="
                        << oracle.moe_runtime_movement_epoch << "/"
                        << candidate.moe_runtime_movement_epoch;
                return mismatch(message.str());
            }
            if (!std::isfinite(minimum_cosine) || minimum_cosine < -1.0 ||
                minimum_cosine > 1.0)
            {
                return mismatch(
                    std::string("terminal ") + label +
                    " numerical comparison has an invalid cosine threshold");
            }
            if (oracle_bytes % sizeof(float) != 0u)
            {
                return mismatch(
                    std::string("terminal ") + label +
                    " numerical comparison requires FP32 byte geometry");
            }

            const size_t expected_elements = oracle_bytes / sizeof(float);
            if (oracle_values.size() != expected_elements ||
                candidate_values.size() != expected_elements)
            {
                std::ostringstream message;
                message << "terminal " << label
                        << " placement-aware comparison requires complete "
                           "retained values: expected="
                        << expected_elements << " actual="
                        << oracle_values.size() << "/"
                        << candidate_values.size();
                return mismatch(message.str());
            }

            double dot = 0.0;
            double oracle_norm_sq = 0.0;
            double candidate_norm_sq = 0.0;
            double difference_norm_sq = 0.0;
            double max_abs = 0.0;
            bool finite = true;
            for (size_t index = 0; index < expected_elements; ++index)
            {
                const double lhs = static_cast<double>(oracle_values[index]);
                const double rhs = static_cast<double>(candidate_values[index]);
                finite = finite && std::isfinite(lhs) && std::isfinite(rhs);
                const double difference = lhs - rhs;
                dot += lhs * rhs;
                oracle_norm_sq += lhs * lhs;
                candidate_norm_sq += rhs * rhs;
                difference_norm_sq += difference * difference;
                max_abs = std::max(max_abs, std::abs(difference));
            }

            const double norm_product =
                std::sqrt(oracle_norm_sq) * std::sqrt(candidate_norm_sq);
            const bool both_zero =
                oracle_norm_sq <= 1e-30 && candidate_norm_sq <= 1e-30;
            const double cosine =
                both_zero ? 1.0
                          : (norm_product > 1e-30 ? dot / norm_product : 0.0);
            const double relative_l2 =
                std::sqrt(difference_norm_sq) /
                std::max(1e-30, std::sqrt(oracle_norm_sq));
            *numerical = {
                .compared = true,
                .passed = finite && cosine >= minimum_cosine,
                .elements = expected_elements,
                .cosine = cosine,
                .relative_l2 = relative_l2,
                .max_abs = max_abs,
            };
            if (!numerical->passed)
            {
                std::ostringstream message;
                message << "terminal " << label
                        << " numerical mismatch after MoE placement change: "
                           "epochs="
                        << oracle.moe_runtime_movement_epoch << "/"
                        << candidate.moe_runtime_movement_epoch
                        << " elements=" << expected_elements
                        << " finite=" << (finite ? "true" : "false")
                        << " cosine=" << cosine
                        << " required_cosine=" << minimum_cosine
                        << " rel_l2=" << relative_l2
                        << " max_abs=" << max_abs;
                return mismatch(message.str());
            }
            return MTPStateValidationResult::success();
        };

        if (auto hidden_result = compare_terminal_payload(
                "hidden",
                oracle.terminal_hidden_hash_available,
                candidate.terminal_hidden_hash_available,
                oracle.terminal_hidden_bytes,
                candidate.terminal_hidden_bytes,
                oracle.terminal_hidden_hash,
                candidate.terminal_hidden_hash,
                oracle.terminal_hidden_values,
                candidate.terminal_hidden_values,
                options.terminal_hidden_policy,
                options.terminal_hidden_min_cosine,
                &terminal_hidden_numerical);
            !hidden_result)
        {
            return hidden_result;
        }
        if (auto logits_result = compare_terminal_payload(
                "logits",
                oracle.terminal_logits_hash_available,
                candidate.terminal_logits_hash_available,
                oracle.terminal_logits_bytes,
                candidate.terminal_logits_bytes,
                oracle.terminal_logits_hash,
                candidate.terminal_logits_hash,
                oracle.terminal_logits_values,
                candidate.terminal_logits_values,
                options.terminal_logits_policy,
                options.terminal_logits_min_cosine,
                &terminal_logits_numerical);
            !logits_result)
        {
            return logits_result;
        }
        if (oracle.gdn_layers.size() != candidate.gdn_layers.size())
            return mismatch("GDN layer count mismatch");

        auto compare_main_kv_suffix_payload =
            [&](const std::vector<uint8_t> &lhs_payload,
                const std::vector<uint8_t> &rhs_payload,
                ActivationPrecision precision,
                size_t cache_idx,
                int global_layer,
                int seq_idx,
                const char *kind) -> MTPStateValidationResult
        {
            if (!std::isfinite(options.main_kv_suffix_min_cosine) ||
                options.main_kv_suffix_min_cosine < -1.0 ||
                options.main_kv_suffix_min_cosine > 1.0)
            {
                return mismatch(
                    "placement-aware main KV comparison has an invalid cosine threshold");
            }

            std::vector<float> lhs_values;
            std::vector<float> rhs_values;
            std::string decode_reason;
            if (!decodeFloatingKVPayload(
                    lhs_payload,
                    precision,
                    &lhs_values,
                    &decode_reason))
            {
                return mismatch(
                    std::string("oracle ") + kind +
                    " suffix decode failed at cache " +
                    std::to_string(cache_idx) + " layer " +
                    std::to_string(global_layer) + " seq " +
                    std::to_string(seq_idx) + ": " + decode_reason);
            }
            if (!decodeFloatingKVPayload(
                    rhs_payload,
                    precision,
                    &rhs_values,
                    &decode_reason))
            {
                return mismatch(
                    std::string("candidate ") + kind +
                    " suffix decode failed at cache " +
                    std::to_string(cache_idx) + " layer " +
                    std::to_string(global_layer) + " seq " +
                    std::to_string(seq_idx) + ": " + decode_reason);
            }
            if (lhs_values.size() != rhs_values.size())
            {
                return mismatch(
                    std::string(kind) +
                    " suffix numerical geometry mismatch at cache " +
                    std::to_string(cache_idx) + " layer " +
                    std::to_string(global_layer) + " seq " +
                    std::to_string(seq_idx));
            }

            double dot = 0.0;
            double lhs_norm_sq = 0.0;
            double rhs_norm_sq = 0.0;
            double difference_norm_sq = 0.0;
            double max_abs = 0.0;
            bool finite = true;
            for (size_t index = 0; index < lhs_values.size(); ++index)
            {
                const double lhs = static_cast<double>(lhs_values[index]);
                const double rhs = static_cast<double>(rhs_values[index]);
                const double difference = lhs - rhs;
                finite = finite && std::isfinite(lhs) && std::isfinite(rhs);
                dot += lhs * rhs;
                lhs_norm_sq += lhs * lhs;
                rhs_norm_sq += rhs * rhs;
                difference_norm_sq += difference * difference;
                max_abs = std::max(max_abs, std::abs(difference));
            }
            const bool both_zero =
                lhs_norm_sq <= 1e-30 && rhs_norm_sq <= 1e-30;
            const double norm_product =
                std::sqrt(lhs_norm_sq) * std::sqrt(rhs_norm_sq);
            const double cosine =
                both_zero ? 1.0
                          : (norm_product > 1e-30 ? dot / norm_product : 0.0);
            const double relative_l2 =
                std::sqrt(difference_norm_sq) /
                std::max(1e-30, std::sqrt(lhs_norm_sq));
            const bool passed =
                finite && cosine >= options.main_kv_suffix_min_cosine;

            if (!main_kv_numerical.compared)
            {
                main_kv_numerical.compared = true;
                main_kv_numerical.passed = true;
            }
            main_kv_numerical.passed =
                main_kv_numerical.passed && passed;
            ++main_kv_numerical.numerical_suffix_payloads;
            main_kv_numerical.elements += lhs_values.size();
            main_kv_numerical.minimum_cosine = std::min(
                main_kv_numerical.minimum_cosine,
                cosine);
            main_kv_numerical.maximum_relative_l2 = std::max(
                main_kv_numerical.maximum_relative_l2,
                relative_l2);
            main_kv_numerical.maximum_abs = std::max(
                main_kv_numerical.maximum_abs,
                max_abs);

            if (!passed)
            {
                std::ostringstream message;
                message << kind
                        << " suffix numerical mismatch after MoE placement change at cache "
                        << cache_idx << " layer " << global_layer
                        << " seq " << seq_idx
                        << " precision=" << activationPrecisionToString(precision)
                        << " elements=" << lhs_values.size()
                        << " finite=" << (finite ? "true" : "false")
                        << " cosine=" << cosine
                        << " required_cosine="
                        << options.main_kv_suffix_min_cosine
                        << " rel_l2=" << relative_l2
                        << " max_abs=" << max_abs;
                return mismatch(message.str());
            }
            return MTPStateValidationResult::success();
        };

        auto compare_kv_caches = [&](const std::vector<PrefixKVCacheProbe> &lhs_caches,
                                     const std::vector<PrefixKVCacheProbe> &rhs_caches,
                                     const char *label,
                                     bool compare_payload_hashes,
                                     MTPMainKVPayloadComparisonPolicy payload_policy) -> MTPStateValidationResult
        {
            if (lhs_caches.size() != rhs_caches.size())
            {
                std::ostringstream msg;
                msg << label << " KV cache count mismatch";
                return mismatch(msg.str());
            }
            for (size_t cache_idx = 0; cache_idx < lhs_caches.size(); ++cache_idx)
            {
                const PrefixKVCacheProbe &lhs_cache = lhs_caches[cache_idx];
                const PrefixKVCacheProbe &rhs_cache = rhs_caches[cache_idx];
                if (lhs_cache.owner != rhs_cache.owner ||
                    lhs_cache.device != rhs_cache.device ||
                    lhs_cache.first_layer_index != rhs_cache.first_layer_index ||
                    lhs_cache.n_layers != rhs_cache.n_layers ||
                    lhs_cache.max_seq_len != rhs_cache.max_seq_len ||
                    lhs_cache.n_kv_heads != rhs_cache.n_kv_heads ||
                    lhs_cache.local_n_kv_heads != rhs_cache.local_n_kv_heads ||
                    lhs_cache.kv_head_start != rhs_cache.kv_head_start ||
                    lhs_cache.k_precision != rhs_cache.k_precision ||
                    lhs_cache.v_precision != rhs_cache.v_precision)
                {
                    std::ostringstream msg;
                    msg << label << " KV cache metadata mismatch for cache "
                        << cache_idx;
                    return mismatch(msg.str());
                }
                if (lhs_cache.layers.size() != rhs_cache.layers.size())
                {
                    std::ostringstream msg;
                    msg << label << " KV layer-probe count mismatch for cache "
                        << cache_idx;
                    return mismatch(msg.str());
                }
                for (size_t layer_idx = 0; layer_idx < lhs_cache.layers.size(); ++layer_idx)
                {
                    const PrefixKVLayerProbe &lhs = lhs_cache.layers[layer_idx];
                    const PrefixKVLayerProbe &rhs = rhs_cache.layers[layer_idx];
                    if (lhs.cache_layer != rhs.cache_layer ||
                        lhs.global_layer != rhs.global_layer ||
                        lhs.seq_idx != rhs.seq_idx ||
                        lhs.cached_tokens != rhs.cached_tokens ||
                        lhs.ring_head != rhs.ring_head)
                    {
                        std::ostringstream msg;
                        msg << label << " KV layer metadata mismatch at cache "
                            << cache_idx
                            << " layer " << lhs.global_layer
                            << " seq " << lhs.seq_idx
                            << " tokens=" << lhs.cached_tokens << "/"
                            << rhs.cached_tokens
                            << " ring_head=" << lhs.ring_head << "/"
                            << rhs.ring_head;
                        return mismatch(msg.str());
                    }
                    if (!compare_payload_hashes)
                    {
                        continue;
                    }
                    if (lhs.payload_hash_available != rhs.payload_hash_available)
                    {
                        std::ostringstream msg;
                        msg << label << " KV payload hash availability mismatch at cache "
                            << cache_idx << " layer " << lhs.global_layer;
                        return mismatch(msg.str());
                    }
                    if (!lhs.payload_hash_available)
                    {
                        continue;
                    }
                    if (lhs.k_payload_bytes != rhs.k_payload_bytes ||
                        lhs.v_payload_bytes != rhs.v_payload_bytes ||
                        lhs.k_payload_hash != rhs.k_payload_hash ||
                        lhs.v_payload_hash != rhs.v_payload_hash)
                    {
                        const bool placement_changed =
                            oracle.moe_runtime_movement_epoch !=
                            candidate.moe_runtime_movement_epoch;
                        const bool placement_aware_suffix =
                            payload_policy ==
                                MTPMainKVPayloadComparisonPolicy::
                                    ExactPrefixNumericalSuffixAfterMoEPlacementChange &&
                            placement_changed;
                        if (placement_aware_suffix)
                        {
                            if (lhs.k_payload_bytes != rhs.k_payload_bytes ||
                                lhs.v_payload_bytes != rhs.v_payload_bytes)
                            {
                                std::ostringstream message;
                                message
                                    << label
                                    << " KV placement-aware comparison requires identical "
                                       "full-payload geometry at cache "
                                    << cache_idx << " layer " << lhs.global_layer
                                    << " seq " << lhs.seq_idx
                                    << " k_bytes=" << lhs.k_payload_bytes << "/"
                                    << rhs.k_payload_bytes
                                    << " v_bytes=" << lhs.v_payload_bytes << "/"
                                    << rhs.v_payload_bytes;
                                return mismatch(message.str());
                            }
                            if (options.main_kv_exact_prefix_tokens < 0)
                            {
                                return mismatch(
                                    "placement-aware main KV comparison requires an "
                                    "explicit exact-prefix token count");
                            }
                            if (lhs.cached_tokens <=
                                options.main_kv_exact_prefix_tokens)
                            {
                                std::ostringstream message;
                                message
                                    << label
                                    << " KV placement-aware comparison has no recomputed "
                                       "suffix at cache "
                                    << cache_idx << " layer " << lhs.global_layer
                                    << " seq " << lhs.seq_idx
                                    << " cached_tokens=" << lhs.cached_tokens
                                    << " exact_prefix_tokens="
                                    << options.main_kv_exact_prefix_tokens;
                                return mismatch(message.str());
                            }

                            /*
                             * The leading digest proves that cache restore did
                             * not modify one byte of the cached prefix.  Only
                             * the disjoint, explicitly retained suffix below
                             * may cross a floating-point comparison boundary.
                             */
                            const bool exact_prefix =
                                lhs.leading_segment_hash_available &&
                                rhs.leading_segment_hash_available &&
                                lhs.leading_segment_tokens ==
                                    options.main_kv_exact_prefix_tokens &&
                                rhs.leading_segment_tokens ==
                                    options.main_kv_exact_prefix_tokens &&
                                lhs.leading_k_payload_bytes ==
                                    rhs.leading_k_payload_bytes &&
                                lhs.leading_v_payload_bytes ==
                                    rhs.leading_v_payload_bytes &&
                                lhs.leading_k_payload_hash ==
                                    rhs.leading_k_payload_hash &&
                                lhs.leading_v_payload_hash ==
                                    rhs.leading_v_payload_hash;
                            if (!exact_prefix)
                            {
                                std::ostringstream message;
                                message
                                    << label
                                    << " KV cached-prefix bytes changed at cache "
                                    << cache_idx << " layer " << lhs.global_layer
                                    << " seq " << lhs.seq_idx
                                    << " available="
                                    << (lhs.leading_segment_hash_available ? "yes" : "no")
                                    << "/"
                                    << (rhs.leading_segment_hash_available ? "yes" : "no")
                                    << " tokens=" << lhs.leading_segment_tokens << "/"
                                    << rhs.leading_segment_tokens
                                    << " required_tokens="
                                    << options.main_kv_exact_prefix_tokens
                                    << " k_bytes="
                                    << lhs.leading_k_payload_bytes << "/"
                                    << rhs.leading_k_payload_bytes
                                    << " v_bytes="
                                    << lhs.leading_v_payload_bytes << "/"
                                    << rhs.leading_v_payload_bytes
                                    << " k_hash=" << lhs.leading_k_payload_hash << "/"
                                    << rhs.leading_k_payload_hash
                                    << " v_hash=" << lhs.leading_v_payload_hash << "/"
                                    << rhs.leading_v_payload_hash;
                                return mismatch(message.str());
                            }

                            const int suffix_tokens =
                                lhs.cached_tokens -
                                options.main_kv_exact_prefix_tokens;
                            auto find_suffix =
                                [&](const std::vector<PrefixKVSegmentProbe> &segments)
                                -> const PrefixKVSegmentProbe *
                            {
                                const PrefixKVSegmentProbe *found = nullptr;
                                for (const auto &segment : segments)
                                {
                                    if (segment.token_start !=
                                            options.main_kv_exact_prefix_tokens ||
                                        segment.token_count != suffix_tokens)
                                    {
                                        continue;
                                    }
                                    if (found != nullptr)
                                    {
                                        return nullptr;
                                    }
                                    found = &segment;
                                }
                                return found;
                            };
                            const PrefixKVSegmentProbe *lhs_suffix =
                                find_suffix(lhs.segments);
                            const PrefixKVSegmentProbe *rhs_suffix =
                                find_suffix(rhs.segments);
                            if (!lhs_suffix || !rhs_suffix)
                            {
                                std::ostringstream message;
                                message
                                    << label
                                    << " KV placement-aware comparison requires one "
                                       "unambiguous retained suffix segment at cache "
                                    << cache_idx << " layer " << lhs.global_layer
                                    << " seq " << lhs.seq_idx
                                    << " start="
                                    << options.main_kv_exact_prefix_tokens
                                    << " tokens=" << suffix_tokens;
                                return mismatch(message.str());
                            }
                            const bool complete_suffix =
                                lhs_suffix->hash_available &&
                                rhs_suffix->hash_available &&
                                lhs_suffix->k_payload_bytes ==
                                    rhs_suffix->k_payload_bytes &&
                                lhs_suffix->v_payload_bytes ==
                                    rhs_suffix->v_payload_bytes &&
                                lhs_suffix->k_payload.size() ==
                                    lhs_suffix->k_payload_bytes &&
                                rhs_suffix->k_payload.size() ==
                                    rhs_suffix->k_payload_bytes &&
                                lhs_suffix->v_payload.size() ==
                                    lhs_suffix->v_payload_bytes &&
                                rhs_suffix->v_payload.size() ==
                                    rhs_suffix->v_payload_bytes;
                            if (!complete_suffix)
                            {
                                std::ostringstream message;
                                message
                                    << label
                                    << " KV placement-aware comparison requires complete "
                                       "retained suffix bytes at cache "
                                    << cache_idx << " layer " << lhs.global_layer
                                    << " seq " << lhs.seq_idx
                                    << " hash_available="
                                    << (lhs_suffix->hash_available ? "yes" : "no")
                                    << "/"
                                    << (rhs_suffix->hash_available ? "yes" : "no")
                                    << " k_bytes=" << lhs_suffix->k_payload.size()
                                    << "/" << lhs_suffix->k_payload_bytes << " vs "
                                    << rhs_suffix->k_payload.size() << "/"
                                    << rhs_suffix->k_payload_bytes
                                    << " v_bytes=" << lhs_suffix->v_payload.size()
                                    << "/" << lhs_suffix->v_payload_bytes << " vs "
                                    << rhs_suffix->v_payload.size() << "/"
                                    << rhs_suffix->v_payload_bytes;
                                return mismatch(message.str());
                            }

                            ++main_kv_numerical.exact_prefix_segments;
                            if (auto result = compare_main_kv_suffix_payload(
                                    lhs_suffix->k_payload,
                                    rhs_suffix->k_payload,
                                    lhs_cache.k_precision,
                                    cache_idx,
                                    lhs.global_layer,
                                    lhs.seq_idx,
                                    "K");
                                !result)
                            {
                                return result;
                            }
                            if (auto result = compare_main_kv_suffix_payload(
                                    lhs_suffix->v_payload,
                                    rhs_suffix->v_payload,
                                    lhs_cache.v_precision,
                                    cache_idx,
                                    lhs.global_layer,
                                    lhs.seq_idx,
                                    "V");
                                !result)
                            {
                                return result;
                            }
                            continue;
                        }

                        auto append_segment_hashes =
                            [](std::ostringstream &msg,
                               const char *name,
                               bool lhs_available,
                               bool rhs_available,
                               int lhs_start,
                               int rhs_start,
                               int lhs_tokens,
                               int rhs_tokens,
                               uint64_t lhs_k_hash,
                               uint64_t rhs_k_hash,
                               uint64_t lhs_v_hash,
                               uint64_t rhs_v_hash)
                        {
                            if (!lhs_available && !rhs_available)
                                return;
                            msg << " " << name
                                << "_available=" << (lhs_available ? "yes" : "no")
                                << "/" << (rhs_available ? "yes" : "no")
                                << " start=" << lhs_start << "/" << rhs_start
                                << " tokens=" << lhs_tokens << "/" << rhs_tokens
                                << " k_hash=" << lhs_k_hash << "/" << rhs_k_hash
                                << " v_hash=" << lhs_v_hash << "/" << rhs_v_hash;
                        };
                        auto append_named_segment_hashes =
                            [](std::ostringstream &msg,
                               const std::vector<PrefixKVSegmentProbe> &lhs_segments,
                               const std::vector<PrefixKVSegmentProbe> &rhs_segments)
                        {
                            if (lhs_segments.empty() && rhs_segments.empty())
                                return;
                            msg << " named_segments=";
                            const size_t count =
                                std::max(lhs_segments.size(), rhs_segments.size());
                            for (size_t i = 0; i < count; ++i)
                            {
                                const PrefixKVSegmentProbe *lhs_segment =
                                    i < lhs_segments.size() ? &lhs_segments[i] : nullptr;
                                const PrefixKVSegmentProbe *rhs_segment =
                                    i < rhs_segments.size() ? &rhs_segments[i] : nullptr;
                                if (i > 0)
                                    msg << ";";
                                msg << "[";
                                msg << "name="
                                    << (lhs_segment
                                            ? lhs_segment->name
                                            : (rhs_segment ? rhs_segment->name : "<missing>"));
                                msg << " available="
                                    << (lhs_segment && lhs_segment->hash_available ? "yes" : "no")
                                    << "/"
                                    << (rhs_segment && rhs_segment->hash_available ? "yes" : "no");
                                msg << " start="
                                    << (lhs_segment ? lhs_segment->token_start : -1)
                                    << "/"
                                    << (rhs_segment ? rhs_segment->token_start : -1);
                                msg << " tokens="
                                    << (lhs_segment ? lhs_segment->token_count : -1)
                                    << "/"
                                    << (rhs_segment ? rhs_segment->token_count : -1);
                                msg << " k_hash="
                                    << (lhs_segment ? lhs_segment->k_payload_hash : 0)
                                    << "/"
                                    << (rhs_segment ? rhs_segment->k_payload_hash : 0);
                                msg << " v_hash="
                                    << (lhs_segment ? lhs_segment->v_payload_hash : 0)
                                    << "/"
                                    << (rhs_segment ? rhs_segment->v_payload_hash : 0);
                                msg << "]";
                            }
                        };
                        std::ostringstream msg;
                        msg << label << " KV payload hash mismatch at cache "
                            << cache_idx
                            << " layer " << lhs.global_layer
                            << " seq " << lhs.seq_idx
                            << " k_bytes=" << lhs.k_payload_bytes << "/"
                            << rhs.k_payload_bytes
                            << " v_bytes=" << lhs.v_payload_bytes << "/"
                            << rhs.v_payload_bytes
                            << " k_hash=" << lhs.k_payload_hash << "/"
                            << rhs.k_payload_hash
                            << " v_hash=" << lhs.v_payload_hash << "/"
                            << rhs.v_payload_hash;
                        append_segment_hashes(
                            msg,
                            "leading",
                            lhs.leading_segment_hash_available,
                            rhs.leading_segment_hash_available,
                            0,
                            0,
                            lhs.leading_segment_tokens,
                            rhs.leading_segment_tokens,
                            lhs.leading_k_payload_hash,
                            rhs.leading_k_payload_hash,
                            lhs.leading_v_payload_hash,
                            rhs.leading_v_payload_hash);
                        append_segment_hashes(
                            msg,
                            "trailing",
                            lhs.trailing_segment_hash_available,
                            rhs.trailing_segment_hash_available,
                            lhs.trailing_segment_start,
                            rhs.trailing_segment_start,
                            lhs.trailing_segment_tokens,
                            rhs.trailing_segment_tokens,
                            lhs.trailing_k_payload_hash,
                            rhs.trailing_k_payload_hash,
                            lhs.trailing_v_payload_hash,
                            rhs.trailing_v_payload_hash);
                        append_named_segment_hashes(
                            msg,
                            lhs.segments,
                            rhs.segments);
                        return mismatch(msg.str());
                    }
                }
            }
            return MTPStateValidationResult::success();
        };

        if (auto kv_result =
                compare_kv_caches(
                    oracle.kv_caches,
                    candidate.kv_caches,
                    "main",
                    options.compare_main_kv_payload_hashes,
                    options.main_kv_payload_policy);
            !kv_result)
        {
            return kv_result;
        }
        if (options.compare_shifted_mtp_kv)
        {
            if (auto mtp_kv_result =
                    compare_kv_caches(
                        oracle.mtp_kv_caches,
                        candidate.mtp_kv_caches,
                        "shifted MTP",
                        /*compare_payload_hashes=*/true,
                        MTPMainKVPayloadComparisonPolicy::ExactBytes);
                !mtp_kv_result)
            {
                return mtp_kv_result;
            }
        }

        const bool placement_changed =
            oracle.moe_runtime_movement_epoch !=
            candidate.moe_runtime_movement_epoch;
        const bool placement_aware_gdn =
            options.gdn_state_policy ==
            MTPGDNStateComparisonPolicy::
                ExactUnlessMoEPlacementChanged;
        const bool always_numerical_gdn =
            options.gdn_state_policy ==
            MTPGDNStateComparisonPolicy::NumericalValues;

        auto compare_gdn_values =
            [&](const std::vector<float> &lhs,
                const std::vector<float> &rhs,
                size_t expected_values,
                int layer,
                const char *state_name) -> MTPStateValidationResult
        {
            if (lhs.size() != expected_values ||
                rhs.size() != expected_values)
            {
                std::ostringstream msg;
                msg << "GDN " << state_name
                    << " numerical comparison requires complete retained values at layer "
                    << layer << ": expected=" << expected_values
                    << " actual=" << lhs.size() << "/" << rhs.size();
                return mismatch(msg.str());
            }

            double sq_diff = 0.0;
            double sq_oracle = 0.0;
            double dot = 0.0;
            double sq_candidate = 0.0;
            double max_abs = 0.0;
            bool finite = true;
            for (size_t i = 0; i < expected_values; ++i)
            {
                const double a = static_cast<double>(lhs[i]);
                const double b = static_cast<double>(rhs[i]);
                finite = finite && std::isfinite(a) && std::isfinite(b);
                const double diff = a - b;
                sq_diff += diff * diff;
                sq_oracle += a * a;
                sq_candidate += b * b;
                dot += a * b;
                max_abs = std::max(max_abs, std::abs(diff));
            }

            const double rel_l2 =
                std::sqrt(sq_diff) /
                std::max(1e-30, std::sqrt(sq_oracle));
            const double norm_product =
                std::sqrt(sq_oracle) * std::sqrt(sq_candidate);
            const bool both_zero =
                sq_oracle <= 1e-30 && sq_candidate <= 1e-30;
            const double cosine =
                both_zero
                    ? 1.0
                    : (norm_product > 1e-30 ? dot / norm_product : 0.0);
            const bool relative_l2_passed =
                !options.gdn_relative_l2_tolerance ||
                rel_l2 <= *options.gdn_relative_l2_tolerance;
            const bool max_abs_passed =
                !options.gdn_max_abs_tolerance ||
                max_abs <= *options.gdn_max_abs_tolerance;
            const bool passed =
                finite && relative_l2_passed && max_abs_passed &&
                cosine >= options.gdn_min_cosine;

            gdn_numerical.compared = true;
            gdn_numerical.passed =
                gdn_numerical.payloads == 0u
                    ? passed
                    : gdn_numerical.passed && passed;
            ++gdn_numerical.payloads;
            gdn_numerical.elements += expected_values;
            gdn_numerical.minimum_cosine = std::min(
                gdn_numerical.minimum_cosine, cosine);
            gdn_numerical.maximum_relative_l2 = std::max(
                gdn_numerical.maximum_relative_l2, rel_l2);
            gdn_numerical.maximum_abs = std::max(
                gdn_numerical.maximum_abs, max_abs);

            if (!passed)
            {
                std::ostringstream msg;
                msg << "GDN " << state_name << " numerical mismatch at layer "
                    << layer
                    << " epochs=" << oracle.moe_runtime_movement_epoch
                    << "/" << candidate.moe_runtime_movement_epoch
                    << " elements=" << expected_values
                    << " finite=" << (finite ? "true" : "false")
                    << " rel_l2=" << rel_l2
                    << " max_abs=" << max_abs
                    << " cosine=" << cosine
                    << " tolerances(rel_l2=";
                if (options.gdn_relative_l2_tolerance)
                    msg << *options.gdn_relative_l2_tolerance;
                else
                    msg << "not_gated";
                msg << ", max_abs=";
                if (options.gdn_max_abs_tolerance)
                    msg << *options.gdn_max_abs_tolerance;
                else
                    msg << "not_gated";
                msg << ", min_cosine=" << options.gdn_min_cosine << ")";
                return mismatch(msg.str());
            }
            return MTPStateValidationResult::success();
        };

        for (size_t i = 0; i < oracle.gdn_layers.size(); ++i)
        {
            const PrefixGDNLayerProbe &lhs = oracle.gdn_layers[i];
            const PrefixGDNLayerProbe &rhs = candidate.gdn_layers[i];
            if (lhs.global_layer != rhs.global_layer)
                return mismatch("GDN layer id mismatch");
            if (lhs.recurrence_values != rhs.recurrence_values)
                return mismatch("GDN recurrence value count mismatch");
            if (lhs.conv_values != rhs.conv_values)
                return mismatch("GDN short-conv value count mismatch");
            if (options.gdn_state_policy ==
                MTPGDNStateComparisonPolicy::LogicalMetadataOnly)
            {
                continue;
            }
            if (lhs.device_state_hash_available !=
                rhs.device_state_hash_available)
            {
                return mismatch(
                    "GDN device-state hash availability mismatch at layer " +
                    std::to_string(lhs.global_layer));
            }
            const bool compare_device_gdn_hashes =
                lhs.device_state_hash_available;
            const size_t recurrence_expected_values =
                compare_device_gdn_hashes
                    ? lhs.recurrence_device_bytes / sizeof(float)
                    : lhs.recurrence_values;
            const size_t conv_expected_values =
                compare_device_gdn_hashes
                    ? lhs.conv_device_bytes / sizeof(float)
                    : lhs.conv_values;
            const uint64_t lhs_recurrence_hash =
                compare_device_gdn_hashes
                    ? lhs.recurrence_device_hash
                    : lhs.recurrence_hash;
            const uint64_t rhs_recurrence_hash =
                compare_device_gdn_hashes
                    ? rhs.recurrence_device_hash
                    : rhs.recurrence_hash;
            const uint64_t lhs_conv_hash =
                compare_device_gdn_hashes
                    ? lhs.conv_device_hash
                    : lhs.conv_hash;
            const uint64_t rhs_conv_hash =
                compare_device_gdn_hashes
                    ? rhs.conv_device_hash
                    : rhs.conv_hash;
            const bool recurrence_hash_changed =
                lhs_recurrence_hash != rhs_recurrence_hash;
            const bool conv_hash_changed = lhs_conv_hash != rhs_conv_hash;
            const bool compare_recurrence_numerically =
                always_numerical_gdn ||
                (placement_aware_gdn && placement_changed &&
                 recurrence_hash_changed);
            const bool compare_conv_numerically =
                always_numerical_gdn ||
                (placement_aware_gdn && placement_changed &&
                 conv_hash_changed);
            if (compare_device_gdn_hashes)
            {
                if (lhs.recurrence_device_bytes != rhs.recurrence_device_bytes)
                {
                    std::ostringstream msg;
                    msg << "GDN recurrence device byte count mismatch at layer "
                        << lhs.global_layer;
                    return mismatch(msg.str());
                }
                if (lhs.conv_device_bytes != rhs.conv_device_bytes)
                {
                    std::ostringstream msg;
                    msg << "GDN short-conv device byte count mismatch at layer "
                        << lhs.global_layer;
                    return mismatch(msg.str());
                }
            }

            if (compare_recurrence_numerically)
            {
                if (auto result = compare_gdn_values(
                        lhs.recurrence_sample_values,
                        rhs.recurrence_sample_values,
                        recurrence_expected_values,
                        lhs.global_layer,
                        "recurrence");
                    !result)
                {
                    return result;
                }
            }
            else if (recurrence_hash_changed)
            {
                std::ostringstream msg;
                msg << "GDN recurrence "
                    << (compare_device_gdn_hashes ? "device " : "")
                    << "hash mismatch at layer "
                    << lhs.global_layer;
                return mismatch(msg.str());
            }

            if (compare_conv_numerically)
            {
                if (auto result = compare_gdn_values(
                        lhs.conv_sample_values,
                        rhs.conv_sample_values,
                        conv_expected_values,
                        lhs.global_layer,
                        "short-conv");
                    !result)
                {
                    return result;
                }
            }
            else if (conv_hash_changed)
            {
                std::ostringstream msg;
                msg << "GDN short-conv "
                    << (compare_device_gdn_hashes ? "device " : "")
                    << "hash mismatch at layer "
                    << lhs.global_layer;
                return mismatch(msg.str());
            }

            /* GPU host mirrors may lag device-only publication.  When device
             * bytes are authoritative, their exact/numerical proof supersedes
             * the host-mirror zero flags. */
            if (!compare_device_gdn_hashes)
            {
                if (lhs.recurrence_all_zero != rhs.recurrence_all_zero)
                    return mismatch("GDN recurrence zero-state flag mismatch");
                if (lhs.conv_all_zero != rhs.conv_all_zero)
                    return mismatch("GDN short-conv zero-state flag mismatch");
            }
        }

        MTPStateValidationResult result =
            MTPStateValidationResult::success();
        result.terminal_hidden_numerical = terminal_hidden_numerical;
        result.terminal_logits_numerical = terminal_logits_numerical;
        result.main_kv_numerical = main_kv_numerical;
        result.gdn_numerical = gdn_numerical;
        return result;
    }

} // namespace llaminar2
