#include "MTPStateTransaction.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

namespace llaminar2
{

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
        auto mismatch = [](std::string reason)
        {
            return MTPStateValidationResult::failure(std::move(reason));
        };

        if (oracle.initialized != candidate.initialized)
            return mismatch("initialized flag mismatch");
        if (!oracle.initialized)
            return MTPStateValidationResult::success();
        if (oracle.current_position != candidate.current_position)
            return mismatch("current position mismatch");
        if (oracle.positions != candidate.positions)
            return mismatch("per-sequence position vector mismatch");
        if (oracle.sequence_lengths != candidate.sequence_lengths)
            return mismatch("per-sequence length vector mismatch");
        if (oracle.totalCachedTokens() != candidate.totalCachedTokens())
            return mismatch("main KV cached token count mismatch");
        if (options.compare_shifted_mtp_kv &&
            oracle.totalMTPCachedTokens() != candidate.totalMTPCachedTokens())
            return mismatch("shifted MTP cached token count mismatch");
        if (oracle.has_hidden != candidate.has_hidden)
            return mismatch("terminal hidden availability mismatch");
        if (oracle.has_logits != candidate.has_logits)
            return mismatch("terminal logits availability mismatch");
        if (oracle.gdn_layers.size() != candidate.gdn_layers.size())
            return mismatch("GDN layer count mismatch");

        auto compare_kv_caches = [&](const std::vector<PrefixKVCacheProbe> &lhs_caches,
                                     const std::vector<PrefixKVCacheProbe> &rhs_caches,
                                     const char *label,
                                     bool compare_payload_hashes) -> MTPStateValidationResult
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
                    options.compare_main_kv_payload_hashes);
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
                        /*compare_payload_hashes=*/true);
                !mtp_kv_result)
            {
                return mtp_kv_result;
            }
        }

        auto compare_gdn_values =
            [&](const std::vector<float> &lhs,
                const std::vector<float> &rhs,
                int layer,
                const char *state_name) -> MTPStateValidationResult
        {
            if (lhs.empty() || rhs.empty())
                return MTPStateValidationResult::success();
            if (lhs.size() != rhs.size())
            {
                std::ostringstream msg;
                msg << "GDN " << state_name << " value sample count mismatch at layer "
                    << layer << ": " << lhs.size() << "/" << rhs.size();
                return mismatch(msg.str());
            }

            double sq_diff = 0.0;
            double sq_rhs = 0.0;
            double dot = 0.0;
            double sq_lhs = 0.0;
            double max_abs = 0.0;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                const double a = static_cast<double>(lhs[i]);
                const double b = static_cast<double>(rhs[i]);
                const double diff = a - b;
                sq_diff += diff * diff;
                sq_lhs += a * a;
                sq_rhs += b * b;
                dot += a * b;
                max_abs = std::max(max_abs, std::abs(diff));
            }

            const double rel_l2 =
                std::sqrt(sq_diff) / std::max(1e-30, std::sqrt(sq_rhs));
            const double cosine =
                dot / std::max(1e-30, std::sqrt(sq_lhs) * std::sqrt(sq_rhs));
            if (rel_l2 > options.gdn_relative_l2_tolerance ||
                max_abs > options.gdn_max_abs_tolerance ||
                cosine < options.gdn_min_cosine)
            {
                std::ostringstream msg;
                msg << "GDN " << state_name << " value mismatch at layer "
                    << layer
                    << " rel_l2=" << rel_l2
                    << " max_abs=" << max_abs
                    << " cosine=" << cosine
                    << " tolerances(rel_l2="
                    << options.gdn_relative_l2_tolerance
                    << ", max_abs=" << options.gdn_max_abs_tolerance
                    << ", min_cosine=" << options.gdn_min_cosine << ")";
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
            if (options.compare_gdn_values_if_available)
            {
                if (auto recurrence_result = compare_gdn_values(
                        lhs.recurrence_sample_values,
                        rhs.recurrence_sample_values,
                        lhs.global_layer,
                        "recurrence");
                    !recurrence_result)
                {
                    return recurrence_result;
                }
                if (auto conv_result = compare_gdn_values(
                        lhs.conv_sample_values,
                        rhs.conv_sample_values,
                        lhs.global_layer,
                        "short-conv");
                    !conv_result)
                {
                    return conv_result;
                }
            }
            if (options.compare_gdn_hashes &&
                lhs.recurrence_hash != rhs.recurrence_hash)
            {
                std::ostringstream msg;
                msg << "GDN recurrence hash mismatch at layer "
                    << lhs.global_layer;
                return mismatch(msg.str());
            }
            if (options.compare_gdn_hashes &&
                lhs.conv_hash != rhs.conv_hash)
            {
                std::ostringstream msg;
                msg << "GDN short-conv hash mismatch at layer "
                    << lhs.global_layer;
                return mismatch(msg.str());
            }
            if (lhs.recurrence_all_zero != rhs.recurrence_all_zero)
                return mismatch("GDN recurrence zero-state flag mismatch");
            if (lhs.conv_all_zero != rhs.conv_all_zero)
                return mismatch("GDN short-conv zero-state flag mismatch");
        }

        return MTPStateValidationResult::success();
    }

} // namespace llaminar2
