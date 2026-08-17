/**
 * @file SnapshotCapture.cpp
 * @brief Converts concrete graph-stage observations into stable parity keys.
 *
 * Snapshot capture is diagnostic-only: graph execution publishes each selected
 * stage's already-produced tensor through its declared stream/event edge, and
 * this component copies that host-visible observation into the semantic names
 * consumed by model parity tests and their CSV evidence.  It deliberately
 * maps by the real producing graph node rather than by a historical model
 * convention so optimized, captured, and heterogeneous paths expose the same
 * numerical checkpoints without changing production arithmetic. Segmented
 * prefill additionally retains each live bucket under an explicit context and
 * joins sequence-shaped values back into one full-prompt checkpoint only after
 * all chunks have published their diagnostic copies.
 *
 * Extracted from DeviceGraphOrchestrator.h (Phase 2 of DGO refactor).
 */

#include "SnapshotCapture.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <string_view>

namespace llaminar2
{
    namespace
    {
        std::string snapshotContextPrefix(const std::string &context)
        {
            std::string result;
            result.reserve(context.size());
            bool previous_underscore = false;
            for (char c : context)
            {
                const unsigned char uc = static_cast<unsigned char>(c);
                if (std::isalnum(uc))
                {
                    result.push_back(static_cast<char>(std::toupper(uc)));
                    previous_underscore = false;
                }
                else if (!previous_underscore)
                {
                    result.push_back('_');
                    previous_underscore = true;
                }
            }
            while (!result.empty() && result.back() == '_')
                result.pop_back();
            return result.empty() ? "CONTEXT" : result;
        }

        /**
         * @brief Name one ordered cumulative ExpertOverlay contribution.
         *
         * Each graph-native overlay ticket consume copies the routed-expert
         * accumulator back to the continuation participant after exactly one
         * target has returned its sparse rows.  The ordinary
         * `MOE_EXPERT_OUTPUT` key intentionally keeps only the final consume,
         * which is the semantic model checkpoint.  This diagnostic key retains
         * each intermediate cumulative value as well, allowing a parity report
         * to isolate the participant that first introduces a numerical error.
         *
         * @param stage_name Concrete ticket-consume graph node name.
         * @return Stable semantic key, or an empty string for a malformed name.
         */
        std::string overlayTicketCumulativeSnapshotKey(
            const std::string &stage_name)
        {
            constexpr std::string_view kTicketConsume =
                "_moe_overlay_ticket_consume_";
            const size_t marker = stage_name.find(kTicketConsume);
            if (marker == std::string::npos)
                return {};

            const std::string prefix = stage_name.substr(0, marker);
            const std::string participant = stage_name.substr(
                marker + kTicketConsume.size());
            if (prefix.empty() || participant.empty())
                return {};

            return prefix + "_MOE_OVERLAY_CUMULATIVE_" +
                   snapshotContextPrefix(participant);
        }
    } // namespace

    // =========================================================================
    // Stage capture routing
    // =========================================================================

    void SnapshotCapture::captureStage(const std::string &name, const StageDumpInfo &dump)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (const size_t context_sep = name.find("::"); context_sep != std::string::npos)
        {
            SnapshotCapture scoped_capture;
            scoped_capture.captureStage(name.substr(context_sep + 2), dump);
            const std::string prefix = snapshotContextPrefix(name.substr(0, context_sep));
            for (const auto &entry : scoped_capture.all())
            {
                snapshots_[prefix + "_" + entry.first] = entry.second;
            }
            return;
        }

        LOG_TRACE("[Snapshot] Callback invoked for stage: " << name
                                                            << " outputs.size=" << dump.outputs.size());

        // Handle fused QKV stage — split into separate Q, K, V snapshots
        if (name.find("_qkv_proj") != std::string::npos)
        {
            size_t qkv_pos = name.find("_qkv_proj");
            std::string prefix = name.substr(0, qkv_pos);

            if (dump.outputs.size() >= 3)
            {
                storeOutput(prefix + "_Q_PROJECTION", dump.outputs[0]);
                storeOutput(prefix + "_K_PROJECTION", dump.outputs[1]);
                storeOutput(prefix + "_V_PROJECTION", dump.outputs[2]);
            }
            return;
        }

        /*
         * A K/V-only MTP catch-up graph deliberately omits query projection,
         * but the fused stage still owns two independently meaningful outputs.
         * Never collapse output K into an ambiguous `KV_PROJ` alias and drop V:
         * the canonical K/V names retain their column-parallel schema contract
         * and line up with the same PyTorch checkpoints as the full sidecar.
         */
        if (name.find("_kv_proj") != std::string::npos)
        {
            const size_t kv_pos = name.find("_kv_proj");
            const std::string prefix = name.substr(0, kv_pos);
            if (dump.outputs.size() >= 2)
            {
                storeOutput(prefix + "_K_PROJECTION", dump.outputs[0]);
                storeOutput(prefix + "_V_PROJECTION", dump.outputs[1]);
            }
            return;
        }

        // Handle fused Gate/Up stage — split into separate GATE and UP snapshots
        if (name.find("_gate_up") != std::string::npos)
        {
            size_t pos = name.find("_gate_up");
            std::string prefix = name.substr(0, pos);

            if (dump.outputs.size() >= 2)
            {
                storeOutput(prefix + "_FFN_GATE", dump.outputs[0]);
                storeOutput(prefix + "_FFN_UP", dump.outputs[1]);
            }
            return;
        }

        // Handle fused RoPE stage — captures Q_ROPE and K_ROPE
        if (name.find("_rope") != std::string::npos &&
            name.find("_q_rope") == std::string::npos &&
            name.find("_k_rope") == std::string::npos)
        {
            size_t pos = name.find("_rope");
            std::string prefix = name.substr(0, pos);

            if (dump.outputs.size() >= 2)
            {
                storeOutput(prefix + "_Q_ROPE", dump.outputs[0]);
                storeOutput(prefix + "_K_ROPE", dump.outputs[1]);
            }
            return;
        }

        // Handle GDN 4-way projection — split into QKV, Z, alpha, beta snapshots
        if (name.find("_gdn_proj") != std::string::npos)
        {
            size_t pos = name.find("_gdn_proj");
            std::string prefix = name.substr(0, pos);

            // outputs: [0]=output_qkv, [1]=output_z, [2]=output_a, [3]=output_b
            if (dump.outputs.size() >= 1 && dump.outputs[0].data)
                storeOutput(prefix + "_QKV_PROJECTION", dump.outputs[0]);
            if (dump.outputs.size() >= 2 && dump.outputs[1].data)
                storeOutput(prefix + "_GDN_Z_PROJECTION", dump.outputs[1]);
            if (dump.outputs.size() >= 3 && dump.outputs[2].data)
                storeOutput(prefix + "_GDN_ALPHA", dump.outputs[2]);
            if (dump.outputs.size() >= 4 && dump.outputs[3].data)
                storeOutput(prefix + "_GDN_BETA", dump.outputs[3]);
            return;
        }

        // Handle Qwen3.5 full-attention Q/gate split. The raw q_proj snapshot
        // remains Q_PROJECTION because it matches HuggingFace q_proj output;
        // this extra key exposes the sigmoid gate consumed before Wo.
        if (name.find("_q_gate_split") != std::string::npos)
        {
            size_t pos = name.find("_q_gate_split");
            std::string prefix = name.substr(0, pos);

            if (dump.outputs.size() >= 2 && dump.outputs[1].data)
                storeOutput(prefix + "_FA_GATE", dump.outputs[1]);
            return;
        }

        // Handle debug-only KV append cache snapshots. These outputs are
        // populated only when LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT is enabled and
        // expose the post-append persistent cache rows that attention consumes.
        if (name.find("_kv_append") != std::string::npos)
        {
            size_t pos = name.find("_kv_append");
            std::string prefix = name.substr(0, pos);

            auto storeRowsForSmallAppendSource = [&](const std::string &base_key,
                                                     const StageDumpInfo::OutputBuffer &output)
            {
                if (!output.data || output.rows == 0 || output.cols == 0 || output.rows > 16)
                    return;
                if (std::string(output.dtype ? output.dtype : "") != "FP32")
                    return;

                const auto *fp32 = static_cast<const float *>(output.data);
                for (size_t row = 0; row < output.rows; ++row)
                {
                    auto row_output = output;
                    row_output.data = fp32 + row * output.cols;
                    row_output.rows = 1;
                    row_output.byte_size = output.cols * sizeof(float);
                    row_output.element_size = sizeof(float);
                    storeOutput(base_key + "_ROW" + std::to_string(row), row_output);
                }
            };

            for (const auto &output : dump.outputs)
            {
                const std::string output_name = output.name ? output.name : "";
                if (output_name == "cache_k" && output.data)
                    storeOutput(prefix + "_KV_CACHE_K", output);
                else if (output_name == "cache_v" && output.data)
                    storeOutput(prefix + "_KV_CACHE_V", output);
                else if (output_name == "source_k" && output.data)
                {
                    storeOutput(prefix + "_KV_APPEND_SOURCE_K", output);
                    storeRowsForSmallAppendSource(prefix + "_KV_APPEND_SOURCE_K", output);
                }
                else if (output_name == "source_v" && output.data)
                {
                    storeOutput(prefix + "_KV_APPEND_SOURCE_V", output);
                    storeRowsForSmallAppendSource(prefix + "_KV_APPEND_SOURCE_V", output);
                }
            }
            return;
        }

        // Handle Qwen3.5 full-attention output gate. This is the gated context
        // immediately before Wo, matching HuggingFace o_proj's pre-hook input.
        if (name.find("_attn_output_gate") != std::string::npos)
        {
            size_t pos = name.find("_attn_output_gate");
            std::string prefix = name.substr(0, pos);

            if (!dump.outputs.empty() && dump.outputs[0].data)
                storeOutput(prefix + "_ATTENTION_CONTEXT_GATED", dump.outputs[0]);
            return;
        }

        // Handle attention stage debug snapshots. The normal output is the
        // attention context; optional named outputs expose effective K/V after
        // cache read/conversion, immediately before the attention kernel.
        if (name.find("_attention") != std::string::npos)
        {
            size_t pos = name.find("_attention");
            std::string prefix = name.substr(0, pos);

            auto storeRowsForSmallEffectiveKV = [&](const std::string &base_key,
                                                    const StageDumpInfo::OutputBuffer &output)
            {
                if (!output.data || output.rows == 0 || output.cols == 0 || output.rows > 16)
                    return;
                if (std::string(output.dtype ? output.dtype : "") != "FP32")
                    return;

                const auto *fp32 = static_cast<const float *>(output.data);
                for (size_t row = 0; row < output.rows; ++row)
                {
                    auto row_output = output;
                    row_output.data = fp32 + row * output.cols;
                    row_output.rows = 1;
                    row_output.byte_size = output.cols * sizeof(float);
                    row_output.element_size = sizeof(float);
                    storeOutput(base_key + "_ROW" + std::to_string(row), row_output);
                }
            };

            for (const auto &output : dump.outputs)
            {
                const std::string output_name = output.name ? output.name : "";
                if (output_name == "output" && output.data)
                    storeOutput(prefix + "_ATTENTION_CONTEXT", output);
                else if (output_name == "effective_k" && output.data)
                {
                    storeOutput(prefix + "_ATTENTION_EFFECTIVE_K", output);
                    storeRowsForSmallEffectiveKV(prefix + "_ATTENTION_EFFECTIVE_K", output);
                }
                else if (output_name == "effective_v" && output.data)
                {
                    storeOutput(prefix + "_ATTENTION_EFFECTIVE_V", output);
                    storeRowsForSmallEffectiveKV(prefix + "_ATTENTION_EFFECTIVE_V", output);
                }
                else if (output_name.starts_with("device_kv_count_request_") &&
                         output.data)
                {
                    storeOutput(
                        prefix + "_ATTENTION_DEVICE_KV_COUNT_REQUEST_" +
                            output_name.substr(
                                std::string("device_kv_count_request_").size()),
                        output);
                }
                else if (output_name.starts_with("device_kv_head_request_") &&
                         output.data)
                {
                    storeOutput(
                        prefix + "_ATTENTION_DEVICE_KV_HEAD_REQUEST_" +
                            output_name.substr(
                                std::string("device_kv_head_request_").size()),
                        output);
                }
            }
            return;
        }

        // Handle lm_head_allgather — overwrites partial LM_HEAD with full vocab
        if (name == "lm_head_allgather")
        {
            if (!dump.outputs.empty() && dump.outputs[0].data)
            {
                const auto &out = dump.outputs[0];
                auto data = extractFp32FromOutput(out);
                LOG_DEBUG("[Snapshot] lm_head_allgather handler: storing as LM_HEAD (overwriting partial), count=" << data.size());
                if (!data.empty())
                    storeSnapshot("LM_HEAD", std::move(data), out.rows, out.cols);
            }
            return;
        }

        // Handle FusedResidualNormStage. The normalized output keeps the long-standing
        // semantic key, while residual_out exposes the hidden-state handoff that
        // happens inside this fused stage. That handoff is essential for grouped
        // verifier diagnostics because non-terminal FFN residual adds are often
        // represented by the next layer's fused attention norm rather than by a
        // standalone ResidualAddStage node.
        if ((name.find("_attn_norm") != std::string::npos ||
             name.find("_ffn_norm") != std::string::npos) &&
            dump.outputs.size() >= 2)
        {
            std::string key = convertStageNameToSnapshotKey(name);

            if (dump.outputs[0].data)
            {
                auto data = extractFp32FromOutput(dump.outputs[0]);
                LOG_DEBUG("[Snapshot] FusedResidualNorm: storing residual_out as key="
                          << key << "_RESIDUAL_OUT count=" << data.size());
                if (!data.empty())
                    storeSnapshot(
                        key + "_RESIDUAL_OUT",
                        std::move(data),
                        dump.outputs[0].rows,
                        dump.outputs[0].cols);
            }

            if (dump.outputs[1].data)
            {
                auto data = extractFp32FromOutput(dump.outputs[1]);
                LOG_DEBUG("[Snapshot] FusedResidualNorm: storing norm_output as key="
                          << key << " count=" << data.size());
                if (!data.empty())
                    storeSnapshot(
                        key,
                        std::move(data),
                        dump.outputs[1].rows,
                        dump.outputs[1].cols);
            }
            return;
        }

        // Handle fused MoE FFN stage — split into expert output + routing data
        if (name.find("_moe_ffn") != std::string::npos && dump.outputs.size() >= 4)
        {
            size_t pos = name.find("_moe_ffn");
            std::string prefix = name.substr(0, pos);

            // outputs[0] = expert output [seq_len, d_model]
            // outputs[1] = router logits [seq_len, num_experts]
            // outputs[2] = routing indices [seq_len, top_k] (int as float)
            // outputs[3] = routing weights [seq_len, top_k]
            storeOutput(prefix + "_MOE_EXPERT_OUTPUT", dump.outputs[0]);
            storeOutput(prefix + "_MOE_ROUTER_OUTPUT", dump.outputs[1]);
            storeOutput(prefix + "_MOE_ROUTING_INDICES", dump.outputs[2]);
            storeOutput(prefix + "_MOE_ROUTING_WEIGHTS", dump.outputs[3]);
            return;
        }

        /*
         * Canonical LocalTP publication owns three semantically distinct values
         * at one rooted finalizer. Route each named output explicitly so the
         * snapshot contract follows the graph's real producer rather than a
         * historical stage-name convention.
         */
        if (name.find("_moe_canonical_publication_finalize") !=
            std::string::npos)
        {
            const size_t pos =
                name.find("_moe_canonical_publication_finalize");
            const std::string prefix = name.substr(0, pos);
            for (const auto &output : dump.outputs)
            {
                const std::string output_name =
                    output.name ? output.name : "";
                if (output_name == "routed_output" && output.data)
                    storeOutput(prefix + "_MOE_EXPERT_OUTPUT", output);
                else if (output_name == "shared_output" && output.data)
                    storeOutput(prefix + "_MOE_SHARED_GATE_OUTPUT", output);
                else if (output_name == "combined_output" && output.data)
                    storeOutput(prefix + "_MOE_COMBINED_OUTPUT", output);
            }
            return;
        }

        /*
         * The Qwen3.6 MoE combined shared-verifier path can fuse routed expert
         * and shared expert output inside MoEExpertComputeStage.  The stage name
         * is still `_moe_expert_ffn`, so route by output name before the generic
         * suffix map labels it as routed-only expert output.
         */
        if (name.find("_moe_expert_ffn") != std::string::npos)
        {
            size_t pos = name.find("_moe_expert_ffn");
            std::string prefix = name.substr(0, pos);
            bool handled_named_output = false;
            for (const auto &output : dump.outputs)
            {
                const std::string output_name = output.name ? output.name : "";
                if (output_name == "combined_output" && output.data)
                {
                    storeOutput(prefix + "_MOE_COMBINED_OUTPUT", output);
                    handled_named_output = true;
                }
                else if (output_name == "canonical_route_contributions" &&
                         output.data)
                {
                    storeOutput(
                        prefix + "_MOE_CANONICAL_ROUTE_CONTRIBUTIONS",
                        output);
                    handled_named_output = true;
                }
                else if (output_name == "output" && output.data)
                {
                    storeOutput(prefix + "_MOE_EXPERT_OUTPUT", output);
                    handled_named_output = true;
                }
            }
            if (handled_named_output)
                return;
        }

        // Handle shared-expert gate. In the ordinary path the stage has one
        // output, the gated shared contribution. In the fused gate-add path it
        // publishes both that gated contribution and the final routed+shared
        // combined row. Route by output name so both paths keep the same
        // semantic snapshot keys.
        if (name.find("_shared_expert_gate") != std::string::npos)
        {
            size_t pos = name.find("_shared_expert_gate");
            std::string prefix = name.substr(0, pos);

            for (const auto &output : dump.outputs)
            {
                const std::string output_name = output.name ? output.name : "";
                if (output_name == "shared_output" && output.data)
                    storeOutput(prefix + "_MOE_SHARED_GATE_OUTPUT", output);
                else if (output_name == "combined_output" && output.data)
                    storeOutput(prefix + "_MOE_COMBINED_OUTPUT", output);
            }
            return;
        }

        // Handle standalone MoE routing stage. Eager snapshot builds may
        // provide host-stashed router logits plus routing vectors, while
        // graph-captured runs publish only the tensor-backed routing outputs
        // after replay. Route by output name so both contracts preserve the
        // same parity keys without forcing host mirrors into graph capture.
        if (name.find("_moe_routing") != std::string::npos)
        {
            size_t pos = name.find("_moe_routing");
            std::string prefix = name.substr(0, pos);

            const StageDumpInfo::OutputBuffer *router_logits = nullptr;
            const StageDumpInfo::OutputBuffer *routing_indices = nullptr;
            const StageDumpInfo::OutputBuffer *routing_weights = nullptr;

            for (const auto &output : dump.outputs)
            {
                const std::string output_name = output.name ? output.name : "";
                if (output_name == "router_logits" || output_name == "logits")
                    router_logits = &output;
                else if (output_name == "routing_indices" ||
                         output_name == "indices" ||
                         output_name == "output_indices_tensor")
                    routing_indices = &output;
                else if (output_name == "routing_weights" ||
                         output_name == "weights" ||
                         output_name == "output_weights_tensor")
                    routing_weights = &output;
            }

            if (!router_logits && !routing_indices && !routing_weights &&
                dump.outputs.size() >= 3)
            {
                router_logits = &dump.outputs[0];
                routing_indices = &dump.outputs[1];
                routing_weights = &dump.outputs[2];
            }

            if (router_logits && router_logits->data)
                storeOutput(prefix + "_MOE_ROUTER_OUTPUT", *router_logits);
            if (routing_indices && routing_indices->data)
            {
                LOG_DEBUG("[Snapshot] MoE routing indices stage=" << name
                                                                     << " shape=["
                                                                     << routing_indices->rows
                                                                     << ','
                                                                     << routing_indices->cols
                                                                     << "] bytes="
                                                                     << routing_indices->byte_size);
                storeOutput(prefix + "_MOE_ROUTING_INDICES", *routing_indices);
            }
            if (routing_weights && routing_weights->data)
            {
                LOG_DEBUG("[Snapshot] MoE routing weights stage=" << name
                                                                     << " shape=["
                                                                     << routing_weights->rows
                                                                     << ','
                                                                     << routing_weights->cols
                                                                     << "] bytes="
                                                                     << routing_weights->byte_size);
                storeOutput(prefix + "_MOE_ROUTING_WEIGHTS", *routing_weights);
            }

            if (router_logits || routing_indices || routing_weights)
                return;
        }

        /*
         * Preserve the real per-target accumulator before the generic mapping
         * below overwrites `layerN_MOE_EXPERT_OUTPUT` with the final target's
         * result.  This creates diagnostic-only snapshots; it neither changes
         * the ticket protocol nor adds a producer-side copy.
         */
        if (const std::string cumulative_key =
                overlayTicketCumulativeSnapshotKey(name);
            !cumulative_key.empty() && !dump.outputs.empty() &&
            dump.outputs.front().data)
        {
            storeOutput(cumulative_key, dump.outputs.front());
        }

        // Standard single-output stages
        LOG_DEBUG("[Snapshot] Standard path: stage=" << name
                                                     << " outputs.size=" << dump.outputs.size()
                                                     << " out[0].data=" << (dump.outputs.empty() ? nullptr : dump.outputs[0].data));
        if (!dump.outputs.empty() && dump.outputs[0].data)
        {
            const auto &out = dump.outputs[0];
            auto data = extractFp32FromOutput(out);
            std::string key = convertStageNameToSnapshotKey(name);
            LOG_DEBUG("[Snapshot] Storing key=" << key << " count=" << data.size());

            if (data.size() >= 8 && key == "EMBEDDING")
            {
                LOG_DEBUG("[Snapshot] " << key << " first 8 values: "
                                        << data[0] << "," << data[1] << "," << data[2] << "," << data[3] << ","
                                        << data[4] << "," << data[5] << "," << data[6] << "," << data[7]);
            }

            if (!data.empty())
                storeSnapshot(key, std::move(data), out.rows, out.cols);
        }
    }

    /**
     * @brief Join context-qualified live rows into the prompt-wide parity view.
     *
     * The graph executor has already waited on each producing stream before
     * this diagnostic boundary. We therefore only copy host-visible FP32
     * values here. The bare semantic key remains the public parity API, while
     * each qualified key stays available for a CSV/artifact consumer that must
     * inspect the exact chunk where a numerical divergence began.
     */
    SnapshotChunkSequenceAggregation
    SnapshotCapture::aggregateSequentialChunkSnapshots(
        const std::vector<SnapshotChunkSequencePart> &chunks)
    {
        SnapshotChunkSequenceAggregation result;
        if (chunks.empty())
        {
            result.error = "cannot aggregate an empty prefill chunk sequence";
            return result;
        }

        struct SequencePieces
        {
            std::vector<StoredSnapshotHandle> snapshots;
            std::vector<bool> present;
        };

        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> prefixes;
        prefixes.reserve(chunks.size());
        for (size_t index = 0; index < chunks.size(); ++index)
        {
            const auto &chunk = chunks[index];
            if (chunk.context.empty() || chunk.logical_rows == 0)
            {
                result.error = "prefill chunk snapshot context is incomplete";
                return result;
            }

            const std::string prefix = snapshotContextPrefix(chunk.context) + "_";
            if (std::find(prefixes.begin(), prefixes.end(), prefix) != prefixes.end())
            {
                result.error = "prefill chunk snapshot contexts are not unique";
                return result;
            }
            prefixes.push_back(prefix);
        }

        std::unordered_map<std::string, SequencePieces> sequences;
        for (const auto &[key, snapshot] : snapshots_)
        {
            for (size_t chunk_index = 0;
                 chunk_index < prefixes.size();
                 ++chunk_index)
            {
                const std::string &prefix = prefixes[chunk_index];
                if (key.compare(0, prefix.size(), prefix) != 0)
                    continue;

                const std::string semantic_key = key.substr(prefix.size());
                if (semantic_key.empty())
                {
                    result.error = "prefill chunk snapshot has an empty semantic key";
                    return result;
                }

                auto [it, inserted] = sequences.try_emplace(semantic_key);
                if (inserted)
                {
                    it->second.snapshots.resize(chunks.size());
                    it->second.present.assign(chunks.size(), false);
                }
                if (it->second.present[chunk_index])
                {
                    result.error = "duplicate prefill chunk snapshot for '" +
                                   semantic_key + "'";
                    return result;
                }

                /*
                 * Keep a shared immutable publication while we insert final
                 * bare aggregates below.  That insertion can rehash the map,
                 * but it cannot invalidate this handle or its FP32 vector.
                 */
                it->second.snapshots[chunk_index] = snapshot;
                it->second.present[chunk_index] = true;
                break;
            }
        }

        for (const auto &[semantic_key, pieces] : sequences)
        {
            if (!pieces.present.front())
            {
                result.error = "prefill chunk sequence for '" + semantic_key +
                               "' has no first chunk";
                return result;
            }

            const StoredSnapshot &first = *pieces.snapshots.front();
            if (first.rows != chunks.front().logical_rows)
            {
                /*
                 * Last-token logits and other terminal-state values are
                 * legitimately narrower than an input chunk. Publish the
                 * latest context-qualified observation under the bare key.
                 * DeviceGraphOrchestrator normally captures an unscoped copy
                 * too, but making the aggregation self-contained keeps an
                 * artifact consumer from depending on callback ordering.
                 */
                size_t latest_chunk = pieces.present.size();
                while (latest_chunk > 0 && !pieces.present[latest_chunk - 1])
                    --latest_chunk;
                if (latest_chunk == 0)
                {
                    result.error = "terminal prefill snapshot for '" +
                                   semantic_key + "' has no captured chunk";
                    return result;
                }
                snapshots_[semantic_key] = pieces.snapshots[latest_chunk - 1];
                ++result.terminal_or_nonsequence_keys;
                continue;
            }

            if (first.cols == 0 ||
                first.rows > std::numeric_limits<size_t>::max() / first.cols ||
                first.data.size() != first.rows * first.cols)
            {
                result.error = "prefill chunk sequence for '" + semantic_key +
                               "' has an invalid first-chunk shape";
                return result;
            }

            size_t total_rows = 0;
            std::vector<float> joined;
            for (size_t chunk_index = 0;
                 chunk_index < chunks.size();
                 ++chunk_index)
            {
                if (!pieces.present[chunk_index])
                {
                    result.error = "prefill chunk sequence for '" + semantic_key +
                                   "' is missing chunk " +
                                   std::to_string(chunk_index);
                    return result;
                }

                const StoredSnapshot &piece = *pieces.snapshots[chunk_index];
                if (piece.rows != chunks[chunk_index].logical_rows ||
                    piece.cols != first.cols ||
                    piece.rows > std::numeric_limits<size_t>::max() / piece.cols ||
                    piece.data.size() != piece.rows * piece.cols ||
                    total_rows > std::numeric_limits<size_t>::max() - piece.rows)
                {
                    result.error = "prefill chunk sequence for '" + semantic_key +
                                   "' has inconsistent chunk " +
                                   std::to_string(chunk_index) +
                                   " (expected rows=" +
                                   std::to_string(chunks[chunk_index].logical_rows) +
                                   ", cols=" + std::to_string(first.cols) +
                                   "; got rows=" + std::to_string(piece.rows) +
                                   ", cols=" + std::to_string(piece.cols) +
                                   ", elements=" + std::to_string(piece.data.size()) +
                                   ")";
                    return result;
                }
                total_rows += piece.rows;

                if (joined.size() > std::numeric_limits<size_t>::max() -
                                        piece.data.size())
                {
                    result.error = "prefill chunk sequence for '" + semantic_key +
                                   "' overflows diagnostic storage";
                    return result;
                }
                joined.insert(joined.end(), piece.data.begin(), piece.data.end());
            }

            storeSnapshot(
                semantic_key,
                std::move(joined),
                total_rows,
                first.cols);
            ++result.aggregated_sequence_keys;
        }

        result.ok = true;
        return result;
    }

    // =========================================================================
    // FP32 extraction from various tensor formats
    // =========================================================================

    std::vector<float> SnapshotCapture::extractFp32FromOutput(const StageDumpInfo::OutputBuffer &out)
    {
        if (!out.data)
            return {};

        size_t count = out.rows * out.cols;
        if (count == 0)
            return {};

        std::vector<float> data(count);
        std::string dtype_str = out.dtype ? out.dtype : "FP32";

        LOG_TRACE("[extractFp32FromOutput] name=" << (out.name ? out.name : "?")
                                                  << " dtype=" << dtype_str
                                                  << " rows=" << out.rows << " cols=" << out.cols);

        // FP32: direct copy
        if (dtype_str == "FP32")
        {
            std::memcpy(data.data(), out.data, count * sizeof(float));
            return data;
        }

        if (dtype_str == "INT32")
        {
            const auto *int_data = static_cast<const int32_t *>(out.data);
            for (size_t i = 0; i < count; ++i)
                data[i] = static_cast<float>(int_data[i]);
            return data;
        }

        // Q8_1: dequantize blocks
        if (dtype_str == "Q8_1")
        {
            const Q8_1Block *blocks = static_cast<const Q8_1Block *>(out.data);
            constexpr int BLOCK_SIZE = 32;
            size_t num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (size_t b = 0; b < num_blocks; ++b)
            {
                const Q8_1Block &block = blocks[b];
                float scale = fp16_to_fp32(block.d);
                for (int i = 0; i < BLOCK_SIZE && b * BLOCK_SIZE + i < count; ++i)
                {
                    data[b * BLOCK_SIZE + i] = static_cast<float>(block.qs[i]) * scale;
                }
            }
            return data;
        }

        // Q16_1 variants: dequantize blocks (block sizes 32, 64, 128)
        if (dtype_str.find("Q16_1") == 0)
        {
            int block_size = 32;
            if (dtype_str.find("_64") != std::string::npos)
                block_size = 64;
            else if (dtype_str.find("_128") != std::string::npos)
                block_size = 128;

            const Q16_1Block *blocks = static_cast<const Q16_1Block *>(out.data);
            size_t num_blocks = (count + block_size - 1) / block_size;

            for (size_t b = 0; b < num_blocks; ++b)
            {
                const Q16_1Block &block = blocks[b];
                float scale = fp16_to_fp32(block.d);
                for (int i = 0; i < block_size && b * block_size + i < count; ++i)
                {
                    data[b * block_size + i] = static_cast<float>(block.qs[i]) * scale;
                }
            }
            return data;
        }

        // BF16 or FP16: convert to FP32
        if (dtype_str == "BF16" || dtype_str == "FP16")
        {
            const uint16_t *half_data = static_cast<const uint16_t *>(out.data);
            for (size_t i = 0; i < count; ++i)
            {
                if (dtype_str == "BF16")
                    data[i] = simd::bf16_to_fp32(half_data[i]);
                else
                    data[i] = simd::fp16_to_fp32(half_data[i]);
            }
            return data;
        }

        // Unknown dtype — warn and try FP32 (may be garbage)
        LOG_WARN("[extractFp32FromOutput] Unknown dtype '" << dtype_str << "', assuming FP32");
        std::memcpy(data.data(), out.data, count * sizeof(float));
        return data;
    }

    // =========================================================================
    // Stage name → snapshot key conversion
    // =========================================================================

    std::string SnapshotCapture::convertStageNameToSnapshotKey(const std::string &stage_name)
    {
        if (stage_name.find("_moe_expert_ffn_tier") != std::string::npos)
        {
            std::string result = stage_name;
            for (char &c : result)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            return result;
        }

        if (stage_name.find("_moe_sparse_return_reduce") != std::string::npos &&
            stage_name.find("_allreduce") != std::string::npos)
        {
            const size_t pos = stage_name.find("_moe_sparse_return_reduce");
            return stage_name.substr(0, pos) + "_MOE_EXPERT_OUTPUT_ALLREDUCED";
        }

        /*
         * The MTP graph builder names its embedding collective with the
         * implementation-facing lower-case prefix `mtp<depth>`, whereas all
         * other depth-qualified diagnostic keys use the canonical upper-case
         * `MTP<depth>` grammar.  Normalize this one boundary before the generic
         * suffix table runs.  Leaving the prefix lower-case makes
         * extractStageType() correctly reject it as an operational key, so the
         * post-allreduce embedding silently becomes UNKNOWN during LocalTP
         * snapshot combination.
         */
        constexpr std::string_view kEmbeddingAllreduce =
            "_embedding_allreduce";
        if (stage_name.ends_with(kEmbeddingAllreduce))
        {
            std::string prefix = stage_name.substr(
                0, stage_name.size() - kEmbeddingAllreduce.size());
            if (prefix.starts_with("mtp") && prefix.size() > 3)
            {
                const bool numeric_depth = std::all_of(
                    prefix.begin() + 3,
                    prefix.end(),
                    [](unsigned char character)
                    {
                        return std::isdigit(character) != 0;
                    });
                if (numeric_depth)
                    prefix.replace(0, 3, "MTP");
            }
            return prefix + "_EMBEDDING_ALLREDUCED";
        }

        // Ordered vector: longest/most-specific suffixes FIRST to ensure correct
        // prefix extraction. E.g. "_gdn_wo_allreduce" must match before "_wo_allreduce"
        // so the prefix is "layerN" (not "layerN_gdn").
        //
        // Collective stages intentionally publish diagnostic *_ALLREDUCED keys.
        // Canonical row-parallel keys such as ATTENTION_OUTPUT and FFN_DOWN are
        // produced by the per-device partial projection stages and combined by
        // TPSnapshot. This prevents graph-captured collective diagnostics from
        // overwriting the semantic stage snapshot used by parity tests.
        static const std::vector<std::pair<std::string, std::string>> suffix_map = {
            // MTP-prefixed embedding collectives retain the same canonical
            // post-reduction name as the unprefixed model graph.
            {"_embedding_allreduce", "_EMBEDDING_ALLREDUCED"},
            // GDN (Gated Delta Net) linear attention stages — longest suffixes first
            {"_gdn_wo_allreduce", "_ATTENTION_OUTPUT_ALLREDUCED"},
            {"_gdn_out_proj", "_ATTENTION_OUTPUT"},
            {"_gdn_proj", "_QKV_PROJECTION"},
            {"_short_conv", "_GDN_CONV1D_OUTPUT"},
            {"_gdn_recurrence", "_GDN_DELTA_RULE_OUTPUT"},
            {"_gated_norm", "_GDN_NORM_GATE_OUTPUT"},
            // Standard attention stages
            {"_attn_norm", "_ATTENTION_NORM"},
            {"_attn_residual", "_ATTENTION_RESIDUAL"},
            {"_attn_allreduce", "_ATTENTION_OUTPUT_ALLREDUCED"},
            {"_wo_allreduce", "_ATTENTION_OUTPUT_ALLREDUCED"},
            {"_wo_proj", "_ATTENTION_OUTPUT"},
            {"_q_norm", "_Q_NORM"},
            {"_k_norm", "_K_NORM"},
            {"_q_gate_split", "_FA_GATE"},
            {"_q_proj", "_Q_PROJECTION"},
            {"_k_proj", "_K_PROJECTION"},
            {"_v_proj", "_V_PROJECTION"},
            {"_q_rope", "_Q_ROPE"},
            {"_k_rope", "_K_ROPE"},
            {"_attn_output_gate", "_ATTENTION_CONTEXT_GATED"},
            {"_attention", "_ATTENTION_CONTEXT"},
            // FFN stages
            {"_down_allreduce", "_FFN_DOWN_ALLREDUCED"},
            {"_ffn_norm", "_FFN_NORM"},
            {"_ffn_gate", "_FFN_GATE"},
            {"_ffn_up", "_FFN_UP"},
            {"_swiglu", "_FFN_SWIGLU"},
            {"_down_proj", "_FFN_DOWN"},
            {"_ffn_residual", "_FFN_RESIDUAL"},
            // MoE stages
            {"_moe_expert_overlay_fast_allreduce", "_MOE_EXPERT_OUTPUT_ALLREDUCED"},
            {"_moe_overlay_continuation_broadcast", "_MOE_EXPERT_OUTPUT_ALLREDUCED"},
            {"_moe_canonical_publication_finalize", "_MOE_COMBINED_OUTPUT"},
            /*
             * A graph-native ExpertOverlay return is accumulated by CPU/MPI
             * sparse boundaries and copied back to the continuation GPU by a
             * ticket-consume stage.  The final consume in layer order holds
             * the complete routed-expert sum, before shared-expert addition.
             * Mapping it explicitly keeps heterogeneous parity evidence at
             * the same semantic checkpoint as the ordinary MoE expert stage.
             */
            {"_moe_overlay_ticket_consume", "_MOE_EXPERT_OUTPUT"},
            {"_shared_expert_allreduce", "_MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED"},
            {"_moe_sparse_return_reduce", "_MOE_EXPERT_OUTPUT"},
            {"_shared_expert_gate", "_MOE_SHARED_GATE_OUTPUT"},
            {"_shared_expert", "_MOE_SHARED_EXPERT_OUTPUT"},
            {"_moe_routed_expert_partial_reduce", "_MOE_EXPERT_OUTPUT"},
            {"_moe_expert_allreduce", "_MOE_EXPERT_OUTPUT_ALLREDUCED"},
            {"_moe_expert_ffn", "_MOE_EXPERT_OUTPUT"},
            {"_moe_combine", "_MOE_COMBINED_OUTPUT"},
            {"_moe_ffn", "_MOE_EXPERT_OUTPUT"},
            {"_moe_add", "_MOE_COMBINED_OUTPUT"},
        };

        // Global stages
        if (stage_name == "embedding")
            return "EMBEDDING";
        if (stage_name == "embedding_allreduce")
            return "EMBEDDING_ALLREDUCED";
        if (stage_name == "final_norm")
            return "FINAL_NORM";
        if (stage_name == "lm_head")
            return "LM_HEAD";

        // Layer-specific stages: extract layer prefix and convert suffix.
        // Uses ordered iteration so longer/more-specific suffixes match first.
        for (const auto &[suffix, replacement] : suffix_map)
        {
            size_t pos = stage_name.find(suffix);
            if (pos != std::string::npos)
            {
                std::string prefix = stage_name.substr(0, pos);
                return prefix + replacement;
            }
        }

        // Fallback: return original name (uppercase)
        std::string result = stage_name;
        for (char &c : result)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return result;
    }

    std::vector<std::string> SnapshotCapture::possibleKeysForStageName(const std::string &stage_name)
    {
        auto prefixBefore = [&](const std::string &needle) -> std::string
        {
            const size_t pos = stage_name.find(needle);
            return pos == std::string::npos ? stage_name : stage_name.substr(0, pos);
        };

        if (stage_name.find("_qkv_proj") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_qkv_proj");
            return {prefix + "_Q_PROJECTION", prefix + "_K_PROJECTION", prefix + "_V_PROJECTION"};
        }
        if (stage_name.find("_kv_proj") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_kv_proj");
            return {prefix + "_K_PROJECTION", prefix + "_V_PROJECTION"};
        }
        if (stage_name.find("_gate_up") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_gate_up");
            return {prefix + "_FFN_GATE", prefix + "_FFN_UP"};
        }
        if (stage_name.find("_rope") != std::string::npos &&
            stage_name.find("_q_rope") == std::string::npos &&
            stage_name.find("_k_rope") == std::string::npos)
        {
            const std::string prefix = prefixBefore("_rope");
            return {prefix + "_Q_ROPE", prefix + "_K_ROPE"};
        }
        if (stage_name.find("_gdn_proj") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_gdn_proj");
            return {prefix + "_QKV_PROJECTION",
                    prefix + "_GDN_Z_PROJECTION",
                    prefix + "_GDN_ALPHA",
                    prefix + "_GDN_BETA"};
        }
        if (stage_name.find("_q_gate_split") != std::string::npos)
        {
            return {prefixBefore("_q_gate_split") + "_FA_GATE"};
        }
        if (stage_name.find("_kv_append") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_kv_append");
            return {prefix + "_KV_CACHE_K",
                    prefix + "_KV_CACHE_V",
                    prefix + "_KV_APPEND_SOURCE_K",
                    prefix + "_KV_APPEND_SOURCE_V"};
        }
        if (stage_name.find("_attn_output_gate") != std::string::npos)
        {
            return {prefixBefore("_attn_output_gate") + "_ATTENTION_CONTEXT_GATED"};
        }
        if (stage_name.find("_gdn_wo_allreduce") != std::string::npos)
        {
            /*
             * Filtered captures are used by the grouped-verifier parity suite to
             * keep long-context diagnostics small.  Collective nodes publish
             * their own post-reduction keys, so the filter must advertise those
             * keys explicitly; otherwise the allreduce stage is skipped and the
             * CSV jumps from a local row-parallel partial to the next replicated
             * consumer.
             */
            return {prefixBefore("_gdn_wo_allreduce") + "_ATTENTION_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_wo_allreduce") != std::string::npos)
        {
            return {prefixBefore("_wo_allreduce") + "_ATTENTION_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_attn_allreduce") != std::string::npos)
        {
            return {prefixBefore("_attn_allreduce") + "_ATTENTION_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_down_allreduce") != std::string::npos)
        {
            return {prefixBefore("_down_allreduce") + "_FFN_DOWN_ALLREDUCED"};
        }
        if (stage_name.find("_shared_expert_allreduce") != std::string::npos)
        {
            return {prefixBefore("_shared_expert_allreduce") + "_MOE_SHARED_EXPERT_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_moe_overlay_continuation_broadcast") !=
            std::string::npos)
        {
            return {
                prefixBefore("_moe_overlay_continuation_broadcast") +
                "_MOE_EXPERT_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_moe_expert_overlay_fast_allreduce") != std::string::npos)
        {
            return {prefixBefore("_moe_expert_overlay_fast_allreduce") + "_MOE_EXPERT_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_moe_expert_allreduce") != std::string::npos)
        {
            return {prefixBefore("_moe_expert_allreduce") + "_MOE_EXPERT_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_moe_sparse_return_reduce") != std::string::npos &&
            stage_name.find("_allreduce") != std::string::npos)
        {
            const size_t pos = stage_name.find("_moe_sparse_return_reduce");
            return {stage_name.substr(0, pos) + "_MOE_EXPERT_OUTPUT_ALLREDUCED"};
        }
        if (stage_name.find("_attention") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_attention");
            return {prefix + "_ATTENTION_CONTEXT",
                    prefix + "_ATTENTION_EFFECTIVE_K",
                    prefix + "_ATTENTION_EFFECTIVE_V"};
        }
        if (stage_name == "lm_head_allgather")
        {
            return {"LM_HEAD"};
        }
        if ((stage_name.find("_attn_norm") != std::string::npos ||
             stage_name.find("_ffn_norm") != std::string::npos))
        {
            const std::string key = convertStageNameToSnapshotKey(stage_name);
            return {key, key + "_RESIDUAL_OUT"};
        }
        if (stage_name.find("_moe_ffn") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_moe_ffn");
            return {prefix + "_MOE_EXPERT_OUTPUT",
                    prefix + "_MOE_ROUTER_OUTPUT",
                    prefix + "_MOE_ROUTING_INDICES",
                    prefix + "_MOE_ROUTING_WEIGHTS"};
        }
        if (stage_name.find("_moe_expert_ffn") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_moe_expert_ffn");
            return {prefix + "_MOE_EXPERT_OUTPUT",
                    prefix + "_MOE_COMBINED_OUTPUT"};
        }
        if (stage_name.find("_moe_canonical_publication_finalize") !=
            std::string::npos)
        {
            const std::string prefix =
                prefixBefore("_moe_canonical_publication_finalize");
            return {prefix + "_MOE_EXPERT_OUTPUT",
                    prefix + "_MOE_SHARED_GATE_OUTPUT",
                    prefix + "_MOE_COMBINED_OUTPUT"};
        }
        if (stage_name.find("_shared_expert_gate") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_shared_expert_gate");
            return {prefix + "_MOE_SHARED_GATE_OUTPUT",
                    prefix + "_MOE_COMBINED_OUTPUT"};
        }
        if (const std::string cumulative_key =
                overlayTicketCumulativeSnapshotKey(stage_name);
            !cumulative_key.empty())
        {
            return {convertStageNameToSnapshotKey(stage_name), cumulative_key};
        }
        if (stage_name.find("_moe_routing") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_moe_routing");
            return {prefix + "_MOE_ROUTER_OUTPUT",
                    prefix + "_MOE_ROUTING_INDICES",
                    prefix + "_MOE_ROUTING_WEIGHTS"};
        }

        return {convertStageNameToSnapshotKey(stage_name)};
    }

    std::vector<std::string> SnapshotCapture::possibleKeysForStage(
        const std::string &stage_name,
        const StageDumpInfo &dump_info)
    {
        auto prefixBefore = [&](const std::string &needle) -> std::string
        {
            const size_t pos = stage_name.find(needle);
            return pos == std::string::npos
                       ? stage_name
                       : stage_name.substr(0, pos);
        };

        auto outputNamesToKeys = [&](const std::string &prefix,
                                     const bool canonical_finalizer)
        {
            std::vector<std::string> keys;
            keys.reserve(dump_info.outputs.size());
            for (const auto &output : dump_info.outputs)
            {
                const std::string output_name =
                    output.name ? output.name : "";
                if (output_name == "combined_output")
                    keys.push_back(prefix + "_MOE_COMBINED_OUTPUT");
                else if (output_name == "canonical_route_contributions")
                {
                    keys.push_back(
                        prefix + "_MOE_CANONICAL_ROUTE_CONTRIBUTIONS");
                }
                else if (canonical_finalizer &&
                         output_name == "routed_output")
                {
                    keys.push_back(prefix + "_MOE_EXPERT_OUTPUT");
                }
                else if (canonical_finalizer &&
                         output_name == "shared_output")
                {
                    keys.push_back(prefix + "_MOE_SHARED_GATE_OUTPUT");
                }
                else if (!canonical_finalizer && output_name == "output")
                    keys.push_back(prefix + "_MOE_EXPERT_OUTPUT");
            }
            return keys;
        };

        if (stage_name.find("_moe_canonical_publication_finalize") !=
            std::string::npos)
        {
            return outputNamesToKeys(
                prefixBefore("_moe_canonical_publication_finalize"),
                /*canonical_finalizer=*/true);
        }
        if (stage_name.find("_moe_expert_ffn") != std::string::npos)
        {
            return outputNamesToKeys(
                prefixBefore("_moe_expert_ffn"),
                /*canonical_finalizer=*/false);
        }
        if (stage_name.find("_shared_expert_gate") != std::string::npos)
        {
            const std::string prefix = prefixBefore("_shared_expert_gate");
            std::vector<std::string> keys;
            keys.reserve(dump_info.outputs.size());
            for (const auto &output : dump_info.outputs)
            {
                const std::string output_name =
                    output.name ? output.name : "";
                if (output_name == "shared_output" || output_name == "output")
                    keys.push_back(prefix + "_MOE_SHARED_GATE_OUTPUT");
                else if (output_name == "combined_output")
                    keys.push_back(prefix + "_MOE_COMBINED_OUTPUT");
            }
            return keys;
        }

        return possibleKeysForStageName(stage_name);
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    void SnapshotCapture::storeSnapshot(
        const std::string &key,
        std::vector<float> data,
        size_t rows,
        size_t cols)
    {
        snapshots_[key] = std::make_shared<const StoredSnapshot>(
            StoredSnapshot{
                .data = std::move(data),
                .rows = rows,
                .cols = cols,
            });
    }

    void SnapshotCapture::storeOutput(const std::string &key, const StageDumpInfo::OutputBuffer &out)
    {
        if (!out.data)
            return;
        auto data = extractFp32FromOutput(out);
        if (!data.empty())
            storeSnapshot(key, std::move(data), out.rows, out.cols);
    }

} // namespace llaminar2
