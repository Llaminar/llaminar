/**
 * @file GlobalOrchestrator.cpp
 * @brief Implementation of cross-machine MPI cluster inference orchestrator
 *
 * Named-domain runners own participant-local execution; this layer joins
 * their immutable request results. Prefix coordination preserves the epoch
 * selected by each lookup rather than relabeling it from later live state.
 * One explicit stage registry owns every runner; remote endpoints cannot be
 * replaced by an anonymous rank runner or the nearest local layer scope.
 * Global checkpoint envelopes retain one child per local stage; forwarding a
 * terminal operation unwraps exactly that stage while its nested TP owners,
 * payload handles and ready events remain independently authenticated.
 *
 * @author David Sanftenberg
 * @date April 2026
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 * Hosted activation banks retain the declared residual matrix geometry even
 * when a smaller request consumes only a prefix of their admitted rows.
 */

#include "GlobalOrchestrator.h"
#include "OriginalPipelineStageDomain.h"
#include "HostedPipelineActivationStorage.h"
#include "../local_execution/orchestrators/PipelineDeviceGeneration.h"
#include "../../transfer/CapturedTransferChannelProtocol.h"
#include <cstring>
#include "../local_execution/engine/PrefillBucketUtils.h"
#include <type_traits>
#include "../local_execution/orchestrators/PipelineForwardGraphEdges.h"
#include "../local_execution/orchestrators/PipelineActivationFanout.h"
#include "planning/PipelineTransferMemory.h"
#include "planning/PlanningPublication.h"
#include "utils/PerfStatsCollector.h"
#include "transfer/CapturedTransferChannel.h"
#include "transfer/TransferEngine.h"
#include <nlohmann/json.hpp>
#include <limits>
#include <array>
#include "../moe/MoEOverlayPipelineStageBinding.h"
#include "../moe/MoEOverlayResidencyAuthority.h"
#include "../local_execution/orchestrators/IRankOrchestrator.h"
#include "../global_pp/GlobalPPRankPlanBuilder.h"
#include "../prefix_cache/PrefixCacheCoordinator.h"
#include "../../tensors/TensorClasses.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"

#include <stdexcept>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <optional>

namespace llaminar2
{
    const ILocalTPContext *StageRunnerEntry::localTPContext() const noexcept
    {
        const auto *rank_runner = dynamic_cast<const IRankOrchestrator *>(runner.get());
        return rank_runner ? rank_runner->localTPContext() : nullptr;
    }

    namespace
    {
        struct ParsedLayerSnapshotKey
        {
            int layer = -1;
            std::string suffix;
        };

        std::optional<ParsedLayerSnapshotKey> parseLayerSnapshotKey(const std::string &key)
        {
            constexpr const char *prefix = "layer";
            constexpr size_t prefix_len = 5;
            if (key.rfind(prefix, 0) != 0 || key.size() <= prefix_len)
            {
                return std::nullopt;
            }

            size_t pos = prefix_len;
            if (!std::isdigit(static_cast<unsigned char>(key[pos])))
            {
                return std::nullopt;
            }

            int layer = 0;
            while (pos < key.size() && std::isdigit(static_cast<unsigned char>(key[pos])))
            {
                layer = layer * 10 + (key[pos] - '0');
                ++pos;
            }

            if (pos >= key.size() || key[pos] != '_')
            {
                return std::nullopt;
            }

            return ParsedLayerSnapshotKey{layer, key.substr(pos)};
        }

        std::string makeLayerSnapshotKey(int layer, const std::string &suffix)
        {
            return "layer" + std::to_string(layer) + suffix;
        }

        int greedyArgmaxToken(const float *logits, int vocab_size)
        {
            if (!logits || vocab_size <= 0)
            {
                return -1;
            }

            int token = 0;
            float best = logits[0];
            for (int i = 1; i < vocab_size; ++i)
            {
                if (logits[i] > best)
                {
                    best = logits[i];
                    token = i;
                }
            }
            return token;
        }

        bool stageOwnsGlobalLayer(const StageRunnerEntry &entry, int layer)
        {
            if (!entry.pp_stage_config.has_value())
            {
                return true;
            }
            const auto &stage = entry.pp_stage_config.value();
            return layer >= stage.first_layer && layer < stage.last_layer;
        }

        std::string globalizeStageSnapshotKey(const StageRunnerEntry &entry,
                                              const std::string &key)
        {
            auto parsed = parseLayerSnapshotKey(key);
            if (!parsed || !entry.pp_stage_config.has_value())
            {
                return key;
            }

            const auto &stage = entry.pp_stage_config.value();
            if (parsed->layer >= stage.first_layer && parsed->layer < stage.last_layer)
            {
                return key;
            }

            if (parsed->layer >= 0 && parsed->layer < stage.layerCount())
            {
                return makeLayerSnapshotKey(stage.first_layer + parsed->layer,
                                            parsed->suffix);
            }

            return key;
        }

        std::optional<std::string> localStageSnapshotKey(const StageRunnerEntry &entry,
                                                         const ParsedLayerSnapshotKey &parsed)
        {
            if (!entry.pp_stage_config.has_value())
            {
                return std::nullopt;
            }

            const auto &stage = entry.pp_stage_config.value();
            const int local_layer = parsed.layer - stage.first_layer;
            if (local_layer < 0 || local_layer >= stage.layerCount())
            {
                return std::nullopt;
            }

            return makeLayerSnapshotKey(local_layer, parsed.suffix);
        }
    }

    // =========================================================================
    // StageRunnerRegistry
    // =========================================================================

    void StageRunnerRegistry::add(StageRunnerEntry entry)
    {
        if (!entry.runner)
        {
            throw std::invalid_argument("StageRunnerRegistry: runner is required");
        }

        if (entry.stage_id < 0 && entry.action.stage_id >= 0)
        {
            entry.stage_id = entry.action.stage_id;
        }
        if (entry.action.stage_id < 0 && entry.stage_id >= 0)
        {
            entry.action.stage_id = entry.stage_id;
        }
        if (entry.stage_id < 0)
        {
            throw std::invalid_argument("StageRunnerRegistry: stage_id is required");
        }
        if (entry.action.stage_id >= 0 && entry.action.stage_id != entry.stage_id)
        {
            throw std::invalid_argument("StageRunnerRegistry: stage_id/action mismatch");
        }
        if (entry.domain_name.empty())
        {
            entry.domain_name = entry.action.domain_name;
        }
        if (entry.domain_name != entry.action.domain_name)
            throw std::invalid_argument("StageRunnerRegistry: domain name/action mismatch");

        for (const auto &existing : entries_)
        {
            if (existing.stage_id == entry.stage_id)
            {
                throw std::invalid_argument("StageRunnerRegistry: duplicate stage_id");
            }
            if ((existing.action.has_embedding && entry.action.has_embedding) ||
                (existing.action.has_lm_head && entry.action.has_lm_head))
                throw std::invalid_argument("StageRunnerRegistry: duplicate pipeline endpoint owner");
        }

        entries_.push_back(std::move(entry));
    }

    void StageRunnerRegistry::releaseAllAfterTerminalDrain()
    {
        entries_.clear();
    }

    bool StageRunnerRegistry::empty() const
    {
        return entries_.empty();
    }

    size_t StageRunnerRegistry::size() const
    {
        return entries_.size();
    }

    bool StageRunnerRegistry::hasRunnerForStage(int stage_id) const
    {
        return runnerForStage(stage_id) != nullptr;
    }

    StageRunnerEntry *StageRunnerRegistry::entryForStage(int stage_id)
    {
        for (auto &entry : entries_)
        {
            if (entry.stage_id == stage_id)
            {
                return &entry;
            }
        }
        return nullptr;
    }

    const StageRunnerEntry *StageRunnerRegistry::entryForStage(int stage_id) const
    {
        for (const auto &entry : entries_)
        {
            if (entry.stage_id == stage_id)
            {
                return &entry;
            }
        }
        return nullptr;
    }

    StageRunnerEntry *StageRunnerRegistry::entryForDomain(const std::string &domain_name)
    {
        for (auto &entry : entries_)
        {
            if (entry.domain_name == domain_name)
            {
                return &entry;
            }
        }
        return nullptr;
    }

    const StageRunnerEntry *StageRunnerRegistry::entryForDomain(const std::string &domain_name) const
    {
        for (const auto &entry : entries_)
        {
            if (entry.domain_name == domain_name)
            {
                return &entry;
            }
        }
        return nullptr;
    }

    IInferenceRunner *StageRunnerRegistry::runnerForStage(int stage_id)
    {
        for (auto &entry : entries_)
        {
            if (entry.stage_id == stage_id)
            {
                return entry.runner.get();
            }
        }
        return nullptr;
    }

    const IInferenceRunner *StageRunnerRegistry::runnerForStage(int stage_id) const
    {
        for (const auto &entry : entries_)
        {
            if (entry.stage_id == stage_id)
            {
                return entry.runner.get();
            }
        }
        return nullptr;
    }

    IInferenceRunner *StageRunnerRegistry::runnerForDomain(const std::string &domain_name)
    {
        if (auto *entry = entryForDomain(domain_name))
        {
            return entry->runner.get();
        }
        return nullptr;
    }

    const IInferenceRunner *StageRunnerRegistry::runnerForDomain(const std::string &domain_name) const
    {
        if (const auto *entry = entryForDomain(domain_name))
        {
            return entry->runner.get();
        }
        return nullptr;
    }

    IInferenceRunner *StageRunnerRegistry::pipelineHeadRunner()
    {
        for (auto &entry : entries_)
        {
            if (entry.action.has_embedding)
            {
                return entry.runner.get();
            }
        }
        return nullptr;
    }

    const IInferenceRunner *StageRunnerRegistry::pipelineHeadRunner() const
    {
        for (const auto &entry : entries_)
        {
            if (entry.action.has_embedding)
            {
                return entry.runner.get();
            }
        }
        return nullptr;
    }

    IInferenceRunner *StageRunnerRegistry::pipelineTailRunner()
    {
        for (auto entry_it = entries_.rbegin(); entry_it != entries_.rend(); ++entry_it)
        {
            if (entry_it->action.has_lm_head)
            {
                return entry_it->runner.get();
            }
        }
        return nullptr;
    }

    const IInferenceRunner *StageRunnerRegistry::pipelineTailRunner() const
    {
        for (auto entry_it = entries_.rbegin(); entry_it != entries_.rend(); ++entry_it)
        {
            if (entry_it->action.has_lm_head)
            {
                return entry_it->runner.get();
            }
        }
        return nullptr;
    }

    IInferenceRunner *StageRunnerRegistry::defaultRunner()
    {
        if (!entries_.empty())
        {
            return entries_.front().runner.get();
        }
        return nullptr;
    }

    const IInferenceRunner *StageRunnerRegistry::defaultRunner() const
    {
        if (!entries_.empty())
        {
            return entries_.front().runner.get();
        }
        return nullptr;
    }

    IInferenceRunner *StageRunnerRegistry::lastLocalRunner()
    {
        if (!entries_.empty())
        {
            return entries_.back().runner.get();
        }
        return nullptr;
    }

    const IInferenceRunner *StageRunnerRegistry::lastLocalRunner() const
    {
        if (!entries_.empty())
        {
            return entries_.back().runner.get();
        }
        return nullptr;
    }

    void StageRunnerRegistry::clearCacheAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->resetInferenceState(
                InferenceStateResetRequest::requestBoundary("global-stage-registry"));
        }
    }

    bool StageRunnerRegistry::purgePrefixCacheAll()
    {
        for (auto &entry : entries_)
        {
            if (!entry.runner || !entry.runner->purgePrefixCache())
                return false;
        }
        return true;
    }

    bool StageRunnerRegistry::configureMTPRequestStopTokensAll(
        const std::vector<int32_t> &stop_tokens)
    {
        bool saw_runner = false;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            if (!entry.runner)
            {
                throw std::logic_error(
                    "Cannot configure MTP request stop tokens on null global "
                    "stage runner " +
                    std::to_string(entry.stage_id) + " (domain='" +
                    entry.domain_name + "')");
            }
            if (!entry.runner->configureMTPRequestStopTokens(stop_tokens))
            {
                throw std::runtime_error(
                    "MTP request stop-token configuration was rejected by "
                    "global stage runner " +
                    std::to_string(entry.stage_id) + " (domain='" +
                    entry.domain_name + "')");
            }
        }


        if (!saw_runner)
        {
            throw std::logic_error(
                "Cannot configure MTP request stop tokens without a local "
                "global stage runner");
        }
        return true;
    }

    bool StageRunnerRegistry::configureMTPRequestPenaltyPolicyAll(
        const MTPRequestPenaltyPolicy &policy)
    {
        bool saw_runner = false;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            if (!entry.runner)
            {
                throw std::logic_error(
                    "Cannot configure MTP request penalty policy on null global "
                    "stage runner " +
                    std::to_string(entry.stage_id) + " (domain='" +
                    entry.domain_name + "')");
            }
            if (!entry.runner->configureMTPRequestPenaltyPolicy(policy))
            {
                throw std::runtime_error(
                    "MTP request penalty-policy configuration was rejected by "
                    "global stage runner " +
                    std::to_string(entry.stage_id) + " (domain='" +
                    entry.domain_name + "')");
            }
        }


        if (!saw_runner)
        {
            throw std::logic_error(
                "Cannot configure MTP request penalty policy without a local "
                "global stage runner");
        }
        return true;
    }

    void StageRunnerRegistry::setSkipLogitsGatherDecodeAll(bool skip)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setSkipLogitsGatherDecode(skip);
        }
    }

    void StageRunnerRegistry::setSkipLogitsGatherPrefillAll(bool skip)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setSkipLogitsGatherPrefill(skip);
        }
    }

    void StageRunnerRegistry::setSuppressTimelineAll(bool suppress)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setSuppressTimeline(suppress);
        }
    }

    void StageRunnerRegistry::setAccumulatePrefillAll(bool accumulate)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setAccumulatePrefill(accumulate);
        }
    }

    void StageRunnerRegistry::flushStageTimelineAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->flushStageTimeline();
        }
    }

    bool StageRunnerRegistry::supportsChainedMTPDraftsAll() const
    {
        bool saw_runner = false;
        for (const auto &entry : entries_)
        {
            saw_runner = true;
            if (!entry.runner->supportsChainedMTPDrafts())
            {
                return false;
            }
        }
        return saw_runner;
    }

    bool StageRunnerRegistry::forwardMTPAll(int32_t draft_condition_token)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->forwardMTP(draft_condition_token) && ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::forwardMTPFromLastDraftAll(
        int32_t draft_condition_token,
        int position_id)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->forwardMTPFromLastDraft(
                     draft_condition_token,
                     position_id) &&
                 ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::commitMTPShiftedRowsFromLastForwardAll(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->commitMTPShiftedRowsFromLastForward(
                     tokens,
                     token_count,
                     already_appended_tokens) &&
                 ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::commitMTPShiftedRowsFromPartialForwardAll(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard,
        int position_offset_override,
        int already_appended_shifted_kv_tokens)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->commitMTPShiftedRowsFromPartialForward(
                     tokens,
                     token_count,
                     already_appended_tokens,
                     main_forward_token_count,
                     allow_speculative_discard,
                     position_offset_override,
                     already_appended_shifted_kv_tokens) &&
                 ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::commitMTPShiftedRowFromCurrentTerminalHiddenAll(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->commitMTPShiftedRowFromCurrentTerminalHidden(
                     token,
                     already_appended_tokens,
                     allow_speculative_discard,
                     position_offset_override) &&
                 ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::commitMTPShiftedRowFromCheckpointTerminalHiddenAll(
        const PrefixStateSnapshot &checkpoint,
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!checkpoint.valid || checkpoint.participant_snapshots.empty())
            return false;

        size_t participant_index = 0;
        bool saw_runner = false;
        bool ok = true;
        auto commit_runner = [&](IInferenceRunner &runner)
        {
            saw_runner = true;
            if (participant_index >= checkpoint.participant_snapshots.size())
            {
                ok = false;
                return;
            }
            ok = runner.commitMTPShiftedRowFromCheckpointTerminalHidden(
                     checkpoint.participant_snapshots[participant_index++],
                     token,
                     already_appended_tokens,
                     allow_speculative_discard,
                     position_offset_override) &&
                 ok;
        };

        for (auto &entry : entries_)
        {
            commit_runner(*entry.runner);
        }
        return saw_runner && ok &&
               participant_index == checkpoint.participant_snapshots.size();
    }

    bool StageRunnerRegistry::setComputeAllPositionLogitsAll(bool enabled)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->setComputeAllPositionLogits(enabled) && ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::setComputeRowIndexedAllPositionLogitsAll(
        bool enabled,
        int row_count)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->setComputeRowIndexedAllPositionLogits(
                     enabled,
                     row_count) &&
                 ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::setMTPSpecVerifierInputPlanAll(
        const MTPSpecDecodeVerifierInputPlan &plan)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->setMTPSpecVerifierInputPlan(plan) && ok;
        }
        return saw_runner && ok;
    }

    void StageRunnerRegistry::clearMTPSpecVerifierInputPlanAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->clearMTPSpecVerifierInputPlan();
        }
    }

    bool StageRunnerRegistry::publishGroupedDecodeEquivalentMTPSpecStateBatchAll(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        bool saw_runner = false;
        bool ok = true;
        std::string first_error;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            std::string local_error;
            const bool runner_ok =
                entry.runner->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                    plans,
                    &local_error);
            if (!runner_ok && first_error.empty())
                first_error = local_error;
            ok = runner_ok && ok;
        }
        if ((!saw_runner || !ok) && error)
        {
            *error = first_error.empty()
                         ? "global grouped decode-equivalent publication had no local runner"
                         : first_error;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::ensureMTPCheckpointTerminalHiddenAll()
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner && entry.runner->ensureMTPCheckpointTerminalHidden() && ok;
        }
        return saw_runner && ok;
    }

    uint64_t StageRunnerRegistry::moePlacementEpochAll() const
    {
        uint64_t epoch = 0;
        for (const auto &entry : entries_)
        {
            epoch = std::max(epoch, entry.runner->moePlacementEpoch());
        }
        return epoch;
    }

    uint64_t StageRunnerRegistry::moeRuntimeMovementEpochAll() const
    {
        uint64_t epoch = 0;
        for (const auto &entry : entries_)
        {
            epoch = std::max(epoch, entry.runner->moeRuntimeMovementEpoch());
        }
        return epoch;
    }

    PrefixLookupResult StageRunnerRegistry::lookupPrefixAll(
        const std::vector<int32_t> &tokens)
    {
        last_prefix_hits_.clear();

        std::vector<PrefixParticipantLookup> participants;
        int block_size = 0;
        int participant_id = 0;
        for (auto &entry : entries_)
        {
            PrefixLookupResult hit = entry.runner->lookupPrefix(tokens);
            if (block_size <= 0 && hit.block_size > 0)
                block_size = hit.block_size;
            participants.push_back(makePrefixParticipantLookup(
                participant_id++,
                entry.runner->primaryDeviceId(),
                hit,
                entry.domain_name,
                PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally));
            last_prefix_hits_.push_back(std::move(hit));
        }

        if (participants.empty())
            return {};

        PrefixLookupResult aggregate = makePrefixLookupResult(
            coordinatePrefixLookups(std::move(participants)), block_size);
        const int common_tokens = std::max(0, aggregate.cached_tokens);
        if (common_tokens > 0)
        {
            auto copy_representative = [&](const PrefixLookupResult &candidate)
            {
                if (candidate.cached_tokens < common_tokens || candidate.blocks.empty())
                    return false;
                aggregate.blocks = candidate.clampedTo(common_tokens).blocks;
                return true;
            };
            for (const auto &hit : last_prefix_hits_)
            {
                if (copy_representative(hit))
                {
                    break;
                }
            }
        }
        return aggregate;
    }

    PrefixRestoreMetadata StageRunnerRegistry::prefixRestoreMetadataAll(const PrefixLookupResult &hit) const
    {
        PrefixRestoreMetadata metadata;
        if (hit.cached_tokens <= 0) return metadata;
        if (last_prefix_hits_.size() != entries_.size())
            throw std::logic_error("Prefix restore metadata has no complete stage admission");
        for (size_t index = 0; index < entries_.size(); ++index)
        {
            const auto selected = last_prefix_hits_[index].clampedTo(hit.cached_tokens);
            if (selected.cached_tokens != hit.cached_tokens)
                throw std::logic_error("Prefix restore metadata disagrees with the stage frontier");
            metadata.merge(entries_[index].runner->prefixRestoreMetadata(selected));
        }
        return metadata;
    }

    bool StageRunnerRegistry::preparePrefixHarvestAll(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        const PrefixHarvestSchedule &schedule)
    {
        if (!admission.supported || !admission.cache_enabled ||
            entries_.empty() || last_prefix_hits_.size() != entries_.size())
            return false;
        for (size_t index = 0u; index < entries_.size(); ++index)
        {
            auto &child_admission = last_prefix_hits_[index];
            child_admission = child_admission.forHarvest(admission.cached_tokens);
            if (!entries_[index].runner->preparePrefixHarvest(
                    child_admission, tokens, schedule))
                return false;
        }
        return true;
    }

    bool StageRunnerRegistry::populatePrefixAll(
        const PrefixLookupResult &hit,
        int seq_idx)
    {
        (void)hit.restoreBlocks();
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0)
            return true;
        if (last_prefix_hits_.size() != entries_.size())
        {
            return false;
        }

        for (size_t index = 0; index < entries_.size(); ++index)
        {
            PrefixLookupResult child_hit =
                last_prefix_hits_[index].clampedTo(common_tokens);
            child_hit.restore_model_runtime_state = hit.restore_model_runtime_state;
            child_hit.restore_hybrid_state_for_suffix_prefill =
                hit.restore_hybrid_state_for_suffix_prefill;
            if (!entries_[index].runner->populatePrefix(child_hit, seq_idx))
            {
                clearCacheAll();
                return false;
            }
        }
        return true;
    }

    bool StageRunnerRegistry::harvestPrefixAll(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        int prompt_token_count)
    {
        if (!admission.cache_enabled || !admission.supported ||
            last_prefix_hits_.size() != entries_.size())
        {
            return false;
        }

        bool saw_runner = false;
        bool ok = true;
        for (size_t index = 0; index < entries_.size(); ++index)
        {
            saw_runner = true;
            ok = entries_[index].runner->harvestPrefix(
                     last_prefix_hits_[index],
                     tokens,
                     prompt_token_count) &&
                 ok;
        }
        return saw_runner && ok;
    }

    bool StageRunnerRegistry::restorePrefixTerminalStateAll(
        const PrefixLookupResult &hit)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0 ||
            (hit.requires_terminal_logits && !hit.has_terminal_logits))
        {
            return false;
        }
        if (last_prefix_hits_.size() != entries_.size())
        {
            return false;
        }

        for (size_t index = 0; index < entries_.size(); ++index)
        {
            if (!entries_[index].runner->restorePrefixTerminalState(
                    last_prefix_hits_[index].clampedTo(common_tokens)))
            {
                return false;
            }
        }
        return true;
    }

    PrefixRuntimeStateSnapshot StageRunnerRegistry::prefixStateProbeAll(
        const PrefixProbeCapturePolicy &capture_policy) const
    {
        std::vector<PrefixRuntimeStateSnapshot> children;
        children.reserve(entries_.size());
        for (const auto &entry : entries_)
            children.push_back(entry.runner->prefixStateProbe(capture_policy));
        if (children.empty())
            return {};
        if (children.size() == 1u)
            return std::move(children.front());

        std::vector<PrefixMovementEpochSnapshot> movement_epochs;
        movement_epochs.reserve(children.size());
        for (const auto &child : children) movement_epochs.push_back(child.moe_runtime_movement_epoch);
        PrefixRuntimeStateSnapshot aggregate = std::move(children.front());
        aggregate.moe_runtime_movement_epoch = PrefixMovementEpochSnapshot::pipeline(std::move(movement_epochs));
        aggregate.execution_path = "global-stage-registry";

        constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
        constexpr uint64_t kFnvPrime = 1099511628211ull;
        auto fold_value = [](uint64_t digest, uint64_t value)
        {
            for (unsigned byte = 0; byte < sizeof(value); ++byte)
            {
                digest ^= value & 0xffull;
                digest *= kFnvPrime;
                value >>= 8;
            }
            return digest;
        };
        auto fold_hash = [&](bool child_available,
                             size_t child_bytes,
                             uint64_t child_hash,
                             bool *available,
                             size_t *bytes,
                             uint64_t *hash)
        {
            if (!child_available)
                return;
            uint64_t digest = *available ? *hash : kFnvOffsetBasis;
            digest = fold_value(digest, static_cast<uint64_t>(child_bytes));
            digest = fold_value(digest, child_hash);
            *available = true;
            *bytes += child_bytes;
            *hash = digest;
        };

        for (size_t index = 1; index < children.size(); ++index)
        {
            const auto &child = children[index];
            aggregate.initialized = aggregate.initialized && child.initialized;
            aggregate.prefill_logits_ready =
                aggregate.prefill_logits_ready || child.prefill_logits_ready;
            aggregate.has_hidden = aggregate.has_hidden || child.has_hidden;
            aggregate.has_logits = aggregate.has_logits || child.has_logits;
            aggregate.current_position =
                std::max(aggregate.current_position, child.current_position);
            aggregate.session_epoch =
                std::max(aggregate.session_epoch, child.session_epoch);
            aggregate.live_state_epoch =
                std::max(aggregate.live_state_epoch, child.live_state_epoch);
            aggregate.live_state_mutations += child.live_state_mutations;
            aggregate.live_state_accepted_publications +=
                child.live_state_accepted_publications;
            aggregate.live_state_rejected_corrections +=
                child.live_state_rejected_corrections;
            aggregate.live_state_prefix_restores += child.live_state_prefix_restores;
            aggregate.live_state_prefix_truncates += child.live_state_prefix_truncates;
            aggregate.live_state_session_resets += child.live_state_session_resets;

            aggregate.prefix_cache_config_enabled =
                aggregate.prefix_cache_config_enabled &&
                child.prefix_cache_config_enabled;
            aggregate.prefix_cache_ready =
                aggregate.prefix_cache_ready && child.prefix_cache_ready;
            aggregate.prefix_cache_bypassed =
                aggregate.prefix_cache_bypassed || child.prefix_cache_bypassed;
            if (aggregate.prefix_cache_bypass_reason.empty())
                aggregate.prefix_cache_bypass_reason = child.prefix_cache_bypass_reason;

            auto sum = [&](uint64_t PrefixRuntimeStateSnapshot::*member)
            {
                aggregate.*member += child.*member;
            };
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_lookups);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_hits);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_partial_hits);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_misses);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_matched_blocks);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_matched_tokens);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_stores);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_inserts);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_evictions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_promotions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_ram_to_disk_demotions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_device_hot_promotions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_device_hot_repromotions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_device_hot_evictions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_disk_evictions);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_device_hot_direct_hits);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_disk_hydrations);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_terminal_state_hits);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_ram_bytes);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_device_bytes);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_disk_bytes);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_hybrid_state_bytes);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_mtp_state_bytes);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_bypasses);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_unsupported_backend_bypasses);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_fingerprint_bypasses);
            sum(&PrefixRuntimeStateSnapshot::prefix_cache_terminal_state_bypasses);

            aggregate.mtp_config_enabled =
                aggregate.mtp_config_enabled || child.mtp_config_enabled;
            aggregate.mtp_bypassed = aggregate.mtp_bypassed || child.mtp_bypassed;
            if (aggregate.mtp_bypass_reason.empty())
                aggregate.mtp_bypass_reason = child.mtp_bypass_reason;
            sum(&PrefixRuntimeStateSnapshot::mtp_draft_steps);
            sum(&PrefixRuntimeStateSnapshot::mtp_accepted_tokens);
            sum(&PrefixRuntimeStateSnapshot::mtp_rejected_tokens);
            sum(&PrefixRuntimeStateSnapshot::mtp_rollbacks);
            sum(&PrefixRuntimeStateSnapshot::mtp_bypasses);
            sum(&PrefixRuntimeStateSnapshot::mtp_verifier_runs);
            sum(&PrefixRuntimeStateSnapshot::mtp_verifier_token_count);
            sum(&PrefixRuntimeStateSnapshot::mtp_stochastic_accept_tests);
            sum(&PrefixRuntimeStateSnapshot::mtp_stochastic_accepts);
            sum(&PrefixRuntimeStateSnapshot::mtp_stochastic_residual_samples);
            sum(&PrefixRuntimeStateSnapshot::mtp_stochastic_terminal_samples);
            sum(&PrefixRuntimeStateSnapshot::mtp_transaction_commits);
            sum(&PrefixRuntimeStateSnapshot::mtp_transaction_rollbacks);
            sum(&PrefixRuntimeStateSnapshot::mtp_transaction_validation_failures);
            sum(&PrefixRuntimeStateSnapshot::mtp_unsafe_verifier_state_rejections);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_windows);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_updates);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_promotions);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_demotions);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_utility_returns);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_utility_keeps);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_utility_probes);
            sum(&PrefixRuntimeStateSnapshot::mtp_depth_policy_observe_recommendations);
            sum(&PrefixRuntimeStateSnapshot::prefill_chunk_schedules);
            sum(&PrefixRuntimeStateSnapshot::prefill_chunk_successful_schedules);
            sum(&PrefixRuntimeStateSnapshot::prefill_chunks);
            sum(&PrefixRuntimeStateSnapshot::prefill_chunk_real_tokens);
            sum(&PrefixRuntimeStateSnapshot::prefill_chunk_padded_tokens);
            sum(&PrefixRuntimeStateSnapshot::prefill_chunk_failures);

            fold_hash(
                child.terminal_hidden_hash_available,
                child.terminal_hidden_bytes,
                child.terminal_hidden_hash,
                &aggregate.terminal_hidden_hash_available,
                &aggregate.terminal_hidden_bytes,
                &aggregate.terminal_hidden_hash);
            aggregate.terminal_hidden_values.insert(
                aggregate.terminal_hidden_values.end(),
                child.terminal_hidden_values.begin(),
                child.terminal_hidden_values.end());
            fold_hash(
                child.terminal_logits_hash_available,
                child.terminal_logits_bytes,
                child.terminal_logits_hash,
                &aggregate.terminal_logits_hash_available,
                &aggregate.terminal_logits_bytes,
                &aggregate.terminal_logits_hash);
            aggregate.terminal_logits_values.insert(
                aggregate.terminal_logits_values.end(),
                child.terminal_logits_values.begin(),
                child.terminal_logits_values.end());
            if (aggregate.primary_device.is_cpu() && !child.primary_device.is_cpu())
                aggregate.primary_device = child.primary_device;
            aggregate.kv_caches.insert(
                aggregate.kv_caches.end(),
                child.kv_caches.begin(), child.kv_caches.end());
            aggregate.mtp_kv_caches.insert(
                aggregate.mtp_kv_caches.end(),
                child.mtp_kv_caches.begin(), child.mtp_kv_caches.end());
            aggregate.gdn_layers.insert(
                aggregate.gdn_layers.end(),
                child.gdn_layers.begin(), child.gdn_layers.end());
            aggregate.prefill_graphs.insert(
                aggregate.prefill_graphs.end(),
                child.prefill_graphs.begin(), child.prefill_graphs.end());
        }
        if (aggregate.prefix_cache_bypassed)
            aggregate.prefix_cache_ready = false;
        return aggregate;
    }

    PrefixStateSnapshot StageRunnerRegistry::captureLivePrefixStateAll(int seq_idx) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        bool have_common_provenance = false;
        bool same_provenance = true;
        int common_tokens = 0;
        PrefixStateProvenance common_provenance = PrefixStateProvenance::Unknown;

        auto capture_runner = [&](const IInferenceRunner &runner)
        {
            saw_runner = true;
            PrefixStateSnapshot child = runner.captureLivePrefixState(seq_idx);
            if (!child.valid)
                return false;

            if (!have_common_tokens)
            {
                common_tokens = child.cached_tokens;
                have_common_tokens = true;
            }
            else if (child.cached_tokens != common_tokens)
            {
                return false;
            }

            if (!have_common_provenance)
            {
                common_provenance = child.provenance;
                have_common_provenance = true;
            }
            else if (child.provenance != common_provenance)
            {
                same_provenance = false;
            }

            aggregate.participant_snapshots.push_back(std::move(child));
            return true;
        };

        for (const auto &entry : entries_)
        {
            if (!capture_runner(*entry.runner))
                return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            return {};
        }

        aggregate.valid = true;
        aggregate.cached_tokens = common_tokens;
        aggregate.provenance = same_provenance ? common_provenance
                                               : PrefixStateProvenance::PayloadCheckpoint;
        return aggregate;
    }

    PrefixStateSnapshot StageRunnerRegistry::captureLivePrefixCheckpointAll(
        const PrefixCheckpointCaptureRequest &request) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        bool have_common_provenance = false;
        bool same_provenance = true;
        bool all_logical = true;
        int common_tokens = 0;
        PrefixStateProvenance common_provenance = PrefixStateProvenance::Unknown;

        auto capture_runner = [&](const IInferenceRunner &runner)
        {
            saw_runner = true;
            PrefixStateSnapshot child =
                runner.captureLivePrefixCheckpoint(request);
            if (!child.valid)
                return false;

            if (!have_common_tokens)
            {
                common_tokens = child.cached_tokens;
                have_common_tokens = true;
            }
            else if (child.cached_tokens != common_tokens)
            {
                return false;
            }

            all_logical = all_logical && child.logical_checkpoint;
            if (!have_common_provenance)
            {
                common_provenance = child.provenance;
                have_common_provenance = true;
            }
            else if (child.provenance != common_provenance)
            {
                same_provenance = false;
            }
            aggregate.participant_snapshots.push_back(std::move(child));
            return true;
        };

        for (const auto &entry : entries_)
        {
            if (!capture_runner(*entry.runner))
                return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            return {};
        }

        aggregate.valid = true;
        aggregate.logical_checkpoint = all_logical;
        aggregate.cached_tokens = common_tokens;
        aggregate.provenance = same_provenance
                                   ? common_provenance
                                   : (all_logical
                                          ? PrefixStateProvenance::LogicalCheckpoint
                                          : PrefixStateProvenance::PayloadCheckpoint);
        return aggregate;
    }

    const PrefixStateSnapshot *StageRunnerRegistry::prefixCheckpointForRunner(
        const PrefixStateSnapshot &snapshot, const IInferenceRunner &runner) const
    {
        if (!snapshot.valid || snapshot.participant_snapshots.size() != entries_.size())
            return nullptr;
        for (size_t index = 0; index < entries_.size(); ++index)
        {
            if (entries_[index].runner.get() != &runner)
                continue;
            const auto &child = snapshot.participant_snapshots[index];
            if (!child.valid || child.cached_tokens != snapshot.cached_tokens)
                return nullptr;
            return &child;
        }
        return nullptr;
    }

    bool StageRunnerRegistry::restoreLivePrefixStateAll(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        if (!snapshot.valid || snapshot.participant_snapshots.empty())
            return false;

        size_t participant_index = 0;
        bool saw_runner = false;
        bool ok = true;

        auto restore_runner = [&](IInferenceRunner &runner)
        {
            saw_runner = true;
            if (participant_index >= snapshot.participant_snapshots.size())
            {
                ok = false;
                return;
            }
            ok = runner.restoreLivePrefixState(
                     snapshot.participant_snapshots[participant_index++],
                     seq_idx) &&
                 ok;
        };

        for (auto &entry : entries_)
        {
            restore_runner(*entry.runner);
        }

        return saw_runner && ok && participant_index == snapshot.participant_snapshots.size();
    }

    bool StageRunnerRegistry::truncateLivePrefixStateAll(int cached_tokens, int seq_idx)
    {
        bool saw_runner = false;
        bool ok = true;
        for (auto &entry : entries_)
        {
            saw_runner = true;
            ok = entry.runner->truncateLivePrefixState(cached_tokens, seq_idx) && ok;
        }
        return saw_runner && ok;
    }

    std::string StageRunnerRegistry::mtpDecodeUnsupportedReasonAll() const
    {
        bool saw_runner = false;
        for (const auto &entry : entries_)
        {
            saw_runner = true;
            std::string reason = entry.runner->mtpDecodeUnsupportedReason();
            if (!reason.empty())
                return reason;
        }
        return saw_runner ? std::string{} : "MTP decode requires a global stage runner";
    }

    void StageRunnerRegistry::enableSnapshotCaptureAll(const std::string &output_dir)
    {
        for (auto &entry : entries_)
        {
            entry.runner->enableSnapshotCapture(output_dir);
        }
    }

    void StageRunnerRegistry::setSnapshotCaptureFilterAll(
        const std::vector<std::string> &keys)
    {
        for (auto &entry : entries_)
        {
            if (keys.empty())
            {
                entry.runner->setSnapshotCaptureFilter({});
                continue;
            }

            std::vector<std::string> stage_keys;
            stage_keys.reserve(keys.size() * 2u);
            for (const auto &key : keys)
            {
                auto parsed = parseLayerSnapshotKey(key);
                if (!parsed)
                {
                    stage_keys.push_back(key);
                    continue;
                }
                if (!stageOwnsGlobalLayer(entry, parsed->layer))
                    continue;

                // Stage graph builders may expose either global layer names or
                // their stage-local ordinal. Accept both identities at this
                // topology boundary; snapshot lookup later globalizes the one
                // actually published.
                stage_keys.push_back(key);
                if (auto local_key = localStageSnapshotKey(entry, *parsed);
                    local_key && *local_key != key)
                {
                    stage_keys.push_back(std::move(*local_key));
                }
            }
            std::sort(stage_keys.begin(), stage_keys.end());
            stage_keys.erase(
                std::unique(stage_keys.begin(), stage_keys.end()),
                stage_keys.end());
            entry.runner->setSnapshotCaptureFilter(stage_keys);
        }
    }

    void StageRunnerRegistry::disableSnapshotCaptureAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->disableSnapshotCapture();
        }
    }

    void StageRunnerRegistry::clearSnapshotsAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->clearSnapshots();
        }
    }

    const float *StageRunnerRegistry::getSnapshot(const std::string &key, size_t &out_size) const
    {
        auto parsed = parseLayerSnapshotKey(key);
        for (const auto &entry : entries_)
        {
            if (parsed && !stageOwnsGlobalLayer(entry, parsed->layer))
            {
                continue;
            }

            const float *snapshot = entry.runner->getSnapshot(key, out_size);
            if (snapshot)
            {
                return snapshot;
            }

            if (parsed)
            {
                auto local_key = localStageSnapshotKey(entry, *parsed);
                if (local_key && *local_key != key)
                {
                    snapshot = entry.runner->getSnapshot(*local_key, out_size);
                    if (snapshot)
                    {
                        return snapshot;
                    }
                }
            }
        }
        out_size = 0;
        return nullptr;
    }

    SnapshotInfo StageRunnerRegistry::getSnapshotWithShape(const std::string &key) const
    {
        auto parsed = parseLayerSnapshotKey(key);
        for (const auto &entry : entries_)
        {
            if (parsed && !stageOwnsGlobalLayer(entry, parsed->layer))
            {
                continue;
            }

            SnapshotInfo snapshot = entry.runner->getSnapshotWithShape(key);
            if (snapshot)
            {
                return snapshot;
            }

            if (parsed)
            {
                auto local_key = localStageSnapshotKey(entry, *parsed);
                if (local_key && *local_key != key)
                {
                    snapshot = entry.runner->getSnapshotWithShape(*local_key);
                    if (snapshot)
                    {
                        return snapshot;
                    }
                }
            }
        }
        return {};
    }

    std::vector<std::string> StageRunnerRegistry::snapshotKeysAll() const
    {
        std::vector<std::string> keys;
        auto append_unique = [&keys](const std::vector<std::string> &runner_keys) {
            for (const auto &key : runner_keys)
            {
                if (std::find(keys.begin(), keys.end(), key) == keys.end())
                {
                    keys.push_back(key);
                }
            }
        };

        for (const auto &entry : entries_)
        {
            auto runner_keys = entry.runner->getSnapshotKeys();
            for (auto &key : runner_keys)
            {
                key = globalizeStageSnapshotKey(entry, key);
            }
            append_unique(runner_keys);
        }
        return keys;
    }

    // =========================================================================
    // StageActivationRouter
    // =========================================================================

    size_t StageActivationRouter::transferElementCount(int last_seq_len, int d_model)
    {
        const int effective_seq_len = last_seq_len > 0 ? last_seq_len : 1;
        return static_cast<size_t>(effective_seq_len) * static_cast<size_t>(d_model);
    }

    void StageActivationRouter::ensureActivationBufferSize(std::shared_ptr<TensorBase> &activation_buffer,
                                                           size_t num_elements)
    {
        if(!activation_buffer || activation_buffer->numel()<num_elements)
            throw std::logic_error("Hosted pipeline transfer exceeds its original cold-admitted activation bank");
    }

    bool StageActivationRouter::executeTransfer(const RankTransferAction &action,
                                                StageRunnerRegistry &registry,
                                                IMPIContext &mpi_ctx,
                                                int rank,
                                                int last_seq_len,
                                                int d_model,
                                                std::shared_ptr<TensorBase> &activation_buffer,
                                                ForwardExecutionPhase phase) const
    {
        if (action.direction == RankTransferAction::Direction::NONE)
            return true;

        const auto &mpi_log = debugEnv().mpi_logging;
        const size_t count = transferElementCount(last_seq_len, d_model);
        const size_t bytes = count * sizeof(float);
        const auto recordOutgoing = [&](DeviceId source, const char *transport)
        {
            // The original sender or local handoff records only a completed
            // live payload, after its host copy and any MPI send have succeeded.
            PerfStatsCollector::addCounterWithSequence(
                "transfer", "hosted_pipeline_activation_bytes", static_cast<double>(bytes),
                {static_cast<std::uint64_t>(last_seq_len), static_cast<std::uint64_t>(bytes)},
                phase == ForwardExecutionPhase::Prefill ? "prefill" : "decode", source.toString(),
                {{"source_stage", std::to_string(action.from_stage)},
                 {"consumer_stage", std::to_string(action.to_stage)}, {"transport", transport}});
        };

        if (action.direction == RankTransferAction::Direction::LOCAL_HANDOFF)
        {
            IInferenceRunner *source_runner = registry.runnerForStage(action.from_stage);
            IInferenceRunner *destination_runner = registry.runnerForStage(action.to_stage);
            if (!source_runner || !destination_runner)
            {
                LOG_ERROR("StageActivationRouter: missing runner for local handoff stage "
                          << action.from_stage << " -> " << action.to_stage
                          << " on rank " << rank);
                return false;
            }

            TensorBase *hidden = source_runner->getHiddenState();
            if (!hidden)
            {
                LOG_ERROR("StageActivationRouter: rank " << rank
                          << " has no hidden state for local handoff stage "
                          << action.from_stage << " -> " << action.to_stage);
                return false;
            }
            if (hidden->numel() < count)
            {
                LOG_ERROR("StageActivationRouter: hidden state too small for local handoff: have "
                          << hidden->numel() << " elements, need " << count);
                return false;
            }

            ensureActivationBufferSize(activation_buffer, count);
            const auto copied = TransferEngine::instance().copyActivation(
                hidden, activation_buffer.get(), DeviceId::cpu(), bytes);
            if (!copied.success)
            {
                LOG_ERROR("StageActivationRouter: exact local handoff failed: " << copied.error);
                return false;
            }
            destination_runner->setHiddenState(activation_buffer.get());
            recordOutgoing(source_runner->primaryDeviceId(), "local");

            LOG_DEBUG("StageActivationRouter: rank " << rank
                      << " LOCAL_HANDOFF " << count << " floats stage "
                      << action.from_stage << " -> " << action.to_stage);
            return true;
        }

        if (action.direction == RankTransferAction::Direction::SEND)
        {
            IInferenceRunner *source_runner = registry.runnerForStage(action.from_stage);
            if (!source_runner)
            {
                LOG_ERROR("StageActivationRouter: missing source runner for SEND stage "
                          << action.from_stage << " on rank " << rank);
                return false;
            }

            TensorBase *hidden = source_runner->getHiddenState();
            if (!hidden)
            {
                LOG_ERROR("StageActivationRouter: rank " << rank
                          << " has no hidden state to send to rank " << action.peer_rank
                          << " (stage " << action.from_stage << " -> " << action.to_stage << ")");
                return false;
            }
            if (hidden->numel() < count)
            {
                LOG_ERROR("StageActivationRouter: hidden state too small for SEND: have "
                          << hidden->numel() << " elements, need " << count);
                return false;
            }

            const float *send_data = nullptr;
            if (source_runner->primaryDeviceId().is_gpu())
            {
                ensureActivationBufferSize(activation_buffer, count);
                const auto copied = TransferEngine::instance().copyActivation(
                    hidden, activation_buffer.get(), DeviceId::cpu(), bytes);
                if (!copied.success)
                {
                    LOG_ERROR("StageActivationRouter: exact MPI send staging failed: " << copied.error);
                    return false;
                }
                send_data = activation_buffer->data();
            }
            else
            {
                send_data = hidden->data();
            }
            if (mpi_log.log_collectives)
            {
                LOG_DEBUG("[MPI] rank " << rank << " SEND " << count
                         << " floats (" << (count * sizeof(float) / 1024) << " KB)"
                         << " to rank " << action.peer_rank << " tag=" << action.mpi_tag);
            }
            else
            {
                LOG_DEBUG("StageActivationRouter: rank " << rank
                          << " SEND " << count << " floats to rank " << action.peer_rank
                          << " tag=" << action.mpi_tag);
            }

            try
            {
                auto start_time = std::chrono::steady_clock::now();
                mpi_ctx.sendFloat(send_data, count, action.peer_rank, action.mpi_tag);
                if (mpi_log.log_timing)
                {
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                    double bw_mbps = (count * sizeof(float)) / (ms * 1000.0);
                    LOG_DEBUG("[MPI] rank " << rank << " SEND complete: "
                             << ms << "ms, " << bw_mbps << " MB/s");
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("StageActivationRouter: rank " << rank
                          << " sendFloat failed (peer=" << action.peer_rank
                          << " count=" << count << " tag=" << action.mpi_tag
                          << "): " << e.what());
                return false;
            }
            recordOutgoing(source_runner->primaryDeviceId(), "mpi");
            return true;
        }

        IInferenceRunner *destination_runner = registry.runnerForStage(action.to_stage);
        if (!destination_runner)
        {
            LOG_ERROR("StageActivationRouter: missing destination runner for RECV stage "
                      << action.to_stage << " on rank " << rank);
            return false;
        }

        ensureActivationBufferSize(activation_buffer, count);
        float *recv_data = activation_buffer->mutable_data();

        if (mpi_log.log_collectives)
        {
            LOG_DEBUG("[MPI] rank " << rank << " RECV " << count
                     << " floats (" << (count * sizeof(float) / 1024) << " KB)"
                     << " from rank " << action.peer_rank << " tag=" << action.mpi_tag);
        }
        else
        {
            LOG_DEBUG("StageActivationRouter: rank " << rank
                      << " RECV " << count << " floats from rank " << action.peer_rank
                      << " tag=" << action.mpi_tag);
        }

        try
        {
            auto start_time = std::chrono::steady_clock::now();

            int timeout_ms = mpi_log.recv_timeout_ms;
            if (timeout_ms > 0)
            {
                bool ready = false;
                auto deadline = start_time + std::chrono::milliseconds(timeout_ms);
                bool warned = false;

                while (!ready)
                {
                    MPI_Status probe_status;
                    ready = mpi_ctx.iprobe(action.peer_rank, action.mpi_tag, &probe_status);
                    if (ready) break;

                    auto now = std::chrono::steady_clock::now();
                    if (!warned && now >= deadline)
                    {
                        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
                        LOG_WARN("StageActivationRouter: rank " << rank
                                 << " RECV from rank " << action.peer_rank
                                 << " blocked for " << elapsed_ms << "ms"
                                 << " (expected " << count << " floats, "
                                 << (count * sizeof(float) / 1024) << " KB"
                                 << ", tag=" << action.mpi_tag << ")");
                        warned = true;
                    }
                }
            }

            mpi_ctx.recvFloat(recv_data, count, action.peer_rank, action.mpi_tag, nullptr);

            if (mpi_log.log_timing)
            {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                double bw_mbps = (count * sizeof(float)) / (ms * 1000.0);
                LOG_DEBUG("[MPI] rank " << rank << " RECV complete: "
                         << ms << "ms, " << bw_mbps << " MB/s");
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("StageActivationRouter: rank " << rank
                      << " recvFloat failed (peer=" << action.peer_rank
                      << " count=" << count << " tag=" << action.mpi_tag
                      << "): " << e.what());
            return false;
        }

        destination_runner->setHiddenState(activation_buffer.get());
        return true;
    }

    // =========================================================================
    // Construction
    // =========================================================================

    GlobalOrchestrator::GlobalOrchestrator(Config config)
        : config_(std::move(config))
    {
        // Validate required fields
        if (!config_.mpi_ctx)
        {
            throw std::invalid_argument("GlobalOrchestrator: mpi_ctx is required");
        }
        if (config_.topology.stages.empty())
        {
            throw std::invalid_argument("GlobalOrchestrator: topology must have at least one stage");
        }
        if (config_.rank < 0 || config_.rank >= config_.world_size)
        {
            throw std::invalid_argument("GlobalOrchestrator: rank out of range [0, world_size)");
        }
        if (config_.vocab_size <= 0)
        {
            throw std::invalid_argument("GlobalOrchestrator: vocab_size must be positive");
        }
        if (config_.d_model <= 0)
        {
            throw std::invalid_argument("GlobalOrchestrator: d_model must be positive");
        }

        // Build this rank's execution plan from topology
        rank_plan_ = GlobalPPRankPlanBuilder::build(config_.topology, config_.rank);

        const auto local_actions = rank_plan_.executeStages();
        for (auto &entry : config_.stage_runners)
        {
            const auto expected = std::find_if(local_actions.begin(), local_actions.end(),
                [&](const auto *action) { return action->stage_id == entry.stage_id; });
            if (expected == local_actions.end() ||
                entry.action.stage_id != (*expected)->stage_id ||
                entry.action.role != RankStageAction::Role::EXECUTE ||
                entry.action.first_layer != (*expected)->first_layer ||
                entry.action.last_layer != (*expected)->last_layer ||
                entry.action.has_embedding != (*expected)->has_embedding ||
                entry.action.has_lm_head != (*expected)->has_lm_head ||
                entry.action.domain_name != (*expected)->domain_name)
                throw std::invalid_argument("GlobalOrchestrator: stage runner does not match this rank's topology action");
            // A runner cannot claim a different local layer interval from the
            // action that routes its activations and determines terminal authority.
            if (entry.pp_stage_config &&
                (entry.pp_stage_config->first_layer != (*expected)->first_layer ||
                 entry.pp_stage_config->last_layer != (*expected)->last_layer + 1 ||
                 entry.pp_stage_config->has_embedding != (*expected)->has_embedding ||
                 entry.pp_stage_config->has_lm_head != (*expected)->has_lm_head))
                throw std::invalid_argument("GlobalOrchestrator: stage runner scope disagrees with its topology action");
            stage_runners_.add(std::move(entry));
        }
        config_.stage_runners.clear();

        if (stage_runners_.empty())
        {
            throw std::invalid_argument("GlobalOrchestrator: explicit stage runners are required");
        }

        for (const auto *action : rank_plan_.executeStages())
        {
            if (!stage_runners_.hasRunnerForStage(action->stage_id))
            {
                throw std::invalid_argument("GlobalOrchestrator: missing runner for executable stage");
            }
        }

        LOG_DEBUG("GlobalOrchestrator: rank " << config_.rank << "/" << config_.world_size
                 << ", " << config_.topology.numStages() << " PP stages, "
                 << rank_plan_.steps.size() << " execution steps");
        LOG_DEBUG("GlobalOrchestrator plan:\n" << rank_plan_.toString());

        // Cache pipeline head/tail
        tail_rank_ = findTailRank();
        for (const auto *action : rank_plan_.executeStages())
        {
            if (action->has_embedding)
                is_pipeline_head_ = true;
            if (action->has_lm_head)
                is_pipeline_tail_ = true;
        }

        if(!requiresOriginalRankGPUTransport() &&
           HostedPipelineActivationStorage::requiresBank(config_.topology, rank_plan_)) {
            if(!config_.physical_memory || config_.physical_memory->worldRank()!=config_.rank)
                throw std::invalid_argument("Hosted pipeline activation requires its original rank PMA");
            const auto root=HostedPipelineActivationStorage::forRows(config_.hosted_activation_rows,config_.d_model);
            // The original allocator owns capacity, while the model owns row
            // geometry. Flattening this bank loses the residual stride required
            // by the next stage's exact-live activation copy contract.
            activation_buffer_=std::make_shared<FP32Tensor>(
                std::vector<std::size_t>{static_cast<std::size_t>(config_.hosted_activation_rows),
                    static_cast<std::size_t>(config_.d_model)},root.allocate(config_.physical_memory));
        }

        LOG_DEBUG("GlobalOrchestrator: pipeline_head=" << is_pipeline_head_
                 << " pipeline_tail=" << is_pipeline_tail_
                 << " tail_rank=" << tail_rank_);
    }

    GlobalOrchestrator::~GlobalOrchestrator()
    {
        if (rank_active_command_) abortOriginalRankGenerationCommand();
        // The follower retains native completion fences borrowing the original
        // graph/workspace owners. Close that service before either the pipeline
        // pool or the stage registry releases those owners.
        rank_transaction_follower_.reset();
        if (rank_generation_) rank_generation_->retireForwardOnlyBeforeStageRelease();
        if(config_.original_cpu_domains) {
            // Original CPU retirement is a native domain collective. All actual
            // local graph/worker/completion borrowers must disappear first;
            // ordinary context RAII cannot discharge this admitted duty.
            rank_generation_.reset();
            stage_runners_.releaseAllAfterTerminalDrain();
            config_.stage_runners.clear();
            try {config_.original_cpu_domains->retireAfterGraphRelease();}
            catch(...) {std::terminate();}
        }
    }

    bool GlobalOrchestrator::materializeServingGraphFamilyWithoutLaunch(
        const ServingGraphFamilyMaterializationPlan &plan)
    {
        if (!requiresOriginalRankGPUTransport()) {
            if(!plan.valid()) return false;
            if (hosted_serving_family_plan_ && !hosted_serving_family_plan_->sameCaptureIdentity(plan))
                throw std::invalid_argument("Hosted pipeline changed its original retained serving family");
            const auto actions=rank_plan_.executeStages();
            if(actions.empty()) return false;
            for(const auto* action:actions) {
                auto* entry=stage_runners_.entryForStage(action->stage_id);
                if(!entry || !entry->runner) return false;
                auto child=plan;
                child.pipeline_hidden_input=action->has_embedding?nullptr:activation_buffer_.get();
                if(!action->has_embedding && !child.pipeline_hidden_input)
                    throw std::logic_error("Hosted pipeline child lost its original stable activation input");
                if(!entry->runner->materializeServingGraphFamilyWithoutLaunch(child)) return false;
            }
            hosted_serving_family_plan_ = plan;
            return true;
        }
        prepareOriginalRankPipelinePorts(plan);
        if (!rank_generation_)
        {
            std::vector<PipelineDeviceGeneration::RankStageBinding> originals;
            for (const auto *action : rank_plan_.executeStages())
            {
                auto *entry = stage_runners_.entryForStage(action->stage_id);
                if (!entry || !entry->runner)
                    throw std::logic_error("Original global cohort lost a declared stage runner");
                originals.push_back({entry->runner.get(), action});
            }
            rank_generation_ = std::make_unique<PipelineDeviceGeneration>(config_.topology,
                rank_plan_, originals, rank_forward_edges_);
        }
        if (!rank_generation_->prepareServing(plan)) return false;
        prepareOriginalRankGenerationService();
        return true;
    }

    bool GlobalOrchestrator::supportsPrefillChunkSchedule(int seq_len) const
    {
        if (seq_len <= 0) return false;
        if (requiresOriginalRankGPUTransport() &&
            (!rank_generation_ || !rank_generation_->prefillPrepared())) return false;
        for (const auto *action : rank_plan_.executeStages())
        {
            const auto *entry = stage_runners_.entryForStage(action->stage_id);
            if (!entry || !entry->runner || !entry->runner->supportsPrefillChunkSchedule(seq_len)) return false;
        }
        return true;
    }

    bool GlobalOrchestrator::forwardPrefillChunkSchedule(const int *tokens, int seq_len,
        const PrefillChunkSchedulerPolicy &policy, int pad_token_id, bool allow_padded_execution)
    {
        if (!supportsPrefillChunkSchedule(seq_len) || !tokens) return false;
        if (requiresOriginalRankGPUTransport())
            return rank_generation_->prefill(tokens, seq_len, policy, pad_token_id, allow_padded_execution);
        if(config_.topology.numStages()==1) {
            auto* runner=stage_runners_.defaultRunner();
            return runner && runner->forwardPrefillChunkSchedule(tokens,seq_len,policy,pad_token_id,allow_padded_execution);
        }
        const auto schedule = planPrefillChunkSchedule(policy);
        if (!schedule || schedule.chunks.empty() || policy.real_token_count != seq_len ||
            policy.real_token_start != get_position())
            return false;
        for (const auto &chunk : schedule.chunks)
        {
            const int offset = chunk.token_offset - policy.real_token_start;
            if (offset < 0 || chunk.real_count <= 0 || offset > seq_len - chunk.real_count ||
                chunk.real_count > config_.hosted_activation_rows)
                return false;
            // Preserve the original retained prefill shape at every child.
            // A one-row tail is still prefill; auto-forward would select decode.
            const auto widths = policy.width_policy == PrefillChunkWidthPolicy::ExactRetained
                ? exactPrefillGraphWidths(chunk.bucket_seq_len) : std::vector<int>{chunk.bucket_seq_len};
            auto child = PrefillChunkSchedulerPolicy::forRetainedBucketFamily(
                widths, chunk.token_offset, chunk.real_count, policy.width_policy);
            // The parent has already resolved cumulative maintenance windows.
            // A child executes one chunk, so its real-count threshold carries
            // that original decision without restarting the parent's interval.
            child.min_rebalance_interval_tokens = chunk.rebalance_allowed_after ? chunk.real_count : 0;
            child.max_rebalance_interval_tokens = chunk.rebalance_required_after ? chunk.real_count : 0;
            if (!forwardHosted(tokens + offset, chunk.real_count, ForwardExecutionPhase::Prefill, &child,
                               pad_token_id, allow_padded_execution))
                return false;
        }
        return true;
    }

    bool GlobalOrchestrator::requiresOriginalRankGPUTransport() const noexcept
    {
        return config_.topology.stages.size() > 1 &&
            std::all_of(config_.topology.stages.begin(), config_.topology.stages.end(), [](const auto &stage) {
                return !stage.devices.empty() && std::all_of(stage.devices.begin(), stage.devices.end(),
                    [](const auto &member) { return member.toLocalDeviceId().is_gpu(); });
            });
    }

    void GlobalOrchestrator::validateOriginalRankGPUTransportContext(const std::shared_ptr<IMPIContext> &original) const
    {
        if (!requiresOriginalRankGPUTransport() || !original || !config_.original_mpi_ctx ||
            original.get() != config_.mpi_ctx || original.get() != config_.original_mpi_ctx.get() ||
            original.owner_before(config_.original_mpi_ctx) || config_.original_mpi_ctx.owner_before(original) ||
            original->rank() != config_.rank || original->world_size() != config_.world_size ||
            config_.topology.world_size != original->world_size() || !original->clusterInventory())
            throw std::invalid_argument("Global GPU pipeline requires its exact original shared MPI/rank publication");
        for (const auto *action : rank_plan_.executeStages())
        {
            const auto *entry = stage_runners_.entryForStage(action->stage_id);
            if (!entry || !entry->original_mpi_ctx || entry->original_mpi_ctx.get() != original.get() ||
                entry->original_mpi_ctx.owner_before(original) || original.owner_before(entry->original_mpi_ctx))
                throw std::invalid_argument("Global GPU pipeline stage substituted its original MPI owner");
        }
    }

    void GlobalOrchestrator::bindOriginalRankGPUTransportSession(const std::shared_ptr<IMPIContext> &original,
        std::shared_ptr<CapturedTransferRankSession> session)
    {
        validateOriginalRankGPUTransportContext(original);
        if (!session || !rank_boundaries_.empty() || !rank_forward_edges_.empty() || rank_port_family_ ||
            (rank_transfer_session_ && (rank_transfer_session_.get() != session.get() ||
                rank_transfer_session_.owner_before(session) || session.owner_before(rank_transfer_session_))))
            throw std::invalid_argument("Global GPU pipeline cannot replace its cold original transport session");
        session->requireOriginalContext(original);
        rank_transfer_session_ = std::move(session);
    }

    void GlobalOrchestrator::prepareOriginalRankPipelinePorts(const ServingGraphFamilyMaterializationPlan &plan)
    {
        // OR's distinct validate/bind initialization phases already proved
        // this actual shared context on every rank. Never construct an alias
        // when that original owner is absent; no safe collective is possible.
        const auto mpi = config_.original_mpi_ctx;
        if (!mpi) throw std::invalid_argument("Global pipeline lost its previously validated original MPI owner");
        const auto count = config_.topology.stages.size();
        const bool reuse = rank_port_family_.has_value();
        std::vector<std::optional<PipelineDeviceGeneration::RankStageEndpoint>> local;
        std::vector<std::array<int, 3>> envelope; // Immutable cold observations, not a live byte ledger.
        exchangePlanningSamples(mpi, [&](int ranks) {
            const auto text = nlohmann::json{plan.prefill_bucket_rows, plan.prefill_pad_token_id,
                int(plan.main_decode_graph)}.dump();
            return std::vector<std::vector<std::uint8_t>>(ranks, std::vector<std::uint8_t>(text.begin(), text.end()));
        }, [&](auto declaration) {
            // Every fallible local role/session/family check participates in
            // this existing consensus BEFORE any peer enters channel setup.
            validateOriginalRankGPUTransportContext(mpi);
            if (!rank_transfer_session_ || !plan.valid() || plan.pipeline_hidden_input)
                throw std::invalid_argument("Global GPU pipeline requires its original session and complete root serving family");
            rank_transfer_session_->requireOriginalContext(mpi);
            if (nlohmann::json::parse(declaration) != nlohmann::json{plan.prefill_bucket_rows,
                    plan.prefill_pad_token_id, int(plan.main_decode_graph)})
                throw std::invalid_argument("Global pipeline original serving declaration disagrees");
            if (reuse)
            {
                if (!rank_port_family_->sameCaptureIdentity(plan))
                    throw std::logic_error("Global pipeline cannot rebind its retained serving geometry/ports");
                return std::vector<std::uint8_t>{'R'};
            }
            if (!rank_boundaries_.empty() || !rank_forward_edges_.empty())
                throw std::logic_error("Failed original pipeline port setup cannot be reused as a ready family");
            local.resize(count);
            envelope.resize(count);
            size_t members = 0;
            for (const auto *action : rank_plan_.executeStages())
            {
                if (action->devices.size() > std::numeric_limits<size_t>::max() - members)
                    throw std::overflow_error("Global pipeline local member count overflow");
                members += action->devices.size();
            }
            rank_forward_edges_.reserve(members); // No allocation may follow publication of an installed DGO pointer.
            nlohmann::json observation = nlohmann::json::array();
            for (const auto *action : rank_plan_.executeStages())
            {
                if (action->stage_id < 0 || size_t(action->stage_id) >= count)
                    throw std::invalid_argument("Global pipeline has a foreign original stage ID");
                auto *entry = stage_runners_.entryForStage(action->stage_id);
                if (!entry || !entry->runner)
                    throw std::invalid_argument("Global pipeline omitted its actual local runner");
                auto endpoint = PipelineDeviceGeneration::inspectOriginalRankStage(*entry->runner, *action);
                if (endpoint.row_capacity <= 0 || endpoint.verifier_rows <= 0 || endpoint.residual_width <= 0 ||
                    endpoint.memory->worldRank() != mpi->rank() ||
                    endpoint.leader != action->devices.front().toLocalDeviceId())
                    throw std::invalid_argument("Global pipeline cold endpoint changed its original authority or rows");
                local[size_t(action->stage_id)] = endpoint;
                observation.push_back({action->stage_id, endpoint.row_capacity, endpoint.verifier_rows, endpoint.residual_width});
            }
            const auto text = observation.dump();
            return std::vector<std::uint8_t>(text.begin(), text.end());
        }, [&](auto receipts) {
            for (const auto &receipt : receipts)
                if ((receipt.bytes.size() == 1 && receipt.bytes[0] == 'R') != reuse)
                    throw std::invalid_argument("Global pipeline peers disagree on original retained port lifecycle");
            if (reuse) return;
            std::vector<bool> seen(count, false);
            for (const auto &receipt : receipts)
                for (const auto &row : nlohmann::json::parse(receipt.bytes))
                {
                    if (!row.is_array() || row.size() != 4)
                        throw std::invalid_argument("Global pipeline cold endpoint observation is incomplete");
                    const int stage = row.at(0).template get<int>();
                    if (stage < 0 || size_t(stage) >= count || seen[size_t(stage)] ||
                        config_.topology.stages[size_t(stage)].owning_rank != receipt.discovery_rank)
                        throw std::invalid_argument("Global pipeline cold observation has foreign or duplicate original stage ownership");
                    seen[size_t(stage)] = true;
                    envelope[size_t(stage)] = {row.at(1).template get<int>(), row.at(2).template get<int>(), row.at(3).template get<int>()};
                }
            if (std::find(seen.begin(), seen.end(), false) != seen.end())
                throw std::invalid_argument("Global pipeline lacks an original endpoint observation");
        });
        if (reuse) return;
        exchangePlanningArtifact(mpi, PlanningArtifact::TransferSamples, [&] {
            const auto text = nlohmann::json(envelope).dump();
            return std::vector<std::uint8_t>(text.begin(), text.end());
        }, [&](auto wire) {
            envelope = nlohmann::json::parse(wire).template get<std::vector<std::array<int, 3>>>();
            if (envelope.size() != count || std::any_of(envelope.begin(), envelope.end(), [&](const auto &rows) {
                    return rows[0] <= 0 || rows[1] <= 0 || rows[2] <= 0 ||
                        rows[1] != envelope.front()[1] || rows[2] != envelope.front()[2] ||
                        *std::max_element(plan.prefill_bucket_rows.begin(), plan.prefill_bucket_rows.end()) > rows[0];
                }))
                throw std::invalid_argument("Global pipeline serving family exceeds an original stage envelope");
            rank_boundaries_.resize(count - 1);
            rank_graph_capacity_rows_ = 0;
            for (const auto &capacity : envelope)
                rank_graph_capacity_rows_ = std::max(rank_graph_capacity_rows_, std::max(capacity[0], capacity[1]));
            rank_draft_capacity_ = envelope.front()[1] - 1;
        });
        std::vector<std::array<std::optional<CapturedTransferRankPlan>, 2>> declarations;
        std::vector<PipelineTransferMemory> geometries;
        exchangePlanningSamples(mpi, [&](int ranks) {
            return std::vector<std::vector<std::uint8_t>>(ranks, std::vector<std::uint8_t>{1});
        }, [&](auto wire) {
            if (wire.size() != 1 || wire[0] != 1)
                throw std::invalid_argument("Original pipeline boundary setup declaration disagrees");
            declarations.resize(count - 1);
            geometries.reserve(count - 1);
            const auto inventory = mpi->clusterInventory();
            for (size_t boundary = 0; boundary + 1 < count; ++boundary)
            {
                const auto &source = config_.topology.stages[boundary];
                const auto &target = config_.topology.stages[boundary + 1];
                const auto *edge = config_.topology.transferBetween(int(boundary), int(boundary + 1));
                if (!edge || edge->physicalConnection().locality() == RankConnectionLocality::CrossNode ||
                    source.is_global_tp || target.is_global_tp || source.devices.empty() || target.devices.empty())
                    throw std::invalid_argument("Original captured GPU pipeline requires authenticated node-local single-rank stages");
                geometries.push_back(PipelineTransferMemory::forRows(size_t(envelope[boundary][2]),
                    size_t(envelope[boundary][0]), size_t(envelope[boundary][1])));
                if (source.owning_rank != target.owning_rank)
                    for (size_t reverse = 0; reverse < 2; ++reverse)
                    {
                        const auto &producer = reverse ? target : source;
                        const auto &consumer = reverse ? source : target;
                        declarations[boundary][reverse] = CapturedTransferRankPlan::fromInventory(*inventory,
                            {rank_transfer_session_->originalGeneration(), 2 * boundary + (reverse ? 2 : 1)},
                            {producer.owning_rank, producer.devices.front().toLocalDeviceId()},
                            {consumer.owning_rank, consumer.devices.front().toLocalDeviceId()},
                            reverse ? geometries.back().metadata_capacity_bytes : geometries.back().activation_capacity_bytes,
                            reverse ? CapturedTransferEndpoint::Consumer : CapturedTransferEndpoint::Producer);
                    }
            }
            return std::vector<std::uint8_t>{1};
        }, [](auto receipts) {
            for (const auto &receipt : receipts)
                if (receipt.bytes.size() != 1 || receipt.bytes[0] != 1)
                    throw std::invalid_argument("Original pipeline boundary declaration omitted a rank");
        });
        auto &transfer = TransferEngine::instance();
        for (size_t boundary = 0; boundary + 1 < count; ++boundary)
        {
            const auto &source = config_.topology.stages[boundary];
            const auto &target = config_.topology.stages[boundary + 1];
            const auto geometry = geometries[boundary];
            for (const bool reverse : {false, true})
            {
                const auto producer = reverse ? boundary + 1 : boundary;
                const auto consumer = reverse ? boundary : boundary + 1;
                const auto &producer_stage = config_.topology.stages[producer];
                const auto &consumer_stage = config_.topology.stages[consumer];
                auto &channel = reverse ? rank_boundaries_[boundary].metadata : rank_boundaries_[boundary].activation;
                const size_t capacity = reverse ? geometry.metadata_capacity_bytes : geometry.activation_capacity_bytes;
                if (source.owning_rank != target.owning_rank)
                {
                    const auto &declaration = *declarations[boundary][reverse ? 1 : 0];
                    const auto role = declaration.localRole(mpi->rank());
                    const auto member = role ? (*role == CapturedTransferEndpoint::Producer ? producer : consumer) : count;
                    channel = transfer.createRankCapturedTransferChannel(declaration, rank_transfer_session_,
                        role ? local[member]->memory : nullptr, role ? local[member]->setup_stream : nullptr);
                }
                else
                {
                    // Same-rank domains use the same original channel body and
                    // admitted authority, never a host FP32 activation bounce.
                    exchangePlanningSamples(mpi, [&](int ranks) {
                        return std::vector<std::vector<std::uint8_t>>(ranks, std::vector<std::uint8_t>{1});
                    }, [&](auto wire) {
                        if (wire.size() != 1 || wire[0] != 1)
                            throw std::invalid_argument("Original local pipeline boundary declaration disagrees");
                        if (mpi->rank() == source.owning_rank)
                        {
                            if (!local[producer] || !local[consumer] || local[producer]->memory != local[consumer]->memory)
                                throw std::invalid_argument("Same-rank pipeline stages lost their one original PMA");
                            channel = transfer.createCapturedTransferChannel(*local[boundary]->memory, DeviceId::cpu(),
                                local[producer]->leader, local[producer]->setup_stream,
                                local[consumer]->leader, local[consumer]->setup_stream, capacity);
                        }
                        return std::vector<std::uint8_t>{1};
                    }, [](auto receipts) {
                        for (const auto &receipt : receipts)
                            if (receipt.bytes.size() != 1 || receipt.bytes[0] != 1)
                                throw std::invalid_argument("Original local pipeline boundary setup omitted an endpoint");
                    });
                }
            }
        }
        // A second setup consensus contains all fallible member/bank creation.
        // No rank begins graph capture while another rank has unready aliases.
        exchangePlanningSamples(mpi, [&](int ranks) {
            return std::vector<std::vector<std::uint8_t>>(ranks, std::vector<std::uint8_t>{1});
        }, [&](auto wire) {
            if (wire.size() != 1 || wire[0] != 1)
                throw std::invalid_argument("Original pipeline port installation declaration disagrees");
            for (const auto *action : rank_plan_.executeStages())
            {
                const auto stage = size_t(action->stage_id);
                auto *entry = stage_runners_.entryForStage(action->stage_id);
                PipelineForwardGraphEdges::CapturedDomain ports{
                    .stage_index = stage, .stage_count = count,
                    .activation_in = stage ? rank_boundaries_[stage - 1].activation : nullptr,
                    .activation_out = stage + 1 < count ? rank_boundaries_[stage].activation : nullptr,
                    .metadata_in = stage + 1 < count ? rank_boundaries_[stage].metadata : nullptr,
                    .metadata_out = stage ? rank_boundaries_[stage - 1].metadata : nullptr};
                auto installed = PipelineDeviceGeneration::installOriginalRankStageGenerationEdges(*entry->runner, *action, ports);
                for (auto &owner : installed) rank_forward_edges_.push_back(std::move(owner));
            }
            if (PerfStatsCollector::isRecordFamilyEnabled("forward_graph", "original_rank_pipeline_activation_capacity"))
                for (size_t boundary = 0; boundary < rank_boundaries_.size(); ++boundary)
                    if (const auto &channel = rank_boundaries_[boundary].activation; channel)
                        PerfStatsCollector::addCounter("forward_graph", "original_rank_pipeline_activation_capacity",
                            double(channel->capacityBytes()), "setup", {},
                            {{"source_stage", std::to_string(boundary)},
                             {"consumer_stage", std::to_string(boundary + 1)},
                             {"source_rank", std::to_string(config_.topology.stages[boundary].owning_rank)},
                             {"consumer_rank", std::to_string(config_.topology.stages[boundary + 1].owning_rank)},
                             {"local_rank", std::to_string(mpi->rank())},
                             {"embedding_width", std::to_string(config_.d_model)},
                             {"residual_width", std::to_string(envelope[boundary][2])},
                             {"admitted_source_rows", std::to_string(envelope[boundary][0])},
                             {"retained_verifier_rows", std::to_string(envelope[boundary][1])},
                             {"actual_capacity_bytes", std::to_string(channel->capacityBytes())},
                             {"extent_kind", "storage_capacity_only"}});
            rank_port_family_ = plan; // Any metadata allocation failure still joins this setup consensus.
            return std::vector<std::uint8_t>{1};
        }, [](auto receipts) {
            for (const auto &receipt : receipts)
                if (receipt.bytes.size() != 1 || receipt.bytes[0] != 1)
                    throw std::invalid_argument("Original pipeline port installation omitted a participant");
        });
    }

    // =========================================================================
    // Core Inference API
    // =========================================================================

    bool GlobalOrchestrator::forwardPrefill(const int *tokens, int seq_len)
    {
        if (!tokens || seq_len <= 0) return false;
        const auto &family = requiresOriginalRankGPUTransport() ? rank_port_family_ : hosted_serving_family_plan_;
        if (!family)
        {
            const bool has_gpu = std::any_of(config_.topology.stages.begin(), config_.topology.stages.end(),
                [](const auto &stage) {
                    return std::any_of(stage.devices.begin(), stage.devices.end(),
                        [](const auto &device) { return device.toLocalDeviceId().is_gpu(); });
                });
            if (has_gpu)
                throw std::logic_error("GPU pipeline prefill lost its original retained serving family");
            return forwardHosted(tokens, seq_len, ForwardExecutionPhase::Prefill);
        }
        const auto policy = PrefillChunkSchedulerPolicy::forRetainedBucketFamily(
            family->prefill_bucket_rows, get_position(), seq_len);
        return forwardPrefillChunkSchedule(tokens, seq_len, policy, family->prefill_pad_token_id, true);
    }

    bool GlobalOrchestrator::forward(const int *tokens, int seq_len)
    {
        last_seq_len_ = seq_len;
        if (requiresOriginalRankGPUTransport())
        {
            if (!rank_service_ready_ || !rank_generation_ || !tokens || seq_len <= 0 || rank_active_command_)
                throw std::logic_error("Original GPU request input requires cold serving ownership outside a generation command");
            if (seq_len == 1)
            {
                if (!rank_generation_->forwardDecodeInput(tokens[0])) return false;
                PerfStatsCollector::addCounter("forward_graph", "pipeline_scalar_input_transactions", 1.0,
                    "decode", "rank", {{"boundary_authority", "captured_original_rank_pipeline_edges"},
                        {"state_transition", "main_only"}, {"world_rank", std::to_string(config_.rank)}});
                return true;
            }
            const auto policy = PrefillChunkSchedulerPolicy::forRetainedBucketFamily(
                rank_port_family_->prefill_bucket_rows, get_position(), seq_len);
            return rank_generation_->prefill(tokens, seq_len, policy, rank_port_family_->prefill_pad_token_id, true);
        }

        return forwardHosted(tokens, seq_len,
            seq_len > 1 ? ForwardExecutionPhase::Prefill : ForwardExecutionPhase::Decode);
    }

    bool GlobalOrchestrator::forwardHosted(const int *tokens, int seq_len, ForwardExecutionPhase phase,
        const PrefillChunkSchedulerPolicy *prefill, int pad_token_id, bool allow_padding)
    {
        last_seq_len_ = seq_len;
        for (const auto &step : rank_plan_.steps)
        {
            switch (step.type)
            {
            case GlobalPPRankPlan::Step::Type::EXECUTE_STAGE:
            {
                if (step.stage_action.role == RankStageAction::Role::EXECUTE)
                {
                    if (!executeStage(step.stage_action, tokens, seq_len, phase, prefill, pad_token_id, allow_padding))
                    {
                        LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                                  << " failed to execute stage " << step.stage_action.stage_id);
                        return false;
                    }
                }
                // IDLE role: nothing to do
                break;
            }
            case GlobalPPRankPlan::Step::Type::TRANSFER:
            {
                if (step.transfer_action.direction != RankTransferAction::Direction::NONE)
                {
                    if (!executeTransfer(step.transfer_action, phase))
                    {
                        LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                                  << " failed transfer (peer=" << step.transfer_action.peer_rank
                                  << " tag=" << step.transfer_action.mpi_tag << ")");
                        return false;
                    }
                }
                break;
            }
            }
        }
        return true;
    }

    bool GlobalOrchestrator::forwardGroupedMTPVerifierWithHostTokenIds(
        const std::vector<std::vector<int>> &token_batches)
    {
        if (token_batches.empty() ||
            std::any_of(
                token_batches.begin(),
                token_batches.end(),
                [](const auto &row) { return row.empty(); }))
        {
            LOG_ERROR(
                "GlobalOrchestrator: grouped MTP verifier requires one or "
                "more non-empty request rows");
            return false;
        }

        last_seq_len_ = static_cast<int>(
            std::max_element(
                token_batches.begin(),
                token_batches.end(),
                [](const auto &lhs, const auto &rhs)
                {
                    return lhs.size() < rhs.size();
                })
                ->size());

        for (const auto &step : rank_plan_.steps)
        {
            switch (step.type)
            {
            case GlobalPPRankPlan::Step::Type::EXECUTE_STAGE:
                if (step.stage_action.role == RankStageAction::Role::EXECUTE &&
                    !executeGroupedMTPVerifierStage(
                        step.stage_action,
                        token_batches))
                {
                    LOG_ERROR(
                        "GlobalOrchestrator: rank " << config_.rank
                        << " failed grouped MTP verifier stage "
                        << step.stage_action.stage_id);
                    return false;
                }
                break;
            case GlobalPPRankPlan::Step::Type::TRANSFER:
                if (step.transfer_action.direction !=
                        RankTransferAction::Direction::NONE &&
                    !executeTransfer(step.transfer_action))
                {
                    LOG_ERROR(
                        "GlobalOrchestrator: rank " << config_.rank
                        << " failed grouped MTP verifier transfer (peer="
                        << step.transfer_action.peer_rank
                        << " tag=" << step.transfer_action.mpi_tag << ")");
                    return false;
                }
                break;
            }
        }
        return true;
    }

    const float *GlobalOrchestrator::logits() const
    {
        // Only the tail rank (with LM head) has valid logits
        if (is_pipeline_tail_)
        {
            const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            return runner ? runner->logits() : nullptr;
        }
        return nullptr;
    }

    int GlobalOrchestrator::vocab_size() const
    {
        return config_.vocab_size;
    }

    void GlobalOrchestrator::clear_cache()
    {
        stage_runners_.clearCacheAll();
        // Synchronize across ranks to ensure all caches are cleared
        try
        {
            config_.mpi_ctx->barrier();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                      << " barrier in clear_cache failed: " << e.what());
        }
    }

    bool GlobalOrchestrator::purgePrefixCache()
    {
        const bool local_success = stage_runners_.purgePrefixCacheAll();
        /*
         * Global orchestration already owns rank coordination. Preserve that
         * boundary so no rank begins a new lookup while a peer still exposes
         * the old archive generation.
         */
        try
        {
            config_.mpi_ctx->barrier();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                      << " barrier in purgePrefixCache failed: " << e.what());
            return false;
        }
        return local_success;
    }

    bool StageRunnerRegistry::admitSparseQueryExecutionProgramAll(const DeviceGenerationAdmissionRequest &request)
    {
        if (!request.valid() || entries_.empty()) return false;
        for (const auto &entry : entries_)
            if (!entry.runner || !entry.runner->admitSparseQueryExecutionProgram(request)) return false;
        return true;
    }

    void StageRunnerRegistry::endSparseQueryExecutionProgramAll() noexcept
    {
        for (const auto &entry : entries_) if (entry.runner) entry.runner->endSparseQueryExecutionProgram();
    }

    bool GlobalOrchestrator::admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &request)
    {
        return stage_runners_.admitSparseQueryExecutionProgramAll(request);
    }

    void GlobalOrchestrator::endSparseQueryExecutionProgram() noexcept
    {
        stage_runners_.endSparseQueryExecutionProgramAll();
    }

    int GlobalOrchestrator::get_position() const
    {
        const IInferenceRunner *runner = stage_runners_.defaultRunner();
        return runner ? runner->get_position() : 0;
    }

    ExecutionPath GlobalOrchestrator::executionPath() const
    {
        const IInferenceRunner *runner = stage_runners_.defaultRunner();
        return runner ? runner->executionPath() : ExecutionPath::GRAPH;
    }

    const char *GlobalOrchestrator::architecture() const
    {
        return config_.architecture_name.c_str();
    }

    bool GlobalOrchestrator::forwardMTP(int32_t draft_condition_token)
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.forwardMTPAll(draft_condition_token);
    }

    bool GlobalOrchestrator::supportsChainedMTPDrafts() const
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.supportsChainedMTPDraftsAll();
    }

    bool GlobalOrchestrator::forwardMTPFromLastDraft(
        int32_t draft_condition_token,
        int position_id)
    {
        if (!supportsChainedMTPDrafts())
            return false;
        return stage_runners_.forwardMTPFromLastDraftAll(
            draft_condition_token,
            position_id);
    }

    bool GlobalOrchestrator::commitMTPShiftedRowsFromLastForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens)
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.commitMTPShiftedRowsFromLastForwardAll(
            tokens,
            token_count,
            already_appended_tokens);
    }

    bool GlobalOrchestrator::commitMTPShiftedRowsFromPartialForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard,
        int position_offset_override,
        int already_appended_shifted_kv_tokens)
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.commitMTPShiftedRowsFromPartialForwardAll(
            tokens,
            token_count,
            already_appended_tokens,
            main_forward_token_count,
            allow_speculative_discard,
            position_offset_override,
            already_appended_shifted_kv_tokens);
    }

    bool GlobalOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.commitMTPShiftedRowFromCurrentTerminalHiddenAll(
            token,
            already_appended_tokens,
            allow_speculative_discard,
            position_offset_override);
    }

    bool GlobalOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden(
        const PrefixStateSnapshot &checkpoint,
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.commitMTPShiftedRowFromCheckpointTerminalHiddenAll(
            checkpoint,
            token,
            already_appended_tokens,
            allow_speculative_discard,
            position_offset_override);
    }

    bool GlobalOrchestrator::ensureMTPCheckpointTerminalHidden()
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.ensureMTPCheckpointTerminalHiddenAll();
    }

    const float *GlobalOrchestrator::mtpLogits() const
    {
        if (!is_pipeline_tail_)
            return nullptr;
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->mtpLogits() : nullptr;
    }

    bool GlobalOrchestrator::configureMTPRequestStopTokens(
        const std::vector<int32_t> &stop_tokens)
    {
        return stage_runners_.configureMTPRequestStopTokensAll(stop_tokens);
    }

    bool GlobalOrchestrator::publishMTPDirectTokenHistory(int target_sample_slot,
        MTPDirectTokenHistoryDisposition disposition)
    {
        if (target_sample_slot != 0 ||
            (disposition != MTPDirectTokenHistoryDisposition::NewlyEmitted &&
             disposition != MTPDirectTokenHistoryDisposition::AlreadyPublished))
            return false;
        // Only the declared vocabulary-tail owner samples and owns generated
        // history. Other ranks must not manufacture a scalar source or shadow.
        if (!is_pipeline_tail_) return true;
        IInferenceRunner *tail = stage_runners_.pipelineTailRunner();
        return tail && tail->publishMTPDirectTokenHistory(target_sample_slot, disposition);
    }

    bool GlobalOrchestrator::configureMTPRequestPenaltyPolicy(
        const MTPRequestPenaltyPolicy &policy)
    {
        return stage_runners_.configureMTPRequestPenaltyPolicyAll(policy);
    }

    bool GlobalOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().setComputeAllPositionLogits(enabled);
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.setComputeAllPositionLogitsAll(enabled);
    }

    bool GlobalOrchestrator::setComputeRowIndexedAllPositionLogits(
        bool enabled,
        int row_count)
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().setComputeRowIndexedAllPositionLogits(enabled, row_count);
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.setComputeRowIndexedAllPositionLogitsAll(
            enabled,
            row_count);
    }

    bool GlobalOrchestrator::setMTPSpecVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan)
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().setMTPSpecVerifierInputPlan(plan);
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.setMTPSpecVerifierInputPlanAll(plan);
    }

    void GlobalOrchestrator::clearMTPSpecVerifierInputPlan()
    {
        if (requiresOriginalRankGPUTransport()) { generationTail().clearMTPSpecVerifierInputPlan(); return; }
        stage_runners_.clearMTPSpecVerifierInputPlanAll();
    }

    const float *GlobalOrchestrator::getAllPositionLogits() const
    {
        if (!is_pipeline_tail_)
            return nullptr;
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->getAllPositionLogits() : nullptr;
    }

    bool GlobalOrchestrator::hasMTPLogitsLocal() const
    {
        if (!is_pipeline_tail_)
            return false;
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->hasMTPLogitsLocal() : false;
    }

    LogitsLocalInfo GlobalOrchestrator::getMTPLogitsLocalInfo() const
    {
        if (!is_pipeline_tail_)
            return {};
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->getMTPLogitsLocalInfo() : LogitsLocalInfo{};
    }

    LogitsLocalInfo GlobalOrchestrator::consumeMTPLogitsLocalInfoForSampling()
    {
        if (!is_pipeline_tail_)
            return {};
        IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner
                   ? runner->consumeMTPLogitsLocalInfoForSampling()
                   : LogitsLocalInfo{};
    }

    LogitsLocalInfo GlobalOrchestrator::consumeMTPLogitsLocalInfoForHostGather()
    {
        if (!is_pipeline_tail_)
            return {};
        IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner
                   ? runner->consumeMTPLogitsLocalInfoForHostGather()
                   : LogitsLocalInfo{};
    }

    bool GlobalOrchestrator::hasAllPositionLogitsLocal() const
    {
        if (!is_pipeline_tail_)
            return false;
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->hasAllPositionLogitsLocal() : false;
    }

    LogitsLocalInfo GlobalOrchestrator::getAllPositionLogitsLocalInfo() const
    {
        if (!is_pipeline_tail_)
            return {};
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->getAllPositionLogitsLocalInfo() : LogitsLocalInfo{};
    }

    LogitsLocalInfo GlobalOrchestrator::consumeAllPositionLogitsLocalInfoForSampling()
    {
        if (!is_pipeline_tail_)
            return {};
        IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner
                   ? runner->consumeAllPositionLogitsLocalInfoForSampling()
                   : LogitsLocalInfo{};
    }

    LogitsLocalInfo GlobalOrchestrator::consumeAllPositionLogitsLocalInfoForHostGather()
    {
        if (!is_pipeline_tail_)
            return {};
        IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner
                   ? runner->consumeAllPositionLogitsLocalInfoForHostGather()
                   : LogitsLocalInfo{};
    }

    bool GlobalOrchestrator::publishGroupedDecodeEquivalentMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        if (!mtpDecodeUnsupportedReason().empty())
        {
            if (error)
                *error = mtpDecodeUnsupportedReason();
            return false;
        }
        return stage_runners_.publishGroupedDecodeEquivalentMTPSpecStateBatchAll(
            plans,
            error);
    }

    std::string GlobalOrchestrator::mtpDecodeUnsupportedReason() const
    {
        if (config_.topology.numStages() != 1 && !requiresOriginalRankGPUTransport())
            return "MTP global pipeline requires original GPU stage owners and captured rank transport";
        return stage_runners_.mtpDecodeUnsupportedReasonAll();
    }

    bool GlobalOrchestrator::supportsMTPTokenCoordination() const
    {
        return mtpDecodeUnsupportedReason().empty();
    }

    uint64_t GlobalOrchestrator::moePlacementEpoch() const
    {
        return stage_runners_.moePlacementEpochAll();
    }

    uint64_t GlobalOrchestrator::moeRuntimeMovementEpoch() const
    {
        return stage_runners_.moeRuntimeMovementEpochAll();
    }

    PrefixLookupResult GlobalOrchestrator::lookupPrefix(
        const std::vector<int32_t> &tokens)
    {
        auto local = stage_runners_.lookupPrefixAll(tokens);
        if (requiresOriginalRankGPUTransport())
        {
            // Prefix admission is host-owned immutable metadata. Every actual
            // rank restores the same safe prefix extent from its own retained
            // payload plan; no remote handles or cache payloads cross ranks.
            const std::array<int32_t, 2> observation{local.hit() ? local.cached_tokens : 0, local.block_size};
            std::vector<std::array<int32_t, 2>> observations(config_.world_size);
            config_.original_mpi_ctx->allgather_bytes(observation.data(), observations.data(), sizeof(observation));
            int common = observation[0];
            for (const auto &other : observations)
            {
                if (other[0] < 0 || (other[0] > 0 && other[1] != local.block_size))
                    throw std::logic_error("Original rank prefix admission has incompatible cache geometry");
                common = std::min(common, other[0]);
            }
            local = local.clampedTo(common);
        }
        return local;
    }

    PrefixRestoreMetadata GlobalOrchestrator::prefixRestoreMetadata(const PrefixLookupResult &hit) const
    {
        return stage_runners_.prefixRestoreMetadataAll(hit);
    }

    bool GlobalOrchestrator::preparePrefixHarvest(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        const PrefixHarvestSchedule &schedule)
    {
        return stage_runners_.preparePrefixHarvestAll(admission, tokens, schedule);
    }

    bool GlobalOrchestrator::populatePrefix(
        const PrefixLookupResult &hit,
        int seq_idx)
    {
        return stage_runners_.populatePrefixAll(hit, seq_idx);
    }

    bool GlobalOrchestrator::harvestPrefix(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        int prompt_token_count)
    {
        return stage_runners_.harvestPrefixAll(
            admission,
            tokens,
            prompt_token_count);
    }

    bool GlobalOrchestrator::restorePrefixTerminalState(
        const PrefixLookupResult &hit)
    {
        return stage_runners_.restorePrefixTerminalStateAll(hit);
    }

    int GlobalOrchestrator::sampleGreedyFromMTPLogitsOnDevice()
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().sampleGreedyFromMTPLogitsOnDevice();
        int32_t token = -1;

        if (is_pipeline_tail_)
        {
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            token = runner ? runner->sampleGreedyFromMTPLogitsOnDevice() : -1;
            if (token < 0 && runner && !runner->hasMTPLogitsLocal())
            {
                token = greedyArgmaxToken(runner ? runner->mtpLogits() : nullptr,
                                          config_.vocab_size);
            }
        }

        if (config_.mpi_ctx)
        {
            try
            {
                config_.mpi_ctx->broadcast_int32(&token, 1, tail_rank_);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                          << " MTP token broadcast_int32 failed: " << e.what());
                return -1;
            }
        }
        return token;
    }

    int GlobalOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(int row)
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().sampleGreedyFromAllPositionLogitsOnDevice(row);
        int32_t token = -1;

        if (is_pipeline_tail_)
        {
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            token = runner ? runner->sampleGreedyFromAllPositionLogitsOnDevice(row) : -1;
            if (token < 0 && runner && row >= 0 && !runner->hasAllPositionLogitsLocal())
            {
                const float *all_logits = runner->getAllPositionLogits();
                if (all_logits)
                {
                    token = greedyArgmaxToken(
                        all_logits + static_cast<size_t>(row) * static_cast<size_t>(config_.vocab_size),
                        config_.vocab_size);
                }
            }
        }

        if (config_.mpi_ctx)
        {
            try
            {
                config_.mpi_ctx->broadcast_int32(&token, 1, tail_rank_);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                          << " MTP verifier token broadcast_int32 failed: " << e.what());
                return -1;
            }
        }
        return token;
    }

    PrefixStateSnapshot GlobalOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        return stage_runners_.captureLivePrefixStateAll(seq_idx);
    }

    PrefixStateSnapshot GlobalOrchestrator::captureLivePrefixCheckpoint(
        const PrefixCheckpointCaptureRequest &request) const
    {
        return stage_runners_.captureLivePrefixCheckpointAll(request);
    }

    bool GlobalOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        return stage_runners_.restoreLivePrefixStateAll(snapshot, seq_idx);
    }

    bool GlobalOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        return stage_runners_.truncateLivePrefixStateAll(cached_tokens, seq_idx);
    }

    PrefixRuntimeStateSnapshot GlobalOrchestrator::prefixStateProbe(
        const PrefixProbeCapturePolicy &capture_policy) const
    {
        return stage_runners_.prefixStateProbeAll(capture_policy);
    }

    // =========================================================================
    // GPU-side Sampling with Cross-Rank Broadcast
    // =========================================================================

    bool GlobalOrchestrator::requiresMPICoordinatedDecodeSampling(
        const SamplingParams &params) const
    {
        if (requiresOriginalRankGPUTransport()) return false;
        // Every rank must enter sampleGreedyOnDevice's terminal-token Bcast.
        // Stochastic CPU sampling consumes only the tail's complete host row.
        return params.is_greedy();
    }

    int GlobalOrchestrator::sampleGreedyOnDevice()
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().sampleGreedyOnDevice();
        int32_t token = -1;
        const auto &mpi_log = debugEnv().mpi_logging;

        if (is_pipeline_tail_)
        {
            // Tail rank: sample locally
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            token = runner ? runner->sampleGreedyOnDevice() : -1;
            if (token < 0 && runner && !runner->primaryDeviceId().is_gpu())
            {
                // CPU-only host sampling. GPU runners must surface the device
                // sampling failure so callers do not silently copy logits.
                const float *log = logits();
                if (log)
                {
                    token = 0;
                    float best = log[0];
                    for (int i = 1; i < config_.vocab_size; ++i)
                    {
                        if (log[i] > best)
                        {
                            best = log[i];
                            token = i;
                        }
                    }
                }
            }
        }

        // Broadcast sampled token from tail rank to all ranks
        try
        {
            auto t0 = std::chrono::steady_clock::now();
            config_.mpi_ctx->broadcast_int32(&token, 1, tail_rank_);
            if (mpi_log.log_collectives)
            {
                auto dt = std::chrono::steady_clock::now() - t0;
                double ms = std::chrono::duration<double, std::milli>(dt).count();
                LOG_DEBUG("[MPI] rank " << config_.rank << " broadcast_int32 token="
                         << token << " root=" << tail_rank_
                         << (mpi_log.log_timing ? (" " + std::to_string(ms) + "ms") : ""));
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                      << " broadcast_int32 failed: " << e.what());
            return -1;
        }
        return token;
    }

    int GlobalOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        if (requiresOriginalRankGPUTransport()) return generationTail().sampleOnDevice(params);
        int32_t token = -1;

        if (is_pipeline_tail_)
        {
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            token = runner ? runner->sampleOnDevice(params) : -1;
        }

        // Broadcast sampled token from tail rank to all ranks
        try
        {
            config_.mpi_ctx->broadcast_int32(&token, 1, tail_rank_);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                      << " broadcast_int32 failed: " << e.what());
            return -1;
        }
        return token;
    }

    // =========================================================================
    // Logits Gather Control (delegate)
    // =========================================================================

    void GlobalOrchestrator::setSkipLogitsGatherDecode(bool skip)
    {
        stage_runners_.setSkipLogitsGatherDecodeAll(skip);
    }

    void GlobalOrchestrator::setSkipLogitsGatherPrefill(bool skip)
    {
        stage_runners_.setSkipLogitsGatherPrefillAll(skip);
    }

    // =========================================================================
    // Timeline/Profiling (delegate)
    // =========================================================================

    void GlobalOrchestrator::setSuppressTimeline(bool suppress)
    {
        stage_runners_.setSuppressTimelineAll(suppress);
    }

    void GlobalOrchestrator::setAccumulatePrefill(bool accumulate)
    {
        stage_runners_.setAccumulatePrefillAll(accumulate);
    }

    void GlobalOrchestrator::flushStageTimeline()
    {
        stage_runners_.flushStageTimelineAll();
    }

    // =========================================================================
    // Hidden State API (delegate)
    // =========================================================================

    TensorBase *GlobalOrchestrator::getHiddenState()
    {
        IInferenceRunner *runner = stage_runners_.lastLocalRunner();
        return runner ? runner->getHiddenState() : nullptr;
    }

    const TensorBase *GlobalOrchestrator::getHiddenState() const
    {
        const IInferenceRunner *runner = stage_runners_.lastLocalRunner();
        return runner ? runner->getHiddenState() : nullptr;
    }

    void GlobalOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        IInferenceRunner *runner = stage_runners_.pipelineHeadRunner();
        if (runner)
        {
            runner->setHiddenState(hidden_state);
        }
    }

    bool GlobalOrchestrator::hasHiddenStateInput() const
    {
        const IInferenceRunner *runner = stage_runners_.pipelineHeadRunner();
        return runner ? runner->hasHiddenStateInput() : false;
    }

    void GlobalOrchestrator::clearHiddenStateInput()
    {
        if (IInferenceRunner *runner = stage_runners_.pipelineHeadRunner())
        {
            runner->clearHiddenStateInput();
        }
    }

    // =========================================================================
    // Snapshot API (delegate)
    // =========================================================================

    void GlobalOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        stage_runners_.enableSnapshotCaptureAll(output_dir);
    }

    void GlobalOrchestrator::setSnapshotCaptureFilter(
        const std::vector<std::string> &keys)
    {
        stage_runners_.setSnapshotCaptureFilterAll(keys);
    }

    void GlobalOrchestrator::disableSnapshotCapture()
    {
        stage_runners_.disableSnapshotCaptureAll();
    }

    void GlobalOrchestrator::clearSnapshots()
    {
        stage_runners_.clearSnapshotsAll();
    }

    const float *GlobalOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        return stage_runners_.getSnapshot(key, out_size);
    }

    SnapshotInfo GlobalOrchestrator::getSnapshotWithShape(const std::string &key) const
    {
        return stage_runners_.getSnapshotWithShape(key);
    }

    std::vector<std::string> GlobalOrchestrator::getSnapshotKeys() const
    {
        return stage_runners_.snapshotKeysAll();
    }

    // =========================================================================
    // Device & Logits Local (delegate)
    // =========================================================================

    DeviceId GlobalOrchestrator::primaryDeviceId() const
    {
        const IInferenceRunner *runner = is_pipeline_tail_ ?
            stage_runners_.pipelineTailRunner() : stage_runners_.defaultRunner();
        return runner ? runner->primaryDeviceId() : DeviceId::cpu();
    }

    bool GlobalOrchestrator::hasLogitsLocal() const
    {
        if (is_pipeline_tail_)
        {
            const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            return runner ? runner->hasLogitsLocal() : false;
        }
        return false;
    }

    LogitsLocalInfo GlobalOrchestrator::getLogitsLocalInfo() const
    {
        if (is_pipeline_tail_)
        {
            const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            return runner ? runner->getLogitsLocalInfo() : LogitsLocalInfo{};
        }
        return {};
    }

    LogitsLocalInfo GlobalOrchestrator::consumeLogitsLocalInfoForSampling()
    {
        if (is_pipeline_tail_)
        {
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            return runner
                       ? runner->consumeLogitsLocalInfoForSampling()
                       : LogitsLocalInfo{};
        }
        return {};
    }

    LogitsLocalInfo GlobalOrchestrator::consumeLogitsLocalInfoForHostGather()
    {
        if (is_pipeline_tail_)
        {
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            return runner
                       ? runner->consumeLogitsLocalInfoForHostGather()
                       : LogitsLocalInfo{};
        }
        return {};
    }

    // =========================================================================
    // Query API
    // =========================================================================

    bool GlobalOrchestrator::isPipelineHead() const { return is_pipeline_head_; }
    bool GlobalOrchestrator::isPipelineTail() const { return is_pipeline_tail_; }
    int GlobalOrchestrator::pipelineDepth() const { return config_.topology.numStages(); }
    const GlobalPPRankPlan &GlobalOrchestrator::rankPlan() const { return rank_plan_; }
    const GlobalPPTopology &GlobalOrchestrator::topology() const { return config_.topology; }

    void GlobalOrchestrator::requireMoEPipelineRuntimeOwnership(MoEPipelineOwnershipValidation frontier) const
    {
        if (frontier != MoEPipelineOwnershipValidation::OwnersBound &&
            frontier != MoEPipelineOwnershipValidation::ServingMaterialized)
            throw std::invalid_argument("Global MoE pipeline has an unknown setup frontier");
        const auto actions = rank_plan_.executeStages();
        if (actions.empty() || actions.size() != stage_runners_.size())
            throw std::logic_error("Global MoE pipeline lost its complete local stage inventory");
        for (const auto *action : actions)
        {
            const auto *entry = stage_runners_.entryForStage(action->stage_id);
            if (!entry || !entry->runner || !entry->moe_runtime || !entry->pp_stage_config)
                throw std::logic_error("Global MoE pipeline stage has no original prepared runtime owner");
            const auto spec=std::find_if(config_.topology.stages.begin(),config_.topology.stages.end(),
                [&](const auto& stage){return stage.stage_id==action->stage_id;});
            if(spec==config_.topology.stages.end()) throw std::logic_error("Global expert stage lost its original domain");
            const auto original=OriginalPipelineStageDomain::borrow(*spec,*action);
            if(action->is_global_tp && (!entry->global_tp_ctx ||
                entry->global_tp_ctx->worldRanks()!=spec->participating_ranks ||
                entry->global_tp_ctx->myIndex()!=action->tp_rank_in_domain ||
                entry->global_tp_ctx->localDevice()!=action->device))
                throw std::logic_error("CPU pipeline stage lost its original MPI/NUMA domain context");
            entry->moe_runtime->requireDestination(*entry->pp_stage_config,original.participants,
                original.weights,original.backend,entry->moe_runtime->owners().authority->maintenanceMode());
            entry->moe_runtime->requireRuntimeOwner(*entry->runner, frontier);
        }
    }

    bool GlobalOrchestrator::setMoEOverlayCollectiveRequestGeneration(uint64_t generation_id)
    {
        if (generation_id == 0 || stage_runners_.empty()) return false;
        for (const auto *action : rank_plan_.executeStages())
        {
            auto *entry = stage_runners_.entryForStage(action->stage_id);
            if (!entry || !entry->runner || !entry->moe_runtime ||
                !entry->runner->setMoEOverlayCollectiveRequestGeneration(generation_id)) return false;
        }
        return true;
    }

    size_t GlobalOrchestrator::stageRunnerCount() const
    {
        return stage_runners_.size();
    }

    StageRunnerEntry *GlobalOrchestrator::stageRunnerEntryForStage(int stage_id)
    {
        return stage_runners_.entryForStage(stage_id);
    }

    const StageRunnerEntry *GlobalOrchestrator::stageRunnerEntryForStage(int stage_id) const
    {
        return stage_runners_.entryForStage(stage_id);
    }

    StageRunnerEntry *GlobalOrchestrator::stageRunnerEntryForDomain(const std::string &domain_name)
    {
        return stage_runners_.entryForDomain(domain_name);
    }

    const StageRunnerEntry *GlobalOrchestrator::stageRunnerEntryForDomain(const std::string &domain_name) const
    {
        return stage_runners_.entryForDomain(domain_name);
    }

    IInferenceRunner *GlobalOrchestrator::stageRunnerForStage(int stage_id)
    {
        if (auto *entry = stage_runners_.entryForStage(stage_id))
        {
            return entry->runner.get();
        }
        return nullptr;
    }

    const IInferenceRunner *GlobalOrchestrator::stageRunnerForStage(int stage_id) const
    {
        if (const auto *entry = stage_runners_.entryForStage(stage_id))
        {
            return entry->runner.get();
        }
        return nullptr;
    }

    IInferenceRunner *GlobalOrchestrator::stageRunnerForDomain(const std::string &domain_name)
    {
        return stage_runners_.runnerForDomain(domain_name);
    }

    const IInferenceRunner *GlobalOrchestrator::stageRunnerForDomain(const std::string &domain_name) const
    {
        return stage_runners_.runnerForDomain(domain_name);
    }

    ITPContext *GlobalOrchestrator::globalTPContext() const
    {
        return config_.global_tp_ctx;
    }

    const WeightShardInfo *GlobalOrchestrator::weightShardForStage(int stage_id) const
    {
        for (const auto &step : rank_plan_.steps)
        {
            if (step.type == GlobalPPRankPlan::Step::Type::EXECUTE_STAGE &&
                step.stage_action.role == RankStageAction::Role::EXECUTE &&
                step.stage_action.stage_id == stage_id)
            {
                return &step.stage_action.weight_shard;
            }
        }
        return nullptr;
    }

    // =========================================================================
    // Internal: Execute Stage
    // =========================================================================

    bool GlobalOrchestrator::executeStage(const RankStageAction &action, const int *tokens, int seq_len,
        ForwardExecutionPhase phase, const PrefillChunkSchedulerPolicy *prefill, int pad_token_id, bool allow_padding)
    {
        if (action.is_global_tp)
        {
            LOG_DEBUG("GlobalOrchestrator: rank " << config_.rank
                      << " executing global TP stage " << action.stage_id
                      << " (tp_rank=" << action.tp_rank_in_domain
                      << " of " << action.tp_domain_size << ")");
        }

        IInferenceRunner *runner = stage_runners_.runnerForStage(action.stage_id);
        if (!runner)
        {
            LOG_ERROR("GlobalOrchestrator: rank " << config_.rank
                      << " has no runner for stage " << action.stage_id);
            return false;
        }

        // Nonembedding native stages also admit the actual token row for
        // their captured consumers. The graph still takes its hidden input
        // from the preceding original transfer.
        if (prefill)
            return runner->forwardPrefillChunkSchedule(tokens, seq_len, *prefill, pad_token_id, allow_padding);
        if (phase == ForwardExecutionPhase::Prefill)
            return runner->forwardPrefill(tokens, seq_len);
        return runner->forward(action.has_embedding || runner->primaryDeviceId().is_gpu() ? tokens : nullptr,
                               seq_len);
    }

    bool GlobalOrchestrator::executeGroupedMTPVerifierStage(
        const RankStageAction &action,
        const std::vector<std::vector<int>> &token_batches)
    {
        IInferenceRunner *runner =
            stage_runners_.runnerForStage(action.stage_id);
        if (!runner)
        {
            LOG_ERROR(
                "GlobalOrchestrator: rank " << config_.rank
                << " has no grouped-verifier runner for stage "
                << action.stage_id);
            return false;
        }
        return runner->forwardGroupedMTPVerifierWithHostTokenIds(
            token_batches);
    }

    // =========================================================================
    // Internal: Execute Transfer
    // =========================================================================

    bool GlobalOrchestrator::executeTransfer(const RankTransferAction &action, ForwardExecutionPhase phase)
    {
        StageActivationRouter router;
        return router.executeTransfer(action,
                                      stage_runners_,
                                      *config_.mpi_ctx,
                                      config_.rank,
                                      last_seq_len_,
                                      config_.d_model,
                                      activation_buffer_, phase);
    }

    // =========================================================================
    // Internal: Find Tail Rank
    // =========================================================================

    int GlobalOrchestrator::findTailRank() const
    {
        std::optional<int> owner;
        for (const auto &stage : config_.topology.stages)
        {
            if (stage.has_lm_head)
            {
                if (owner)
                    throw std::invalid_argument("GlobalOrchestrator: multiple vocabulary-head stages");
                if (stage.is_global_tp)
                {
                    // The domain leader is shared by all ranks' resolved plan;
                    // it is not necessarily the last physical rank in the job.
                    if (stage.participating_ranks.empty())
                        throw std::invalid_argument("GlobalOrchestrator: vocabulary domain has no participants");
                    owner = stage.participating_ranks.front();
                }
                else
                    owner = stage.owning_rank;
            }
        }
        if (!owner || *owner < 0 || *owner >= config_.world_size)
            throw std::invalid_argument("GlobalOrchestrator: no valid vocabulary-head request authority");
        return *owner;
    }

    /** @copydoc StageRunnerRegistry::collectPrefixCacheShutdownAll */
    void StageRunnerRegistry::collectPrefixCacheShutdownAll(PrefixCacheShutdown &shutdown) const
    {
        for (const auto &entry : entries_)
            if (entry.runner) entry.runner->collectPrefixCacheShutdown(shutdown);
    }

    PrefixCacheTelemetrySources StageRunnerRegistry::prefixCacheTelemetrySourcesAll() const
    {
        PrefixCacheTelemetrySources result;
        for (const auto &entry : entries_)
        {
            const auto sources = entry.runner->prefixCacheTelemetrySources();
            result.insert(result.end(), sources.begin(), sources.end());
        }
        return result;
    }

    PrefixCacheTelemetrySources GlobalOrchestrator::prefixCacheTelemetrySources() const
    {
        return stage_runners_.prefixCacheTelemetrySourcesAll();
    }

    /** @copydoc GlobalOrchestrator::collectPrefixCacheShutdown */
    void GlobalOrchestrator::collectPrefixCacheShutdown(PrefixCacheShutdown &shutdown) const
    {
        stage_runners_.collectPrefixCacheShutdownAll(shutdown);
    }


    namespace
    {
        /** @brief Original immutable command provenance precedes its ordered policy/model transactions. */
        struct OriginalRankGenerationCommandWire
        {
            uint32_t abi = 2; ///< Reject incompatible command provenance/placement layouts.
            MoEOverlayInferenceCommandIdentity command; ///< Existing source-owned command identity.
            int base = -1; ///< Completed initial scheduler boundary, never a live KV mirror.
        };

        /** @brief Versioned immutable policy envelope; no device-owned state or pointers cross this boundary. */
        struct OriginalRankGenerationAdmissionWire
        {
            uint32_t abi = 1;
            uint32_t admitted = 0;
            int budget = 0;
            int base = -1;
            sampling_math::DeviceGenerationAdmissionKind kind{};
            sampling_math::DeviceGenerationLeadingRowDisposition leading{};
            DeviceGenerationSamplingMode sampling{};
            sampling_math::DeviceGenerationPolicy policy{};
        };

        /** @brief Publish one fixed immutable host-owned command/policy/terminal object on the original context.
         * @tparam T Trivially copyable versioned metadata with no device pointers.
         * @param context Original admitted MPI owner, never an alias communicator.
         * @param source Real vocabulary-tail world rank.
         * @param value Source-owned immutable metadata, overwritten only at other ranks.
         */
        template<class T>
        void broadcastOriginalRankMetadata(const std::shared_ptr<IMPIContext> &context, int source, T &value)
        {
            static_assert(std::is_trivially_copyable_v<T>);
            std::array<int32_t, (sizeof(T) + sizeof(int32_t) - 1) / sizeof(int32_t)> wire{};
            if (context->rank() == source) std::memcpy(wire.data(), &value, sizeof(T));
            context->broadcast_int32(wire.data(), wire.size(), source);
            if (context->rank() != source) std::memcpy(&value, wire.data(), sizeof(T));
        }
    }

    void GlobalOrchestrator::prepareOriginalRankGenerationService()
    {
        if (rank_service_ready_) return;
        if (!rank_generation_ || !rank_generation_->prefillPrepared() || !rank_transfer_session_ ||
            !config_.original_mpi_ctx || rank_graph_capacity_rows_ <= 0 || rank_draft_capacity_ < 0)
            throw std::logic_error("Original rank generation service requires its complete cold graph/transport owners");
        const auto original = config_.original_mpi_ctx;
        validateOriginalRankGPUTransportContext(original);
        // Every rank derives the same graph-stable namespace from the actual
        // topology, retained serving geometry and original admitted session.
        // Local workspace identities remain independently checked by receipts.
        uint64_t fingerprint = 1469598103934665603ULL;
        const auto topology = config_.topology.toString();
        for (const unsigned char byte : topology) fingerprint = (fingerprint ^ byte) * 1099511628211ULL;
        for (const int bucket : rank_port_family_->prefill_bucket_rows)
            fingerprint = (fingerprint ^ uint64_t(bucket)) * 1099511628211ULL;
        const auto session = rank_transfer_session_->originalGeneration();
        const auto slots = moeOverlayInferenceTransactionSlotCount(rank_draft_capacity_);
        rank_transaction_slot_capacity_ = slots;
        rank_generation_->prepareOriginalRankCompletionSlots(slots);

        std::vector<int> followers;
        for (const auto &stage : config_.topology.stages)
            if (stage.owning_rank != tail_rank_ &&
                std::find(followers.begin(), followers.end(), stage.owning_rank) == followers.end())
                followers.push_back(stage.owning_rank);
        for (const int follower : followers)
        {
            if (config_.rank != tail_rank_ && config_.rank != follower) continue;
            auto channel = std::make_shared<MoEOverlayMPIInferenceTransactionChannel>(
                MoEOverlayMPIInferenceTransactionChannel::Config{
                    .mpi_ctx = original, .source_world_rank = tail_rank_,
                    .target_world_rank = follower, .send_slot_count = slots});
            const MoEOverlayInferenceTransactionProtocol::Config protocol{
                .topology = {.workspace_generation = session,
                    .topology_fingerprint_low = fingerprint,
                    .topology_fingerprint_high = fingerprint ^ session ^ 0x72616e6b50504735ULL,
                    .source_world_rank = tail_rank_, .target_world_rank = follower},
                .slot_count = slots, .max_request_count = 1,
                .max_rows_per_request = rank_graph_capacity_rows_, .max_mtp_draft_depth = rank_draft_capacity_};
            if (config_.rank == tail_rank_)
                rank_publishers_.push_back(std::make_shared<MoEOverlayInferenceTransactionPublisher>(
                    MoEOverlayInferenceTransactionPublisher::Config{.channel = std::move(channel), .protocol = protocol}));
            else
                rank_transaction_follower_ = std::make_unique<MoEOverlayInferenceTransactionFollower>(
                    MoEOverlayInferenceTransactionFollower::Config{
                        .channel = std::move(channel), .executor = rank_generation_.get(), .protocol = protocol,
                        .retired_decode_progress_sink = [this](uint64_t count, std::string *error) {
                            return !rank_retired_decode_progress_sink_ || rank_retired_decode_progress_sink_(count, error);
                        }});
        }
        if (is_pipeline_tail_)
        {
            rank_tail_capabilities_[0] = generationTail().supportsMTPSidecarSampleFusion() ? 1 : 0;
            rank_tail_capabilities_[1] = generationTail().supportsMTPSidecarLogitsStreamHandoff() ? 1 : 0;
            rank_tail_capabilities_[2] = generationTail().supportsMTPDeviceDraftTokenInput() ? 1 : 0;
            rank_tail_capabilities_[3] = generationTail().supportsMTPSidecarPreservesMainState() ? 1 : 0;
            rank_tail_capabilities_[4] = generationTail().supportsLogicalMTPVerifierBaseCheckpoint() ? 1 : 0;
            rank_tail_capabilities_[5] = generationTail().supportsMTPShiftedRowReuseFromSidecar() ? 1 : 0;
            rank_tail_capabilities_[6] = generationTail().requiresMTPDecodeEquivalentVerifierReplay() ? 1 : 0;
            rank_tail_capabilities_[7] = generationTail().supportsMTPSpecStatePublication() ? 1 : 0;
            rank_tail_capabilities_[8] = generationTail().supportsDeviceResidentMTPSpecStatePublication() ? 1 : 0;
            rank_tail_capabilities_[9] = generationTail().supportsGreedyAllPositionBatchOutcomeOnDevice() ? 1 : 0;
            rank_tail_capabilities_[10] = generationTail().supportsRowLocalAllPositionPenaltyApplication() ? 1 : 0;
            rank_tail_capabilities_[11] = generationTail().supportsDeviceStochasticMTPVerification() ? 1 : 0;
        }
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, rank_tail_capabilities_);
        if (is_pipeline_tail_) rank_tail_verifier_capability_ = generationTail().mtpVerifierRowCapability();
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, rank_tail_verifier_capability_);
        rank_pending_transactions_.resize(rank_publishers_.size());
        rank_deferred_transactions_.resize(rank_publishers_.size());
        for (auto &lane : rank_deferred_transactions_) lane.reserve(slots);

        if (is_pipeline_tail_)
            rank_generation_->bindOriginalRankTransactionCallbacks(
                [this](auto role, int rows, int depth) { return publishOriginalRankTransaction(role, rows, depth); },
                [this](uint64_t frontier) { return retireOriginalRankTransaction(frontier); });
        else
        {
            if (!rank_transaction_follower_)
                throw std::logic_error("Original non-tail rank lost its actual source transaction lane");
            rank_generation_->rank_receive_admission_ = [this](std::string *error) {
                if (!rank_active_command_ || rank_admission_published_) return false;
                OriginalRankGenerationAdmissionWire wire;
                broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, wire);
                if (wire.abi != 1 || wire.admitted != 1) return false;
                const DeviceGenerationAdmissionRequest admission{
                    .request_count = 1, .max_new_tokens = wire.budget, .kind = wire.kind,
                    .depth_policy = wire.policy, .initial_leading_row_disposition = wire.leading,
                    .initial_target_counts = DeviceGenerationInitialTargetCounts(std::vector<int>{wire.base})};
                if (!admission.valid() || !rank_generation_->beginOriginalRankFollowerCommand(admission, wire.sampling, error))
                    return false;
                if (wire.base < rank_admission_base_) return false;
                rank_response_admission_base_ = wire.base;
                rank_admission_budget_ = wire.budget;
                rank_admission_published_ = true;
                return true;
            };
        }
        rank_service_ready_ = true;
    }

    bool GlobalOrchestrator::originalRankGenerationFollower() const noexcept
    {
        return rank_service_ready_ && !is_pipeline_tail_ && rank_transaction_follower_ != nullptr;
    }

    uint64_t GlobalOrchestrator::originalRankRequestGeneration() const
    {
        if (!rank_service_ready_ || !is_pipeline_tail_)
            throw std::logic_error("Original request generation belongs only to the admitted vocabulary-tail source");
        return rank_generation_->originalRankRequestGeneration();
    }

    MoEOverlayInferenceCommandIdentity GlobalOrchestrator::originalRankCommandIdentity(uint64_t command_id) const
    {
        if (!rank_service_ready_ || !is_pipeline_tail_)
            throw std::logic_error("Original command identity belongs only to the admitted vocabulary-tail source");
        return rank_generation_->originalRankCommandIdentity(command_id);
    }

    void GlobalOrchestrator::beginOriginalRankChannelInvocations()
    {
        if (!rank_active_command_ || rank_channels_admitted_)
            throw std::logic_error("Original rank channels have missing or overlapping parent admission");
        rank_channels_admitted_ = true; // Any partial admission owes cancellation of every admitted direction.
        const CapturedTransferInvocation invocation{rank_active_command_->request_generation, rank_active_command_->command_id};
        auto &transfer = TransferEngine::instance();
        for (const auto &boundary : rank_boundaries_)
            for (const auto &channel : {boundary.activation, boundary.metadata})
                if (channel) transfer.beginCapturedTransferInvocation(channel, invocation);
    }

    bool GlobalOrchestrator::finishOriginalRankChannelInvocations()
    {
        if (!rank_active_command_ || !rank_channels_admitted_) return false;
        const CapturedTransferInvocation invocation{rank_active_command_->request_generation, rank_active_command_->command_id};
        auto &transfer = TransferEngine::instance();
        for (const auto &boundary : rank_boundaries_)
            for (const auto &channel : {boundary.activation, boundary.metadata})
                if (channel && transfer.finishCapturedTransferInvocation(channel, invocation) !=
                        CapturedTransferInvocationRetirement::Completed) return false;
        rank_channels_admitted_ = false;
        return true;
    }

    bool GlobalOrchestrator::beginOriginalRankGenerationCommand(
        const MoEOverlayInferenceCommandIdentity &command, int admission_base, DeviceGenerationSamplingMode sampling)
    {
        if (!rank_service_ready_ || !is_pipeline_tail_ || !command.valid() ||
            rank_active_command_ || admission_base < 0 || !isValidDeviceGenerationSamplingMode(sampling)) return false;
        if (command != originalRankCommandIdentity(command.command_id)) return false;
        rank_active_command_ = command;
        rank_admission_base_ = admission_base;
        rank_generation_sampling_mode_ = sampling;
        rank_admission_budget_ = 0;
        rank_retired_decode_scheduler_frontier_ = 0;
        rank_admission_published_ = false;
        rank_terminal_observation_.reset();
        OriginalRankGenerationCommandWire publication{.command = command, .base = admission_base};
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, publication);
        beginOriginalRankChannelInvocations();
        for (const auto &publisher : rank_publishers_)
            if (!publisher->beginCommand(command)) return false;
        return true;
    }

    bool GlobalOrchestrator::publishOriginalRankAdmission(const DeviceGenerationAdmissionRequest &request)
    {
        if (!rank_active_command_ || !is_pipeline_tail_ || rank_admission_published_ ||
            !request.valid() || request.request_count != 1 || !request.initial_target_counts ||
            request.initial_target_counts->values().front() < rank_admission_base_) return false;
        OriginalRankGenerationAdmissionWire wire{
            .admitted = 1, .budget = request.max_new_tokens, .base = request.initial_target_counts->values().front(),
            .kind = request.kind, .leading = request.initial_leading_row_disposition,
            .sampling = request.ordinary_sampling && !request.ordinary_sampling->is_greedy() ?
                DeviceGenerationSamplingMode::Stochastic : DeviceGenerationSamplingMode::Greedy,
            .policy = request.depth_policy};
        // Speculative admission has no ordinary sampler law: its already-frozen
        // outcome topology is selected by the source materialization call below.
        if (!request.depth_policy.isOrdinary())
            wire.sampling = rank_generation_sampling_mode_;
        if (!publishOriginalRankTransaction(MoEOverlayInferenceGraphRole::GenerationAdmission, 1, -1)) return false;
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, wire);
        if (!retireOriginalRankTransaction(rank_retired_decode_scheduler_frontier_)) return false;
        rank_admission_published_ = true;
        rank_response_admission_base_ = wire.base;
        rank_admission_budget_ = request.max_new_tokens;
        return true;
    }

    bool GlobalOrchestrator::publishOriginalRankTransaction(MoEOverlayInferenceGraphRole role, int rows, int depth)
    {
        if (!is_pipeline_tail_ || !rank_active_command_ ||
            (!rank_admission_published_ && role != MoEOverlayInferenceGraphRole::GenerationAdmission &&
             role != MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation &&
             role != MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken) ||
            rank_transaction_pending_ || rows <= 0) return false;
        if (!progressOriginalRankTransactions(false)) return false;
        // Ring pressure is the only nonterminal retirement boundary. Retain
        // every old slot until its actual native ACK; never synthesize completion
        // from a source enqueue or an unrelated stream's scheduler ticket.
        for (size_t i = 0; i < rank_deferred_transactions_.size(); ++i)
        {
            auto &lane = rank_deferred_transactions_[i];
            if (lane.size() == rank_transaction_slot_capacity_)
            {
                if (!rank_publishers_[i]->retire(lane.front())) return false;
                lane.erase(lane.begin());
            }
        }
        const auto placement = originalRankCommandIdentity(rank_active_command_->command_id);
        if (placement.request_generation != rank_active_command_->request_generation ||
            placement.placement_identity_kind != rank_active_command_->placement_identity_kind ||
            placement.initial_placement_epoch < rank_active_command_->initial_placement_epoch) return false;
        const MoEOverlayInferenceExecutionDescriptor descriptor{
            .graph_role = role, .logical_step_id = 0,
            .placement_epoch = placement.initial_placement_epoch,
            .request_count = 1, .logical_rows_per_request = rows,
            .physical_rows_per_request = rows, .draft_depth = role == MoEOverlayInferenceGraphRole::MTPGroupedVerifier ? depth : -1,
            .sidecar_depth = -1, .retired_decode_progress_tokens = rank_retired_decode_scheduler_frontier_};
        // Every remote lane receives its exact descriptor before any local
        // model/transport body is allowed to enqueue. No endpoint waits here.
        rank_transaction_pending_ = true;
        for (size_t i = 0; i < rank_publishers_.size(); ++i)
        {
            rank_pending_transactions_[i] = rank_publishers_[i]->publish(descriptor);
            if (!rank_pending_transactions_[i].ok) return false;
        }
        return true;
    }

    bool GlobalOrchestrator::retireOriginalRankTransaction(uint64_t scheduler_frontier)
    {
        if (scheduler_frontier < rank_retired_decode_scheduler_frontier_) return false;
        if (!rank_transaction_pending_)
        {
            rank_retired_decode_scheduler_frontier_ = scheduler_frontier;
            return true;
        }
        for (size_t i = 0; i < rank_publishers_.size(); ++i)
        {
            if (!rank_pending_transactions_[i].ok ||
                rank_deferred_transactions_[i].size() == rank_transaction_slot_capacity_) return false;
            rank_deferred_transactions_[i].push_back(rank_pending_transactions_[i]);
        }
        for (auto &transaction : rank_pending_transactions_) transaction = {};
        rank_transaction_pending_ = false;
        rank_retired_decode_scheduler_frontier_ = scheduler_frontier;
        return progressOriginalRankTransactions(false);
    }

    bool GlobalOrchestrator::progressOriginalRankTransactions(bool terminal)
    {
        for (size_t i = 0; i < rank_deferred_transactions_.size(); ++i)
        {
            auto &lane = rank_deferred_transactions_[i];
            for (auto it = lane.begin(); it != lane.end();)
            {
                if (terminal)
                {
                    if (!rank_publishers_[i]->retire(*it)) return false;
                    it = lane.erase(it);
                    continue;
                }
                const auto progress = rank_publishers_[i]->tryRetire(*it);
                if (progress == MoEOverlayInferenceCompletionFenceProgress::Failed) return false;
                if (progress == MoEOverlayInferenceCompletionFenceProgress::Ready) it = lane.erase(it);
                else ++it;
            }
        }
        return !terminal || rank_generation_->retireOriginalRankCompletions(nullptr);
    }

    bool GlobalOrchestrator::serviceOriginalRankGenerationCommand(
        OriginalRankGenerationTerminal *terminal, std::string *error)
    {
        if (!originalRankGenerationFollower() || !terminal || rank_active_command_) return false;
        OriginalRankGenerationCommandWire publication;
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, publication);
        if (publication.abi != 2 || publication.base < 0 || !publication.command.valid()) return false;
        const auto command = publication.command;
        rank_active_command_ = command;
        beginOriginalRankChannelInvocations();
        rank_admission_published_ = false;
        rank_admission_base_ = publication.base;
        rank_response_admission_base_ = -1;
        rank_admission_budget_ = 0;
        const auto followed = rank_transaction_follower_->runOneCommand(command);
        if (!followed.ok)
        {
            if (error) *error = followed.error;
            return false;
        }
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, *terminal);
        if (terminal->abi != 1 || terminal->request_generation != command.request_generation ||
            terminal->command_id != command.command_id ||
            terminal->published_state_commit_count < 0 || terminal->cached_tokens < 0 ||
            terminal->admission_base < 0 || terminal->admission_budget < 0 ||
            !sampling_math::valid_device_generation_leading_row_disposition(terminal->next_leading_row_disposition))
            return false;
        if (rank_admission_published_)
        {
            if (terminal->admission_base != rank_admission_base_ || terminal->admission_budget != rank_admission_budget_ ||
                !rank_generation_->finishOriginalRankFollowerCommand(terminal->published_state_commit_count -
                        (rank_response_admission_base_ - rank_admission_base_),
                    terminal->cached_tokens, error)) return false;
        }
        else
        {
            const auto cached = rank_generation_->observeOriginalRankKVTerminal(error);
            if (!cached || *cached != terminal->cached_tokens) return false;
        }
        if (!finishOriginalRankChannelInvocations()) return false;
        rank_active_command_.reset();
        return true;
    }

    bool GlobalOrchestrator::completeOriginalRankGenerationCommand(bool model_stopped,
        sampling_math::DeviceGenerationLeadingRowDisposition terminal_leading)
    {
        if (!is_pipeline_tail_ || !rank_active_command_ || rank_transaction_pending_ ||
            !progressOriginalRankTransactions(true)) return false;
        if (!rank_admission_published_)
        {
            const auto cached = rank_generation_->observeOriginalRankKVTerminal(nullptr);
            if (!cached || *cached < rank_admission_base_) return false;
            rank_terminal_observation_ = OriginalRankGenerationTerminal{
                .admission_base = rank_admission_base_, .admission_budget = 0,
                .published_state_commit_count = *cached - rank_admission_base_, .cached_tokens = *cached,
                .next_leading_row_disposition = terminal_leading,
                .model_stopped = model_stopped};
        }
        if (!rank_terminal_observation_) return false;
        for (const auto &publisher : rank_publishers_)
            if (!publisher->complete(publisher->currentPlacementEpoch(), rank_retired_decode_scheduler_frontier_)) return false;
        auto observation = *rank_terminal_observation_;
        observation.request_generation = rank_active_command_->request_generation;
        observation.command_id = rank_active_command_->command_id;
        observation.next_leading_row_disposition = terminal_leading;
        observation.model_stopped = model_stopped;
        broadcastOriginalRankMetadata(config_.original_mpi_ctx, tail_rank_, observation);
        if (!finishOriginalRankChannelInvocations()) return false;
        rank_active_command_.reset();
        rank_terminal_observation_.reset();
        return true;
    }

    void GlobalOrchestrator::abortOriginalRankGenerationCommand() noexcept
    {
        if (!rank_active_command_) return;
        const CapturedTransferInvocation invocation{rank_active_command_->request_generation, rank_active_command_->command_id};
        // Both directions are cancelled before any native completion join or
        // protocol teardown. Cancellation is absorbing; no failed graph or
        // channel can be retried under a new local counter.
        auto &transfer = TransferEngine::instance();
        for (const auto &boundary : rank_boundaries_)
            for (const auto &channel : {boundary.activation, boundary.metadata})
                if (channel)
                    try { (void)transfer.cancelCapturedTransferInvocation(channel, invocation); }
                    catch (...) { LOG_ERROR("Original rank directed-channel cancellation failed"); }
        if (is_pipeline_tail_)
            for (const auto &publisher : rank_publishers_)
                try { (void)publisher->abort(publisher->currentPlacementEpoch(), 1); }
                catch (...) { LOG_ERROR("Original rank source abort publication failed"); }
    }

    IInferenceRunner &GlobalOrchestrator::generationTail()
    {
        auto *tail = stage_runners_.pipelineTailRunner();
        if (!is_pipeline_tail_ || !tail)
            throw std::logic_error("Original GPU generation execution requires the actual local vocabulary-tail authority");
        return *tail;
    }

    const IInferenceRunner &GlobalOrchestrator::generationTail() const
    {
        auto *tail = stage_runners_.pipelineTailRunner();
        if (!is_pipeline_tail_ || !tail)
            throw std::logic_error("Original GPU generation execution requires the actual local vocabulary-tail authority");
        return *tail;
    }

    bool GlobalOrchestrator::forwardBatchWithDeviceTokenIds(const std::vector<std::vector<int>> &token_batches,
            const void *token_ids_device,
            int padded_seq_len)
    {
        return generationTail().forwardBatchWithDeviceTokenIds(token_batches, token_ids_device, padded_seq_len);
    }
    const void * GlobalOrchestrator::prepareMTPVerifierInputTokensOnDevice(int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens)
    {
        return generationTail().prepareMTPVerifierInputTokensOnDevice(first_token, first_draft_slot, draft_token_count, total_verifier_input_tokens);
    }
    const void * GlobalOrchestrator::prepareMTPVerifierInputTokenBatchOnDevice(const DeviceMTPVerifierInputBatchRequest *requests,
            int request_count,
            int logical_padded_seq_len)
    {
        return generationTail().prepareMTPVerifierInputTokenBatchOnDevice(requests, request_count, logical_padded_seq_len);
    }
    const void * GlobalOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromHostRow(const int32_t *verifier_tokens,
            int total_verifier_input_tokens,
            int draft_token_count)
    {
        return generationTail().prepareMTPVerifierInputTokensOnDeviceFromHostRow(verifier_tokens, total_verifier_input_tokens, draft_token_count);
    }
    bool GlobalOrchestrator::forwardMTPForDeviceSampling(int32_t draft_condition_token)
    {
        return generationTail().forwardMTPForDeviceSampling(draft_condition_token);
    }
    bool GlobalOrchestrator::supportsMTPSidecarSampleFusion() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[0] != 0;
    }
    bool GlobalOrchestrator::supportsMTPSidecarLogitsStreamHandoff() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[1] != 0;
    }
    bool GlobalOrchestrator::supportsMTPDeviceDraftTokenInput() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[2] != 0;
    }
    bool GlobalOrchestrator::supportsMTPSidecarPreservesMainState() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[3] != 0;
    }
    bool GlobalOrchestrator::supportsLogicalMTPVerifierBaseCheckpoint() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[4] != 0;
    }
    bool GlobalOrchestrator::supportsMTPShiftedRowReuseFromSidecar() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[5] != 0;
    }
    bool GlobalOrchestrator::requiresMTPDecodeEquivalentVerifierReplay() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[6] != 0;
    }
    bool GlobalOrchestrator::supportsMTPSpecStatePublication() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[7] != 0;
    }
    bool GlobalOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[8] != 0;
    }
    bool GlobalOrchestrator::publishAcceptedMTPSpecState(const MTPSpecStepPlan &plan,
            std::string *error)
    {
        return generationTail().publishAcceptedMTPSpecState(plan, error);
    }
    bool GlobalOrchestrator::publishAcceptedMTPSpecStateBatch(const MTPSpecStepPlanBatch &plans,
            std::string *error)
    {
        return generationTail().publishAcceptedMTPSpecStateBatch(plans, error);
    }
    bool GlobalOrchestrator::forwardMTPFromLastDraftForDeviceSampling(int32_t draft_condition_token,
            int position_id)
    {
        return generationTail().forwardMTPFromLastDraftForDeviceSampling(draft_condition_token, position_id);
    }
    bool GlobalOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(int draft_sample_slot,
            int position_offset)
    {
        return generationTail().forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(draft_sample_slot, position_offset);
    }
    bool GlobalOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(int target_sample_slot)
    {
        return generationTail().forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(target_sample_slot);
    }
    bool GlobalOrchestrator::forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot(int target_sample_slot,
            int draft_sample_slot,
            int32_t *out_token)
    {
        return generationTail().forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot(target_sample_slot, draft_sample_slot, out_token);
    }
    bool GlobalOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index)
    {
        return generationTail().forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(logical_state, request_index);
    }
    bool GlobalOrchestrator::forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int draft_sample_slot,
            int32_t *out_token)
    {
        return generationTail().forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(logical_state, request_index, draft_sample_slot, out_token);
    }
    bool GlobalOrchestrator::forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token)
    {
        return generationTail().forwardMTPAndSampleGreedy(draft_condition_token, out_token);
    }
    bool GlobalOrchestrator::forwardMTPAndSampleGreedyToDeviceDraftSlot(int32_t draft_condition_token,
            int draft_sample_slot,
            int32_t *out_token)
    {
        return generationTail().forwardMTPAndSampleGreedyToDeviceDraftSlot(draft_condition_token, draft_sample_slot, out_token);
    }
    bool GlobalOrchestrator::forwardMTPBatchAndSampleGreedy(const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens)
    {
        return generationTail().forwardMTPBatchAndSampleGreedy(draft_condition_tokens, position_ids, request_batch, out_tokens);
    }
    bool GlobalOrchestrator::advanceMTPRequestBatchConditionOnDevice(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds)
    {
        return generationTail().advanceMTPRequestBatchConditionOnDevice(logical_state, request_batch, params, stochastic_position_seeds);
    }
    bool GlobalOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_draft_slot,
            int slot_stride)
    {
        return generationTail().forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(logical_state, request_batch, first_draft_slot, slot_stride);
    }
    bool GlobalOrchestrator::forwardMTPBatchFromLastDraftAndSampleGreedy(const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens)
    {
        return generationTail().forwardMTPBatchFromLastDraftAndSampleGreedy(draft_condition_tokens, position_ids, request_batch, out_tokens);
    }
    bool GlobalOrchestrator::forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_batch,
            int first_condition_slot,
            int condition_slot_stride,
            int position_offset,
            int first_draft_slot,
            int draft_slot_stride)
    {
        return generationTail().forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(logical_state, request_batch, first_condition_slot, condition_slot_stride, position_offset, first_draft_slot, draft_slot_stride);
    }
    bool GlobalOrchestrator::forwardMTPFromLastDraftAndSampleGreedy(int32_t draft_condition_token,
            int position_id,
            int32_t *out_token)
    {
        return generationTail().forwardMTPFromLastDraftAndSampleGreedy(draft_condition_token, position_id, out_token);
    }
    bool GlobalOrchestrator::forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(int32_t draft_condition_token,
            int position_id,
            int draft_sample_slot,
            int32_t *out_token)
    {
        return generationTail().forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(draft_condition_token, position_id, draft_sample_slot, out_token);
    }
    bool GlobalOrchestrator::forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot(int draft_condition_sample_slot,
            int position_offset,
            int draft_sample_slot,
            int32_t *out_token)
    {
        return generationTail().forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot(draft_condition_sample_slot, position_offset, draft_sample_slot, out_token);
    }
    bool GlobalOrchestrator::flushPendingMTPWork()
    {
        return generationTail().flushPendingMTPWork();
    }
    void GlobalOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)
    {
        if(requiresOriginalRankGPUTransport())
            generationTail().setMTPAllPositionVerifierSyncDeferralEnabled(enabled);
        else
            stage_runners_.setMTPAllPositionVerifierSyncDeferralAll(enabled);
    }
    void GlobalOrchestrator::setMTPMainDecodeSyncDeferralEnabled(bool enabled)
    {
        if(requiresOriginalRankGPUTransport())
            generationTail().setMTPMainDecodeSyncDeferralEnabled(enabled);
        else
            stage_runners_.setMTPMainDecodeSyncDeferralAll(enabled);
    }
    void StageRunnerRegistry::setMTPAllPositionVerifierSyncDeferralAll(bool enabled)
    {
        // Hosted CPU ranks enter the same forward command even when their
        // vocabulary stage is remote. Each real local graph owns its ordering.
        for(auto& entry:entries_)
            entry.runner->setMTPAllPositionVerifierSyncDeferralEnabled(enabled);
    }
    void StageRunnerRegistry::setMTPMainDecodeSyncDeferralAll(bool enabled)
    {
        for(auto& entry:entries_)
            entry.runner->setMTPMainDecodeSyncDeferralEnabled(enabled);
    }
    bool GlobalOrchestrator::commitMTPInitialShiftedRowFromDeviceOutcome(const PrefixStateSnapshot &checkpoint,
            const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            bool allow_speculative_discard)
    {
        auto &tail = generationTail();
        const auto *tail_checkpoint = stage_runners_.prefixCheckpointForRunner(checkpoint, tail);
        if (!tail_checkpoint)
        {
            LOG_ERROR("[GlobalOrchestrator] Initial shifted MTP commit requires its exact local stage checkpoint envelope");
            return false;
        }
        // Global capture wraps local stage runners once. The terminal Rank
        // runner must receive its own untouched checkpoint, whose inner TP
        // cardinality and per-participant ready/source identity remain intact.
        return tail.commitMTPInitialShiftedRowFromDeviceOutcome(
            *tail_checkpoint, outcome, request_index, allow_speculative_discard);
    }
    bool GlobalOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard)
    {
        return generationTail().commitMTPShiftedRowFromDeviceTargetSample(target_sample_slot, already_appended_tokens, allow_speculative_discard);
    }
    bool GlobalOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard)
    {
        return generationTail().commitMTPShiftedRowFromDeviceResidentLogicalState(logical_state, request_index, already_appended_tokens, allow_speculative_discard);
    }
    bool GlobalOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(const DeviceSpeculativeOutcomeHandle &outcome,
            int request_index,
            int already_appended_tokens,
            int max_state_commit_rows,
            int main_forward_token_count,
            bool allow_speculative_discard)
    {
        return generationTail().commitMTPShiftedRowsFromDeviceOutcome(outcome, request_index, already_appended_tokens, max_state_commit_rows, main_forward_token_count, allow_speculative_discard);
    }
    bool GlobalOrchestrator::publishesFullVocabularyMTPLogits() const
    {
        return generationTail().publishesFullVocabularyMTPLogits();
    }
    bool GlobalOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot(int draft_sample_slot,
            int32_t *out_token)
    {
        return generationTail().sampleGreedyFromMTPLogitsToDeviceDraftSlot(draft_sample_slot, out_token);
    }
    bool GlobalOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(int start_row,
            int row_count,
            int32_t *out_tokens)
    {
        return generationTail().sampleGreedyFromAllPositionLogitsOnDeviceRows(start_row, row_count, out_tokens);
    }
    bool GlobalOrchestrator::supportsGreedyAllPositionBatchOutcomeOnDevice() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[9] != 0;
    }
    bool GlobalOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeVerifyBatchOutcome *out)
    {
        return generationTail().verifyGreedyAllPositionBatchOutcomeOnDevice(draft_tokens, draft_token_count, stop_tokens, stop_token_count, out);
    }
    bool GlobalOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle)
    {
        return generationTail().verifyGreedyAllPositionBatchOutcomeOnDeviceResident(draft_tokens, draft_token_count, stop_tokens, stop_token_count, out_handle);
    }
    bool GlobalOrchestrator::prepareGreedyAllPositionBatchOutcomeGraph(int verifier_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            const MTPRequestPenaltyPolicy &penalty_policy)
    {
        return generationTail().prepareGreedyAllPositionBatchOutcomeGraph(verifier_token_count, stop_tokens, stop_token_count, penalty_policy);
    }
    bool GlobalOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(const DeviceGreedyBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle)
    {
        return generationTail().verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(requests, request_count, out_handle);
    }
    DeviceResidentLogicalSequenceStateHandle GlobalOrchestrator::deviceResidentLogicalSequenceState() const
    {
        return generationTail().deviceResidentLogicalSequenceState();
    }
    bool GlobalOrchestrator::observeDeviceResidentNextConditionTokens(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_count,
            int32_t *out_tokens)
    {
        return generationTail().observeDeviceResidentNextConditionTokens(logical_state, request_count, out_tokens);
    }
    bool GlobalOrchestrator::sampleGreedyFromMainLogitsToDeviceTargetSlot(int target_sample_slot,
            int32_t *out_token)
    {
        return generationTail().sampleGreedyFromMainLogitsToDeviceTargetSlot(target_sample_slot, out_token);
    }
    int GlobalOrchestrator::sampleOnDeviceAtLogicalPosition(const SamplingParams &params,
            int logical_position)
    {
        return generationTail().sampleOnDeviceAtLogicalPosition(params, logical_position);
    }
    bool GlobalOrchestrator::publishMainLogitsBatchSamplesToDeviceResidentState(int request_count,
            const SamplingParams &params,
            const uint64_t *stochastic_position_seeds)
    {
        return generationTail().publishMainLogitsBatchSamplesToDeviceResidentState(request_count, params, stochastic_position_seeds);
    }
    std::vector<MoEOverlayDeviceControllerRuntimeBinding> GlobalOrchestrator::moeOverlayDeviceControllerRuntimeBindings() const
    {
        return generationTail().moeOverlayDeviceControllerRuntimeBindings();
    }
    bool GlobalOrchestrator::applyPenaltiesOnDevice(const LogitPenaltyBatch &penalties,
                                            int vocab_size)
    {
        // Hosted CPU greedy sampling is collective even though only the tail
        // owns logits. A non-tail rank has no row to mutate, but must remain a
        // successful participant so it enters the existing token broadcast.
        // Original GPU generation retains its strict local-tail authority.
        if (!requiresOriginalRankGPUTransport() && !is_pipeline_tail_)
            return true;
        return generationTail().applyPenaltiesOnDevice(penalties, vocab_size);
    }
    bool GlobalOrchestrator::applyPenaltiesToMTPLogitsOnDevice(const LogitPenaltyBatch &penalties,
                                                       int vocab_size)
    {
        return generationTail().applyPenaltiesToMTPLogitsOnDevice(penalties, vocab_size);
    }
    bool GlobalOrchestrator::applyPenaltiesToAllPositionLogitsOnDeviceRow(int row,
            const LogitPenaltyBatch &penalties,
            int vocab_size)
    {
        return generationTail().applyPenaltiesToAllPositionLogitsOnDeviceRow(row, penalties, vocab_size);
    }
    bool GlobalOrchestrator::applyDeviceOwnedMTPPenaltiesToLogitRows(DeviceLogitsSource source,
            int row_count,
            const MTPRequestPenaltyPolicy &penalty_policy)
    {
        return generationTail().applyDeviceOwnedMTPPenaltiesToLogitRows(source, row_count, penalty_policy);
    }
    bool GlobalOrchestrator::applyDeviceOwnedMTPBranchPenaltiesToLogits(int prior_draft_count,
            const MTPRequestPenaltyPolicy &penalty_policy)
    {
        return generationTail().applyDeviceOwnedMTPBranchPenaltiesToLogits(prior_draft_count, penalty_policy);
    }
    bool GlobalOrchestrator::supportsRowLocalAllPositionPenaltyApplication() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[10] != 0;
    }
    bool GlobalOrchestrator::supportsDeviceStochasticMTPVerification() const
    {
        return rank_service_ready_ && rank_tail_capabilities_[11] != 0;
    }
    bool GlobalOrchestrator::buildStochasticDistributionOnDevice(DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size)
    {
        return generationTail().buildStochasticDistributionOnDevice(source, row, buffer, slot, params, vocab_size);
    }
    bool GlobalOrchestrator::buildStochasticDistributionsOnDevice(DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size)
    {
        return generationTail().buildStochasticDistributionsOnDevice(source, first_row, buffer, first_slot, row_count, params, vocab_size);
    }
    bool GlobalOrchestrator::buildCapturedStochasticVerifierTargetDistributions(int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size)
    {
        return generationTail().buildCapturedStochasticVerifierTargetDistributions(row_count, params, penalty_policy, vocab_size);
    }
    bool GlobalOrchestrator::buildStochasticProbabilityRowsOnDevice(DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size)
    {
        return generationTail().buildStochasticProbabilityRowsOnDevice(source, first_row, buffer, first_slot, row_count, params, vocab_size);
    }
    bool GlobalOrchestrator::buildStochasticProcessedLogitRowsOnDevice(DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size)
    {
        return generationTail().buildStochasticProcessedLogitRowsOnDevice(source, first_row, buffer, first_slot, row_count, params, vocab_size);
    }
    int GlobalOrchestrator::sampleStochasticDraftProposalOnDevice(DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold)
    {
        return generationTail().sampleStochasticDraftProposalOnDevice(source, row, slot, params, vocab_size, threshold);
    }
    bool GlobalOrchestrator::publishCapturedMTPDraftToken(int row,
            int slot,
            const MTPRequestPenaltyPolicy &penalty_policy)
    {
        return generationTail().publishCapturedMTPDraftToken(row, slot, penalty_policy);
    }
    bool GlobalOrchestrator::sampleStochasticDraftProposalOnDeviceDeferred(DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold)
    {
        return generationTail().sampleStochasticDraftProposalOnDeviceDeferred(source, row, slot, params, vocab_size, threshold);
    }
    DeviceStochasticTargetSampleSlotHandle GlobalOrchestrator::deviceStochasticTargetSampleProducerSlot(int slot)
    {
        return generationTail().deviceStochasticTargetSampleProducerSlot(slot);
    }
    DeviceStochasticTargetSampleSlotHandle GlobalOrchestrator::deviceStochasticTargetSampleBroadcastDestinationSlot(int slot)
    {
        return generationTail().deviceStochasticTargetSampleBroadcastDestinationSlot(slot);
    }
    bool GlobalOrchestrator::recordStochasticTargetSampleSlotReadyFromDevice(int slot,
            void *producer_stream,
            bool verifier_consumer_pending)
    {
        return generationTail().recordStochasticTargetSampleSlotReadyFromDevice(slot, producer_stream, verifier_consumer_pending);
    }
    bool GlobalOrchestrator::stageStochasticDraftTokensForDeviceVerification(const int32_t *draft_tokens,
            int draft_token_count,
            int first_draft_slot)
    {
        return generationTail().stageStochasticDraftTokensForDeviceVerification(draft_tokens, draft_token_count, first_draft_slot);
    }
    bool GlobalOrchestrator::stageStochasticTargetTokenForDeviceSampling(int32_t target_token,
            int target_sample_slot)
    {
        return generationTail().stageStochasticTargetTokenForDeviceSampling(target_token, target_sample_slot);
    }
    bool GlobalOrchestrator::publishForcedDeviceResidentConditionToken(int32_t token,
            int target_sample_slot)
    {
        return generationTail().publishForcedDeviceResidentConditionToken(token, target_sample_slot);
    }
    bool GlobalOrchestrator::publishDeviceResidentConditionTokenToTargetSampleSlot(const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int target_sample_slot)
    {
        return generationTail().publishDeviceResidentConditionTokenToTargetSampleSlot(logical_state, request_index, target_sample_slot);
    }
    int GlobalOrchestrator::sampleStochasticDistributionOnDevice(DeviceDistributionBuffer buffer,
            int slot,
            float threshold)
    {
        return generationTail().sampleStochasticDistributionOnDevice(buffer, slot, threshold);
    }
    bool GlobalOrchestrator::sampleStochasticDistributionOnDeviceDeferred(DeviceDistributionBuffer buffer,
            int slot,
            float threshold)
    {
        return generationTail().sampleStochasticDistributionOnDeviceDeferred(buffer, slot, threshold);
    }
    const void * GlobalOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens)
    {
        return generationTail().prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(first_target_sample_slot, first_draft_slot, draft_token_count, total_verifier_input_tokens);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsOnDevice(int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out)
    {
        return generationTail().verifyStochasticDistributionsOnDevice(target_slot, draft_slot, draft_token, accept_threshold, residual_threshold, out);
    }
    bool GlobalOrchestrator::verifyStochasticProbabilityRowOnDevice(int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            uint64_t inverse_sample_seed,
            int inverse_sample_logical_position,
            DeviceSpeculativeVerifyResult *out)
    {
        return generationTail().verifyStochasticProbabilityRowOnDevice(target_slot, draft_slot, draft_token, accept_threshold, inverse_sample_seed, inverse_sample_logical_position, out);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsBatchOnDevice(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            DeviceSpeculativeVerifyResult *out)
    {
        return generationTail().verifyStochasticDistributionsBatchOnDevice(first_target_slot, first_draft_slot, draft_tokens, accept_thresholds, residual_thresholds, row_count, out);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDevice(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            bool use_vllm_probability_rejection)
    {
        return generationTail().verifyStochasticDistributionsBatchOutcomeOnDevice(first_target_slot, first_draft_slot, draft_tokens, accept_thresholds, residual_thresholds, row_count, first_token, stop_tokens, stop_token_count, bonus_target_slot, bonus_threshold, out, inverse_sample_seed, inverse_sample_first_logical_position, use_vllm_probability_rejection);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            bool use_vllm_probability_rejection)
    {
        return generationTail().verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(first_target_slot, first_draft_slot, draft_tokens, accept_thresholds, residual_thresholds, row_count, first_target_sample_slot, stop_tokens, stop_token_count, bonus_target_slot, bonus_threshold, out, inverse_sample_seed, inverse_sample_first_logical_position, use_vllm_probability_rejection);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceResident(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int32_t first_token,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeOutcomeHandle *out_handle,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            bool use_vllm_probability_rejection)
    {
        return generationTail().verifyStochasticDistributionsBatchOutcomeOnDeviceResident(first_target_slot, first_draft_slot, draft_tokens, accept_thresholds, residual_thresholds, row_count, first_token, stop_tokens, stop_token_count, bonus_target_slot, bonus_threshold, out_handle, inverse_sample_seed, inverse_sample_first_logical_position, use_vllm_probability_rejection);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeOutcomeHandle *out_handle,
            uint64_t inverse_sample_seed,
            int inverse_sample_first_logical_position,
            bool use_vllm_probability_rejection)
    {
        return generationTail().verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(first_target_slot, first_draft_slot, draft_tokens, accept_thresholds, residual_thresholds, row_count, first_target_sample_slot, stop_tokens, stop_token_count, bonus_target_slot, bonus_threshold, out_handle, inverse_sample_seed, inverse_sample_first_logical_position, use_vllm_probability_rejection);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDevice(const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        return generationTail().verifyStochasticDistributionsRequestBatchOutcomesOnDevice(requests, request_count, outcomes);
    }
    bool GlobalOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle)
    {
        return generationTail().verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(requests, request_count, out_handle);
    }
    bool GlobalOrchestrator::observeDeviceGenerationDispatchTicket(sampling_math::DeviceGenerationDispatchTicket *out_ticket)
    {
        return generationTail().observeDeviceGenerationDispatchTicket(out_ticket);
    }
    bool GlobalOrchestrator::beginHostScheduledDeviceGenerationAdvance(const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t *out_fragment_count)
    {
        return generationTail().beginHostScheduledDeviceGenerationAdvance(ticket, out_fragment_count);
    }
    bool GlobalOrchestrator::submitHostScheduledDeviceGenerationFragment(const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t fragment_index)
    {
        return generationTail().submitHostScheduledDeviceGenerationFragment(ticket, fragment_index);
    }
    bool GlobalOrchestrator::finishHostScheduledDeviceGenerationAdvance(const sampling_math::DeviceGenerationDispatchTicket &ticket)
    {
        return generationTail().finishHostScheduledDeviceGenerationAdvance(ticket);
    }
    bool GlobalOrchestrator::submitHostScheduledDeviceGenerationAdvance(const sampling_math::DeviceGenerationDispatchTicket &ticket)
    {
        return generationTail().submitHostScheduledDeviceGenerationAdvance(ticket);
    }
    bool GlobalOrchestrator::copyDeviceSpeculativeOutcomesToHostForDiagnostics(const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        return generationTail().copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, outcomes);
    }

    bool GlobalOrchestrator::beginDeviceResidentGeneration(const DeviceGenerationAdmissionRequest &request)
    {
        return rank_service_ready_ && is_pipeline_tail_ && publishOriginalRankAdmission(request) && rank_generation_->begin(request);
    }

    DeviceGenerationExecutionPolicy GlobalOrchestrator::deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology topology) const noexcept
    {
        (void)topology;
        return rank_service_ready_ ? DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions :
            DeviceGenerationExecutionPolicy::Unsupported;
    }

    bool GlobalOrchestrator::materializeDeviceResidentGeneration(int request_count, int draft_depth, DeviceGenerationLoopTopology topology, DeviceGenerationSamplingMode sampling_mode)
    {
        return rank_service_ready_ && is_pipeline_tail_ && sampling_mode == rank_generation_sampling_mode_ &&
            rank_generation_->materialize(request_count, draft_depth, topology, sampling_mode);
    }

    bool GlobalOrchestrator::prepareDeviceResidentGenerationContinuation(DeviceGenerationSamplingMode sampling_mode)
    {
        return rank_service_ready_ && is_pipeline_tail_ && sampling_mode == rank_generation_sampling_mode_ &&
            rank_generation_->prepareContinuation(sampling_mode);
    }

    bool GlobalOrchestrator::launchDeviceResidentGeneration()
    {
        return rank_service_ready_ && is_pipeline_tail_ && rank_generation_->launch();
    }

    bool GlobalOrchestrator::finishDeviceResidentGeneration(DeviceGenerationTerminalResult *result)
    {
        if (!rank_service_ready_ || !is_pipeline_tail_ || !rank_active_command_ ||
            !rank_generation_->finish(result) || !result || !result->valid() || result->requests.size() != 1 ||
            !rank_generation_->originalTerminalCachedTokens()) return false;
        const auto &row = result->requests.front();
        const int cached = *rank_generation_->originalTerminalCachedTokens();
        if (rank_response_admission_base_ < rank_admission_base_ || cached < rank_response_admission_base_ ||
            cached - rank_response_admission_base_ != row.published_state_commit_count) return false;
        rank_terminal_observation_ = OriginalRankGenerationTerminal{
            .admission_base = rank_admission_base_, .admission_budget = rank_admission_budget_,
            .published_state_commit_count = cached - rank_admission_base_,
            .cached_tokens = *rank_generation_->originalTerminalCachedTokens(),
            .next_leading_row_disposition = row.next_leading_row_disposition, .model_stopped = row.model_stopped};
        return true;
    }

    bool GlobalOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds(const int *token_shadow, const void *token_ids_device, int seq_len)
    {
        if (!rank_service_ready_ || !is_pipeline_tail_ || seq_len < 2 ||
            !publishOriginalRankTransaction(MoEOverlayInferenceGraphRole::MTPGroupedVerifier, seq_len, seq_len - 1)) return false;
        // The remote verifier remains live through the later accepted-state
        // publication. Retiring here would deadlock its reverse metadata wait.
        return rank_generation_->forwardMTPVerifier(seq_len, [&](IInferenceRunner &tail) {
            return tail.forwardGroupedMTPVerifierWithDeviceTokenIds(token_shadow, token_ids_device, seq_len);
        });
    }

    bool GlobalOrchestrator::advanceMTPMainConditionFromDeviceResidentLogicalState(int32_t token_shadow, const DeviceResidentLogicalSequenceStateHandle &logical_state, MTPConditionForwardPurpose purpose, int request_index)
    {
        if (!rank_service_ready_ || !is_pipeline_tail_ ||
            !publishOriginalRankTransaction(moeOverlayPipelineConditionGraphRole(purpose), 1, -1)) return false;
        if (!rank_generation_->forwardMTPCondition(purpose, [&](IInferenceRunner &tail) {
                return tail.advanceMTPMainConditionFromDeviceResidentLogicalState(token_shadow, logical_state, purpose, request_index);
            }) || !rank_generation_->completeOriginalRankSourceTransaction(moeOverlayPipelineConditionGraphRole(purpose), nullptr)) return false;
        return retireOriginalRankTransaction(rank_retired_decode_scheduler_frontier_);
    }

    bool GlobalOrchestrator::advanceMTPMainConditionFromDeviceTargetSample(int32_t token_shadow, int target_sample_slot, MTPConditionForwardPurpose purpose)
    {
        if (!rank_service_ready_ || !is_pipeline_tail_ ||
            !publishOriginalRankTransaction(moeOverlayPipelineConditionGraphRole(purpose), 1, -1)) return false;
        if (!rank_generation_->forwardMTPCondition(purpose, [&](IInferenceRunner &tail) {
                return tail.advanceMTPMainConditionFromDeviceTargetSample(token_shadow, target_sample_slot, purpose);
            }) || !rank_generation_->completeOriginalRankSourceTransaction(moeOverlayPipelineConditionGraphRole(purpose), nullptr)) return false;
        return retireOriginalRankTransaction(rank_retired_decode_scheduler_frontier_);
    }

    bool GlobalOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(const DeviceSpeculativePublicationRequest &request, std::string *error)
    {
        if (!rank_service_ready_ || !is_pipeline_tail_ || !rank_transaction_pending_ ||
            !rank_generation_->publishMTPOutcome(request, error) ||
            !rank_generation_->completeOriginalRankSourceTransaction(MoEOverlayInferenceGraphRole::MTPGroupedVerifier, error)) return false;
        return retireOriginalRankTransaction(rank_retired_decode_scheduler_frontier_);
    }

    bool GlobalOrchestrator::advanceOrdinaryMainConditionFromDeviceResidentLogicalState(int32_t token_shadow, const DeviceResidentLogicalSequenceStateHandle &logical_state, int request_index)
    {
        (void)token_shadow;
        // Forced ordinary conditions have their own existing no-sampling
        // one-shot owner. They must never masquerade as response-loop decode.
        // Its source/follower command role is intentionally separate from an
        // ordinary sampler transaction; no host token is uploaded here.
        const DeviceGenerationAdmissionRequest admission{
            .request_count = 1, .max_new_tokens = 0,
            .depth_policy = sampling_math::DeviceGenerationPolicy::forwardOnly(),
            .initial_leading_row_disposition = sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted,
            .initial_target_counts = DeviceGenerationInitialTargetCounts(std::vector<int>{rank_admission_base_})};
        if (!rank_service_ready_ || !is_pipeline_tail_ || !rank_active_command_ ||
            !publishOriginalRankAdmission(admission) ||
            !publishOriginalRankTransaction(MoEOverlayInferenceGraphRole::OrdinaryForwardOnly, 1, -1) ||
            !rank_generation_->advanceOrdinaryCondition(logical_state, request_index) ||
            !rank_generation_->originalTerminalCachedTokens() ||
            !retireOriginalRankTransaction(rank_retired_decode_scheduler_frontier_)) return false;
        rank_terminal_observation_ = OriginalRankGenerationTerminal{
            .admission_base = rank_admission_base_, .admission_budget = 0,
            .published_state_commit_count = 1,
            .cached_tokens = *rank_generation_->originalTerminalCachedTokens(),
            .next_leading_row_disposition = sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse};
        return true;
    }

    MTPVerifierRowCapability GlobalOrchestrator::mtpVerifierRowCapability() const
    {
        return rank_service_ready_ ? rank_tail_verifier_capability_ : MTPVerifierRowCapability{};
    }


    ServingGraphPreparationKind GlobalOrchestrator::servingGraphPreparationKind() const noexcept
    {
        if (requiresOriginalRankGPUTransport())
        {
            for (const auto *action : rank_plan_.executeStages())
            {
                const auto *entry = stage_runners_.entryForStage(action->stage_id);
                if (!entry || !entry->runner || entry->runner->servingGraphPreparationKind() !=
                        ServingGraphPreparationKind::NativeDeviceExecutableFamily)
                    return ServingGraphPreparationKind::Unresolved;
            }
            return ServingGraphPreparationKind::NativeDeviceExecutableFamily;
        }
        if(config_.topology.numStages()>1) {
            // A real host pipeline declares the lifecycle of every original
            // local child. Native GPU composition retains its separate complete
            // captured contract above; CPU stages never imply CPU capture.
            const bool cpu_only=std::all_of(config_.topology.stages.begin(),config_.topology.stages.end(),[](const auto& stage){
                const auto& members=stage.is_global_tp?stage.per_rank_devices:stage.devices;
                return !members.empty() && std::all_of(members.begin(),members.end(),[](const auto& member){return member.isCPU();});
            });
            if(cpu_only && !stage_runners_.empty()) {
                for(const auto* action:rank_plan_.executeStages()) {
                    const auto* entry=stage_runners_.entryForStage(action->stage_id);
                    if(!entry || !entry->runner || entry->runner->servingGraphPreparationKind()!=ServingGraphPreparationKind::EagerHostGraph)
                        return ServingGraphPreparationKind::Unresolved;
                }
                return ServingGraphPreparationKind::EagerHostGraph;
            }
        }
        const auto *runner = stage_runners_.defaultRunner();
        return config_.topology.numStages() == 1 && runner ? runner->servingGraphPreparationKind() :
            ServingGraphPreparationKind::Unresolved;
    }

    bool GlobalOrchestrator::forwardRestoredPrefixMTPDecodeBridge(const RestoredPrefixMTPDecodeBridgeRequest &request)
    {
        if (!requiresOriginalRankGPUTransport())
            return config_.topology.numStages() == 1 && generationTail().forwardRestoredPrefixMTPDecodeBridge(request);
        if (!rank_service_ready_ || rank_active_command_ ||
            !rank_generation_->restoredPrefixMTPDecodeBridge(request)) return false;
        // Count the admitted scalar suffix once per original process, after
        // its actual captured participant transactions have been submitted.
        PerfStatsCollector::addCounter("forward_graph", "pipeline_scalar_input_transactions", 1.0,
            "decode", "rank", {{"boundary_authority", "captured_original_rank_pipeline_edges"},
                {"state_transition", rank_draft_capacity_ > 0 ? "main_and_shifted_mtp" : "main_only"},
                {"world_rank", std::to_string(config_.rank)}});
        return true;
    }

    void GlobalOrchestrator::bindOriginalRankRetiredDecodeProgressSink(MoEOverlayRetiredDecodeProgressSink sink)
    {
        if (!sink || rank_active_command_ || rank_retired_decode_progress_sink_)
            throw std::logic_error("Original rank maintenance notification requires one inactive process-local sink");
        rank_retired_decode_progress_sink_ = std::move(sink);
    }

} // namespace llaminar2
