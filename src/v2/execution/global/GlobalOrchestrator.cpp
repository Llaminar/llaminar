/**
 * @file GlobalOrchestrator.cpp
 * @brief Implementation of cross-machine MPI cluster inference orchestrator
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "GlobalOrchestrator.h"
#include "../global_pp/GlobalPPRankPlanBuilder.h"
#include "../../tensors/TensorClasses.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"

#include <stdexcept>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <optional>

namespace llaminar2
{
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

        for (const auto &existing : entries_)
        {
            if (existing.stage_id == entry.stage_id)
            {
                throw std::invalid_argument("StageRunnerRegistry: duplicate stage_id");
            }
        }

        entries_.push_back(std::move(entry));
    }

    void StageRunnerRegistry::setCompatibilityRunner(std::unique_ptr<IInferenceRunner> runner)
    {
        if (!runner)
        {
            throw std::invalid_argument("StageRunnerRegistry: compatibility runner is required");
        }
        compatibility_runner_ = std::move(runner);
    }

    bool StageRunnerRegistry::empty() const
    {
        return entries_.empty() && !compatibility_runner_;
    }

    size_t StageRunnerRegistry::size() const
    {
        return entries_.size() + (compatibility_runner_ ? 1u : 0u);
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
        return compatibility_runner_.get();
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
        return compatibility_runner_.get();
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
        return defaultRunner();
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
        return defaultRunner();
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
        return lastLocalRunner();
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
        return lastLocalRunner();
    }

    IInferenceRunner *StageRunnerRegistry::defaultRunner()
    {
        if (!entries_.empty())
        {
            return entries_.front().runner.get();
        }
        return compatibility_runner_.get();
    }

    const IInferenceRunner *StageRunnerRegistry::defaultRunner() const
    {
        if (!entries_.empty())
        {
            return entries_.front().runner.get();
        }
        return compatibility_runner_.get();
    }

    IInferenceRunner *StageRunnerRegistry::lastLocalRunner()
    {
        if (!entries_.empty())
        {
            return entries_.back().runner.get();
        }
        return compatibility_runner_.get();
    }

    const IInferenceRunner *StageRunnerRegistry::lastLocalRunner() const
    {
        if (!entries_.empty())
        {
            return entries_.back().runner.get();
        }
        return compatibility_runner_.get();
    }

    void StageRunnerRegistry::clearCacheAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->clear_cache();
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->clear_cache();
        }
    }

    void StageRunnerRegistry::setSkipLogitsGatherDecodeAll(bool skip)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setSkipLogitsGatherDecode(skip);
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->setSkipLogitsGatherDecode(skip);
        }
    }

    void StageRunnerRegistry::setSkipLogitsGatherPrefillAll(bool skip)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setSkipLogitsGatherPrefill(skip);
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->setSkipLogitsGatherPrefill(skip);
        }
    }

    void StageRunnerRegistry::setSuppressTimelineAll(bool suppress)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setSuppressTimeline(suppress);
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->setSuppressTimeline(suppress);
        }
    }

    void StageRunnerRegistry::setAccumulatePrefillAll(bool accumulate)
    {
        for (auto &entry : entries_)
        {
            entry.runner->setAccumulatePrefill(accumulate);
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->setAccumulatePrefill(accumulate);
        }
    }

    void StageRunnerRegistry::flushStageTimelineAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->flushStageTimeline();
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->flushStageTimeline();
        }
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
        if (compatibility_runner_)
        {
            saw_runner = true;
            ok = compatibility_runner_->forwardMTP(draft_condition_token) && ok;
        }
        return saw_runner && ok;
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
        if (compatibility_runner_)
        {
            saw_runner = true;
            ok = compatibility_runner_->setComputeAllPositionLogits(enabled) && ok;
        }
        return saw_runner && ok;
    }

    PrefixStateSnapshot StageRunnerRegistry::captureLivePrefixStateAll(int seq_idx) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        int common_tokens = 0;

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

            aggregate.participant_snapshots.push_back(std::move(child));
            return true;
        };

        for (const auto &entry : entries_)
        {
            if (!capture_runner(*entry.runner))
                return {};
        }
        if (compatibility_runner_ && !capture_runner(*compatibility_runner_))
        {
            return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            return {};
        }

        aggregate.valid = true;
        aggregate.cached_tokens = common_tokens;
        return aggregate;
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
        if (compatibility_runner_)
        {
            restore_runner(*compatibility_runner_);
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
        if (compatibility_runner_)
        {
            saw_runner = true;
            ok = compatibility_runner_->truncateLivePrefixState(cached_tokens, seq_idx) && ok;
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
        if (compatibility_runner_)
        {
            saw_runner = true;
            std::string reason = compatibility_runner_->mtpDecodeUnsupportedReason();
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
        if (compatibility_runner_)
        {
            compatibility_runner_->enableSnapshotCapture(output_dir);
        }
    }

    void StageRunnerRegistry::disableSnapshotCaptureAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->disableSnapshotCapture();
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->disableSnapshotCapture();
        }
    }

    void StageRunnerRegistry::clearSnapshotsAll()
    {
        for (auto &entry : entries_)
        {
            entry.runner->clearSnapshots();
        }
        if (compatibility_runner_)
        {
            compatibility_runner_->clearSnapshots();
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
        if (compatibility_runner_)
        {
            return compatibility_runner_->getSnapshot(key, out_size);
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
        if (compatibility_runner_)
        {
            return compatibility_runner_->getSnapshotWithShape(key);
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
        if (compatibility_runner_)
        {
            append_unique(compatibility_runner_->getSnapshotKeys());
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
        if (!activation_buffer || activation_buffer->numel() != num_elements)
        {
            activation_buffer = std::make_shared<FP32Tensor>(std::vector<size_t>{num_elements});
            LOG_DEBUG("StageActivationRouter: allocated activation buffer with "
                      << num_elements << " elements");
        }
    }

    bool StageActivationRouter::executeTransfer(const RankTransferAction &action,
                                                StageRunnerRegistry &registry,
                                                IMPIContext &mpi_ctx,
                                                int rank,
                                                int last_seq_len,
                                                int d_model,
                                                std::shared_ptr<TensorBase> &activation_buffer) const
    {
        if (action.direction == RankTransferAction::Direction::NONE)
            return true;

        const auto &mpi_log = debugEnv().mpi_logging;
        const size_t count = transferElementCount(last_seq_len, d_model);

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
            std::memmove(activation_buffer->mutable_data(), hidden->data(), count * sizeof(float));
            destination_runner->setHiddenState(activation_buffer.get());

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

            const float *send_data = hidden->data();
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

        for (auto &entry : config_.stage_runners)
        {
            stage_runners_.add(std::move(entry));
        }
        config_.stage_runners.clear();

        if (config_.rank_runner)
        {
            stage_runners_.setCompatibilityRunner(std::move(config_.rank_runner));
        }

        if (stage_runners_.empty())
        {
            throw std::invalid_argument("GlobalOrchestrator: rank_runner or stage_runners is required");
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

        // Allocate activation transfer buffer for PP (if we have any transfers)
        if (!rank_plan_.transferActions().empty() && config_.d_model > 0)
        {
            // Buffer size: max_seq_len * d_model floats
            // Use a conservative max_seq_len estimate; will be resized on demand
            // For now: create a 1D FP32 tensor sized to d_model (single token decode)
            // The actual transfer count is determined at runtime from the hidden state
            activation_buffer_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(config_.d_model)});

            LOG_DEBUG("GlobalOrchestrator: allocated activation buffer ["
                      << config_.d_model << "] for PP transfers");
        }

        LOG_DEBUG("GlobalOrchestrator: pipeline_head=" << is_pipeline_head_
                 << " pipeline_tail=" << is_pipeline_tail_
                 << " tail_rank=" << tail_rank_);
    }

    // =========================================================================
    // Core Inference API
    // =========================================================================

    bool GlobalOrchestrator::forward(const int *tokens, int seq_len)
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
                    if (!executeStage(step.stage_action, tokens, seq_len))
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
                    if (!executeTransfer(step.transfer_action))
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

    const float *GlobalOrchestrator::mtpLogits() const
    {
        if (!is_pipeline_tail_)
            return nullptr;
        const IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
        return runner ? runner->mtpLogits() : nullptr;
    }

    bool GlobalOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        if (!mtpDecodeUnsupportedReason().empty())
            return false;
        return stage_runners_.setComputeAllPositionLogitsAll(enabled);
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

    std::string GlobalOrchestrator::mtpDecodeUnsupportedReason() const
    {
        if (config_.topology.numStages() != 1)
            return "MTP decode is not enabled for GlobalPP topologies";
        return stage_runners_.mtpDecodeUnsupportedReasonAll();
    }

    PrefixStateSnapshot GlobalOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        return stage_runners_.captureLivePrefixStateAll(seq_idx);
    }

    bool GlobalOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        return stage_runners_.restoreLivePrefixStateAll(snapshot, seq_idx);
    }

    bool GlobalOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        return stage_runners_.truncateLivePrefixStateAll(cached_tokens, seq_idx);
    }

    // =========================================================================
    // GPU-side Sampling with Cross-Rank Broadcast
    // =========================================================================

    int GlobalOrchestrator::sampleGreedyOnDevice()
    {
        int32_t token = -1;
        const auto &mpi_log = debugEnv().mpi_logging;

        if (is_pipeline_tail_)
        {
            // Tail rank: sample locally
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            token = runner ? runner->sampleGreedyOnDevice() : -1;
            if (token < 0)
            {
                // Fallback to CPU sampling
                const float *log = logits();
                if (log)
                {
                    // Simple argmax fallback
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
        int32_t token = -1;

        if (is_pipeline_tail_)
        {
            IInferenceRunner *runner = stage_runners_.pipelineTailRunner();
            token = runner ? runner->sampleOnDevice(params) : -1;
            if (token < 0)
            {
                // Fallback: greedy
                token = sampleGreedyOnDevice();
                // Note: sampleGreedyOnDevice already broadcasts, so return directly
                return token;
            }
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
        const IInferenceRunner *runner = stage_runners_.defaultRunner();
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

    // =========================================================================
    // Query API
    // =========================================================================

    bool GlobalOrchestrator::isPipelineHead() const { return is_pipeline_head_; }
    bool GlobalOrchestrator::isPipelineTail() const { return is_pipeline_tail_; }
    int GlobalOrchestrator::pipelineDepth() const { return config_.topology.numStages(); }
    const GlobalPPRankPlan &GlobalOrchestrator::rankPlan() const { return rank_plan_; }
    const GlobalPPTopology &GlobalOrchestrator::topology() const { return config_.topology; }

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

    bool GlobalOrchestrator::executeStage(const RankStageAction &action,
                                          const int *tokens, int seq_len)
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

        // Pipeline head stages consume tokens directly. Middle/tail stages should
        // already have hidden state populated by a preceding transfer/handoff.
        return runner->forward(action.has_embedding ? tokens : nullptr, seq_len);
    }

    // =========================================================================
    // Internal: Execute Transfer
    // =========================================================================

    bool GlobalOrchestrator::executeTransfer(const RankTransferAction &action)
    {
        StageActivationRouter router;
        return router.executeTransfer(action,
                                      stage_runners_,
                                      *config_.mpi_ctx,
                                      config_.rank,
                                      last_seq_len_,
                                      config_.d_model,
                                      activation_buffer_);
    }

    // =========================================================================
    // Internal: Find Tail Rank
    // =========================================================================

    int GlobalOrchestrator::findTailRank() const
    {
        for (const auto &stage : config_.topology.stages)
        {
            if (stage.has_lm_head)
            {
                if (stage.is_global_tp)
                {
                    // For global TP stages, all participating ranks have the LM head.
                    // Use the first participating rank as the "primary" tail for
                    // broadcasting purposes.
                    if (!stage.participating_ranks.empty())
                        return stage.participating_ranks[0];
                }
                return stage.owning_rank;
            }
        }
        // Fallback: last stage's owner
        if (!config_.topology.stages.empty())
        {
            const auto &last = config_.topology.stages.back();
            return last.is_global_tp && !last.participating_ranks.empty()
                       ? last.participating_ranks[0]
                       : last.owning_rank;
        }
        return 0;
    }

} // namespace llaminar2
