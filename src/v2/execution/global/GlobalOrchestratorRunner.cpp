/**
 * @file GlobalOrchestratorRunner.cpp
 * @brief Implementation of GlobalOrchestratorRunner
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include "GlobalOrchestratorRunner.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    GlobalOrchestratorRunner::GlobalOrchestratorRunner(Config config)
        : config_(std::move(config))
    {
    }

    GlobalOrchestratorRunner::~GlobalOrchestratorRunner()
    {
        if (initialized_)
        {
            shutdown();
        }
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool GlobalOrchestratorRunner::initialize()
    {
        if (initialized_)
            return true;

        if (!config_.global_orchestrator)
        {
            return setError("GlobalOrchestrator not provided in config");
        }

        global_orch_ = std::move(config_.global_orchestrator);
        architecture_name_ = global_orch_->architecture();
        initialized_ = true;

        // Print topology table on rank 0
        int my_rank = config_.mpi_ctx ? config_.mpi_ctx->rank() : 0;
        if (my_rank == 0)
        {
            LOG_INFO("GlobalOrchestratorRunner: PP topology:\n" << config_.topology.toTable());
        }

        LOG_INFO("GlobalOrchestratorRunner initialized: " << architecture_name_
                 << " on rank " << my_rank);
        return true;
    }

    void GlobalOrchestratorRunner::shutdown()
    {
        global_orch_.reset();
        initialized_ = false;
    }

    // =========================================================================
    // Inference
    // =========================================================================

    bool GlobalOrchestratorRunner::prefill(const std::vector<int32_t> &tokens)
    {
        if (!initialized_)
            return setError("Not initialized");

        if (tokens.empty())
            return setError("Empty prompt tokens");

        // GlobalOrchestrator::forward() handles PP transfers and TP collectives
        if (!global_orch_->forward(tokens.data(), static_cast<int>(tokens.size())))
            return setError("Forward pass failed during prefill");

        prefill_logits_ready_ = true;
        return true;
    }

    GenerationResult GlobalOrchestratorRunner::decodeStep()
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Not initialized";
            return result;
        }

        if (prefill_logits_ready_)
        {
            // First decode step after prefill: sample from already-computed
            // prefill logits instead of re-feeding the last prompt token.
            prefill_logits_ready_ = false;
        }
        else
        {
            // Run single-token forward with last token
            if (!global_orch_->forward(&last_token_, 1))
            {
                result.error = "Forward pass failed during decode";
                return result;
            }
        }

        // Sample — GlobalOrchestrator handles broadcast from tail rank
        int token = -1;

        if (active_sampling_params_.has_penalties())
        {
            token = -1; // Force CPU fallback
        }
        else if (active_sampling_params_.is_greedy())
        {
            token = global_orch_->sampleGreedyOnDevice();
        }
        else
        {
            token = global_orch_->sampleOnDevice(active_sampling_params_);
        }

        if (token < 0)
        {
            // Fallback: CPU-side sampling
            const float *logits_ptr = global_orch_->logits();
            if (!logits_ptr)
            {
                result.error = "No logits available";
                return result;
            }
            token = sampler_.sample(logits_ptr,
                                    static_cast<size_t>(global_orch_->vocab_size()),
                                    active_sampling_params_);
        }

        sampler_.record_token(token);
        result.tokens.push_back(token);
        last_token_ = token;

        // Check stop tokens
        for (int32_t stop : stop_tokens_)
        {
            if (token == stop)
            {
                result.is_complete = true;
                break;
            }
        }

        return result;
    }

    GenerationResult GlobalOrchestratorRunner::generate(
        const std::vector<int32_t> &prompt_tokens,
        int max_new_tokens,
        const SamplingParams &sampling)
    {
        GenerationResult result;

        if (!initialized_)
        {
            result.error = "Not initialized";
            return result;
        }

        // Prefill
        if (!prefill(prompt_tokens))
        {
            result.error = last_error_;
            return result;
        }

        // Store sampling params and configure GPU-side decode
        active_sampling_params_ = sampling;
        sampler_ = Sampler(sampling.seed);

        global_orch_->setSkipLogitsGatherDecode(true);

        for (int i = 0; i < max_new_tokens; ++i)
        {
            GenerationResult step = decodeStep();

            if (!step.error.empty())
            {
                result.error = step.error;
                break;
            }

            for (int32_t token : step.tokens)
            {
                result.tokens.push_back(token);
            }

            if (step.is_complete)
            {
                result.is_complete = true;
                break;
            }
        }

        // Restore normal logits gathering
        global_orch_->setSkipLogitsGatherDecode(false);

        return result;
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    const RankExecutionPlan &GlobalOrchestratorRunner::executionPlan() const
    {
        return empty_plan_;
    }

    const OrchestrationConfig &GlobalOrchestratorRunner::config() const
    {
        return config_.orchestration_config;
    }

    // =========================================================================
    // Status
    // =========================================================================

    bool GlobalOrchestratorRunner::isInitialized() const
    {
        return initialized_;
    }

    const std::string &GlobalOrchestratorRunner::lastError() const
    {
        return last_error_;
    }

    int GlobalOrchestratorRunner::vocabSize() const
    {
        return global_orch_ ? global_orch_->vocab_size() : 0;
    }

    int GlobalOrchestratorRunner::currentPosition() const
    {
        return global_orch_ ? global_orch_->get_position() : 0;
    }

    void GlobalOrchestratorRunner::clearCache()
    {
        if (global_orch_)
            global_orch_->clear_cache();
        prefill_logits_ready_ = false;
        last_token_ = 0;
        sampler_ = Sampler();
    }

    // =========================================================================
    // Advanced
    // =========================================================================

    const float *GlobalOrchestratorRunner::lastLogits() const
    {
        return global_orch_ ? global_orch_->logits() : nullptr;
    }

    void GlobalOrchestratorRunner::setStopTokens(const std::vector<int32_t> &stop_tokens)
    {
        stop_tokens_ = stop_tokens;
    }

    std::shared_ptr<ITokenizer> GlobalOrchestratorRunner::tokenizer() const
    {
        return config_.tokenizer;
    }

    const std::string &GlobalOrchestratorRunner::architecture() const
    {
        return architecture_name_;
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void GlobalOrchestratorRunner::enableSnapshotCapture(const std::string &output_dir)
    {
        if (global_orch_)
            global_orch_->enableSnapshotCapture(output_dir);
    }

    void GlobalOrchestratorRunner::disableSnapshotCapture()
    {
        if (global_orch_)
            global_orch_->disableSnapshotCapture();
    }

    void GlobalOrchestratorRunner::clearSnapshots()
    {
        if (global_orch_)
            global_orch_->clearSnapshots();
    }

    const float *GlobalOrchestratorRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        return global_orch_ ? global_orch_->getSnapshot(key, out_size) : nullptr;
    }

    std::vector<std::string> GlobalOrchestratorRunner::getSnapshotKeys() const
    {
        return global_orch_ ? global_orch_->getSnapshotKeys() : std::vector<std::string>{};
    }

    // =========================================================================
    // Profiling
    // =========================================================================

    const GraphExecutorStats *GlobalOrchestratorRunner::executorStats() const
    {
        return nullptr; // GlobalOrchestrator doesn't expose executor stats yet
    }

    void GlobalOrchestratorRunner::resetExecutorStats()
    {
        // No-op
    }

    // =========================================================================
    // GPU-side Sampling
    // =========================================================================

    int GlobalOrchestratorRunner::sampleGreedyOnDevice()
    {
        return global_orch_ ? global_orch_->sampleGreedyOnDevice() : -1;
    }

    int GlobalOrchestratorRunner::sampleOnDevice(const SamplingParams &params)
    {
        return global_orch_ ? global_orch_->sampleOnDevice(params) : -1;
    }

    void GlobalOrchestratorRunner::setSkipLogitsGatherDecode(bool skip)
    {
        if (global_orch_)
            global_orch_->setSkipLogitsGatherDecode(skip);
    }

    void GlobalOrchestratorRunner::setSkipLogitsGatherPrefill(bool skip)
    {
        if (global_orch_)
            global_orch_->setSkipLogitsGatherPrefill(skip);
    }

    void GlobalOrchestratorRunner::setSuppressTimeline(bool suppress)
    {
        if (global_orch_)
            global_orch_->setSuppressTimeline(suppress);
    }

    void GlobalOrchestratorRunner::setAccumulatePrefill(bool accumulate)
    {
        if (global_orch_)
            global_orch_->setAccumulatePrefill(accumulate);
    }

    void GlobalOrchestratorRunner::flushStageTimeline()
    {
        if (global_orch_)
            global_orch_->flushStageTimeline();
    }

    void GlobalOrchestratorRunner::setSamplingParams(const SamplingParams &params)
    {
        active_sampling_params_ = params;
    }

    SamplingParams GlobalOrchestratorRunner::getRecommendedSamplingParams() const
    {
        return {};
    }

    // =========================================================================
    // MPI Worker Loop
    // =========================================================================

    void GlobalOrchestratorRunner::runMPIWorkerLoop()
    {
        // TODO Phase 6: Implement MPI worker loop for server mode
    }

    void GlobalOrchestratorRunner::shutdownMPIWorkers()
    {
        // TODO Phase 6: Implement MPI worker shutdown
    }

    void GlobalOrchestratorRunner::setMPICoordinatedMode(bool enabled)
    {
        mpi_coordinated_mode_ = enabled;
    }

    // =========================================================================
    // Internal
    // =========================================================================

    bool GlobalOrchestratorRunner::setError(const std::string &error)
    {
        last_error_ = error;
        LOG_ERROR("GlobalOrchestratorRunner: " << error);
        return false;
    }

} // namespace llaminar2
