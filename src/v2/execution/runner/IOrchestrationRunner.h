/**
 * @file IOrchestrationRunner.h
 * @brief Interface for orchestrated inference execution
 *
 * Provides the main interface for running inference with complex orchestration
 * scenarios including:
 * - Pipeline parallelism (PP) across ranks
 * - Local tensor parallelism (LOCAL TP) across devices within a rank
 * - Global tensor parallelism (GLOBAL TP) across ranks
 * - Heterogeneous device execution (CUDA, ROCm, CPU)
 *
 * This is the Phase 5 entry point that wires together all Phase 0-4 components:
 * - Phase 0: OrchestrationConfig from CLI/YAML
 * - Phase 1-2: RankExecutionPlan from ExecutionPlanBuilder
 * - Phase 3: Pipeline parallel graph building
 * - Phase 4: LOCAL TP context and weight sharding
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "../../config/OrchestrationConfig.h"
#include "../prefix_cache/PrefixCacheStateProbe.h"
#include "../InferenceReadiness.h"
#include "../moe/MoEOptimizationStatus.h"
#include "../moe/MoEOptimizationMovementTopology.h"
#include "../mpi_orchestration/RankExecutionPlan.h"
#include "../../transfer/TransferEngine.h"
#include "../../planning/GraphSnapshotMemoryCapacity.h"
#include "../../utils/Sampler.h"
#include "../../utils/ToolCallTypes.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <optional>

namespace llaminar2
{
    class ITokenizer;          // Forward declaration
    class IMPIContext;         // Forward declaration
    class IModelContext;       // Forward declaration
    class ModelContext;        // Forward declaration
    class PhysicalMemoryAuthority; // CPU/GPU admission and live-allocation owner
    class ReusableExecutionWorkspaceRegistry; // Model-lifetime workspace owner
    struct MoEOverlaySealedMigrationMeasurements;
    struct MoEOverlayResolvedCapacityPlan;
    struct GraphExecutorStats; // Forward declaration
}

namespace llaminar2
{

    /**
     * @brief Exclusive lifecycle for one reusable prepared model authority.
     *
     * A contract can be copied into a process campaign cache, but exactly one
     * runner may own its prepared weights at a time. Dynamic ExpertOverlay also
     * needs a terminal sealing interval in which logical placement is restored
     * and registry bindings are atomically rebased. Encoding those transitions
     * here prevents a caller from constructing a new runner while the previous
     * graph, maintenance wave, or physical slot assignment is still live.
     */
    class ModelContextReuseAuthority final
    {
    public:
        /** @brief Complete reusable-context lifecycle states. */
        enum class State : std::uint8_t
        {
            RunnerExclusive, ///< Exactly one runner owns mutable execution state.
            Sealing,         ///< Teardown is restoring/rebinding prepared weights.
            Reusable,        ///< No runner state remains and a consumer may acquire.
            Invalid,         ///< Sealing failed; reuse is permanently forbidden.
        };

        /** @brief Create authority initially owned by the exporting runner. */
        ModelContextReuseAuthority() = default;

        ModelContextReuseAuthority(const ModelContextReuseAuthority &) = delete;
        ModelContextReuseAuthority &operator=(
            const ModelContextReuseAuthority &) = delete;

        /**
         * @brief Acquire a reusable contract for exactly one new runner.
         * @param error Optional state-specific rejection diagnostic.
         * @return True only for the `Reusable -> RunnerExclusive` transition.
         */
        [[nodiscard]] bool acquireRunnerExclusive(
            std::string *error = nullptr)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::Reusable)
            {
                if (error)
                {
                    *error = state_ == State::Invalid
                        ? (diagnostic_.empty()
                               ? "prepared model context is invalid"
                               : diagnostic_)
                        : "prepared model context is not at its reusable lifecycle boundary";
                }
                return false;
            }
            state_ = State::RunnerExclusive;
            diagnostic_.clear();
            sealed_device_memory_retention_.clear();
            retention_published_ = false;
            ++generation_;
            return true;
        }

        /**
         * @brief Close runner admission and begin terminal physical sealing.
         * @param error Optional transition diagnostic.
         * @return True only for `RunnerExclusive -> Sealing`.
         */
        [[nodiscard]] bool beginSealing(std::string *error = nullptr)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::RunnerExclusive)
            {
                if (error)
                    *error = "prepared model context cannot enter sealing from its current lifecycle state";
                return false;
            }
            state_ = State::Sealing;
            return true;
        }

        /**
         * @brief Publish successful teardown and its exact retained allocation BOM.
         * @param retention Per-GPU allocations still owned after runner teardown.
         * @param error Optional transition diagnostic.
         * @return True only for `Sealing -> Reusable`.
         *
         * Admission describes peak construction and execution storage. Dynamic
         * teardown may release shadow pools before the model becomes reusable,
         * so only this terminal seal may publish the final owner's retirement
         * obligation.
         */
        [[nodiscard]] bool publishReusable(
            std::vector<ModelDeviceMemoryRetention> retention,
            std::string *error = nullptr)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::Sealing)
            {
                if (error)
                    *error = "prepared model context can become reusable only after sealing";
                return false;
            }
            std::vector<DeviceId> devices;
            devices.reserve(retention.size());
            for (const auto &row : retention)
            {
                if (!row.valid() ||
                    std::find(devices.begin(), devices.end(), row.device) !=
                        devices.end())
                {
                    if (error)
                    {
                        *error =
                            "prepared model context cannot publish an invalid or duplicate device-memory retention row";
                    }
                    return false;
                }
                devices.push_back(row.device);
            }
            sealed_device_memory_retention_ = std::move(retention);
            retention_published_ = true;
            state_ = State::Reusable;
            diagnostic_.clear();
            return true;
        }

        /**
         * @brief Read the final-owner allocation BOM at a reusable boundary.
         * @param error Optional lifecycle-specific rejection diagnostic.
         * @return A copy of the sealed BOM, including an empty CPU-only value.
         *
         * The optional distinguishes a valid CPU-only empty BOM from an attempt
         * to inspect mutable runner state. Retirement tickets must be captured
         * before the contract's final allocation owners are dropped.
         */
        [[nodiscard]] std::optional<std::vector<ModelDeviceMemoryRetention>>
        sealedDeviceMemoryRetention(std::string *error = nullptr) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::Reusable || !retention_published_)
            {
                if (error)
                {
                    *error = state_ == State::Invalid
                        ? (diagnostic_.empty()
                               ? "prepared model context is invalid"
                               : diagnostic_)
                        : "prepared model context has no sealed reusable device-memory retention BOM";
                }
                return std::nullopt;
            }
            return sealed_device_memory_retention_;
        }

        /**
         * @brief Permanently reject reuse after an incomplete seal.
         * @param diagnostic Precise failure retained for every later consumer.
         */
        void invalidate(std::string diagnostic)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::Invalid;
            sealed_device_memory_retention_.clear();
            retention_published_ = false;
            diagnostic_ = diagnostic.empty()
                ? "prepared model context sealing failed"
                : std::move(diagnostic);
        }

        /** @return Race-safe current lifecycle state. */
        [[nodiscard]] State state() const noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return state_;
        }

        /** @return Number of runners that have exclusively owned this context. */
        [[nodiscard]] std::uint64_t generation() const noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return generation_;
        }

        /** @return Retained invalidation diagnostic, if any. */
        [[nodiscard]] std::string diagnostic() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return diagnostic_;
        }

    private:
        mutable std::mutex mutex_;
        State state_ = State::RunnerExclusive;
        std::uint64_t generation_ = 1u;
        std::vector<ModelDeviceMemoryRetention>
            sealed_device_memory_retention_;
        bool retention_published_ = false;
        std::string diagnostic_;
    };

    /**
     * @brief Exact model-state effect of a caller-selected generation token.
     *
     * Ordinary decode returns a sampled token before that token necessarily
     * owns a model row. Forced continuations likewise differ in whether the
     * returned token is merely selected for the next transaction or has
     * already been forwarded by a device-resident transaction. Publishing the
     * distinction prevents callers from guessing from backend or MTP policy
     * and accidentally forwarding a token twice.
     */
    enum class ReturnedTokenCommitState : std::uint8_t
    {
        Unspecified = 0, ///< Operation publishes no forced-token contract.
        NoModelRow = 1, ///< Control-only token (for example EOS) made no row.
        Pending = 2, ///< Token is authoritative but has not been forwarded.
        Committed = 3, ///< Model state and terminal logits include the token.
    };

    /**
     * @brief Model authority plus the production plan that prepared its weights.
     *
     * Packed weights may outlive one runner, but their validity is narrower than
     * the underlying GGUF metadata: device ordinal, layer ownership, sharding,
     * and whether trailing MTP weights were required all affect the prepared
     * set.  Carrying the certifying plan with the shared context prevents a new
     * runner from treating an arbitrary preloaded ModelContext as complete
     * device residency.  The consumer still builds its own current plan and
     * validates the weight-affecting fields before using this contract.
     *
     * Mutable graph, arena, stream, controller, and request state are never part
     * of this object. A separately typed workspace registry may retain only an
     * unnamed backing allocation after every graph borrower has been destroyed;
     * named mappings and publications remain runner-owned.
     */
    struct ModelContextReuseContract
    {
        std::shared_ptr<ModelContext> context;
        /** Shared exclusive/sealing authority required by every consumer. */
        std::shared_ptr<ModelContextReuseAuthority> reuse_authority;
        /**
         * Sealed primary workspace blocks retained beside prepared weights.
         * No graph, stream, publication, or request state belongs here.
         */
        std::shared_ptr<ReusableExecutionWorkspaceRegistry>
            reusable_execution_workspaces;
        /**
         * Exact admission/materialization ledger retained by model-owned
         * weights and workspace backing. A consumer must keep this same object
         * identity; rebuilding it from current free-memory telemetry would
         * count those live allocations under two authorities.
         */
        std::shared_ptr<PhysicalMemoryAuthority>
            physical_memory_authority;
        RankExecutionPlan prepared_weight_plan;
        /**
         * Model-frozen quotas and initial placements whose prepared engines are
         * retained by @ref context. Null when no routed authority participated.
         * Consumers may install this immutable physical plan only after the
         * requested identity below matches exactly.
         */
        std::shared_ptr<const MoERoutedExpertPlacementPlan>
            prepared_routed_weight_plan;
        /**
         * Complete CPU/GPU physical-memory certificate that admitted the
         * model-frozen ExpertOverlay plan above.  Capacity resolution and
         * final preflight consume this same immutable value; a reused model
         * context must never reconstruct its bill from allocator telemetry.
         * Null exactly when no ExpertOverlay authority participated.
         */
        std::shared_ptr<const MoEOverlayResolvedCapacityPlan>
            expert_overlay_memory_admission;
        /**
         * Canonical, collision-free identity of the requested routed-weight
         * topology that populated @ref context.
         *
         * The identity is empty when no ExpertOverlay plan participated.  It
         * includes participant/device topology, whole-expert owner order,
         * tier capacities, explicit placements, capacity-affecting maintenance
         * policy, and normalized retained-graph geometry. This prevents a
         * consumer from re-solving automatic capacity against VRAM already
         * occupied by the retained physical plan.
         */
        std::string routed_weight_authority_identity;
        /**
         * Pointer-free physical transfer evidence retained across compatible
         * runner lifetimes. No graph, stream, controller, histogram, service
         * timing, or mutable placement state is shared through this value.
         */
        std::shared_ptr<const MoEOverlaySealedMigrationMeasurements>
            expert_overlay_migration_profile;
        /**
         * Exact model/topology/catalog identity that admits the profile above.
         * A consumer may reuse the evidence only on byte-identical equality.
         */
        std::string expert_overlay_migration_profile_identity;
    };

    /**
     * @brief Typed proof that this runner consumed a prepared-model contract.
     *
     * This observation comes from constructor admission, exact plan
     * validation, and model-owned PreparedWeightStore accounting. Optional
     * PerfStats counters may mirror it for diagnostics, but must never be used
     * to decide whether reuse occurred or whether the retained state is valid.
     */
    struct ModelContextReuseStatus
    {
        bool imported_contract = false;
        bool prepared_weight_plan_validated = false;
        std::size_t prepared_entry_count = 0u;
        std::uint64_t authority_generation = 0u;

        /** @return Whether certified prepared weights were actually consumed. */
        [[nodiscard]] bool reusedPreparedWeights() const noexcept
        {
            return imported_contract && prepared_weight_plan_validated &&
                   prepared_entry_count > 0u && authority_generation > 1u;
        }
    };

    /**
     * @brief Result of a generation step or full generation
     */
    struct GenerationResult
    {
        std::vector<int32_t> tokens; ///< Generated tokens
        std::vector<float> logprobs; ///< Log probabilities (optional)
        bool is_complete{false};     ///< Whether generation is complete (EOS reached)
        std::string error;           ///< Error message if failed (empty on success)
        /** Typed state effect populated by forced-continuation operations. */
        ReturnedTokenCommitState returned_token_commit =
            ReturnedTokenCommitState::Unspecified;

        /**
         * @brief Check if generation was successful
         * @return true if no error occurred
         */
        bool success() const { return error.empty(); }

        /**
         * @brief Get number of tokens generated
         */
        size_t tokenCount() const { return tokens.size(); }
    };

    /**
     * @brief Result of one request-batched decode step.
     *
     * `requests[i]` is the decode result for logical request `i` in the batch.
     * `error` is reserved for batch-wide failures such as scheduler,
     * publication, or collective errors that prevent per-request results from
     * being trustworthy.
     */
    struct GenerationBatchResult
    {
        std::vector<GenerationResult> requests;
        std::string error;

        /**
         * @brief Check if the whole batch step succeeded.
         */
        bool success() const { return error.empty(); }
    };

    /**
     * @brief Interface for orchestrated inference
     *
     * This interface extends the basic IInferenceRunner concept to support
     * complex orchestration scenarios. It is mockable for unit testing.
     *
     * Lifecycle:
     * 1. Factory creates IOrchestrationRunner from config
     * 2. Call initialize() to load model, set up devices, build graphs
     * 3. Call generate() or prefill()/decodeStep() for inference
     * 4. Call shutdown() for clean teardown
     *
     * Thread safety: Not thread-safe. Each runner should be used from one thread.
     * For multi-threaded inference, create separate runners.
     */
    class IOrchestrationRunner
    {
    public:
        virtual ~IOrchestrationRunner() = default;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /**
         * @brief Initialize the runner
         *
         * Performs all initialization steps:
         * - Build execution plan from config
         * - Set up local TP context (if enabled)
         * - Load weights (with sharding)
         * - Build compute graphs
         * - Initialize device contexts
         *
         * @return true on success, false on failure (check lastError())
         */
        virtual bool initialize() = 0;

        /**
         * @brief Initialize far enough to validate and print the execution plan.
         *
         * Default implementations may conservatively fall back to full
         * initialize(). Runners that can separate preflight from graph/materialized
         * weight setup should override this and stop before inference resources are
         * built.
         *
         * @return true on success, false on failure (check lastError())
         */
        virtual bool initializeForDryRun()
        {
            return initialize();
        }

        /**
         * @brief Clean shutdown
         *
         * Releases all resources:
         * - Frees device memory
         * - Closes collective contexts
         * - Synchronizes pending operations
         *
         * Safe to call multiple times or if not initialized.
         */
        virtual void shutdown() = 0;

        // =====================================================================
        // Inference
        // =====================================================================

        /**
         * @brief Prefill phase: process input tokens
         *
         * Processes the prompt tokens and fills the KV cache.
         * After prefill, call decodeStep() to generate new tokens.
         *
         * @param tokens Input token IDs
         * @return true on success, false on failure (check lastError())
         */
        virtual bool prefill(const std::vector<int32_t> &tokens) = 0;

        /**
         * @brief Observe completion of the latest inference transaction for host timing.
         *
         * GPU implementations wait on the exact durable terminal event published
         * by either the complete graph transaction or a full prefix-terminal
         * restore, including shifted MTP KV population during prefill. This is a
         * benchmark result boundary, not permission to add a stream/device
         * synchronization to production inference.
         *
         * @return true when the latest inference transaction has completed.
         */
        virtual bool waitForLastInferenceCompletionForBenchmark()
        {
            return !primaryDeviceId().is_gpu();
        }

        /**
         * @brief Report mandatory production preparation before timing.
         *
         * This is a read-only lifecycle snapshot.  It never advances a
         * controller or waits for a transfer; ordinary inference and the
         * background maintenance owner remain responsible for progress.
         */
        virtual InferenceReadiness inferenceReadiness() const
        {
            return {};
        }

        /**
         * @brief Complete all mandatory internal preparation for inference.
         *
         * The concrete runner owns any production-shaped setup workload and
         * protocol progress. Applications call this same idempotent boundary
         * before serving, benchmarking, parity, or interactive inference and
         * never learn which subsystem required preparation.
         *
         * @return True only when @ref inferenceReadiness is Ready.
         */
        virtual bool prepareForInference()
        {
            return inferenceReadiness().ready();
        }

        /**
         * @brief Whether prefillBatch() is implemented for this runner.
         *
         * A request-batched decode lane is valid only if every request slot has
         * been initialized through the same ownership layer. This capability is
         * intentionally separate from prefill() to avoid loops that mutate a
         * single live request repeatedly.
         */
        virtual bool supportsPrefillBatch(int request_batch) const
        {
            (void)request_batch;
            return false;
        }

        /**
         * @brief Prefill a logical request batch.
         *
         * `token_batches[i]` is the prompt for logical request `i`.
         * Implementations must initialize independent KV/state slots and leave
         * all requests ready for a later decodeStepBatch() call.
         */
        virtual bool prefillBatch(
            const std::vector<std::vector<int32_t>> &token_batches)
        {
            (void)token_batches;
            return false;
        }

        /**
         * @brief Decode step: generate next token
         *
         * Generates a single token using the current KV cache state.
         * Must call prefill() first.
         *
         * Uses greedy sampling (argmax) unless sampling params were set.
         *
         * @return GenerationResult with the generated token
         */
        virtual GenerationResult decodeStep() = 0;

        /**
         * @brief Commit a caller-selected token at the next decode position.
         *
         * This is used for protocol-level forced continuations such as
         * thinking-budget stop prompts. Implementations must advance any live
         * KV/GDN state needed to reach the token's position, discard the
         * naturally sampled token for that position, and make @p token the
         * authoritative last token for the following decode step.
         *
         * Unlike appending text at the HTTP layer, this keeps model state,
         * sampler history, and later decode output coherent.
         * The result publishes whether the returned token's model row was
         * committed by this call or remains pending for the next transaction.
         */
        virtual GenerationResult forceDecodeToken(int32_t token)
        {
            (void)token;
            GenerationResult result;
            result.error = "forceDecodeToken unsupported";
            return result;
        }

        /**
         * @brief Whether decodeStepBatch() is implemented for this runner.
         *
         * Request batching is an explicit capability because it requires
         * scheduler ownership, per-request publication, and rollback semantics
         * that are stricter than a loop around decodeStep().
         *
         * @param request_batch Number of logical requests the caller wants to
         *        advance together.
         */
        virtual bool supportsDecodeStepBatch(int request_batch) const
        {
            (void)request_batch;
            return false;
        }

        /**
         * @brief Decode step for a logical request batch.
         *
         * Implementations must honor setDecodeStepTokenBudget() as a
         * per-request token limit and must publish or roll back all admitted
         * requests atomically.
         *
         * @param request_batch Number of active logical requests.
         */
        virtual GenerationBatchResult decodeStepBatch(int request_batch)
        {
            (void)request_batch;
            return GenerationBatchResult{{}, "decodeStepBatch unsupported"};
        }

        /**
         * @brief Full generation loop
         *
         * Convenience method that runs prefill + decode loop.
         * Equivalent to:
         *   prefill(prompt_tokens);
         *   while (!complete && count < max_new_tokens) {
         *       result = decodeStep();
         *   }
         *
         * @param prompt_tokens Input prompt token IDs
         * @param max_new_tokens Maximum number of tokens to generate
         * @param sampling Sampling parameters (temperature, top_k, top_p)
         * @return GenerationResult with all generated tokens
         */
        virtual GenerationResult generate(
            const std::vector<int32_t> &prompt_tokens,
            int max_new_tokens,
            const SamplingParams &sampling) = 0;

        /**
         * @brief Limit how many tokens the next decodeStep() may accept.
         *
         * A value <= 0 means unlimited. This is a no-op for runners that only
         * ever return one token per decode step, but MTP/speculative runners use
         * it to keep direct decode loops from advancing state past a caller's
         * remaining token budget.
         */
        virtual void setDecodeStepTokenBudget(int max_tokens) { (void)max_tokens; }

        /**
         * @brief Retire a completed decode transaction into maintenance policy.
         *
         * Server and streaming paths drive decodeStep() directly instead of using
         * generate(), so they call this hook after the complete transaction has
         * committed.  The exact token count matters for grouped MTP: one verifier
         * transaction may advance several logical decode tokens, and maintenance
         * cadence must observe every committed token without inspecting MTP state.
         *
         * @param committed_tokens Number of logical decode tokens durably committed
         *        by the completed transaction. Must be positive.
         */
        virtual bool maybeApplyMoERebalance(uint64_t committed_tokens)
        {
            return committed_tokens != 0u;
        }

        /**
         * @brief Observable MoE expert movement epoch for parity/diagnostics.
         *
         * Unlike placement epochs used for graph and prefix-cache keys, this
         * advances for graph-stable device-side runtime-table mutations too.
         */
        virtual uint64_t moeRuntimeMovementEpoch() const { return 0; }

        /**
         * @brief Observe adaptive MoE lifecycle state from its sole authority.
         *
         * This passive diagnostic never polls a controller, advances a wave,
         * or reads PerfStats. Production request admission continues to use
         * @ref inferenceReadiness; correctness tests may use this narrower
         * surface to wait for background optimization without interpreting
         * instrumentation counters as control state.
         */
        virtual MoEOptimizationStatus moeOptimizationStatus() const
        {
            return {};
        }

        /**
         * @brief Snapshot exact completed movement identities from their owner.
         *
         * Unlike PerfStats, this typed ledger is model state evidence. The
         * snapshot is passive and must not poll, advance, or synchronize the
         * movement lifecycle.
         */
        virtual MoEOptimizationMovementLedger
        moeOptimizationMovementLedger() const
        {
            return {};
        }

        /**
         * @brief Describe immutable movement geometry from the admitted model plan.
         * @return Expected authority and expressible axes, even in Static mode.
         * @throws std::invalid_argument if overlay setup has not frozen its plan.
         *
         * This non-virtual projection reuses canonical configuration, not live
         * device placement or PerfStats. It performs no device I/O, maintenance,
         * allocation admission or synchronization and belongs only in opt-in
         * diagnostics, outside the inference loop.
         */
        [[nodiscard]] MoEOptimizationMovementTopology moeOptimizationMovementTopology() const
        {
            return describeMoEOptimizationMovementTopology(config().moe_routed_expert_plan.get());
        }

        // =====================================================================
        // Configuration
        // =====================================================================

        /**
         * @brief Get the execution plan for this rank
         *
         * The execution plan describes what this rank should do:
         * - Which layers to execute
         * - Device assignments
         * - TP configuration
         * - PP neighbors
         *
         * @return Reference to the RankExecutionPlan
         */
        virtual const RankExecutionPlan &executionPlan() const = 0;

        /**
         * @brief Get orchestration configuration
         *
         * Returns the OrchestrationConfig used to create this runner.
         *
         * @return Reference to OrchestrationConfig
         */
        virtual const OrchestrationConfig &config() const = 0;

        /**
         * @brief Select MTP behavior for the next reset request lifetime.
         *
         * The runner's graph width, request-batch capacity, sidecar placement,
         * and terminal-head placement remain immutable. Implementations must
         * reject this transition while request state is live or when the
         * requested depth exceeds the retained physical envelope; they must
         * never allocate, rebind, or recapture in response to this call.
         *
         * Multi-rank implementations apply one identical policy on every
         * participant before admitting the next prefill.
         *
         * @param policy Typed request-selectable execution policy.
         * @return True only when the idle runner accepted the complete policy.
         */
        virtual bool configureMTPRequestPolicy(
            const MTPRequestPolicy &policy) = 0;

        /**
         * @brief Return the request-selectable policy currently installed.
         * @return A value copy independent of immutable physical capacity.
         */
        [[nodiscard]] virtual MTPRequestPolicy mtpRequestPolicy() const = 0;

        // =====================================================================
        // Status
        // =====================================================================

        /**
         * @brief Check if the runner is initialized
         *
         * @return true if initialize() succeeded
         */
        virtual bool isInitialized() const = 0;

        /**
         * @brief Get error message from last failed operation
         *
         * @return Error message, empty string if no error
         */
        virtual const std::string &lastError() const = 0;

        /**
         * @brief Get vocabulary size
         *
         * @return Vocabulary size from loaded model
         */
        virtual int vocabSize() const = 0;

        /**
         * @brief Get current position in KV cache
         *
         * @return Number of tokens processed so far
         */
        virtual int currentPosition() const = 0;

        /**
         * @brief Reset request/session state before starting a new sequence.
         *
         * This is a request-boundary reset. It must not destructively discard
         * reusable graph caches, prepared weights, workspace bindings, or device
         * contexts unless a lower-level implementation has detected a topology or
         * workspace lifetime change.
         */
        virtual void clearCache() = 0;

        /**
         * @brief Destructively retire the reusable prefix archive on all participants.
         *
         * This administrative operation is separate from clearCache(): it does
         * not reset the live request and must not invalidate captured graphs,
         * prepared weights, or model topology. Multi-rank implementations
         * coordinate the operation through their ordinary command authority.
         *
         * @return True after no reusable prefix record remains addressable.
         */
        virtual bool purgePrefixCache() = 0;

        /**
         * @brief Drain completed decode-boundary maintenance diagnostics.
         *
         * This is an epilogue hook, not a request reset. Implementations should
         * export already-completed async maintenance diagnostics without
         * mutating live decode state beyond retiring diagnostic event
         * bookkeeping.
         */
        virtual void drainCompletedDecodeBoundaryMaintenanceDiagnostics() {}

        /**
         * @brief Return existing request/terminal observations without device work.
         * @return Serving metadata, never a live execution-state snapshot.
         *
         * Implementations must not issue transfers, synchronize streams/devices,
         * or delegate to prefixStateProbe. Logging cannot join background work.
         */
        virtual RequestRuntimeSummary requestRuntimeSummary() const { return {}; }

        /**
         * @brief Explicit, potentially intrusive prefix-cache/MTP diagnostic.
         *
         * May observe device state and synchronize. Ordinary serving summaries
         * must use requestRuntimeSummary instead, regardless of logging level.
         * @param capture_policy Immutable diagnostic payload and per-cache range plan.
         * @return Read-only evidence; empty when this interface has no live runner.
         */
        virtual PrefixRuntimeStateSnapshot prefixStateProbe(
            const PrefixProbeCapturePolicy &capture_policy = PrefixProbeCapturePolicy::fromEnvironment()) const { return {}; }

        /**
         * @brief Get the primary compute device backing this orchestration runner.
         *
         * Adapters that expose IOrchestrationRunner through IInferenceRunner use
         * this to preserve GPU-vs-CPU policy decisions such as decode logits
         * gather suppression. Composite runners should report the underlying
         * runner's primary device.
         */
        virtual DeviceId primaryDeviceId() const { return DeviceId::cpu(); }

        // =====================================================================
        // Advanced
        // =====================================================================

        /**
         * @brief Get logits from last forward pass
         *
         * Returns raw logits before sampling. Shape: [vocab_size]
         *
         * @return Pointer to logits, or nullptr if not available
         */
        virtual const float *lastLogits() const = 0;

        /**
         * @brief Set stop tokens for generation
         *
         * Generation stops when any of these tokens is generated.
         *
         * @param stop_tokens Vector of token IDs that trigger stop
         */
        virtual void setStopTokens(const std::vector<int32_t> &stop_tokens) = 0;

        /**
         * @brief Get the tokenizer
         * @return Shared pointer to tokenizer, or nullptr if not initialized
         */
        virtual std::shared_ptr<ITokenizer> tokenizer() const = 0;

        /**
         * @brief Get the model architecture string (e.g., "qwen2", "qwen35")
         *
         * Returned string is valid for the lifetime of the runner. Empty when
         * the runner has not yet been initialized or does not know its arch.
         */
        virtual const std::string &architecture() const = 0;

        /**
         * @brief Read-only model authority for diagnostics and parity tooling.
         *
         * The returned context remains owned by the runner and is valid only
         * for the runner's initialized lifetime.  The const boundary prevents
         * diagnostic callers from loading weights or mutating the runner's
         * sole model authority.  Lightweight mocks may return nullptr.
         */
        virtual const IModelContext *modelContextForDiagnostics() const
        {
            return nullptr;
        }

        /**
         * @brief Retain the exact model authority for a later compatible runner.
         *
         * This is a setup/teardown boundary, never a hot-path accessor.  The
         * returned context owns its WeightManager and additive
         * PreparedWeightStore, while graphs, arenas, streams, controllers, and
         * request state remain owned by this runner.  Callers may retain and
         * pass the pointer back to the factory's preloaded-context overload but
         * must not mutate its weight-placement or loader policy.  Implementations
         * that span multiple independent model authorities return nullptr.
         *
         * @return Shared rank-local model authority, or nullptr when unsupported.
         */
        virtual std::optional<ModelContextReuseContract>
        modelContextReuseContract() const
        {
            return std::nullopt;
        }

        /**
         * @brief Observe imported prepared-model consumption without telemetry.
         * @return Typed reuse proof; all-zero for runners without this feature.
         */
        virtual ModelContextReuseStatus modelContextReuseStatus() const
        {
            return {};
        }

        // =====================================================================
        // Snapshot Capture (for parity testing)
        // =====================================================================

        /**
         * @brief Enable snapshot capture of intermediate activations
         *
         * When enabled, intermediate tensors are captured during forward pass.
         * These can be retrieved via getSnapshot() for comparison with
         * reference implementations (e.g., PyTorch).
         *
         * @param output_dir Optional directory to save snapshots
         */
        virtual void enableSnapshotCapture(const std::string &output_dir = "") = 0;

        /**
         * @brief Restrict snapshot capture to a set of published snapshot keys.
         *
         * Empty means capture all snapshots.
         */
        virtual void setSnapshotCaptureFilter(const std::vector<std::string> &keys)
        {
            (void)keys;
        }

        /**
         * @brief Declare graph-resident diagnostic storage before initialize().
         *
         * The caller that selects checkpoint topology supplies a complete
         * per-accelerator bound. Production runners charge it through the
         * physical-memory authority before graph or expert admission.
         *
         * @param capacity Maximum simultaneously live snapshot backing.
         * @return True only when the pre-initialization declaration is accepted.
         */
        virtual bool setSnapshotMemoryCapacity(
            GraphSnapshotMemoryCapacity capacity)
        {
            (void)capacity;
            return false;
        }

        /**
         * @brief Declare a diagnostic graph family that starts inference inactive.
         *
         * The implementation must materialize both the requested diagnostic
         * topology and the lean topology before request admission. This is a
         * readiness policy, not permission to capture after inference starts.
         * The caller supplies the snapshot filter through
         * @ref setSnapshotCaptureFilter before this transition.
         *
         * @param output_dir Optional diagnostic destination.
         * @return True when the pre-initialization policy was accepted.
         */
        virtual bool prepareInactiveSnapshotCapture(
            const std::string &output_dir = "")
        {
            (void)output_dir;
            return false;
        }

        /**
         * @brief Select a previously materialized diagnostic graph family.
         *
         * This transition may clear request-local snapshot values, but it must
         * not capture, instantiate, allocate, synchronize, or alter serving
         * geometry. Implementations fail when preparation was not completed.
         *
         * @return True only when the selected executable family was setup-ready.
         */
        virtual bool activatePreparedSnapshotCapture()
        {
            return false;
        }

        /**
         * @brief Disable snapshot capture and clear stored snapshots
         */
        virtual void disableSnapshotCapture() = 0;

        /**
         * @brief Clear stored snapshots but keep capture enabled
         */
        virtual void clearSnapshots() = 0;

        /**
         * @brief Retrieve a captured snapshot by key
         *
         * @param key Snapshot identifier (e.g., "layer0_Q_PROJECTION", "EMBEDDING")
         * @param out_size Output parameter for snapshot size in bytes
         * @return Pointer to snapshot data (FP32), or nullptr if key doesn't exist
         */
        virtual const float *getSnapshot(const std::string &key, size_t &out_size) const = 0;

        /**
         * @brief Get list of all captured snapshot keys
         *
         * @return Vector of snapshot identifiers
         */
        virtual std::vector<std::string> getSnapshotKeys() const = 0;

        // =====================================================================
        // Profiling
        // =====================================================================

        /**
         * @brief Get executor profiling statistics
         *
         * Returns per-stage overhead breakdown (coherence, allocation, etc.).
         * The benchmark runner exports populated statistics through PerfStats;
         * `LLAMINAR_EXECUTOR_PROFILING=1` explicitly enables additional legacy
         * executor instrumentation when diagnosing eager execution.
         *
         * @return Pointer to GraphExecutorStats, or nullptr if not available
         */
        virtual const GraphExecutorStats *executorStats() const { return nullptr; }

        /**
         * @brief Reset executor profiling statistics
         */
        virtual void resetExecutorStats() {}

        // =====================================================================
        // GPU-side Sampling (for benchmark optimization)
        // =====================================================================

        /**
         * @brief Perform greedy argmax sampling on device (GPU)
         *
         * Avoids D2H transfer of logits + CPU scan. Each device runs argmax
         * on its local logits shard, then the host picks the global winner.
         *
         * @return Token ID (>= 0) on success, -1 if not supported or failed.
         *         GPU decode callers must treat -1 as a hard failure rather
         *         than silently falling back to host logits.
         */
        virtual int sampleGreedyOnDevice() { return -1; }

        /**
         * @brief GPU-side sampling with full top-k/top-p support
         * @see IInferenceRunner::sampleOnDevice
         */
        virtual int sampleOnDevice(const SamplingParams &params)
        {
            (void)params;
            return -1;
        }

        /**
         * @brief Enable/disable skipping of logits D2H gather during decode
         *
         * When enabled, forwardTP() skips the gatherLogits call for decode
         * (seq_len=1), since sampleGreedyOnDevice() reads GPU logits directly.
         *
         * @param skip true to skip logits gather, false to restore normal behavior
         */
        virtual void setSkipLogitsGatherDecode(bool /*skip*/) {}

        /**
         * @brief Enable/disable skipping of logits D2H gather during prefill
         *
         * Prefill logits are never consumed in the standard generation flow
         * (the first generated token comes from a decode step). Skipping the
         * gather eliminates massive PCIe traffic for multi-device prefill.
         *
         * @param skip true to skip logits gather, false to restore normal behavior
         */
        virtual void setSkipLogitsGatherPrefill(bool /*skip*/) {}

        virtual void setSuppressTimeline(bool /*suppress*/) {}
        virtual void setAccumulatePrefill(bool /*accumulate*/) {}

        virtual void flushStageTimeline() {}

        /**
         * @brief Set active sampling parameters for use in decodeStep()
         *
         * When set, decodeStep() will use these params to decide between
         * GPU-side greedy (argmax) or GPU-side top-k/top-p sampling.
         */
        virtual void setSamplingParams(const SamplingParams & /*params*/) {}

        /**
         * @brief Get model-recommended sampling parameters
         *
         * Returns the model-specific defaults (e.g., Qwen3.5 recommends
         * temp=0.6, presence_penalty=1.5). Callers should merge these
         * as defaults when the user hasn't specified explicit values.
         */
        virtual SamplingParams getRecommendedSamplingParams() const { return {}; }

        /**
         * @brief Get the stop-thinking prompt for thinking budget enforcement
         *
         * Returns the model-specific string that forces the model out of
         * thinking mode. Empty string means the model doesn't support thinking budgets.
         */
        virtual std::string getStopThinkingPrompt() const { return ""; }

        /**
         * @brief Get the tool call output format for this model
         *
         * Returns the format used by the loaded model to emit tool calls
         * in raw text. Used by ChatCompletionHandler to parse tool calls
         * from model output.
         *
         * @return ToolCallFormat enum value (default: HERMES_2_PRO)
         */
        virtual ToolCallFormat getToolCallFormat() const { return ToolCallFormat::HERMES_2_PRO; }

        // =====================================================================
        // MPI Worker Loop (for non-root ranks in server mode)
        // =====================================================================

        /**
         * @brief Run as MPI worker for non-root ranks in server mode.
         *
         * Blocks in a loop, participating in inference collectives when the
         * inventory-resolved continuation authority initiates them. Returns
         * when that authority signals shutdown.
         * Default implementation is a no-op (single-rank doesn't need this).
         */
        virtual void runMPIWorkerLoop() {}

        /**
         * @brief Leave the coordinated worker loop while retaining this runner.
         *
         * This is a nonterminal serving-session boundary. The caller must have
         * reset all request-owned state first. Multi-rank implementations send
         * one typed command that makes every follower return without draining
         * or destroying model-lifetime graph, weight, stream, or maintenance
         * owners. A later @ref setMPICoordinatedMode call reopens command
         * admission on the same participants.
         *
         * @return True only when every participant entered the retained idle
         *         state; the default rejects runners without this lifecycle.
         */
        virtual bool yieldMPIWorkersForRetainedRunner() { return false; }

        /**
         * @brief Close coordinated inference admission and drain maintenance.
         *
         * Called by the continuation authority when serving is stopping.
         * Multi-rank workers return from runMPIWorkerLoop() after receiving the
         * terminal signal. A single-rank implementation has no command to send,
         * but must still drain model-lifetime background maintenance before
         * terminal diagnostics are inspected.
         */
        virtual void shutdownMPIWorkers() {}

        /**
         * @brief Signal MPI worker ranks to leave their worker loops after a fatal root-rank error.
         *
         * Implementations with a richer worker protocol may send an explicit
         * abort command. The default keeps legacy runners safe by falling back
         * to ordinary shutdown.
         */
        virtual void abortMPIWorkers(const std::string & /*reason*/) { shutdownMPIWorkers(); }

        /**
         * @brief Provide the real overlay-domain MPI context to nested runners.
         *
         * Composite overlay roots may intentionally use a local-only MPI context
         * for the dense/root runner while still needing the global MPI context
         * for same-layer expert domain-worker commands inside graph stages.
         */
        virtual void setMoEExpertOverlayMPIContext(std::shared_ptr<IMPIContext> /*mpi_ctx*/) {}

        /**
         * @brief Return the MPI rank that owns coordinated request execution.
         *
         * Ordinary TP/PP runners retain rank zero. A rank-agnostic heterogeneous
         * ExpertOverlay returns its inventory-resolved continuation root so the
         * process owning dense model state, tokenizer output, logits, sampling,
         * prefix state, and terminal results is also the command authority.
         *
         * @return Valid rank in the runner's coordinated MPI communicator.
         */
        virtual int coordinatedRootRank() const { return 0; }

        /**
         * @brief Enable MPI coordinated mode.
         *
         * When enabled, coordinatedRootRank() broadcasts commands so every
         * other rank can participate in inference collectives from its worker
         * loop. The root must enable this before workers enter that loop.
         *
         * Modes where all ranks run the same code (SingleShotChat, Completion)
         * must NOT enable this — they already coordinate inline.
         */
        virtual void setMPICoordinatedMode(bool /*enabled*/) {}
    };

} // namespace llaminar2
