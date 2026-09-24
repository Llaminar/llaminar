/**
 * @file DecodeExpertHistogram.h
 * @brief Per-layer routing demand with one RCU generation and physical owner.
 *
 * Tracks expert activation patterns during MoE decode for socket-aware
 * dynamic rebalancing. Designed for zero allocation on the hot path
 * after initialization. Transaction-enabled banks retain complete route batches
 * together with their marginals; only retired banks may be copied or reset.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "ExpertHistogramSource.h"
#include "MoELayeredExpertOwnership.h"
#include "MoEOptimizationStatus.h"
#include "MoEOverlayTransactionDemand.h"
#include "RuntimeExpertHistogramDrain.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    class ValidatedDecodeExpertHistogramWindowView;
    class PhysicalMemoryAuthority;
    class DecodeExpertTransactionWindow;

    /**
     * @brief Admitted transaction payload, required by transaction-cost consumers.
     *
     * Counter-only diagnostic consumers do not request this storage. A consumer
     * claiming transaction economics must require it explicitly; absent evidence
     * is never permission to substitute marginal-window cost estimates.
     */
    struct ExpertHistogramTransactionConfig
    {
        moe_overlay_economy::TransactionDemandCapacity capacity;
        std::shared_ptr<PhysicalMemoryAuthority> memory;
    };

    /** @brief Immutable histogram geometry, ownership and optional demand payload. */
    struct DecodeExpertHistogramConfig
    {
        int num_layers = 0;
        int num_experts = 0;
        int top_k = 0;
        int window_size = 256; ///< Decode tokens per window epoch
        /// Layer index that advances the decode-token window. A negative value
        /// falls back to num_layers - 1 for dense/all-routed legacy configs.
        int token_boundary_layer_idx = -1;
        std::vector<DeviceId> sockets;
        /// Complete per-layer expert ownership used by load diagnostics.
        /// Updated atomically at accepted Dynamic publication boundaries.
        MoELayeredExpertOwnership ownership;
        /** Complete transaction banks; their maximum row target bounds adaptation. */
        std::optional<ExpertHistogramTransactionConfig> transaction_demand;
    };

    /** @brief Typed availability mask in decode/prefill/MTP-routed order. */
    using ExpertHistogramProductionSourceMask =
        std::array<bool, kExpertHistogramProductionSourceCount>;

    /** Every production phase is enabled unless runtime policy says otherwise. */
    inline constexpr ExpertHistogramProductionSourceMask
        kAllExpertHistogramProductionSources{true, true, true};

    /**
     * @brief Serving-policy role of serial decode in an MTP-capable graph family.
     *
     * Graph reachability and movement economics are intentionally distinct.
     * Positive-depth MTP retains serial decode for bounded catch-up and short
     * tails, but those exceptional rows must not prevent the steady-state
     * prefill/grouped-verifier service profile from becoming ready.  Adaptive
     * depth with a selectable zero-depth state, by contrast, uses serial decode
     * as an ordinary serving phase and must price it before moving experts.
     */
    enum class ExpertHistogramServingRegime : std::uint8_t
    {
        /** MTP execution is disabled; serial decode and prefill are primary. */
        Serial,
        /** MTP depth stays positive; serial decode is catch-up/tail only. */
        PositiveDepthMTP,
        /** Runtime policy may select either serial decode or positive-depth MTP. */
        AdaptiveSerialOrMTP,
    };

    /** @return Whether at least one typed production source is reachable. */
    [[nodiscard]] constexpr bool validExpertHistogramProductionSourceMask(
        const ExpertHistogramProductionSourceMask &mask) noexcept
    {
        return mask[0] || mask[1] || mask[2];
    }

    /**
     * @brief Exact service-phase reachability for every retained routed layer.
     *
     * Retained model storage and executable graph families are different
     * concepts. An MTP-capable model may retain ordinary main-model layers and
     * one or more predictor sidecar layers, while ordinary prefill reaches only
     * the main interval and MTP routed work reaches both intervals. A single
     * global bit mask cannot express that topology and caused economy
     * certification to wait forever for impossible layer/phase observations.
     *
     * This value is the sole semantic authority. Global masks used by bounded
     * MPI/device wire records are derived with @ref activeSources; they are not
     * independently configurable policy.
     */
    class ExpertHistogramProductionTopology final
    {
    public:
        /** Construct an invalid placeholder suitable only for later assignment. */
        ExpertHistogramProductionTopology() = default;

        /**
         * @brief Retain one source mask per contiguous routed layer.
         * @param layer_sources Layer-ordered masks; individual retained but
         *        inactive auxiliary layers may have an all-false mask.
         * @throws std::invalid_argument when the vector is empty or its union
         *         contains no reachable production phase.
         */
        explicit ExpertHistogramProductionTopology(
            std::vector<ExpertHistogramProductionSourceMask> layer_sources)
            : layer_sources_(std::move(layer_sources)),
              service_economy_sources_(layer_sources_)
        {
            if (!valid())
            {
                throw std::invalid_argument(
                    "Expert histogram production topology requires retained layers and at least one reachable phase");
            }
        }

        /**
         * @brief Retain graph reachability and the economy-priced subset.
         * @param layer_sources Exact phases that a retained graph may execute.
         * @param service_economy_sources Phases whose recurring service cost
         *        participates in migration admission. Every set bit must also
         *        be reachable at the same layer.
         * @throws std::invalid_argument for empty, mismatched, or contradictory
         *         topology.
         */
        ExpertHistogramProductionTopology(
            std::vector<ExpertHistogramProductionSourceMask> layer_sources,
            std::vector<ExpertHistogramProductionSourceMask>
                service_economy_sources)
            : layer_sources_(std::move(layer_sources)),
              service_economy_sources_(
                  std::move(service_economy_sources))
        {
            if (!valid())
            {
                throw std::invalid_argument(
                    "Expert histogram production topology requires consistent reachable and economy-priced layer phases");
            }
        }

        /** @return A uniform topology used by phase-symmetric graph families. */
        [[nodiscard]] static ExpertHistogramProductionTopology uniform(
            int layer_count,
            ExpertHistogramProductionSourceMask sources)
        {
            if (layer_count <= 0 ||
                !validExpertHistogramProductionSourceMask(sources))
            {
                throw std::invalid_argument(
                    "Uniform expert histogram topology requires positive layers and a reachable phase");
            }
            return ExpertHistogramProductionTopology(
                std::vector<ExpertHistogramProductionSourceMask>(
                    static_cast<std::size_t>(layer_count), sources));
        }

        /**
         * @brief Describe the graph phases reachable by a retained model.
         *
         * Ordinary main-model decode remains graph-reachable when MTP is
         * enabled: terminal catch-up and short-tail transactions execute the
         * serial one-row graph even though steady-state verification is
         * grouped. @p regime states whether that serial path is recurring
         * service that must participate in movement economics or only an
         * exceptional legal path. Retained predictor-only layers execute only
         * as part of MTP and remain phase-empty when MTP is disabled.
         *
         * @param retained_layer_count Total main plus predictor layer count.
         * @param main_inference_layer_count Exclusive main-layer boundary.
         * @param regime Typed serving policy governing MTP and serial decode.
         * @return Exact immutable per-layer production reachability.
         * @throws std::invalid_argument for an invalid layer boundary.
         */
        [[nodiscard]] static ExpertHistogramProductionTopology
        forRetainedExecution(
            int retained_layer_count,
            int main_inference_layer_count,
            ExpertHistogramServingRegime regime);

        /** @return Whether retained geometry and its derived source union exist. */
        [[nodiscard]] bool valid() const noexcept
        {
            if (layer_sources_.empty() ||
                layer_sources_.size() != service_economy_sources_.size() ||
                !validExpertHistogramProductionSourceMask(activeSources()) ||
                !validExpertHistogramProductionSourceMask(
                    economyActiveSources()))
            {
                return false;
            }
            for (std::size_t layer = 0; layer < layer_sources_.size(); ++layer)
            {
                for (std::size_t source = 0;
                     source < kExpertHistogramProductionSourceCount;
                     ++source)
                {
                    if (service_economy_sources_[layer][source] &&
                        !layer_sources_[layer][source])
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        /** @return Number of contiguous retained routed layers described. */
        [[nodiscard]] std::size_t layerCount() const noexcept
        {
            return layer_sources_.size();
        }

        /**
         * @return Exact reachable-source mask for @p layer.
         * @throws std::out_of_range when the layer is outside retained geometry.
         */
        [[nodiscard]] const ExpertHistogramProductionSourceMask &sources(
            int layer) const
        {
            if (layer < 0)
                throw std::out_of_range("Negative expert histogram layer");
            return layer_sources_.at(static_cast<std::size_t>(layer));
        }

        /** @return Whether one exact layer/phase coordinate can execute. */
        [[nodiscard]] bool reachable(int layer, std::size_t source) const
        {
            if (source >= kExpertHistogramProductionSourceCount)
            {
                throw std::out_of_range(
                    "Expert histogram production source is out of range");
            }
            return sources(layer)[source];
        }

        /**
         * @return Exact economy-priced source mask for @p layer.
         * @throws std::out_of_range when the layer is outside retained geometry.
         */
        [[nodiscard]] const ExpertHistogramProductionSourceMask &
        economySources(int layer) const
        {
            if (layer < 0)
            {
                throw std::out_of_range(
                    "Negative expert histogram economy layer");
            }
            return service_economy_sources_.at(
                static_cast<std::size_t>(layer));
        }

        /**
         * @return Whether one reachable layer/phase needs measured service cost.
         */
        [[nodiscard]] bool requiresServiceEvidence(
            int layer,
            std::size_t source) const
        {
            if (source >= kExpertHistogramProductionSourceCount)
            {
                throw std::out_of_range(
                    "Expert histogram economy source is out of range");
            }
            return economySources(layer)[source];
        }

        /** @return Union of phases reachable by at least one retained layer. */
        [[nodiscard]] ExpertHistogramProductionSourceMask activeSources()
            const noexcept
        {
            ExpertHistogramProductionSourceMask result{};
            for (const auto &layer : layer_sources_)
            {
                for (std::size_t source = 0;
                     source < kExpertHistogramProductionSourceCount;
                     ++source)
                {
                    result[source] = result[source] || layer[source];
                }
            }
            return result;
        }

        /** @return Union of phases included in migration service economics. */
        [[nodiscard]] ExpertHistogramProductionSourceMask
        economyActiveSources() const noexcept
        {
            ExpertHistogramProductionSourceMask result{};
            for (const auto &layer : service_economy_sources_)
            {
                for (std::size_t source = 0;
                     source < kExpertHistogramProductionSourceCount;
                     ++source)
                {
                    result[source] = result[source] || layer[source];
                }
            }
            return result;
        }

        /** @brief Compare the complete retained-layer semantic topology. */
        bool operator==(
            const ExpertHistogramProductionTopology &) const = default;

    private:
        std::vector<ExpertHistogramProductionSourceMask> layer_sources_;
        /** Reachable phases whose recurring cost is part of movement policy. */
        std::vector<ExpertHistogramProductionSourceMask>
            service_economy_sources_;
    };

    struct RoutedExpertHistogramMerge
    {
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        int layer_idx = -1;
        int real_token_count = 0;
        int bucket_token_count = 0;
        int top_k = 0;
        int route_stride = 0;
        bool count_window_tokens = true;
    };

    struct ExpertHistogramMergeResult
    {
        bool ok = false;
        uint64_t tokens_counted = 0;
        uint64_t activations_merged = 0;
        std::string error;

        explicit operator bool() const { return ok; }
    };

    struct ExpertLoadImbalanceStats
    {
        bool valid = false;                    ///< At least one active routed layer contributed.
        int layer_count = 0;                   ///< Total routed layers examined.
        int active_layer_count = 0;            ///< Layers with non-zero routed activations.
        int infinite_ratio_layers = 0;         ///< Active layers where min participant load is zero.
        int worst_layer = -1;                  ///< Layer with the largest normalized spread.
        uint64_t total_activations = 0;        ///< Total routed expert assignments in the scored window.
        double average_ratio = 1.0;            ///< Mean max/min load ratio over active layers.
        double worst_ratio = 1.0;              ///< Worst max/min ratio; infinity if any active layer has min zero.
        double average_spread = 0.0;           ///< Mean (max-min)/max over active layers, finite [0,1].
        double worst_spread = 0.0;             ///< Worst normalized spread, finite [0,1].
    };

    /**
     * @brief Immutable routing-evidence generation retained by a migration wave.
     *
     * Counts are copied only after the old preallocated bank has stopped
     * receiving writers. Inference has already switched to the other bank by
     * then, so a long-running placement/transfer wave cannot erase or mutate
     * this evidence and cannot lose routes observed after rotation.
     */
    struct DecodeExpertHistogramWindow
    {
        uint64_t generation = 0;
        uint64_t token_count = 0;
        /// Logical rows counted in each production phase, in enum order.
        std::array<uint64_t, kExpertHistogramProductionSourceCount>
            source_token_counts{};
        int num_layers = 0;
        int num_experts = 0;
        /// Aggregate [layer][expert] counts retained for existing diagnostics.
        std::vector<uint64_t> expert_counts;
        /**
         * Exact [source][layer][expert] counts for DecodeToken, PrefillChunk,
         * and GroupedVerifier. The aggregate vector must equal their elementwise
         * sum; validation rejects a divergent distributed or restored window.
         */
        std::vector<uint64_t> source_expert_counts;
        /** Observed batches authenticated against these exact counts/generation. */
        std::shared_ptr<const DecodeExpertTransactionWindow> transaction_demand;

        /** @brief Read one frozen layer/expert count, rejecting invalid geometry. */
        [[nodiscard]] uint64_t activationCount(
            int layer_idx,
            int expert_id) const;

        /**
         * @brief Read one frozen phase/layer/expert count.
         * @throws std::invalid_argument For the ingestion-only SyntheticTest source.
         * @throws std::out_of_range For invalid layer or expert geometry.
         */
        [[nodiscard]] uint64_t activationCount(
            ExpertHistogramSource source,
            int layer_idx,
            int expert_id) const;

        /** @brief Return all frozen expert counts for one layer. */
        [[nodiscard]] std::vector<uint64_t> layerHistogram(
            int layer_idx) const;

        /** @brief Return one phase's frozen expert counts for a layer. */
        [[nodiscard]] std::vector<uint64_t> layerHistogram(
            ExpertHistogramSource source,
            int layer_idx) const;

        /** @return Whether dimensions and flattened storage agree. */
        [[nodiscard]] bool valid() const noexcept;

        /**
         * @brief Authenticate this immutable window and borrow an O(1) reader.
         * @return A non-null view whose geometry and redundant phase totals
         *         were checked exactly once.
         * @throws std::invalid_argument when this window is malformed.
         *
         * `activationCount()` deliberately authenticates a freely constructed
         * public window on every standalone call. Planner hot loops must not
         * repeat that O(layers * experts) proof for every element: they create
         * this typed view once at their ownership boundary and retain the
         * underlying immutable window for at least as long as the view.
         */
        [[nodiscard]] ValidatedDecodeExpertHistogramWindowView
        validatedView() const;
    };

    /**
     * @brief Immutable, PMA-owned batches independent of recycled mutable banks.
     *
     * Copies of a frozen histogram share this evidence without copying payload.
     * The last reader releases the physical claim after its arrays are destroyed.
     * A derived/rounded forecast is not interchangeable with observed evidence.
     */
    class DecodeExpertTransactionWindow final
    {
    public:
        /** @brief Retire payload before releasing its physical allocation claim. */
        ~DecodeExpertTransactionWindow();
        DecodeExpertTransactionWindow(const DecodeExpertTransactionWindow &) = delete;
        DecodeExpertTransactionWindow &operator=(const DecodeExpertTransactionWindow &) = delete;

        /** @return Read-only batch descriptors for a checked layer coordinate. */
        [[nodiscard]] std::span<const moe_overlay_economy::TransactionDemandRecord>
        layerTransactions(int layer) const;
        /** @return One complete compact batch; rejects invalid layer/batch indices. */
        [[nodiscard]] moe_overlay_economy::TransactionRoutes routes(
            int layer, std::size_t transaction) const;
        /** @return Exact retained payload bytes carried by the physical ledger. */
        [[nodiscard]] std::size_t allocationBytes() const noexcept;
        /** @return Whether mutable window fields still name this observed sample. */
        [[nodiscard]] bool matches(const DecodeExpertHistogramWindow &window) const noexcept;
        /**
         * @brief Typed worst-case snapshot BOM, not a physical admission decision.
         * @throws std::invalid_argument or std::overflow_error for invalid geometry.
         */
        [[nodiscard]] static std::size_t maximumAllocationBytes(
            moe_overlay_economy::TransactionDemandCapacity capacity,
            int num_layers, int num_experts);
        /** @return Model routing width, including for a completely empty sample. */
        [[nodiscard]] std::uint32_t topK() const noexcept;
        /** @return The routed layer that advances this sample's logical token count. */
        [[nodiscard]] int tokenBoundaryLayer() const noexcept;
        /** @return Compact wire bytes for this sample; unused capacity is not sent. */
        [[nodiscard]] std::size_t wireBytes() const;
        /** @return Maximum packet payload used for setup-owned receive admission. */
        [[nodiscard]] static std::size_t maximumWireBytes(
            moe_overlay_economy::TransactionDemandCapacity capacity,
            int num_layers, int num_experts);
        /** @brief Encode complete batches into exact-size caller-owned packet storage. */
        void encodeWire(std::span<std::uint8_t> destination) const;
        /**
         * @brief Decode a bounded packet and publish only fully authenticated evidence.
         * @param config Local physical admission and model-owned routing capacity.
         * @param window Exact observed phase counts and generation supplied by the envelope.
         * @param packet Compact transaction payload, without capacity padding.
         * @return An immutable physical owner independent of receive-buffer reuse.
         * @throws On malformed geometry, missing admission, or conflicting route/count evidence.
         */
        [[nodiscard]] static std::shared_ptr<const DecodeExpertTransactionWindow> decodeWire(
            const ExpertHistogramTransactionConfig &config,
            const DecodeExpertHistogramWindow &window,
            std::span<const std::uint8_t> packet);

    private:
        friend class DecodeExpertHistogram;
        /**
         * @brief Authenticate and copy retired batches before their bank is reused.
         * @throws On incomplete evidence, mismatched counts or denied PMA admission.
         */
        DecodeExpertTransactionWindow(
            const ExpertHistogramTransactionConfig &config,
            std::span<const moe_overlay_economy::TransactionDemandBank> layers,
            const DecodeExpertHistogramWindow &window, int token_boundary_layer);
        struct Data;
        /** @brief Seal a decoder-owned payload after its common authentication step. */
        explicit DecodeExpertTransactionWindow(std::unique_ptr<Data> data);
        std::unique_ptr<Data> data_;
    };

    /**
     * @brief Borrowed, already-authenticated read view of one frozen window.
     *
     * Construction is private and available only through
     * @ref DecodeExpertHistogramWindow::validatedView, making the expensive
     * aggregate/phase consistency proof a one-time lifecycle transition. Each
     * subsequent accessor still rejects an invalid coordinate or source in
     * constant time. The referenced window must outlive this view.
     */
    class ValidatedDecodeExpertHistogramWindowView final
    {
    public:
        /** @return Authenticated layer count. */
        [[nodiscard]] int numLayers() const noexcept;

        /** @return Authenticated routed-expert count. */
        [[nodiscard]] int numExperts() const noexcept;

        /** @return Immutable histogram generation. */
        [[nodiscard]] std::uint64_t generation() const noexcept;

        /** @return Immutable routed-token count. */
        [[nodiscard]] std::uint64_t tokenCount() const noexcept;

        /**
         * @return Authenticated invocation boundaries and their actual routed rows.
         * @throws std::logic_error if this is a counts-only prediction/window.
         *
         * Service economics requires co-occurrence, which marginal counts cannot
         * reconstruct. The immutable window must outlive this borrowed reference.
         */
        [[nodiscard]] const DecodeExpertTransactionWindow &transactionDemand() const;

        /**
         * @brief Read one aggregate layer/expert count in constant time.
         * @throws std::out_of_range for an invalid coordinate.
         */
        [[nodiscard]] std::uint64_t activationCount(
            int layer_idx,
            int expert_id) const;

        /**
         * @brief Read one phase/layer/expert count in constant time.
         * @throws std::invalid_argument for the SyntheticTest ingestion alias.
         * @throws std::out_of_range for an invalid coordinate.
         */
        [[nodiscard]] std::uint64_t activationCount(
            ExpertHistogramSource source,
            int layer_idx,
            int expert_id) const;

    private:
        friend struct DecodeExpertHistogramWindow;

        /** @brief Bind an already-authenticated immutable window. */
        explicit ValidatedDecodeExpertHistogramWindowView(
            const DecodeExpertHistogramWindow &window) noexcept;

        const DecodeExpertHistogramWindow *window_;
    };

    /** @brief Production routing ingress and sole mutable histogram-bank lifecycle. */
    class DecodeExpertHistogram
    {
    public:
        static constexpr int MAX_TOP_K = 16;

        /** @brief Validate geometry and materialize two admitted, stable banks. */
        explicit DecodeExpertHistogram(DecodeExpertHistogramConfig config);

        // ── Hot path (allocation-free) ────────────────────

        /// Record decode routing result for one token at one layer.
        /// expert_indices: [top_k] selected expert IDs
        /// expert_weights: [top_k] corresponding routing weights
        /// Thread-safe via atomics (counts) and per-layer mutex (weighted sums).
        void record(int layer_idx,
                    const int *expert_indices,
                    const float *expert_weights,
                    int top_k);

        /// Record only the decode-token boundary for a routed layer.
        /// This is used by graph-captured device routing paths where expert
        /// counts stay on device and are merged in batches. The token window is
        /// still advanced once per decode token by the final MoE layer.
        void recordTokenBoundary(
            int layer_idx,
            uint64_t token_count = 1,
            ExpertHistogramSource source =
                ExpertHistogramSource::DecodeToken);

        /// Merge per-expert activation counts that were accumulated outside the
        /// host hot path, for example in a runtime-table device histogram.
        /// Weighted activation sums are intentionally not reconstructed here.
        /// If count_window_tokens is true and layer_idx is the final MoE layer,
        /// windowTokenCount advances by total_count / configured top_k.
        void mergeLayerCounts(int layer_idx,
                              const uint64_t *expert_counts,
                              int num_experts,
                              bool count_window_tokens = false,
                              ExpertHistogramSource source =
                                  ExpertHistogramSource::SyntheticTest);

        /**
         * @brief Merge one validated routed-row prefix without allocating.
         *
         * Only the leading `real_token_count` rows are counted; padded bucket
         * rows are ignored. The caller supplies invocation-exclusive scratch
         * retained for at least `num_experts` counters. Requiring that storage
         * in the type-level interface keeps decode publication allocation-free
         * and makes concurrent stage ownership explicit instead of hiding a
         * heap allocation or shared mutable accumulator inside the histogram.
         * The method clears and reuses the leading `num_experts` entries.
         *
         * @param expert_indices Row-major selected expert identifiers.
         * @param merge Exact logical/physical row and source contract.
         * @param expert_count_scratch Invocation-exclusive count accumulator.
         * @return Typed success evidence or a diagnostic with no mutation.
         */
        ExpertHistogramMergeResult mergeRoutedExpertRows(
            const int *expert_indices,
            const RoutedExpertHistogramMerge &merge,
            std::span<uint64_t> expert_count_scratch);

        using RuntimeHistogramSyncCallback = std::function<bool()>;
        using RuntimeHistogramDrainCallback =
            std::function<RuntimeExpertHistogramDrainResult()>;
        using RuntimeHistogramAdmissionCallback =
            std::function<bool(RuntimeExpertHistogramAdmission)>;

        /// Register a lazy sync source for device/runtime histograms.
        /// Callbacks should merge pending counts into this histogram and reset
        /// their source counters so repeated syncs are idempotent.
        void registerRuntimeHistogramSync(RuntimeHistogramSyncCallback callback);

        /// Merge all registered runtime histogram sources into this host view.
        bool syncRuntimeHistograms();

        /**
         * @brief Register one event-polled device/runtime evidence source.
         *
         * Registration is model-setup work and is rejected after a drain
         * generation has begun. The callback must merge exactly once before
         * returning Ready and must return promptly while device work is pending.
         */
        void registerRuntimeHistogramDrain(
            RuntimeHistogramDrainCallback callback);

        /**
         * @brief Advance every registered source without blocking the caller.
         *
         * A source that has returned Ready is not polled again until every
         * source completes the current aggregate generation. This prevents a
         * fast CPU source from starting a second generation while a GPU DMA is
         * still in flight.
         */
        [[nodiscard]] RuntimeExpertHistogramDrainResult
        progressRuntimeHistogramDrains();

        /**
         * @brief Register a model-lifetime device admission publisher.
         *
         * The callback publishes the supplied phase onto every exact producer
         * stream without synchronizing. Registration is model setup and is
         * rejected after the calibration-to-live transition begins.
         */
        void registerRuntimeHistogramAdmission(
            RuntimeHistogramAdmissionCallback callback);

        /**
         * @brief Close route admission before calibration evidence is drained.
         *
         * Existing writers retain their pinned old bank and are discarded by
         * the following RCU rotation. New CPU writers observe quarantine
         * immediately; device writers observe it through the drain's ordered
         * writer-state publication.
         */
        void beginOptimizationDemandRebase();

        /**
         * @brief Open the first live demand generation at a request boundary.
         *
         * Every registered device publisher is enqueued before the host state
         * becomes live. Failure is fatal and leaves host admission quarantined.
         */
        void activateOptimizationDemand();

        /**
         * @brief Mark setup-provided economics live before graph construction.
         *
         * Pre-certified authorities never enter quarantine. This transition is
         * model setup only and therefore has no registered device publishers.
         */
        void activatePrecertifiedOptimizationDemand();

        /** @return Exact calibration/quarantine/live admission phase. */
        [[nodiscard]] RuntimeExpertHistogramAdmission admissionState() const
            noexcept;

        // ── Queries (read-only, lock-free for counts) ─────

        /// Get activation count for a specific expert at a specific layer
        uint64_t activationCount(int layer_idx, int expert_id) const;

        /** @brief Get one production phase's activation count. */
        uint64_t activationCount(
            ExpertHistogramSource source,
            int layer_idx,
            int expert_id) const;

        /// Get full per-expert activation counts for a layer [num_experts]
        std::vector<uint64_t> layerHistogram(int layer_idx) const;

        /** @brief Get one production phase's per-expert layer counts. */
        std::vector<uint64_t> layerHistogram(
            ExpertHistogramSource source,
            int layer_idx) const;

        /// Get per-socket total activations for a layer [num_sockets]
        std::vector<uint64_t> socketLoads(int layer_idx) const;

        /// Get weighted activation sum for a specific expert at a layer
        float weightedActivation(int layer_idx, int expert_id) const;

        /// Socket imbalance ratio for a layer: max_socket_load / min_socket_load
        /// Returns 1.0 for perfect balance, >1.0 for imbalance.
        /// If min_load == 0: returns infinity when max_load > 0, else 1.0.
        float socketImbalanceRatio(int layer_idx) const;

        /// Average socket imbalance across all layers
        float averageSocketImbalance() const;

        /// Score the current window for an arbitrary layered ownership plan.
        ExpertLoadImbalanceStats placementImbalance(
            const MoELayeredExpertOwnership &ownership) const;

        /// Score the current window using the histogram's active placement.
        ExpertLoadImbalanceStats currentPlacementImbalance() const;

        /// Total tokens recorded in current window
        uint64_t windowTokenCount() const;

        /// Whether the window is full (ready for rebalance decision)
        bool windowFull() const;

        /// Current window generation (incremented each reset)
        uint64_t windowGeneration() const;

        /**
         * @brief Snapshot the active bank's generation, occupancy, and capacity.
         * @return One internally consistent passive demand-window observation.
         *
         * The implementation rechecks the complete RCU generation after reading
         * the pinned bank. A concurrent rotation therefore returns the new active
         * bank rather than pairing an old count with a new lifecycle state.
         */
        [[nodiscard]] MoEOptimizationDemandWindow
        optimizationDemandWindow() const noexcept;

        // ── Window management ─────────────────────────────

        /// Reset all counters and advance window generation
        void resetWindow();

        /**
         * @brief Atomically rotate inference to a clean bank and freeze the old.
         * @return Immutable counts from the generation active before rotation.
         *
         * The caller must run on a maintenance worker. Existing writers finish
         * in the old bank while new inference writers immediately use the new
         * bank; inference never acquires `rotation_mutex_` and never waits.
         */
        [[nodiscard]] DecodeExpertHistogramWindow freezeAndRotateWindow();

        /** @return Race-safe active routed-token capacity for the current bank. */
        [[nodiscard]] int windowSize() const noexcept;

        /**
         * @brief Publish the routed-token capacity used by future rotations.
         *
         * Maintenance owns this operation after an RCU bank rotation. Route
         * writers never read the capacity, while passive status readers use
         * the same atomic value as @ref windowFull, so a campaign or server
         * cannot observe a torn generation/capacity pair.
         *
         * @param new_size Positive routed-token capacity.
         * @throws std::invalid_argument when @p new_size is not positive.
         */
        void setWindowSize(int new_size);

        // ── Placement update ──────────────────────────────

        /// Publish a complete layered ownership plan after accepted rebalancing.
        void updateOwnership(const MoELayeredExpertOwnership &ownership);

        // ── Diagnostics ───────────────────────────────────

        /// Top-N hottest experts for a layer (sorted by activation count desc)
        std::vector<std::pair<int, uint64_t>> topExperts(int layer_idx, int n) const;

        /// Human-readable summary of a layer's histogram
        std::string layerSummary(int layer_idx) const;

        const DecodeExpertHistogramConfig &config() const { return config_; }

    private:
        bool isTokenBoundaryLayer(int layer_idx) const;

        DecodeExpertHistogramConfig config_;
        /** Live adaptive capacity; `config_` remains immutable setup identity. */
        std::atomic<int> active_window_size_;

        struct LayerTransactionData;

        /** @brief Participant-local counts and optional exclusively published batches. */
        struct LayerData
        {
            /// Aggregate atomic counters for lock-free hot path [num_experts].
            std::vector<std::atomic<uint64_t>> expert_counts;
            /// Exact production-phase counters [source][num_experts].
            std::array<std::vector<std::atomic<uint64_t>>,
                       kExpertHistogramProductionSourceCount>
                source_expert_counts;

            /// Protected by mutex (less frequent access)
            mutable std::mutex weight_mutex;
            std::vector<float> weighted_sums;                         // [num_experts]
            std::vector<std::array<uint64_t, MAX_TOP_K>> slot_counts; // [num_experts][MAX_TOP_K]
            std::unique_ptr<LayerTransactionData> transaction_demand;

            /** @brief Materialize stable counters and any admitted transaction arrays. */
            LayerData(int num_experts, const std::optional<ExpertHistogramTransactionConfig> &transactions);
            /** @brief Destroy payload before its PMA claim. */
            ~LayerData();
            /** @brief Move only during setup, never during publication/capture. */
            LayerData(LayerData &&other) noexcept;
            LayerData &operator=(LayerData &&) = delete;
            LayerData(const LayerData &) = delete;
            LayerData &operator=(const LayerData &) = delete;
            /** @brief Reset a retired generation; no writer/reader may retain a pin. */
            void reset();
        };

        /** @brief One of two persistent RCU histogram generations. */
        struct HistogramBank
        {
            std::vector<LayerData> layers;
            std::atomic<uint64_t> token_count{0};
            std::array<std::atomic<uint64_t>,
                       kExpertHistogramProductionSourceCount>
                source_token_counts{};
            std::atomic<uint64_t> active_users{0};
            uint64_t generation = 0;

            /** @brief Build every layer before this bank becomes publishable. */
            explicit HistogramBank(const DecodeExpertHistogramConfig &config);
            /** @brief Recycle data only after the parent RCU retirement edge. */
            void reset();
        };

        /** @brief Short RCU pin preventing a selected bank from being reset. */
        class BankLease final
        {
        public:
            BankLease() = default;
            ~BankLease();
            BankLease(const BankLease &) = delete;
            BankLease &operator=(const BankLease &) = delete;
            BankLease(BankLease &&other) noexcept;
            BankLease &operator=(BankLease &&other) noexcept;

            HistogramBank &mutableBank() noexcept { return *bank_; }
            const HistogramBank &bank() const noexcept { return *bank_; }
            explicit operator bool() const noexcept { return bank_ != nullptr; }

        private:
            friend class DecodeExpertHistogram;
            explicit BankLease(HistogramBank *bank) noexcept : bank_(bank) {}
            void release() noexcept;

            HistogramBank *bank_ = nullptr;
        };

        /** @brief Acquire and recheck the exact active RCU bank. */
        [[nodiscard]] BankLease acquireActiveBank() const noexcept;

        /**
         * @brief Pin the active bank only while route admission is open.
         *
         * Rechecking admission after the RCU pin ensures a writer racing the
         * quarantine edge either finishes in the old bank (which rotation
         * waits for and discards) or performs no mutation at all.
         */
        [[nodiscard]] BankLease acquireAdmittedBank() const noexcept;

        std::array<std::unique_ptr<HistogramBank>, 2> banks_;
        /**
         * Monotonic publication epoch; its low bit selects the active bank.
         *
         * Comparing the complete epoch, rather than only the bank index,
         * prevents an inference writer delayed across two rotations from
         * mistaking a recycled bank for the generation it originally pinned.
         */
        std::atomic<uint64_t> active_bank_epoch_{0};
        mutable std::mutex rotation_mutex_;

        // Ownership is read by diagnostics and replaced only at a safe
        // rebalance boundary, never while route counters are being recorded.
        mutable std::mutex ownership_mutex_;
        MoELayeredExpertOwnership ownership_;

        mutable std::mutex runtime_sync_mutex_;
        std::vector<RuntimeHistogramSyncCallback> runtime_sync_callbacks_;
        std::vector<RuntimeHistogramDrainCallback> runtime_drain_callbacks_;
        std::vector<bool> runtime_drain_completed_;
        std::vector<RuntimeHistogramAdmissionCallback>
            runtime_admission_callbacks_;
        bool runtime_drain_generation_active_ = false;
        std::atomic<RuntimeExpertHistogramAdmission> admission_state_{
            RuntimeExpertHistogramAdmission::CalibrationEvidence};
    };

    using DomainExpertHistogram = DecodeExpertHistogram;

} // namespace llaminar2
