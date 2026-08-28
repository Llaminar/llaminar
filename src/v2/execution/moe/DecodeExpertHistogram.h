/**
 * @file DecodeExpertHistogram.h
 * @brief Per-layer decode expert utilization tracker with sliding window
 *
 * Tracks expert activation patterns during MoE decode for socket-aware
 * dynamic rebalancing. Designed for zero allocation on the hot path
 * after initialization.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "MoELayeredExpertOwnership.h"
#include "MoEOptimizationStatus.h"
#include "RuntimeExpertHistogramDrain.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    class ValidatedDecodeExpertHistogramWindowView;

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
    };

    /**
     * @brief Semantic inference phase that produced routed-expert demand.
     *
     * The three production values are retained independently because one
     * expert activation has a different service cost in serial decode,
     * bucketed prefill, and grouped MTP verification. `SyntheticTest` is an
     * ingestion-only alias for decode demand; it lets device-free fixtures use
     * the historical merge API without creating a fourth production phase.
     */
    enum class ExpertHistogramSource
    {
        DecodeToken,
        PrefillChunk,
        GroupedVerifier,
        SyntheticTest,
    };

    /// Number of separately retained production inference phases.
    inline constexpr std::size_t kExpertHistogramProductionSourceCount = 3;

    /** @brief Typed availability mask in decode/prefill/grouped order. */
    using ExpertHistogramProductionSourceMask =
        std::array<bool, kExpertHistogramProductionSourceCount>;

    /** Every production phase is enabled unless runtime policy says otherwise. */
    inline constexpr ExpertHistogramProductionSourceMask
        kAllExpertHistogramProductionSources{true, true, true};

    /** @return Dense retained-phase index, or the source-count sentinel. */
    [[nodiscard]] constexpr std::size_t expertHistogramProductionSourceIndex(
        ExpertHistogramSource source) noexcept
    {
        switch (source)
        {
        case ExpertHistogramSource::DecodeToken:
            return 0;
        case ExpertHistogramSource::PrefillChunk:
            return 1;
        case ExpertHistogramSource::GroupedVerifier:
            return 2;
        case ExpertHistogramSource::SyntheticTest:
            break;
        }
        return kExpertHistogramProductionSourceCount;
    }

    /** @return Whether decode/prefill are present and every bit is typed. */
    [[nodiscard]] constexpr bool validExpertHistogramProductionSourceMask(
        const ExpertHistogramProductionSourceMask &mask) noexcept
    {
        /* Decode and prefill are universal; grouped verification is optional. */
        return mask[0] && mask[1];
    }

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

    class DecodeExpertHistogram
    {
    public:
        static constexpr int MAX_TOP_K = 16;

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

        /// Merge a routed-token slice into the histogram. Only the leading
        /// real_token_count rows are counted; padded bucket rows are ignored.
        ExpertHistogramMergeResult mergeRoutedExpertRows(
            const int *expert_indices,
            const RoutedExpertHistogramMerge &merge);

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

            explicit LayerData(int num_experts);
            LayerData(LayerData &&other) noexcept;
            LayerData &operator=(LayerData &&) = delete;
            LayerData(const LayerData &) = delete;
            LayerData &operator=(const LayerData &) = delete;
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

            HistogramBank(int num_layers, int num_experts);
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
