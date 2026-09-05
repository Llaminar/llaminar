/**
 * @file DecodeExpertHistogram.cpp
 * @brief Per-layer decode expert utilization tracker implementation
 */

#include "DecodeExpertHistogram.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Ordering for the cross-atomic RCU bank handoff handshake.
         *
         * Bank acquisition publishes a reader count and then validates the
         * active epoch. Rotation publishes the next epoch and then observes
         * the old reader count. Those operations touch two different atomics,
         * so acquire/release ordering alone permits the store-buffering
         * outcome in which both sides observe the old value: a writer may
         * accept the old bank while rotation incorrectly observes it drained.
         * One sequentially consistent order makes those observations mutually
         * exclusive without adding a lock, allocation, or inference wait.
         */
        inline constexpr std::memory_order kHistogramBankHandoffOrder =
            std::memory_order_seq_cst;

        /** @brief Map production sources, with SyntheticTest ingesting as decode. */
        std::size_t ingestionSourceIndex(ExpertHistogramSource source)
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken:
            case ExpertHistogramSource::SyntheticTest:
                return 0;
            case ExpertHistogramSource::PrefillChunk:
                return 1;
            case ExpertHistogramSource::GroupedVerifier:
                return 2;
            }
            throw std::invalid_argument("Unknown expert histogram source");
        }

        /** @brief Reject the test-only alias from source-specific queries. */
        std::size_t querySourceIndex(ExpertHistogramSource source)
        {
            if (source == ExpertHistogramSource::SyntheticTest)
            {
                throw std::invalid_argument(
                    "SyntheticTest is an ingestion alias, not a retained histogram phase");
            }
            return ingestionSourceIndex(source);
        }

        /** @brief Flatten one source/layer/expert coordinate. */
        std::size_t sourceCountOffset(
            std::size_t source,
            int layer_idx,
            int expert_id,
            int num_layers,
            int num_experts)
        {
            return (source * static_cast<std::size_t>(num_layers) +
                    static_cast<std::size_t>(layer_idx)) *
                       static_cast<std::size_t>(num_experts) +
                   static_cast<std::size_t>(expert_id);
        }
    } // namespace

    ExpertHistogramProductionTopology
    ExpertHistogramProductionTopology::forRetainedExecution(
        int retained_layer_count,
        int main_inference_layer_count,
        ExpertHistogramServingRegime regime)
    {
        if (retained_layer_count <= 0 ||
            main_inference_layer_count <= 0 ||
            main_inference_layer_count > retained_layer_count)
        {
            throw std::invalid_argument(
                "Expert histogram production topology requires a valid main/retained layer boundary");
        }

        const bool mtp_enabled =
            regime != ExpertHistogramServingRegime::Serial;
        const bool serial_decode_is_recurring =
            regime != ExpertHistogramServingRegime::PositiveDepthMTP;

        std::vector<ExpertHistogramProductionSourceMask> layers(
            static_cast<std::size_t>(retained_layer_count));
        std::vector<ExpertHistogramProductionSourceMask> economy_layers(
            static_cast<std::size_t>(retained_layer_count));
        for (int layer = 0;
             layer < main_inference_layer_count;
             ++layer)
        {
            /* MTP never removes the serial graph family: it is required for
             * terminal catch-up and short remaining-output tails. */
            layers[static_cast<std::size_t>(layer)] = {
                true,
                true,
                mtp_enabled,
            };
            economy_layers[static_cast<std::size_t>(layer)] = {
                serial_decode_is_recurring,
                true,
                mtp_enabled,
            };
        }
        for (int layer = main_inference_layer_count;
             layer < retained_layer_count;
             ++layer)
        {
            layers[static_cast<std::size_t>(layer)] = {
                false,
                false,
                mtp_enabled,
            };
            economy_layers[static_cast<std::size_t>(layer)] = {
                false,
                false,
                mtp_enabled,
            };
        }
        return ExpertHistogramProductionTopology(
            std::move(layers), std::move(economy_layers));
    }

    bool DecodeExpertHistogramWindow::valid() const noexcept
    {
        if (num_layers <= 0 || num_experts <= 0)
            return false;
        const size_t layers = static_cast<size_t>(num_layers);
        const size_t experts = static_cast<size_t>(num_experts);
        if (layers > std::numeric_limits<size_t>::max() / experts)
            return false;
        const std::size_t entries = layers * experts;
        if (entries > std::numeric_limits<size_t>::max() /
                          kExpertHistogramProductionSourceCount ||
            expert_counts.size() != entries ||
            source_expert_counts.size() !=
                entries * kExpertHistogramProductionSourceCount)
        {
            return false;
        }

        /*
         * The aggregate is deliberately stored as well as the phase banks so
         * old diagnostics remain O(1). Authenticate that redundancy here: a
         * malformed MPI packet or broken producer can never silently steer the
         * placement planner with a different total than its phase evidence.
         */
        for (std::size_t entry = 0; entry < entries; ++entry)
        {
            std::uint64_t phase_sum = 0;
            for (std::size_t source = 0;
                 source < kExpertHistogramProductionSourceCount;
                 ++source)
            {
                phase_sum += source_expert_counts[source * entries + entry];
            }
            if (phase_sum != expert_counts[entry])
                return false;
        }
        return true;
    }

    ValidatedDecodeExpertHistogramWindowView
    DecodeExpertHistogramWindow::validatedView() const
    {
        if (!valid())
        {
            throw std::invalid_argument(
                "Cannot authenticate a malformed frozen expert histogram window");
        }
        return ValidatedDecodeExpertHistogramWindowView(*this);
    }

    ValidatedDecodeExpertHistogramWindowView::
        ValidatedDecodeExpertHistogramWindowView(
            const DecodeExpertHistogramWindow &window) noexcept
        : window_(&window)
    {
    }

    int ValidatedDecodeExpertHistogramWindowView::numLayers() const noexcept
    {
        return window_->num_layers;
    }

    int ValidatedDecodeExpertHistogramWindowView::numExperts() const noexcept
    {
        return window_->num_experts;
    }

    std::uint64_t
    ValidatedDecodeExpertHistogramWindowView::generation() const noexcept
    {
        return window_->generation;
    }

    std::uint64_t
    ValidatedDecodeExpertHistogramWindowView::tokenCount() const noexcept
    {
        return window_->token_count;
    }

    std::uint64_t ValidatedDecodeExpertHistogramWindowView::activationCount(
        int layer_idx,
        int expert_id) const
    {
        if (layer_idx < 0 || layer_idx >= window_->num_layers ||
            expert_id < 0 || expert_id >= window_->num_experts)
        {
            throw std::out_of_range(
                "Validated frozen expert histogram index is outside its geometry");
        }
        return window_->expert_counts[
            static_cast<std::size_t>(layer_idx) *
                static_cast<std::size_t>(window_->num_experts) +
            static_cast<std::size_t>(expert_id)];
    }

    std::uint64_t ValidatedDecodeExpertHistogramWindowView::activationCount(
        ExpertHistogramSource source,
        int layer_idx,
        int expert_id) const
    {
        if (layer_idx < 0 || layer_idx >= window_->num_layers ||
            expert_id < 0 || expert_id >= window_->num_experts)
        {
            throw std::out_of_range(
                "Validated frozen expert histogram phase index is outside its geometry");
        }
        return window_->source_expert_counts[sourceCountOffset(
            querySourceIndex(source),
            layer_idx,
            expert_id,
            window_->num_layers,
            window_->num_experts)];
    }

    uint64_t DecodeExpertHistogramWindow::activationCount(
        int layer_idx,
        int expert_id) const
    {
        if (!valid() || layer_idx < 0 || layer_idx >= num_layers ||
            expert_id < 0 || expert_id >= num_experts)
        {
            throw std::out_of_range(
                "Frozen expert histogram index is outside its geometry");
        }
        return expert_counts[
            static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts) +
            static_cast<size_t>(expert_id)];
    }

    uint64_t DecodeExpertHistogramWindow::activationCount(
        ExpertHistogramSource source,
        int layer_idx,
        int expert_id) const
    {
        if (!valid() || layer_idx < 0 || layer_idx >= num_layers ||
            expert_id < 0 || expert_id >= num_experts)
        {
            throw std::out_of_range(
                "Frozen expert histogram phase index is outside its geometry");
        }
        return source_expert_counts[sourceCountOffset(
            querySourceIndex(source),
            layer_idx,
            expert_id,
            num_layers,
            num_experts)];
    }

    std::vector<uint64_t> DecodeExpertHistogramWindow::layerHistogram(
        int layer_idx) const
    {
        if (!valid() || layer_idx < 0 || layer_idx >= num_layers)
        {
            throw std::out_of_range(
                "Frozen expert histogram layer is outside its geometry");
        }
        const auto begin = expert_counts.begin() +
                           static_cast<ptrdiff_t>(layer_idx) * num_experts;
        return {begin, begin + num_experts};
    }

    std::vector<uint64_t> DecodeExpertHistogramWindow::layerHistogram(
        ExpertHistogramSource source,
        int layer_idx) const
    {
        if (!valid() || layer_idx < 0 || layer_idx >= num_layers)
        {
            throw std::out_of_range(
                "Frozen expert histogram phase layer is outside its geometry");
        }
        const std::size_t begin_offset = sourceCountOffset(
            querySourceIndex(source),
            layer_idx,
            0,
            num_layers,
            num_experts);
        const auto begin = source_expert_counts.begin() +
                           static_cast<std::ptrdiff_t>(begin_offset);
        return {begin, begin + num_experts};
    }

    // ── LayerData ─────────────────────────────────────

    DecodeExpertHistogram::LayerData::LayerData(int num_experts)
        : expert_counts(num_experts),
          source_expert_counts{
              std::vector<std::atomic<uint64_t>>(num_experts),
              std::vector<std::atomic<uint64_t>>(num_experts),
              std::vector<std::atomic<uint64_t>>(num_experts)},
          weighted_sums(num_experts, 0.0f),
          slot_counts(num_experts)
    {
        for (auto &c : expert_counts)
            c.store(0, std::memory_order_relaxed);
        for (auto &source_counts : source_expert_counts)
        {
            for (auto &count : source_counts)
                count.store(0, std::memory_order_relaxed);
        }
        for (auto &s : slot_counts)
            s.fill(0);
    }

    DecodeExpertHistogram::LayerData::LayerData(LayerData &&other) noexcept
        : expert_counts(other.expert_counts.size()),
          source_expert_counts{
              std::vector<std::atomic<uint64_t>>(other.expert_counts.size()),
              std::vector<std::atomic<uint64_t>>(other.expert_counts.size()),
              std::vector<std::atomic<uint64_t>>(other.expert_counts.size())},
          weighted_sums(std::move(other.weighted_sums)),
          slot_counts(std::move(other.slot_counts))
    {
        for (size_t i = 0; i < expert_counts.size(); ++i)
        {
            expert_counts[i].store(other.expert_counts[i].load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
            for (std::size_t source = 0;
                 source < kExpertHistogramProductionSourceCount;
                 ++source)
            {
                source_expert_counts[source][i].store(
                    other.source_expert_counts[source][i].load(
                        std::memory_order_relaxed),
                    std::memory_order_relaxed);
            }
        }
    }

    void DecodeExpertHistogram::LayerData::reset()
    {
        for (auto &c : expert_counts)
            c.store(0, std::memory_order_relaxed);
        for (auto &source_counts : source_expert_counts)
        {
            for (auto &count : source_counts)
                count.store(0, std::memory_order_relaxed);
        }
        std::fill(weighted_sums.begin(), weighted_sums.end(), 0.0f);
        for (auto &s : slot_counts)
            s.fill(0);
    }

    DecodeExpertHistogram::HistogramBank::HistogramBank(
        int num_layers,
        int num_experts)
    {
        layers.reserve(static_cast<size_t>(num_layers));
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
            layers.emplace_back(num_experts);
        for (auto &count : source_token_counts)
            count.store(0, std::memory_order_relaxed);
    }

    void DecodeExpertHistogram::HistogramBank::reset()
    {
        for (auto &layer : layers)
            layer.reset();
        token_count.store(0, std::memory_order_relaxed);
        for (auto &count : source_token_counts)
            count.store(0, std::memory_order_relaxed);
    }

    DecodeExpertHistogram::BankLease::~BankLease()
    {
        release();
    }

    DecodeExpertHistogram::BankLease::BankLease(BankLease &&other) noexcept
        : bank_(std::exchange(other.bank_, nullptr))
    {
    }

    DecodeExpertHistogram::BankLease &
    DecodeExpertHistogram::BankLease::operator=(BankLease &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        bank_ = std::exchange(other.bank_, nullptr);
        return *this;
    }

    void DecodeExpertHistogram::BankLease::release() noexcept
    {
        if (!bank_)
            return;
        const uint64_t previous =
            bank_->active_users.fetch_sub(1, kHistogramBankHandoffOrder);
        if (previous == 0)
            std::terminate();
        bank_ = nullptr;
    }

    DecodeExpertHistogram::BankLease
    DecodeExpertHistogram::acquireActiveBank() const noexcept
    {
        for (;;)
        {
            const uint64_t epoch =
                active_bank_epoch_.load(kHistogramBankHandoffOrder);
            const uint32_t index = static_cast<uint32_t>(epoch & 1u);
            HistogramBank *bank = banks_[index].get();
            bank->active_users.fetch_add(1, kHistogramBankHandoffOrder);
            if (active_bank_epoch_.load(kHistogramBankHandoffOrder) == epoch)
                return BankLease(bank);

            const uint64_t previous =
                bank->active_users.fetch_sub(1, kHistogramBankHandoffOrder);
            if (previous == 0)
                std::terminate();
        }
    }

    DecodeExpertHistogram::BankLease
    DecodeExpertHistogram::acquireAdmittedBank() const noexcept
    {
        if (!admitsRuntimeExpertHistogramRows(
                admission_state_.load(std::memory_order_acquire)))
        {
            return {};
        }

        auto lease = acquireActiveBank();
        if (!admitsRuntimeExpertHistogramRows(
                admission_state_.load(std::memory_order_acquire)))
        {
            return {};
        }
        return lease;
    }

    // ── DecodeExpertHistogram ─────────────────────────

    DecodeExpertHistogram::DecodeExpertHistogram(DecodeExpertHistogramConfig config)
        : config_(std::move(config)),
          active_window_size_(config_.window_size),
          ownership_(config_.ownership)
    {
        if (config_.window_size <= 0)
        {
            throw std::invalid_argument(
                "DecodeExpertHistogram requires a positive window size");
        }
        if (ownership_.layerCount() != config_.num_layers ||
            ownership_.expertCount() != config_.num_experts ||
            ownership_.participantCount() != static_cast<int>(config_.sockets.size()))
        {
            throw std::invalid_argument(
                "DecodeExpertHistogram ownership geometry does not match its layer, expert, and participant config");
        }

        banks_[0] = std::make_unique<HistogramBank>(
            config_.num_layers,
            config_.num_experts);
        banks_[1] = std::make_unique<HistogramBank>(
            config_.num_layers,
            config_.num_experts);
        banks_[0]->generation = 0;
        banks_[1]->generation = 1;
    }

    bool DecodeExpertHistogram::isTokenBoundaryLayer(int layer_idx) const
    {
        int boundary = config_.token_boundary_layer_idx;
        if (boundary < 0 || boundary >= config_.num_layers)
            boundary = config_.num_layers - 1;
        return layer_idx == boundary;
    }

    // ── Hot path ──────────────────────────────────────

    void DecodeExpertHistogram::record(
        int layer_idx,
        const int *expert_indices,
        const float *expert_weights,
        int top_k)
    {
        auto bank_lease = acquireAdmittedBank();
        if (!bank_lease)
            return;
        auto &bank = bank_lease.mutableBank();
        auto &layer = bank.layers[layer_idx];
        const int k = std::min(top_k, static_cast<int>(MAX_TOP_K));

        // Lock-free atomic increments for counts
        for (int s = 0; s < k; ++s)
        {
            const int eid = expert_indices[s];
            layer.expert_counts[eid].fetch_add(1, std::memory_order_relaxed);
            layer.source_expert_counts[
                ingestionSourceIndex(ExpertHistogramSource::DecodeToken)][eid]
                .fetch_add(1, std::memory_order_relaxed);
        }

        // Weighted sums and slot counts — no mutex needed because decode
        // is single-threaded (one token at a time) and rebalance reads
        // only happen when the window is full (after record() stops).
        for (int s = 0; s < k; ++s)
        {
            const int eid = expert_indices[s];
            layer.weighted_sums[eid] += expert_weights[s];
            layer.slot_counts[eid][s] += 1;
        }

        // Increment window counter only on the last MoE layer so that
        // window_size tracks actual decode tokens, not per-layer calls.
        // record() is called once per MoE layer per token; without this
        // guard, window_size=256 fills after only 256/num_layers tokens.
        if (isTokenBoundaryLayer(layer_idx))
        {
            bank.token_count.fetch_add(1, std::memory_order_relaxed);
            bank.source_token_counts[
                ingestionSourceIndex(ExpertHistogramSource::DecodeToken)]
                .fetch_add(1, std::memory_order_relaxed);
        }
    }

    void DecodeExpertHistogram::recordTokenBoundary(
        int layer_idx,
        uint64_t token_count,
        ExpertHistogramSource source)
    {
        if (token_count == 0)
            return;
        if (isTokenBoundaryLayer(layer_idx))
        {
            auto bank_lease = acquireAdmittedBank();
            if (!bank_lease)
                return;
            bank_lease.mutableBank().token_count.fetch_add(
                token_count,
                std::memory_order_relaxed);
            bank_lease.mutableBank()
                .source_token_counts[ingestionSourceIndex(source)]
                .fetch_add(token_count, std::memory_order_relaxed);
        }
    }

    void DecodeExpertHistogram::mergeLayerCounts(
        int layer_idx,
        const uint64_t *expert_counts,
        int num_experts,
        bool count_window_tokens,
        ExpertHistogramSource source)
    {
        if (!expert_counts || layer_idx < 0 || layer_idx >= config_.num_layers || num_experts <= 0)
            return;

        auto bank_lease = acquireAdmittedBank();
        if (!bank_lease)
            return;
        auto &bank = bank_lease.mutableBank();
        auto &layer = bank.layers[layer_idx];
        const std::size_t source_index = ingestionSourceIndex(source);
        const int count = std::min(num_experts, config_.num_experts);
        uint64_t total_activations = 0;
        for (int e = 0; e < count; ++e)
        {
            const uint64_t delta = expert_counts[e];
            if (delta == 0)
                continue;
            layer.expert_counts[e].fetch_add(delta, std::memory_order_relaxed);
            layer.source_expert_counts[source_index][e].fetch_add(
                delta,
                std::memory_order_relaxed);
            total_activations += delta;
        }

        if (count_window_tokens && isTokenBoundaryLayer(layer_idx) && config_.top_k > 0)
        {
            const uint64_t token_delta = total_activations / static_cast<uint64_t>(config_.top_k);
            if (token_delta > 0)
            {
                bank.token_count.fetch_add(token_delta, std::memory_order_relaxed);
                bank.source_token_counts[source_index].fetch_add(
                    token_delta,
                    std::memory_order_relaxed);
            }
        }
    }

    ExpertHistogramMergeResult DecodeExpertHistogram::mergeRoutedExpertRows(
        const int *expert_indices,
        const RoutedExpertHistogramMerge &merge,
        std::span<uint64_t> expert_count_scratch)
    {
        ExpertHistogramMergeResult result;

        if (!expert_indices)
        {
            result.error = "expert_indices must not be null";
            return result;
        }
        if (merge.layer_idx < 0 || merge.layer_idx >= config_.num_layers)
        {
            result.error = "layer_idx is out of range";
            return result;
        }
        if (merge.real_token_count < 0)
        {
            result.error = "real_token_count must be non-negative";
            return result;
        }
        if (merge.bucket_token_count <= 0)
        {
            result.error = "bucket_token_count must be positive";
            return result;
        }
        if (merge.bucket_token_count < merge.real_token_count)
        {
            result.error = "bucket_token_count must cover real_token_count";
            return result;
        }
        if (merge.top_k <= 0 || merge.top_k > config_.top_k || merge.top_k > MAX_TOP_K)
        {
            result.error = "top_k is incompatible with histogram config";
            return result;
        }

        const int route_stride = merge.route_stride > 0 ? merge.route_stride : merge.top_k;
        if (route_stride < merge.top_k)
        {
            result.error = "route_stride must be at least top_k";
            return result;
        }
        if (expert_count_scratch.size() <
            static_cast<std::size_t>(config_.num_experts))
        {
            result.error = "expert_count_scratch does not cover num_experts";
            return result;
        }

        const auto real_rows = static_cast<std::size_t>(
            merge.real_token_count);
        const auto top_k = static_cast<std::size_t>(merge.top_k);
        if (real_rows != 0u &&
            top_k > std::numeric_limits<std::size_t>::max() / real_rows)
        {
            result.error = "real routed prefix geometry overflows size_t";
            return result;
        }
        const std::size_t route_count = real_rows * top_k;

        /*
         * Validate the complete prefix before acquiring the mutable bank. A
         * malformed late route must never leave a partial histogram update.
         * Small decode and MTP prefixes are then cheaper as direct relaxed
         * increments than as a 256-counter clear plus dense scan. Integer
         * addition remains exact even when one expert occurs more than once.
         */
        if (route_count <= static_cast<std::size_t>(config_.num_experts))
        {
            for (int token = 0; token < merge.real_token_count; ++token)
            {
                const int *row = expert_indices +
                    static_cast<std::size_t>(token) *
                        static_cast<std::size_t>(route_stride);
                for (int slot = 0; slot < merge.top_k; ++slot)
                {
                    const int expert_id = row[slot];
                    if (expert_id < 0 || expert_id >= config_.num_experts)
                    {
                        result.error = "expert id is out of range";
                        return result;
                    }
                }
            }

            auto bank_lease = acquireAdmittedBank();
            if (!bank_lease)
            {
                result.activations_merged = route_count;
                result.ok = true;
                return result;
            }
            auto &bank = bank_lease.mutableBank();
            auto &layer = bank.layers[
                static_cast<std::size_t>(merge.layer_idx)];
            const std::size_t source_index =
                ingestionSourceIndex(merge.source);
            for (int token = 0; token < merge.real_token_count; ++token)
            {
                const int *row = expert_indices +
                    static_cast<std::size_t>(token) *
                        static_cast<std::size_t>(route_stride);
                for (int slot = 0; slot < merge.top_k; ++slot)
                {
                    const auto expert = static_cast<std::size_t>(row[slot]);
                    layer.expert_counts[expert].fetch_add(
                        1u, std::memory_order_relaxed);
                    layer.source_expert_counts[source_index][expert]
                        .fetch_add(1u, std::memory_order_relaxed);
                }
            }

            if (merge.count_window_tokens &&
                isTokenBoundaryLayer(merge.layer_idx) &&
                merge.real_token_count > 0)
            {
                const auto token_count = static_cast<std::uint64_t>(
                    merge.real_token_count);
                bank.token_count.fetch_add(
                    token_count, std::memory_order_relaxed);
                bank.source_token_counts[source_index].fetch_add(
                    token_count, std::memory_order_relaxed);
                result.tokens_counted = token_count;
            }
            result.activations_merged = route_count;
            result.ok = true;
            return result;
        }

        /*
         * This storage belongs to the calling stage and was materialized while
         * the graph was built. Clearing a compact 256-counter Qwen table is
         * deterministic bounded work; allocating it for every layer/token was
         * both unnecessary and a measurable Dynamic-mode host tax.
         */
        std::fill_n(
            expert_count_scratch.begin(),
            config_.num_experts,
            uint64_t{0});
        uint64_t candidate_activations = 0;
        for (int token = 0; token < merge.real_token_count; ++token)
        {
            const int *row = expert_indices + static_cast<size_t>(token) * static_cast<size_t>(route_stride);
            for (int slot = 0; slot < merge.top_k; ++slot)
            {
                const int expert_id = row[slot];
                if (expert_id < 0 || expert_id >= config_.num_experts)
                {
                    result.error = "expert id is out of range";
                    return result;
                }
                expert_count_scratch[static_cast<size_t>(expert_id)] += 1;
                candidate_activations += 1;
            }
        }

        auto bank_lease = acquireAdmittedBank();
        if (!bank_lease)
        {
            result.ok = true;
            return result;
        }
        auto &bank = bank_lease.mutableBank();
        auto &layer = bank.layers[static_cast<size_t>(merge.layer_idx)];
        const std::size_t source_index = ingestionSourceIndex(merge.source);
        for (int expert_id = 0; expert_id < config_.num_experts; ++expert_id)
        {
            const uint64_t delta =
                expert_count_scratch[static_cast<size_t>(expert_id)];
            if (delta != 0)
            {
                layer.expert_counts[static_cast<size_t>(expert_id)].fetch_add(
                    delta,
                    std::memory_order_relaxed);
                layer.source_expert_counts[source_index]
                    [static_cast<size_t>(expert_id)]
                        .fetch_add(delta, std::memory_order_relaxed);
            }
        }

        if (merge.count_window_tokens &&
            isTokenBoundaryLayer(merge.layer_idx) &&
            merge.real_token_count > 0)
        {
            bank.token_count.fetch_add(
                static_cast<uint64_t>(merge.real_token_count),
                std::memory_order_relaxed);
            bank.source_token_counts[source_index].fetch_add(
                static_cast<uint64_t>(merge.real_token_count),
                std::memory_order_relaxed);
            result.tokens_counted = static_cast<uint64_t>(merge.real_token_count);
        }

        result.activations_merged = candidate_activations;
        result.ok = true;
        return result;
    }

    void DecodeExpertHistogram::registerRuntimeHistogramSync(RuntimeHistogramSyncCallback callback)
    {
        if (!callback)
            return;
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        runtime_sync_callbacks_.push_back(std::move(callback));
    }

    bool DecodeExpertHistogram::syncRuntimeHistograms()
    {
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_generation_active_)
        {
            throw std::logic_error(
                "Cannot run a blocking runtime histogram sync while an asynchronous drain generation is active");
        }
        bool ok = true;
        for (const auto &callback : runtime_sync_callbacks_)
            ok = callback() && ok;
        return ok;
    }

    void DecodeExpertHistogram::registerRuntimeHistogramDrain(
        RuntimeHistogramDrainCallback callback)
    {
        if (!callback)
            return;
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_generation_active_)
        {
            throw std::logic_error(
                "Runtime histogram drain sources must be registered before maintenance begins");
        }
        runtime_drain_callbacks_.push_back(std::move(callback));
        runtime_drain_completed_.push_back(false);
    }

    RuntimeExpertHistogramDrainResult
    DecodeExpertHistogram::progressRuntimeHistogramDrains()
    {
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_callbacks_.empty())
            return RuntimeExpertHistogramDrainResult::ready();

        if (!runtime_drain_generation_active_)
        {
            std::fill(
                runtime_drain_completed_.begin(),
                runtime_drain_completed_.end(),
                false);
            runtime_drain_generation_active_ = true;
        }

        bool all_ready = true;
        for (std::size_t index = 0;
             index < runtime_drain_callbacks_.size();
             ++index)
        {
            if (runtime_drain_completed_[index])
                continue;

            auto result = runtime_drain_callbacks_[index]();
            if (result.progress ==
                RuntimeExpertHistogramDrainProgress::Failed)
            {
                runtime_drain_generation_active_ = false;
                return result;
            }
            if (result.progress ==
                RuntimeExpertHistogramDrainProgress::Ready)
            {
                runtime_drain_completed_[index] = true;
            }
            else
            {
                all_ready = false;
            }
        }

        if (!all_ready ||
            std::find(
                runtime_drain_completed_.begin(),
                runtime_drain_completed_.end(),
                false) != runtime_drain_completed_.end())
        {
            return RuntimeExpertHistogramDrainResult::pending();
        }

        runtime_drain_generation_active_ = false;
        return RuntimeExpertHistogramDrainResult::ready();
    }

    void DecodeExpertHistogram::registerRuntimeHistogramAdmission(
        RuntimeHistogramAdmissionCallback callback)
    {
        if (!callback)
            return;
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_generation_active_ ||
            admission_state_.load(std::memory_order_acquire) ==
                RuntimeExpertHistogramAdmission::CertificationQuarantine)
        {
            throw std::logic_error(
                "Runtime histogram admission publishers must be registered before certification rebase begins");
        }
        runtime_admission_callbacks_.push_back(std::move(callback));
    }

    void DecodeExpertHistogram::beginOptimizationDemandRebase()
    {
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_generation_active_)
        {
            throw std::logic_error(
                "Optimization-demand quarantine must precede its runtime histogram drain");
        }
        auto current = admission_state_.load(std::memory_order_acquire);
        if (current ==
            RuntimeExpertHistogramAdmission::CertificationQuarantine)
        {
            return;
        }
        if (current !=
            RuntimeExpertHistogramAdmission::CalibrationEvidence)
        {
            throw std::logic_error(
                "Optimization-demand quarantine may begin exactly once from calibration evidence");
        }
        admission_state_.store(
            RuntimeExpertHistogramAdmission::CertificationQuarantine,
            std::memory_order_release);
    }

    void DecodeExpertHistogram::activateOptimizationDemand()
    {
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_generation_active_)
        {
            throw std::logic_error(
                "Optimization demand cannot activate while a runtime histogram drain is active");
        }
        if (admission_state_.load(std::memory_order_acquire) !=
            RuntimeExpertHistogramAdmission::CertificationQuarantine)
        {
            throw std::logic_error(
                "Optimization demand requires a completed certification quarantine");
        }
        for (const auto &callback : runtime_admission_callbacks_)
        {
            if (!callback(
                    RuntimeExpertHistogramAdmission::OptimizationDemand))
            {
                throw std::runtime_error(
                    "A runtime histogram source rejected request-boundary optimization-demand activation");
            }
        }
        admission_state_.store(
            RuntimeExpertHistogramAdmission::OptimizationDemand,
            std::memory_order_release);
    }

    void DecodeExpertHistogram::activatePrecertifiedOptimizationDemand()
    {
        std::lock_guard<std::mutex> lock(runtime_sync_mutex_);
        if (runtime_drain_generation_active_ ||
            !runtime_admission_callbacks_.empty() ||
            admission_state_.load(std::memory_order_acquire) !=
                RuntimeExpertHistogramAdmission::CalibrationEvidence)
        {
            throw std::logic_error(
                "Pre-certified optimization demand must activate before runtime histogram setup");
        }
        admission_state_.store(
            RuntimeExpertHistogramAdmission::OptimizationDemand,
            std::memory_order_release);
    }

    RuntimeExpertHistogramAdmission
    DecodeExpertHistogram::admissionState() const noexcept
    {
        return admission_state_.load(std::memory_order_acquire);
    }

    // ── Queries ───────────────────────────────────────

    uint64_t DecodeExpertHistogram::activationCount(int layer_idx, int expert_id) const
    {
        const auto bank_lease = acquireActiveBank();
        return bank_lease.bank()
            .layers[static_cast<size_t>(layer_idx)]
            .expert_counts[static_cast<size_t>(expert_id)]
            .load(std::memory_order_relaxed);
    }

    uint64_t DecodeExpertHistogram::activationCount(
        ExpertHistogramSource source,
        int layer_idx,
        int expert_id) const
    {
        const auto bank_lease = acquireActiveBank();
        return bank_lease.bank()
            .layers[static_cast<size_t>(layer_idx)]
            .source_expert_counts[querySourceIndex(source)]
                                 [static_cast<size_t>(expert_id)]
            .load(std::memory_order_relaxed);
    }

    std::vector<uint64_t> DecodeExpertHistogram::layerHistogram(int layer_idx) const
    {
        const auto bank_lease = acquireActiveBank();
        const auto &layer =
            bank_lease.bank().layers[static_cast<size_t>(layer_idx)];
        std::vector<uint64_t> result(config_.num_experts);
        for (int e = 0; e < config_.num_experts; ++e)
            result[e] = layer.expert_counts[e].load(std::memory_order_relaxed);
        return result;
    }

    std::vector<uint64_t> DecodeExpertHistogram::layerHistogram(
        ExpertHistogramSource source,
        int layer_idx) const
    {
        const auto bank_lease = acquireActiveBank();
        const auto &counts =
            bank_lease.bank()
                .layers[static_cast<size_t>(layer_idx)]
                .source_expert_counts[querySourceIndex(source)];
        std::vector<uint64_t> result(config_.num_experts);
        for (int expert = 0; expert < config_.num_experts; ++expert)
        {
            result[static_cast<size_t>(expert)] =
                counts[static_cast<size_t>(expert)].load(
                    std::memory_order_relaxed);
        }
        return result;
    }

    std::vector<uint64_t> DecodeExpertHistogram::socketLoads(int layer_idx) const
    {
        const auto bank_lease = acquireActiveBank();
        const auto &layer =
            bank_lease.bank().layers[static_cast<size_t>(layer_idx)];
        std::lock_guard<std::mutex> lock(ownership_mutex_);
        const int num_sockets = static_cast<int>(config_.sockets.size());
        std::vector<uint64_t> loads(num_sockets, 0);
        for (int e = 0; e < config_.num_experts; ++e)
        {
            const int sock = ownership_.owner(layer_idx, e);
            loads[sock] += layer.expert_counts[e].load(std::memory_order_relaxed);
        }
        return loads;
    }

    float DecodeExpertHistogram::weightedActivation(int layer_idx, int expert_id) const
    {
        const auto bank_lease = acquireActiveBank();
        const auto &layer =
            bank_lease.bank().layers[static_cast<size_t>(layer_idx)];
        return layer.weighted_sums[expert_id];
    }

    float DecodeExpertHistogram::socketImbalanceRatio(int layer_idx) const
    {
        auto loads = socketLoads(layer_idx);
        if (loads.empty())
            return 1.0f;

        auto [min_it, max_it] = std::minmax_element(loads.begin(), loads.end());
        const uint64_t min_load = *min_it;
        const uint64_t max_load = *max_it;

        if (min_load == 0)
            return max_load > 0 ? std::numeric_limits<float>::infinity() : 1.0f;

        return static_cast<float>(max_load) / static_cast<float>(min_load);
    }

    float DecodeExpertHistogram::averageSocketImbalance() const
    {
        if (config_.num_layers == 0)
            return 1.0f;

        float sum = 0.0f;
        int finite_count = 0;
        for (int l = 0; l < config_.num_layers; ++l)
        {
            float ratio = socketImbalanceRatio(l);
            if (std::isfinite(ratio))
            {
                sum += ratio;
                ++finite_count;
            }
        }
        return finite_count > 0 ? sum / static_cast<float>(finite_count) : std::numeric_limits<float>::infinity();
    }

    ExpertLoadImbalanceStats DecodeExpertHistogram::placementImbalance(
        const MoELayeredExpertOwnership &ownership) const
    {
        const auto bank_lease = acquireActiveBank();
        const auto &bank = bank_lease.bank();
        ExpertLoadImbalanceStats stats;
        stats.layer_count = config_.num_layers;

        const int num_sockets = static_cast<int>(config_.sockets.size());
        if (config_.num_layers <= 0 ||
            config_.num_experts <= 0 ||
            num_sockets <= 0 ||
            ownership.layerCount() != config_.num_layers ||
            ownership.expertCount() != config_.num_experts ||
            ownership.participantCount() != num_sockets)
        {
            return stats;
        }

        double ratio_sum = 0.0;
        double spread_sum = 0.0;
        for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx)
        {
            std::vector<uint64_t> loads(static_cast<size_t>(num_sockets), 0);
            uint64_t layer_total = 0;
            const auto &layer = bank.layers[static_cast<size_t>(layer_idx)];
            for (int expert_id = 0; expert_id < config_.num_experts; ++expert_id)
            {
                const int socket = ownership.owner(layer_idx, expert_id);

                const uint64_t count =
                    layer.expert_counts[static_cast<size_t>(expert_id)].load(std::memory_order_relaxed);
                loads[static_cast<size_t>(socket)] += count;
                layer_total += count;
            }

            if (layer_total == 0)
                continue;

            ++stats.active_layer_count;
            stats.total_activations += layer_total;

            const auto [min_it, max_it] = std::minmax_element(loads.begin(), loads.end());
            const uint64_t min_load = *min_it;
            const uint64_t max_load = *max_it;
            const double spread =
                max_load > 0
                    ? static_cast<double>(max_load - min_load) / static_cast<double>(max_load)
                    : 0.0;
            spread_sum += spread;

            double ratio = 1.0;
            if (min_load == 0 && max_load > 0)
            {
                ratio = std::numeric_limits<double>::infinity();
                ++stats.infinite_ratio_layers;
            }
            else if (min_load > 0)
            {
                ratio = static_cast<double>(max_load) / static_cast<double>(min_load);
                ratio_sum += ratio;
            }
            else
            {
                ratio_sum += ratio;
            }

            if (stats.worst_layer < 0 || spread > stats.worst_spread)
            {
                stats.worst_spread = spread;
                stats.worst_ratio = ratio;
                stats.worst_layer = layer_idx;
            }
        }

        stats.valid = stats.active_layer_count > 0;
        if (!stats.valid)
            return stats;

        stats.average_spread =
            spread_sum / static_cast<double>(stats.active_layer_count);
        if (stats.infinite_ratio_layers > 0)
        {
            stats.average_ratio = std::numeric_limits<double>::infinity();
            if (!std::isinf(stats.worst_ratio))
                stats.worst_ratio = std::numeric_limits<double>::infinity();
        }
        else
        {
            stats.average_ratio =
                ratio_sum / static_cast<double>(stats.active_layer_count);
        }

        return stats;
    }

    ExpertLoadImbalanceStats DecodeExpertHistogram::currentPlacementImbalance() const
    {
        std::lock_guard<std::mutex> lock(ownership_mutex_);
        return placementImbalance(ownership_);
    }

    uint64_t DecodeExpertHistogram::windowTokenCount() const
    {
        const auto bank_lease = acquireActiveBank();
        return bank_lease.bank().token_count.load(std::memory_order_relaxed);
    }

    bool DecodeExpertHistogram::windowFull() const
    {
        return windowTokenCount() >=
               static_cast<uint64_t>(windowSize());
    }

    int DecodeExpertHistogram::windowSize() const noexcept
    {
        return active_window_size_.load(std::memory_order_acquire);
    }

    void DecodeExpertHistogram::setWindowSize(int new_size)
    {
        if (new_size <= 0)
        {
            throw std::invalid_argument(
                "DecodeExpertHistogram window size must be positive");
        }
        active_window_size_.store(new_size, std::memory_order_release);
    }

    uint64_t DecodeExpertHistogram::windowGeneration() const
    {
        const auto bank_lease = acquireActiveBank();
        return bank_lease.bank().generation;
    }

    MoEOptimizationDemandWindow
    DecodeExpertHistogram::optimizationDemandWindow() const noexcept
    {
        for (;;)
        {
            const auto bank_lease = acquireActiveBank();
            const auto &bank = bank_lease.bank();
            const MoEOptimizationDemandWindow snapshot{
                .generation = bank.generation,
                .collected_routed_rows =
                    bank.token_count.load(std::memory_order_relaxed),
                .capacity_routed_rows =
                    static_cast<std::uint64_t>(windowSize()),
            };

            /* Rotation publishes the complete monotonically increasing epoch.
             * Rechecking it after the count prevents an ABA on the two physical
             * banks and ensures an old pinned bank is never reported as active. */
            if (active_bank_epoch_.load(std::memory_order_acquire) ==
                snapshot.generation)
            {
                return snapshot;
            }
        }
    }

    // ── Window management ─────────────────────────────

    void DecodeExpertHistogram::resetWindow()
    {
        (void)freezeAndRotateWindow();
    }

    DecodeExpertHistogramWindow
    DecodeExpertHistogram::freezeAndRotateWindow()
    {
        std::lock_guard<std::mutex> rotation_lock(rotation_mutex_);

        const uint64_t frozen_epoch =
            active_bank_epoch_.load(kHistogramBankHandoffOrder);
        const uint32_t frozen_index =
            static_cast<uint32_t>(frozen_epoch & 1u);
        const uint32_t next_index = frozen_index == 0 ? 1u : 0u;
        auto &frozen = *banks_[frozen_index];
        auto &next = *banks_[next_index];

        /*
         * A provisional acquire may have pinned the currently inactive bank
         * just before failing its active-index recheck. It performs no data
         * access, but maintenance waits for that atomic pin before recycling
         * the bank. Live inference continues on `frozen` during this wait.
         */
        while (next.active_users.load(kHistogramBankHandoffOrder) != 0)
            std::this_thread::yield();
        next.reset();
        next.generation = frozen.generation + 1;

        /*
         * Publication redirects every new writer before we inspect the old
         * bank. Writers that already passed their acquire/recheck finish in the
         * old generation; they never wait for this maintenance operation.
         */
        active_bank_epoch_.store(
            frozen_epoch + 1u,
            kHistogramBankHandoffOrder);
        while (frozen.active_users.load(kHistogramBankHandoffOrder) != 0)
            std::this_thread::yield();

        DecodeExpertHistogramWindow window;
        window.generation = frozen.generation;
        window.token_count = frozen.token_count.load(std::memory_order_relaxed);
        for (std::size_t source = 0;
             source < kExpertHistogramProductionSourceCount;
             ++source)
        {
            window.source_token_counts[source] =
                frozen.source_token_counts[source].load(
                    std::memory_order_relaxed);
        }
        window.num_layers = config_.num_layers;
        window.num_experts = config_.num_experts;
        const std::size_t entries =
            static_cast<size_t>(config_.num_layers) *
            static_cast<size_t>(config_.num_experts);
        window.expert_counts.resize(entries);
        window.source_expert_counts.resize(
            entries * kExpertHistogramProductionSourceCount);
        for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx)
        {
            const auto &layer = frozen.layers[static_cast<size_t>(layer_idx)];
            for (int expert_id = 0;
                 expert_id < config_.num_experts;
                 ++expert_id)
            {
                window.expert_counts[
                    static_cast<size_t>(layer_idx) *
                        static_cast<size_t>(config_.num_experts) +
                    static_cast<size_t>(expert_id)] =
                    layer.expert_counts[static_cast<size_t>(expert_id)].load(
                        std::memory_order_relaxed);
                for (std::size_t source = 0;
                     source < kExpertHistogramProductionSourceCount;
                     ++source)
                {
                    window.source_expert_counts[sourceCountOffset(
                        source,
                        layer_idx,
                        expert_id,
                        config_.num_layers,
                        config_.num_experts)] =
                        layer.source_expert_counts[source]
                                                  [static_cast<size_t>(expert_id)]
                                                      .load(
                                                          std::memory_order_relaxed);
                }
            }
        }

        frozen.reset();
        return window;
    }

    // ── Ownership update ──────────────────────────────

    void DecodeExpertHistogram::updateOwnership(
        const MoELayeredExpertOwnership &ownership)
    {
        if (ownership.layerCount() != config_.num_layers ||
            ownership.expertCount() != config_.num_experts ||
            ownership.participantCount() != static_cast<int>(config_.sockets.size()))
        {
            throw std::invalid_argument(
                "DecodeExpertHistogram cannot publish ownership with a different geometry");
        }

        std::lock_guard<std::mutex> lock(ownership_mutex_);
        ownership_ = ownership;
        config_.ownership = ownership;
    }

    // ── Diagnostics ───────────────────────────────────

    std::vector<std::pair<int, uint64_t>> DecodeExpertHistogram::topExperts(int layer_idx, int n) const
    {
        auto hist = layerHistogram(layer_idx);
        std::vector<std::pair<int, uint64_t>> pairs;
        pairs.reserve(config_.num_experts);
        for (int e = 0; e < config_.num_experts; ++e)
            pairs.emplace_back(e, hist[e]);

        const int count = std::min(n, static_cast<int>(pairs.size()));
        std::partial_sort(pairs.begin(), pairs.begin() + count, pairs.end(),
                          [](const auto &a, const auto &b)
                          { return a.second > b.second; });
        pairs.resize(count);
        return pairs;
    }

    std::string DecodeExpertHistogram::layerSummary(int layer_idx) const
    {
        auto hist = layerHistogram(layer_idx);
        uint64_t total = std::accumulate(hist.begin(), hist.end(), uint64_t{0});
        auto top = topExperts(layer_idx, 5);

        std::ostringstream ss;
        ss << "Layer " << layer_idx << ": total_activations=" << total
           << ", top5=[";
        for (size_t i = 0; i < top.size(); ++i)
        {
            if (i > 0)
                ss << ", ";
            ss << "e" << top[i].first << ":" << top[i].second;
        }
        ss << "]";

        auto loads = socketLoads(layer_idx);
        ss << ", socket_loads=[";
        for (size_t i = 0; i < loads.size(); ++i)
        {
            if (i > 0)
                ss << ", ";
            ss << loads[i];
        }
        ss << "], imbalance=" << socketImbalanceRatio(layer_idx);

        return ss.str();
    }

} // namespace llaminar2
