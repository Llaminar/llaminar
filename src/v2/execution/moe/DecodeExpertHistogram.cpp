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

namespace llaminar2
{

    // ── LayerData ─────────────────────────────────────

    DecodeExpertHistogram::LayerData::LayerData(int num_experts)
        : expert_counts(num_experts),
          weighted_sums(num_experts, 0.0f),
          slot_counts(num_experts)
    {
        for (auto& c : expert_counts)
            c.store(0, std::memory_order_relaxed);
        for (auto& s : slot_counts)
            s.fill(0);
    }

    DecodeExpertHistogram::LayerData::LayerData(LayerData&& other) noexcept
        : expert_counts(other.expert_counts.size()),
          weighted_sums(std::move(other.weighted_sums)),
          slot_counts(std::move(other.slot_counts))
    {
        for (size_t i = 0; i < expert_counts.size(); ++i)
            expert_counts[i].store(other.expert_counts[i].load(std::memory_order_relaxed),
                                   std::memory_order_relaxed);
    }

    void DecodeExpertHistogram::LayerData::reset()
    {
        for (auto& c : expert_counts)
            c.store(0, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(weight_mutex);
        std::fill(weighted_sums.begin(), weighted_sums.end(), 0.0f);
        for (auto& s : slot_counts)
            s.fill(0);
    }

    // ── DecodeExpertHistogram ─────────────────────────

    DecodeExpertHistogram::DecodeExpertHistogram(DecodeExpertHistogramConfig config)
        : config_(std::move(config)),
          expert_to_socket_(config_.expert_to_socket)
    {
        layer_data_.reserve(config_.num_layers);
        for (int i = 0; i < config_.num_layers; ++i)
            layer_data_.emplace_back(config_.num_experts);
    }

    // ── Hot path ──────────────────────────────────────

    void DecodeExpertHistogram::record(
        int layer_idx,
        const int* expert_indices,
        const float* expert_weights,
        int top_k)
    {
        auto& layer = layer_data_[layer_idx];
        const int k = std::min(top_k, static_cast<int>(MAX_TOP_K));

        // Lock-free atomic increments for counts
        for (int s = 0; s < k; ++s) {
            const int eid = expert_indices[s];
            layer.expert_counts[eid].fetch_add(1, std::memory_order_relaxed);
        }

        // Mutex-protected weighted sums and slot counts
        {
            std::lock_guard<std::mutex> lock(layer.weight_mutex);
            for (int s = 0; s < k; ++s) {
                const int eid = expert_indices[s];
                layer.weighted_sums[eid] += expert_weights[s];
                layer.slot_counts[eid][s] += 1;
            }
        }

        window_token_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── Queries ───────────────────────────────────────

    uint64_t DecodeExpertHistogram::activationCount(int layer_idx, int expert_id) const
    {
        return layer_data_[layer_idx].expert_counts[expert_id].load(std::memory_order_relaxed);
    }

    std::vector<uint64_t> DecodeExpertHistogram::layerHistogram(int layer_idx) const
    {
        const auto& layer = layer_data_[layer_idx];
        std::vector<uint64_t> result(config_.num_experts);
        for (int e = 0; e < config_.num_experts; ++e)
            result[e] = layer.expert_counts[e].load(std::memory_order_relaxed);
        return result;
    }

    std::vector<uint64_t> DecodeExpertHistogram::socketLoads(int layer_idx) const
    {
        const auto& layer = layer_data_[layer_idx];
        std::lock_guard<std::mutex> lock(placement_mutex_);
        const int num_sockets = static_cast<int>(config_.sockets.size());
        std::vector<uint64_t> loads(num_sockets, 0);
        for (int e = 0; e < config_.num_experts; ++e) {
            const int sock = expert_to_socket_[e];
            loads[sock] += layer.expert_counts[e].load(std::memory_order_relaxed);
        }
        return loads;
    }

    float DecodeExpertHistogram::weightedActivation(int layer_idx, int expert_id) const
    {
        const auto& layer = layer_data_[layer_idx];
        std::lock_guard<std::mutex> lock(layer.weight_mutex);
        return layer.weighted_sums[expert_id];
    }

    float DecodeExpertHistogram::socketImbalanceRatio(int layer_idx) const
    {
        auto loads = socketLoads(layer_idx);
        if (loads.empty()) return 1.0f;

        auto [min_it, max_it] = std::minmax_element(loads.begin(), loads.end());
        const uint64_t min_load = *min_it;
        const uint64_t max_load = *max_it;

        if (min_load == 0)
            return max_load > 0 ? std::numeric_limits<float>::infinity() : 1.0f;

        return static_cast<float>(max_load) / static_cast<float>(min_load);
    }

    float DecodeExpertHistogram::averageSocketImbalance() const
    {
        if (config_.num_layers == 0) return 1.0f;

        float sum = 0.0f;
        int finite_count = 0;
        for (int l = 0; l < config_.num_layers; ++l) {
            float ratio = socketImbalanceRatio(l);
            if (std::isfinite(ratio)) {
                sum += ratio;
                ++finite_count;
            }
        }
        return finite_count > 0 ? sum / static_cast<float>(finite_count) : std::numeric_limits<float>::infinity();
    }

    uint64_t DecodeExpertHistogram::windowTokenCount() const
    {
        return window_token_count_.load(std::memory_order_relaxed);
    }

    bool DecodeExpertHistogram::windowFull() const
    {
        return window_token_count_.load(std::memory_order_relaxed) >= static_cast<uint64_t>(config_.window_size);
    }

    uint64_t DecodeExpertHistogram::windowGeneration() const
    {
        return window_generation_.load(std::memory_order_relaxed);
    }

    // ── Window management ─────────────────────────────

    void DecodeExpertHistogram::resetWindow()
    {
        for (auto& layer : layer_data_)
            layer.reset();
        window_token_count_.store(0, std::memory_order_relaxed);
        window_generation_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── Placement update ──────────────────────────────

    void DecodeExpertHistogram::updatePlacement(const std::vector<int>& expert_to_socket)
    {
        std::lock_guard<std::mutex> lock(placement_mutex_);
        expert_to_socket_ = expert_to_socket;
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
                          [](const auto& a, const auto& b) { return a.second > b.second; });
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
        for (size_t i = 0; i < top.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << "e" << top[i].first << ":" << top[i].second;
        }
        ss << "]";

        auto loads = socketLoads(layer_idx);
        ss << ", socket_loads=[";
        for (size_t i = 0; i < loads.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << loads[i];
        }
        ss << "], imbalance=" << socketImbalanceRatio(layer_idx);

        return ss.str();
    }

} // namespace llaminar2
