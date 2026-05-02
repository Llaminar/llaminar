#include "ExpertWeightPayloadProvider.h"

namespace llaminar2
{

    // ── Payload registration ─────────────────────────────────────────

    void ExpertWeightPayloadProvider::registerPayload(int layer, int expert_id, ExpertWeightBlobs blobs)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        payloads_[layer][expert_id] = std::move(blobs);
        auto &state = getOrCreateState(layer, expert_id);
        state.transferred = true;
    }

    void ExpertWeightPayloadProvider::registerPayloads(int layer, std::unordered_map<int, ExpertWeightBlobs> blobs)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[expert_id, blob] : blobs)
        {
            payloads_[layer][expert_id] = std::move(blob);
            auto &state = getOrCreateState(layer, expert_id);
            state.transferred = true;
        }
    }

    // ── Payload queries ──────────────────────────────────────────────

    bool ExpertWeightPayloadProvider::hasPayload(int layer, int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto layer_it = payloads_.find(layer);
        if (layer_it == payloads_.end()) return false;
        auto expert_it = layer_it->second.find(expert_id);
        return expert_it != layer_it->second.end() && !expert_it->second.empty();
    }

    std::optional<ExpertWeightBlobs> ExpertWeightPayloadProvider::payloadFor(int layer, int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto layer_it = payloads_.find(layer);
        if (layer_it == payloads_.end()) return std::nullopt;
        auto expert_it = layer_it->second.find(expert_id);
        if (expert_it == layer_it->second.end()) return std::nullopt;
        return expert_it->second;
    }

    const ExpertWeightBlobs *ExpertWeightPayloadProvider::payloadPtr(int layer, int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto layer_it = payloads_.find(layer);
        if (layer_it == payloads_.end()) return nullptr;
        auto expert_it = layer_it->second.find(expert_id);
        if (expert_it == layer_it->second.end()) return nullptr;
        return &expert_it->second;
    }

    std::unordered_map<int, ExpertWeightBlobs> ExpertWeightPayloadProvider::payloadsForLayer(int layer) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto layer_it = payloads_.find(layer);
        if (layer_it == payloads_.end()) return {};
        return layer_it->second;
    }

    // ── Preparation state tracking ───────────────────────────────────

    void ExpertWeightPayloadProvider::markExpertPrepared(int layer, int expert_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        getOrCreateState(layer, expert_id).prepared = true;
    }

    void ExpertWeightPayloadProvider::markExpertTransferred(int layer, int expert_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        getOrCreateState(layer, expert_id).transferred = true;
    }

    void ExpertWeightPayloadProvider::markExpertRawReleased(int layer, int expert_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        getOrCreateState(layer, expert_id).raw_released = true;
    }

    bool ExpertWeightPayloadProvider::isExpertPrepared(int layer, int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *s = getState(layer, expert_id);
        return s && s->prepared;
    }

    bool ExpertWeightPayloadProvider::isExpertTransferred(int layer, int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *s = getState(layer, expert_id);
        return s && s->transferred;
    }

    bool ExpertWeightPayloadProvider::isRawDataRequired(int layer, int expert_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto *s = getState(layer, expert_id);
        if (!s) return true; // Unknown expert → raw data still needed
        return !s->prepared && !s->transferred;
    }

    bool ExpertWeightPayloadProvider::allExpertsPreparedOrTransferred(int layer, int num_experts) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto layer_it = states_.find(layer);
        if (layer_it == states_.end()) return false;
        const auto &layer_states = layer_it->second;
        for (int e = 0; e < num_experts; ++e)
        {
            auto it = layer_states.find(e);
            if (it == layer_states.end()) return false;
            if (!it->second.prepared && !it->second.transferred) return false;
        }
        return true;
    }

    // ── Cleanup ──────────────────────────────────────────────────────

    void ExpertWeightPayloadProvider::removePayload(int layer, int expert_id)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto layer_it = payloads_.find(layer);
        if (layer_it != payloads_.end())
            layer_it->second.erase(expert_id);
        auto state_it = states_.find(layer);
        if (state_it != states_.end())
            state_it->second.erase(expert_id);
    }

    void ExpertWeightPayloadProvider::removeLayer(int layer)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        payloads_.erase(layer);
        states_.erase(layer);
    }

    void ExpertWeightPayloadProvider::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        payloads_.clear();
        states_.clear();
    }

    size_t ExpertWeightPayloadProvider::totalPayloadCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto &[layer, experts] : payloads_)
            count += experts.size();
        return count;
    }

    size_t ExpertWeightPayloadProvider::totalPayloadBytes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t bytes = 0;
        for (const auto &[layer, experts] : payloads_)
            for (const auto &[expert_id, blobs] : experts)
                bytes += blobs.totalBytes();
        return bytes;
    }

    // ── Private helpers ──────────────────────────────────────────────

    ExpertPreparationState &ExpertWeightPayloadProvider::getOrCreateState(int layer, int expert_id)
    {
        return states_[layer][expert_id];
    }

    const ExpertPreparationState *ExpertWeightPayloadProvider::getState(int layer, int expert_id) const
    {
        auto layer_it = states_.find(layer);
        if (layer_it == states_.end()) return nullptr;
        auto expert_it = layer_it->second.find(expert_id);
        if (expert_it == layer_it->second.end()) return nullptr;
        return &expert_it->second;
    }

} // namespace llaminar2
