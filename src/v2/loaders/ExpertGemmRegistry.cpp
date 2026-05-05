#include "ExpertGemmRegistry.h"

#include <functional>
#include <mutex>
#include <shared_mutex>

namespace llaminar2
{

    bool ExpertGemmRegistry::Key::operator==(const Key &other) const
    {
        return device == other.device &&
               layer == other.layer &&
               expert == other.expert &&
               role == other.role;
    }

    size_t ExpertGemmRegistry::KeyHash::operator()(const Key &k) const
    {
        size_t hash = std::hash<std::string>{}(k.device.to_string());
        hash ^= std::hash<int>{}(k.layer) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.expert) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.role)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }

    void ExpertGemmRegistry::registerEngine(DeviceId device, int layer, int expert, WeightRole role,
                                            ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        std::unique_lock lock(mutex_);
        Key key{device, layer, expert, role};
        engines_[key] = Entry{engine, std::move(ownership)};
    }

    ITensorGemm *ExpertGemmRegistry::getEngine(DeviceId device, int layer, int expert, WeightRole role) const
    {
        std::shared_lock lock(mutex_);
        Key key{device, layer, expert, role};
        auto it = engines_.find(key);
        if (it == engines_.end())
            return nullptr;
        return it->second.engine;
    }

    bool ExpertGemmRegistry::hasCompleteRole(DeviceId device, int layer, int num_experts, WeightRole role) const
    {
        if (num_experts < 0)
            return false;

        std::shared_lock lock(mutex_);
        for (int e = 0; e < num_experts; ++e)
        {
            auto it = engines_.find(Key{device, layer, e, role});
            if (it == engines_.end() || it->second.engine == nullptr)
                return false;
        }

        return true;
    }

    bool ExpertGemmRegistry::hasCompleteLayer(DeviceId device, int layer, int num_experts) const
    {
        if (num_experts < 0)
            return false;

        std::shared_lock lock(mutex_);
        for (int e = 0; e < num_experts; ++e)
        {
            for (WeightRole role : {WeightRole::GATE, WeightRole::UP, WeightRole::DOWN})
            {
                auto it = engines_.find(Key{device, layer, e, role});
                if (it == engines_.end() || it->second.engine == nullptr)
                    return false;
            }
        }

        return true;
    }

    bool ExpertGemmRegistry::populateExpertEngines(DeviceId device, int layer, int num_experts,
                                                   std::vector<ITensorGemm *> &gate_out,
                                                   std::vector<ITensorGemm *> &up_out,
                                                   std::vector<ITensorGemm *> &down_out) const
    {
        if (num_experts < 0)
        {
            gate_out.clear();
            up_out.clear();
            down_out.clear();
            return false;
        }

        gate_out.resize(num_experts, nullptr);
        up_out.resize(num_experts, nullptr);
        down_out.resize(num_experts, nullptr);

        std::shared_lock lock(mutex_);
        bool complete = true;

        for (int e = 0; e < num_experts; ++e)
        {
            auto it_gate = engines_.find(Key{device, layer, e, WeightRole::GATE});
            if (it_gate != engines_.end())
                gate_out[e] = it_gate->second.engine;

            auto it_up = engines_.find(Key{device, layer, e, WeightRole::UP});
            if (it_up != engines_.end())
                up_out[e] = it_up->second.engine;

            auto it_down = engines_.find(Key{device, layer, e, WeightRole::DOWN});
            if (it_down != engines_.end())
                down_out[e] = it_down->second.engine;

            if (gate_out[e] == nullptr || up_out[e] == nullptr || down_out[e] == nullptr)
                complete = false;
        }

        return complete;
    }

    void ExpertGemmRegistry::replaceEngine(DeviceId device, int layer, int expert, WeightRole role,
                                           ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        std::unique_lock lock(mutex_);
        Key key{device, layer, expert, role};
        engines_[key] = Entry{engine, std::move(ownership)};
    }

    bool ExpertGemmRegistry::removeEngine(DeviceId device, int layer, int expert, WeightRole role)
    {
        std::unique_lock lock(mutex_);
        Key key{device, layer, expert, role};
        return engines_.erase(key) > 0;
    }

    size_t ExpertGemmRegistry::size() const
    {
        std::shared_lock lock(mutex_);
        return engines_.size();
    }

    bool ExpertGemmRegistry::hasEnginesForLayer(DeviceId device, int layer) const
    {
        std::shared_lock lock(mutex_);
        for (const auto &[key, entry] : engines_)
        {
            if (key.device == device && key.layer == layer)
                return true;
        }
        return false;
    }

    void ExpertGemmRegistry::clear()
    {
        std::unique_lock lock(mutex_);
        engines_.clear();
    }

} // namespace llaminar2
