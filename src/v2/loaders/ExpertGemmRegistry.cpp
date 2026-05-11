#include "ExpertGemmRegistry.h"

#include <functional>
#include <mutex>
#include <shared_mutex>

namespace llaminar2
{

    bool ExpertGemmRegistry::Key::operator==(const Key &other) const
    {
        return domain_name == other.domain_name &&
               device == other.device &&
               layer == other.layer &&
               expert == other.expert &&
               role == other.role;
    }

    size_t ExpertGemmRegistry::KeyHash::operator()(const Key &k) const
    {
        size_t hash = std::hash<std::string>{}(k.domain_name);
        hash ^= std::hash<std::string>{}(k.device.to_string()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.layer) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int>{}(k.expert) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.role)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }

    void ExpertGemmRegistry::registerEngine(DeviceId device, int layer, int expert, WeightRole role,
                                            ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        registerEngineForDomain({}, device, layer, expert, role, engine, std::move(ownership));
    }

    void ExpertGemmRegistry::registerEngineForDomain(const std::string &domain_name,
                                                     DeviceId device, int layer, int expert, WeightRole role,
                                                     ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        std::unique_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        engines_[key] = Entry{engine, std::move(ownership)};
    }

    ITensorGemm *ExpertGemmRegistry::getEngine(DeviceId device, int layer, int expert, WeightRole role) const
    {
        return getEngineForDomain({}, device, layer, expert, role);
    }

    ITensorGemm *ExpertGemmRegistry::getEngineForDomain(const std::string &domain_name,
                                                        DeviceId device, int layer, int expert, WeightRole role) const
    {
        std::shared_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        auto it = engines_.find(key);
        if (it == engines_.end())
            return nullptr;
        return it->second.engine;
    }

    bool ExpertGemmRegistry::hasCompleteRole(DeviceId device, int layer, int num_experts, WeightRole role) const
    {
        return hasCompleteRoleForDomain({}, device, layer, num_experts, role);
    }

    bool ExpertGemmRegistry::hasCompleteRoleForDomain(const std::string &domain_name,
                                                      DeviceId device, int layer, int num_experts, WeightRole role) const
    {
        if (num_experts < 0)
            return false;

        std::shared_lock lock(mutex_);
        for (int e = 0; e < num_experts; ++e)
        {
            auto it = engines_.find(Key{domain_name, device, layer, e, role});
            if (it == engines_.end() || it->second.engine == nullptr)
                return false;
        }

        return true;
    }

    bool ExpertGemmRegistry::hasCompleteRoleForExperts(DeviceId device, int layer,
                                                       const std::vector<int> &expert_ids,
                                                       WeightRole role) const
    {
        return hasCompleteRoleForExpertsInDomain({}, device, layer, expert_ids, role);
    }

    bool ExpertGemmRegistry::hasCompleteRoleForExpertsInDomain(const std::string &domain_name,
                                                               DeviceId device, int layer,
                                                               const std::vector<int> &expert_ids,
                                                               WeightRole role) const
    {
        std::shared_lock lock(mutex_);
        for (int expert : expert_ids)
        {
            if (expert < 0)
                return false;
            auto it = engines_.find(Key{domain_name, device, layer, expert, role});
            if (it == engines_.end() || it->second.engine == nullptr)
                return false;
        }

        return true;
    }

    bool ExpertGemmRegistry::hasCompleteLayer(DeviceId device, int layer, int num_experts) const
    {
        return hasCompleteLayerInDomain({}, device, layer, num_experts);
    }

    bool ExpertGemmRegistry::hasCompleteLayerInDomain(const std::string &domain_name,
                                                      DeviceId device, int layer, int num_experts) const
    {
        if (num_experts < 0)
            return false;

        std::shared_lock lock(mutex_);
        for (int e = 0; e < num_experts; ++e)
        {
            for (WeightRole role : {WeightRole::GATE, WeightRole::UP, WeightRole::DOWN})
            {
                auto it = engines_.find(Key{domain_name, device, layer, e, role});
                if (it == engines_.end() || it->second.engine == nullptr)
                    return false;
            }
        }

        return true;
    }

    std::vector<int> ExpertGemmRegistry::completeExpertsForLayer(DeviceId device, int layer, int num_experts) const
    {
        return completeExpertsForLayerInDomain({}, device, layer, num_experts);
    }

    std::vector<int> ExpertGemmRegistry::completeExpertsForLayerInDomain(const std::string &domain_name,
                                                                         DeviceId device, int layer, int num_experts) const
    {
        std::vector<int> experts;
        if (num_experts < 0)
            return experts;

        std::shared_lock lock(mutex_);
        for (int expert = 0; expert < num_experts; ++expert)
        {
            bool complete = true;
            for (WeightRole role : {WeightRole::GATE, WeightRole::UP, WeightRole::DOWN})
            {
                auto it = engines_.find(Key{domain_name, device, layer, expert, role});
                if (it == engines_.end() || it->second.engine == nullptr)
                {
                    complete = false;
                    break;
                }
            }
            if (complete)
                experts.push_back(expert);
        }
        return experts;
    }

    size_t ExpertGemmRegistry::countCompleteExpertsForLayer(DeviceId device, int layer, int num_experts) const
    {
        return completeExpertsForLayer(device, layer, num_experts).size();
    }

    size_t ExpertGemmRegistry::countCompleteExpertsForLayerInDomain(const std::string &domain_name,
                                                                    DeviceId device, int layer, int num_experts) const
    {
        return completeExpertsForLayerInDomain(domain_name, device, layer, num_experts).size();
    }

    size_t ExpertGemmRegistry::countEnginesForDevice(DeviceId device) const
    {
        return countEnginesForDeviceInDomain({}, device);
    }

    size_t ExpertGemmRegistry::countEnginesForDeviceInDomain(const std::string &domain_name, DeviceId device) const
    {
        std::shared_lock lock(mutex_);
        size_t count = 0;
        for (const auto &[key, entry] : engines_)
        {
            if (key.domain_name == domain_name && key.device == device && entry.engine != nullptr)
                ++count;
        }
        return count;
    }

    size_t ExpertGemmRegistry::countEnginesForLayer(DeviceId device, int layer) const
    {
        return countEnginesForLayerInDomain({}, device, layer);
    }

    size_t ExpertGemmRegistry::countEnginesForLayerInDomain(const std::string &domain_name, DeviceId device, int layer) const
    {
        std::shared_lock lock(mutex_);
        size_t count = 0;
        for (const auto &[key, entry] : engines_)
        {
            if (key.domain_name == domain_name && key.device == device && key.layer == layer && entry.engine != nullptr)
                ++count;
        }
        return count;
    }

    bool ExpertGemmRegistry::populateExpertEngines(DeviceId device, int layer, int num_experts,
                                                   std::vector<ITensorGemm *> &gate_out,
                                                   std::vector<ITensorGemm *> &up_out,
                                                   std::vector<ITensorGemm *> &down_out) const
    {
        return populateExpertEnginesForDomain({}, device, layer, num_experts,
                                              gate_out, up_out, down_out);
    }

    bool ExpertGemmRegistry::populateExpertEnginesForDomain(const std::string &domain_name,
                                                            DeviceId device, int layer, int num_experts,
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

        gate_out.assign(static_cast<size_t>(num_experts), nullptr);
        up_out.assign(static_cast<size_t>(num_experts), nullptr);
        down_out.assign(static_cast<size_t>(num_experts), nullptr);

        std::shared_lock lock(mutex_);
        bool complete = true;

        for (int e = 0; e < num_experts; ++e)
        {
            auto it_gate = engines_.find(Key{domain_name, device, layer, e, WeightRole::GATE});
            if (it_gate != engines_.end())
                gate_out[e] = it_gate->second.engine;

            auto it_up = engines_.find(Key{domain_name, device, layer, e, WeightRole::UP});
            if (it_up != engines_.end())
                up_out[e] = it_up->second.engine;

            auto it_down = engines_.find(Key{domain_name, device, layer, e, WeightRole::DOWN});
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
        replaceEngineForDomain({}, device, layer, expert, role, engine, std::move(ownership));
    }

    void ExpertGemmRegistry::replaceEngineForDomain(const std::string &domain_name,
                                                    DeviceId device, int layer, int expert, WeightRole role,
                                                    ITensorGemm *engine, std::shared_ptr<ITensorGemm> ownership)
    {
        std::unique_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        engines_[key] = Entry{engine, std::move(ownership)};
    }

    bool ExpertGemmRegistry::removeEngine(DeviceId device, int layer, int expert, WeightRole role)
    {
        return removeEngineForDomain({}, device, layer, expert, role);
    }

    bool ExpertGemmRegistry::removeEngineForDomain(const std::string &domain_name,
                                                   DeviceId device, int layer, int expert, WeightRole role)
    {
        std::unique_lock lock(mutex_);
        Key key{domain_name, device, layer, expert, role};
        return engines_.erase(key) > 0;
    }

    size_t ExpertGemmRegistry::size() const
    {
        std::shared_lock lock(mutex_);
        return engines_.size();
    }

    bool ExpertGemmRegistry::hasEnginesForLayer(DeviceId device, int layer) const
    {
        return hasEnginesForLayerInDomain({}, device, layer);
    }

    bool ExpertGemmRegistry::hasEnginesForLayerInDomain(const std::string &domain_name, DeviceId device, int layer) const
    {
        std::shared_lock lock(mutex_);
        for (const auto &[key, entry] : engines_)
        {
            if (key.domain_name == domain_name && key.device == device && key.layer == layer)
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
