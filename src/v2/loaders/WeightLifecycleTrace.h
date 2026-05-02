#pragma once

#include "WeightIdentity.h"

#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{
    enum class WeightLifecycleEventType
    {
        SourceLoad,
        Slice,
        Clone,
        Preload,
        Prepare,
        RegisterPrepared,
        HostRelease,
        GraphBind,
        GetWeightForDevice,
    };

    enum class WeightInferenceMode
    {
        Unknown,
        SingleDevice,
        LocalTP,
        NodeLocalTP,
        GlobalTP,
        LocalPP,
        HybridPPTP,
    };

    struct WeightLifecycleEvent
    {
        WeightLifecycleEventType type = WeightLifecycleEventType::SourceLoad;
        WeightInferenceMode mode = WeightInferenceMode::Unknown;
        std::string canonical_name;
        WeightRole role = WeightRole::Other;
        WeightDerivationKind derivation = WeightDerivationKind::Source;
        int layer = -1;
        DeviceId device = DeviceId::cpu();
        std::string detail;
    };

    class WeightLifecycleTrace
    {
    public:
        static WeightLifecycleTrace &instance();
        static bool enabled();
        static void setMode(WeightInferenceMode mode);
        static WeightInferenceMode mode();
        static void record(WeightLifecycleEvent event);
        static void record(
            WeightLifecycleEventType type,
            const std::string &canonical_name,
            WeightRole role = WeightRole::Other,
            int layer = -1,
            DeviceId device = DeviceId::cpu(),
            std::string detail = {});
        static std::vector<WeightLifecycleEvent> snapshot();
        static void clear();
        static void logSummary(const std::string &reason = {});

    private:
        mutable std::mutex mutex_;
        WeightInferenceMode mode_ = WeightInferenceMode::Unknown;
        std::vector<WeightLifecycleEvent> events_;
    };

    std::string toString(WeightLifecycleEventType type);
    std::string toString(WeightInferenceMode mode);
}
