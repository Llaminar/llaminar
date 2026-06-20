#include "WeightLifecycleTrace.h"

#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"

#include <map>
#include <sstream>

namespace llaminar2
{
    WeightLifecycleTrace &WeightLifecycleTrace::instance()
    {
        static WeightLifecycleTrace trace;
        return trace;
    }

    bool WeightLifecycleTrace::enabled()
    {
        return debugEnv().weight_lifecycle_trace;
    }

    void WeightLifecycleTrace::setMode(WeightInferenceMode mode)
    {
        if (!enabled())
            return;
        auto &trace = instance();
        std::lock_guard<std::mutex> lock(trace.mutex_);
        trace.mode_ = mode;
    }

    WeightInferenceMode WeightLifecycleTrace::mode()
    {
        auto &trace = instance();
        std::lock_guard<std::mutex> lock(trace.mutex_);
        return trace.mode_;
    }

    void WeightLifecycleTrace::record(WeightLifecycleEvent event)
    {
        if (!enabled())
            return;

        auto &trace = instance();
        std::lock_guard<std::mutex> lock(trace.mutex_);
        if (event.mode == WeightInferenceMode::Unknown)
            event.mode = trace.mode_;
        trace.events_.push_back(event);

        LOG_DEBUG("[WeightLifecycle] " << toString(event.type)
                                        << " mode=" << toString(event.mode)
                                        << " name=" << event.canonical_name
                                        << " role=" << toString(event.role)
                                        << " layer=" << event.layer
                                        << " device=" << event.device.to_string()
                                        << (event.detail.empty() ? "" : " detail=" + event.detail));
    }

    void WeightLifecycleTrace::record(
        WeightLifecycleEventType type,
        const std::string &canonical_name,
        WeightRole role,
        int layer,
        DeviceId device,
        std::string detail)
    {
        WeightLifecycleEvent event;
        event.type = type;
        event.canonical_name = canonical_name;
        event.role = role == WeightRole::Other ? inferWeightRole(canonical_name) : role;
        event.layer = layer >= 0 ? layer : inferWeightLayer(canonical_name);
        event.device = device;
        event.detail = std::move(detail);
        record(std::move(event));
    }

    std::vector<WeightLifecycleEvent> WeightLifecycleTrace::snapshot()
    {
        auto &trace = instance();
        std::lock_guard<std::mutex> lock(trace.mutex_);
        return trace.events_;
    }

    void WeightLifecycleTrace::clear()
    {
        auto &trace = instance();
        std::lock_guard<std::mutex> lock(trace.mutex_);
        trace.events_.clear();
        trace.mode_ = WeightInferenceMode::Unknown;
    }

    void WeightLifecycleTrace::logSummary(const std::string &reason)
    {
        if (!enabled())
            return;

        auto events = snapshot();
        std::map<WeightLifecycleEventType, size_t> by_type;
        std::map<WeightRole, size_t> by_role;
        std::map<std::string, size_t> by_device;
        for (const auto &event : events)
        {
            by_type[event.type]++;
            by_role[event.role]++;
            by_device[event.device.to_string()]++;
        }

        std::ostringstream out;
        out << "[WeightLifecycle] Summary";
        if (!reason.empty())
            out << " (" << reason << ")";
        out << ": events=" << events.size();

        out << " types=";
        bool first = true;
        for (const auto &[type, count] : by_type)
        {
            if (!first) out << ",";
            first = false;
            out << toString(type) << ":" << count;
        }

        out << " roles=";
        first = true;
        for (const auto &[role, count] : by_role)
        {
            if (!first) out << ",";
            first = false;
            out << toString(role) << ":" << count;
        }

        out << " devices=";
        first = true;
        for (const auto &[device, count] : by_device)
        {
            if (!first) out << ",";
            first = false;
            out << device << ":" << count;
        }

        LOG_DEBUG(out.str());
    }

    std::string toString(WeightLifecycleEventType type)
    {
        switch (type)
        {
        case WeightLifecycleEventType::SourceLoad: return "SourceLoad";
        case WeightLifecycleEventType::Slice: return "Slice";
        case WeightLifecycleEventType::Clone: return "Clone";
        case WeightLifecycleEventType::Preload: return "Preload";
        case WeightLifecycleEventType::Prepare: return "Prepare";
        case WeightLifecycleEventType::RegisterPrepared: return "RegisterPrepared";
        case WeightLifecycleEventType::HostRelease: return "HostRelease";
        case WeightLifecycleEventType::GraphBind: return "GraphBind";
        case WeightLifecycleEventType::GetWeightForDevice: return "GetWeightForDevice";
        }
        return "SourceLoad";
    }

    std::string toString(WeightInferenceMode mode)
    {
        switch (mode)
        {
        case WeightInferenceMode::Unknown: return "Unknown";
        case WeightInferenceMode::SingleDevice: return "SingleDevice";
        case WeightInferenceMode::LocalTP: return "LocalTP";
        case WeightInferenceMode::NodeLocalTP: return "NodeLocalTP";
        case WeightInferenceMode::GlobalTP: return "GlobalTP";
        case WeightInferenceMode::LocalPP: return "LocalPP";
        case WeightInferenceMode::HybridPPTP: return "HybridPPTP";
        }
        return "Unknown";
    }
}
