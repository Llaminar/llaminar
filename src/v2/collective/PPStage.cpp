/**
 * @file PPStage.cpp
 * @brief Implementation of hierarchical pipeline parallel stage
 */

#include "PPStage.h"
#include "ILocalTPContext.h"
#include "ILocalPPContext.h"
#include "IGlobalTPContext.h"

namespace llaminar2
{

    GlobalDeviceAddress PPStage::representativeDevice() const
    {
        switch (type_)
        {
        case PPStageType::SINGLE_DEVICE:
            return device_;

        case PPStageType::TP_DOMAIN:
            if (tp_context_ && tp_context_->degree() > 0)
            {
                return tp_context_->deviceAt(0);
            }
            throw std::runtime_error("PPStage: TP domain has no devices");

        case PPStageType::NESTED_PP:
            if (pp_context_ && pp_context_->numStages() > 0)
            {
                // For nested PP, use first stage's device as representative
                return pp_context_->deviceForStage(0);
            }
            throw std::runtime_error("PPStage: Nested PP has no stages");

        case PPStageType::GLOBAL_TP_DOMAIN:
            if (global_tp_context_)
            {
                return global_tp_context_->localDevice();
            }
            throw std::runtime_error("PPStage: Global TP domain has no context");
        }

        throw std::logic_error("PPStage: Unknown stage type");
    }

    std::vector<GlobalDeviceAddress> PPStage::allDevices() const
    {
        switch (type_)
        {
        case PPStageType::SINGLE_DEVICE:
            return {device_};

        case PPStageType::TP_DOMAIN:
            if (tp_context_)
            {
                return tp_context_->devices();
            }
            return {};

        case PPStageType::NESTED_PP:
            if (pp_context_)
            {
                return pp_context_->stageDevices();
            }
            return {};

        case PPStageType::GLOBAL_TP_DOMAIN:
            if (global_tp_context_)
            {
                // Return only local rank's device - PP transfers are per-rank
                return {global_tp_context_->localDevice()};
            }
            return {};
        }

        return {};
    }

    int PPStage::deviceCount() const
    {
        switch (type_)
        {
        case PPStageType::SINGLE_DEVICE:
            return 1;

        case PPStageType::TP_DOMAIN:
            return tp_context_ ? tp_context_->degree() : 0;

        case PPStageType::NESTED_PP:
            return pp_context_ ? pp_context_->numStages() : 0;

        case PPStageType::GLOBAL_TP_DOMAIN:
            // Local rank has 1 device in global TP domain for PP transfers
            return global_tp_context_ ? 1 : 0;
        }

        return 0;
    }

    bool PPStage::containsDevice(const GlobalDeviceAddress& device) const
    {
        switch (type_)
        {
        case PPStageType::SINGLE_DEVICE:
            return device_ == device;

        case PPStageType::TP_DOMAIN:
            if (tp_context_)
            {
                return tp_context_->indexForDevice(device) >= 0;
            }
            return false;

        case PPStageType::NESTED_PP:
            if (pp_context_)
            {
                for (const auto& stage_dev : pp_context_->stageDevices())
                {
                    if (stage_dev == device) return true;
                }
            }
            return false;

        case PPStageType::GLOBAL_TP_DOMAIN:
            if (global_tp_context_)
            {
                return global_tp_context_->localDevice() == device;
            }
            return false;
        }

        return false;
    }

    std::string PPStage::describe() const
    {
        switch (type_)
        {
        case PPStageType::SINGLE_DEVICE:
            return device_.toShortString();

        case PPStageType::TP_DOMAIN:
        {
            std::string result = "TP(";
            if (tp_context_)
            {
                const auto& devs = tp_context_->devices();
                for (size_t i = 0; i < devs.size(); ++i)
                {
                    if (i > 0) result += ", ";
                    result += devs[i].toShortString();
                }
            }
            result += ")";
            return result;
        }

        case PPStageType::NESTED_PP:
        {
            std::string result = "PP(";
            if (pp_context_)
            {
                for (int s = 0; s < pp_context_->numStages(); ++s)
                {
                    if (s > 0) result += ", ";
                    result += pp_context_->deviceForStage(s).toShortString();
                }
            }
            result += ")";
            return result;
        }

        case PPStageType::GLOBAL_TP_DOMAIN:
            if (global_tp_context_)
            {
                return "GlobalTP(degree=" + std::to_string(global_tp_context_->degree()) +
                       ", local=" + global_tp_context_->localDevice().toString() + ")";
            }
            return "GlobalTP(no context)";
        }

        return "???";
    }

} // namespace llaminar2
