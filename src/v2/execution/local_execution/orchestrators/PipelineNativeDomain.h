/**
 * @file PipelineNativeDomain.h
 * @brief Frozen participant identity shared by every captured pipeline edge.
 *
 * A domain is either one GPU or one native homogeneous TP group. This value
 * records capture identity, not a second topology authority: the context owns
 * membership, and any later disagreement is rejected before graph mutation or
 * submission. In particular, a source TP nonleader must be validated even
 * though it does not itself send the inter-domain activation.
 */
#pragma once
#include "collective/ILocalTPContext.h"
#include <set>
#include <stdexcept>

namespace llaminar2
{
/** @brief Immutable binding between a physical GPU and its native domain role. */
class PipelineNativeDomain final
{
public:
    /** @brief Freeze exact membership without allocating device resources.
     * @param device Local physical GPU, not the domain's ordinal.
     * @param context Canonical native communicator, or null for one GPU.
     * @param participant Index in the ordered communicator; zero for one GPU.
     * @throws std::invalid_argument for a foreign member or non-native group. */
    PipelineNativeDomain(DeviceId device, ILocalTPContext *context, int participant)
        : device_(device), context_(context), participant_(participant),
          members_(context ? context->devices() : std::vector<GlobalDeviceAddress>{})
    { validate(); }

    /** @brief Reject any membership or native-backend change before submission.
     * @throws std::logic_error for changed ordered membership.
     * @throws std::invalid_argument for invalid native transport or member identity. */
    void validate() const
    {
        if (!device_.is_gpu() || (!context_ && participant_ != 0))
            throw std::invalid_argument("Pipeline domain requires an exact GPU member");
        if (!context_) return;
        if (context_->devices() != members_)
            throw std::logic_error("Pipeline native domain membership changed after binding");
        const auto backend = device_.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        if (context_->degree() < 2 || participant_ < 0 || participant_ >= context_->degree() ||
            members_.size() != size_t(context_->degree()) || context_->backend() != backend ||
            members_[participant_].toLocalDeviceId() != device_)
            throw std::invalid_argument("Pipeline requires exact native GPU domain membership");
        std::set<DeviceId> unique;
        for (const auto &address : members_)
            if (address.toLocalDeviceId().type != device_.type ||
                !unique.insert(address.toLocalDeviceId()).second)
                throw std::invalid_argument("Pipeline native domain requires distinct homogeneous GPUs");
    }

    /** @return Frozen domain leader, which need not be physical device zero. */
    DeviceId leader() const noexcept
    { return context_ ? members_.front().toLocalDeviceId() : device_; }

private:
    const DeviceId device_;
    ILocalTPContext *const context_;
    const int participant_;
    const std::vector<GlobalDeviceAddress> members_;
};
}
