/**
 * @file VerifierProjectionBundle.cpp
 * @brief Authenticate original fused verifier bindings and worker-local nesting.
 *
 * Storage identity distinguishes a complete operation from copied/reordered
 * policy subgroups. Backend admission separately validates prepared source
 * provenance, operand publications, epilogues and measured arithmetic policy.
 */
#include "VerifierProjectionBundle.h"
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
VerifierProjectionBundleBinding::VerifierProjectionBundleBinding(
    VerifierProjectionBundleKind kind, DeviceId device, const TensorBase *input,
    DeviceRowRange rows, int reduction, Projections projections) noexcept
    : kind_(kind), device_(device), input_(input), rows_(rows),
      reduction_(reduction), projections_(projections)
{}

VerifierProjectionBundleBinding VerifierProjectionBundleBinding::bind(
    VerifierProjectionBundleKind kind, DeviceId device, const TensorBase *input,
    DeviceRowRange rows, int reduction, Projections projections)
{
    const std::size_t members = kind == VerifierProjectionBundleKind::AttentionQKV ? 3 :
        (kind == VerifierProjectionBundleKind::AttentionKV || kind == VerifierProjectionBundleKind::GateUp) ? 2 : 0;
    if (!members || (!device.is_cpu() && !device.is_gpu()) || !input ||
        rows.physicalRows() < 1 || reduction < 1 || projections.size() != members)
        throw std::invalid_argument("Verifier projection bundle requires its complete original operation");
    for (std::size_t i = 0; i < members; ++i)
    {
        const auto &projection = projections[i];
        if (!projection.kernel || !projection.output || projection.n < 1)
            throw std::invalid_argument("Verifier projection bundle has incomplete or inconsistent members");
        // Distinct semantic outputs are mandatory. Immutable weight sharing is
        // legal; native admission identifies each member by its output as well.
        for (std::size_t earlier = 0; earlier < i; ++earlier)
            if (projections[earlier].output == projection.output)
                throw std::invalid_argument("Verifier projection bundle aliases semantic outputs");
    }
    return {kind, device, input, rows, reduction, projections};
}

bool VerifierProjectionBundleBinding::matches(DeviceId device, const TensorBase *input,
    const DeviceRowRange &rows, int reduction, Projections projections) const noexcept
{
    return device_ == device && input_ == input && reduction_ == reduction &&
        rows_.captureIdentity() == rows.captureIdentity() &&
        projections_.data() == projections.data() && projections_.size() == projections.size();
}

thread_local const ScopedVerifierProjectionBundle *ScopedVerifierProjectionBundle::active_ = nullptr;

ScopedVerifierProjectionBundle::ScopedVerifierProjectionBundle(
    std::optional<VerifierProjectionBundleBinding> binding) noexcept
    : binding_(std::move(binding)), previous_(active_)
{
    active_ = this;
}

ScopedVerifierProjectionBundle::~ScopedVerifierProjectionBundle()
{
    if (active_ != this) std::terminate();
    active_ = previous_;
}

const VerifierProjectionBundleBinding *ScopedVerifierProjectionBundle::current() noexcept
{
    return active_ && active_->binding_ ? &*active_->binding_ : nullptr;
}
}
