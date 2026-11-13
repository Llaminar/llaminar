/**
 * @file DeviceGenerationExecutionPolicy.h
 * @brief Immutable generation execution declaration shared by admission and capture.
 *
 * Declare the complete parent's execution topology before allocating its retained
 * graph family. CUDA native domains require conditional graphs; ROCm and explicit
 * mixed/remote boundaries use authenticated hosted submission. Runtime capture
 * validates the declaration and fails on a missing capability; it cannot change
 * the admitted family. This header owns setup metadata, never live device state.
 */
#pragma once
#include "backends/DeviceType.h"
#include "execution/mtp/MTPVerifierGraphFamily.h"
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
/** @brief Native control topology required by the complete generation parent. */
enum class DeviceGenerationLoopTopology : uint8_t
{
    FixedDepth = 0, ///< One immutable transaction body repeated to completion.
    DynamicDepth,   ///< Device-selected transaction body repeated to completion.
};

/**
 * @brief Sole execution policy for a complete generation loop.
 *
 * Hosted submission selects retained complete transactions with a fixed-size
 * authenticated ticket. Model, sampler, verifier and response state stay on
 * device. Unsupported is a fatal incomplete implementation, never a fallback.
 */
enum class DeviceGenerationExecutionPolicy : uint8_t
{
    NativeConditionalGraph = 0,
    HostScheduledCapturedTransactions,
    Unsupported,
};

/** @brief Frozen graph-API boundary, independent of the current request depth. */
enum class DeviceGenerationExecutionDomain : uint8_t
{
    NativeDomain = 0,
    HostedBoundary,
};

/**
 * @brief Compose immutable participant types and explicit remote boundaries.
 *
 * Planning and graph setup supply their existing authored membership. This
 * accumulator owns no devices, graph handles, allocation or live controller state.
 */
class DeviceGenerationDomainDeclaration final
{
public:
    /**
     * @brief Include one actual participant's backend in the complete domain.
     * @param type Authored participant backend; mixed types seal a hosted boundary.
     */
    constexpr void observe(DeviceType type) noexcept
    {
        if (observed_ && first_type_ != type) requireHostedBoundary();
        first_type_ = observed_ ? first_type_ : type;
        observed_ = true;
    }
    /** @brief Include an authored boundary that one native parent cannot cross. */
    constexpr void requireHostedBoundary() noexcept
    {
        domain_ = DeviceGenerationExecutionDomain::HostedBoundary;
    }
    /** @return The complete immutable domain declaration, with no runtime probes. */
    [[nodiscard]] constexpr DeviceGenerationExecutionDomain domain() const noexcept
    {
        return domain_;
    }
private:
    DeviceType first_type_ = DeviceType::CPU;
    bool observed_ = false;
    DeviceGenerationExecutionDomain domain_ = DeviceGenerationExecutionDomain::NativeDomain;
};

/**
 * @brief Declare the supported parent architecture before native graph creation.
 * @param type Actual participant backend.
 * @param domain Frozen complete-parent boundary, including remote expert owners.
 * @return Required policy, or Unsupported for an unknown domain/backend.
 *
 * CUDA's implementation must prove conditional support when it materializes.
 * HIP deliberately retains complete hosted transactions. A future native HIP
 * implementation must change this declaration and admission together.
 */
[[nodiscard]] constexpr DeviceGenerationExecutionPolicy declaredDeviceGenerationExecutionPolicy(
    DeviceType type, DeviceGenerationExecutionDomain domain) noexcept
{
    if (type != DeviceType::CUDA && type != DeviceType::ROCm)
        return DeviceGenerationExecutionPolicy::Unsupported;
    switch (domain)
    {
    case DeviceGenerationExecutionDomain::HostedBoundary:
        return DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    case DeviceGenerationExecutionDomain::NativeDomain:
        return type == DeviceType::CUDA
            ? DeviceGenerationExecutionPolicy::NativeConditionalGraph
            : DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    }
    return DeviceGenerationExecutionPolicy::Unsupported;
}

/**
 * @brief Project a supported parent declaration into its verifier ownership.
 * @param policy Immutable supported execution policy, not an observed capability.
 * @return Exact hosted or bounded native role for the shared family authority.
 * @throws std::invalid_argument for unsupported/unknown policies.
 */
[[nodiscard]] constexpr MTPVerifierExecutionRole mtpVerifierExecutionRole(
    DeviceGenerationExecutionPolicy policy)
{
    switch (policy)
    {
    case DeviceGenerationExecutionPolicy::NativeConditionalGraph:
        return MTPVerifierExecutionRole::NativeConditionalParent;
    case DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions:
        return MTPVerifierExecutionRole::HostedCapturedTransactions;
    case DeviceGenerationExecutionPolicy::Unsupported:
        break;
    }
    throw std::invalid_argument("MTP graph admission requires a declared supported generation policy");
}
} // namespace llaminar2
