/**
 * @file ROCmDownLastProducer.h
 * @brief Paired128-thread down/weighted-route/canonical-fold completion admission.
 *
 * Original descriptor source policy owns partition arithmetic. Named workspace
 * counters carry completion only; no partial sum uses an atomic or host mirror.
 */
#pragma once
#include "kernels/common/NativeTileCompletion.h"
namespace llaminar2::rocm
{
/**
 * @param columns Original H output width.
 * @param reduction Original participant-local F reduction width.
 * @param routes Original fixed selected-slot extent.
 * @param policy_mask Complete immutable original descriptor source-policy mask.
 * @return Exact two-level counter geometry using the original serial-M1 resolver.
 * @throws std::invalid_argument For an unsupported original policy/geometry.
 */
NativeDownCompletionPlan downLastProducerPlan(int columns,int reduction,int routes,std::uint32_t policy_mask);
/** @return Actual compiled paired128-thread kernel identity, with no device query. */
const void* downLastProducerKernelFunction() noexcept;
}
