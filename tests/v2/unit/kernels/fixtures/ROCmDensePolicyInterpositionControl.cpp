/**
 * @file ROCmDensePolicyInterpositionControl.cpp
 * @brief Positive control reproducing the former preemptible generated-table ABI.
 *
 * This deliberately conflicting table exists only in a device-free test DSO.
 * The executable exports the same GNU-unique symbol with different rows. The
 * control must observe that executable's address, proving the loader actually
 * exercises the interposition that production policy now rejects structurally.
 */
#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlayTypes.h"

namespace llaminar2::rocm::generated
{
inline constexpr ROCmDensePrefillOverlayEntry kROCmDensePrefillOverlayEntries[] = {
    {0,23,23,32,DensePrefillWeightFamily::NativeQuantised,
     {64,16,1,2,false,VNNIPrefillProducer::NativeCooperative}}};
}

/** @return The former public-table address, intentionally subject to interposition. */
extern "C" const void *rocmDensePolicyInterpositionControl() noexcept
{ return llaminar2::rocm::generated::kROCmDensePrefillOverlayEntries; }
