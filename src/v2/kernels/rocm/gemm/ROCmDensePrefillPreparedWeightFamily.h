/**
 * @file ROCmDensePrefillPreparedWeightFamily.h
 * @brief Immutable prepared operand ABI in the dense-prefill exact key.
 *
 * Source codebook and arithmetic provenance do not identify physical storage:
 * signed Q8 may own either native per-block payload/scales or expanded INT8
 * VNNI weights with their separate scaling epilogue. Both existing producers
 * retain their own buffers and generic policy. No conversion or runtime probe
 * is performed to choose this value; the prepared consumer supplies it.
 */
#pragma once
#include <cstdint>
#include "ROCmVNNIPrefillLaunch.h"
namespace llaminar2::rocm
{
/** @brief Disjoint original prepared contracts, independent of source codebook. */
enum class DensePrefillWeightFamily : std::uint8_t
{
    NativeVNNI = 0, ///< Native K32 payload and per-block source metadata, including CB19.
    Int8Blockwise = 1, ///< Expanded INT8 VNNI payload plus its weight-scaling epilogue.
};
/**
 * @brief Validate an explicit prepared storage identity.
 * @param family Physical operand ABI supplied by the prepared consumer.
 * @return Whether this family belongs to the compiled contract.
 */
[[nodiscard]] constexpr bool densePrefillWeightFamilyValid(DensePrefillWeightFamily family) noexcept
{
    return family == DensePrefillWeightFamily::NativeVNNI ||
           family == DensePrefillWeightFamily::Int8Blockwise;
}
/**
 * @brief Validate a measured producer against the consumer's physical operand ABI.
 * @param family Explicit prepared storage contract supplied by the consumer.
 * @param producer Physical launch recorded by the measured configuration.
 * @return Whether the producer reads exactly this prepared storage contract.
 */
[[nodiscard]] constexpr bool densePrefillProducerMatchesFamily(
    DensePrefillWeightFamily family, VNNIPrefillProducer producer) noexcept
{
    switch (producer)
    {
    case VNNIPrefillProducer::NativeCooperative:
    case VNNIPrefillProducer::NativeStreaming:
        return family == DensePrefillWeightFamily::NativeVNNI;
    case VNNIPrefillProducer::Int8BlockwiseV3:
    case VNNIPrefillProducer::Int8BlockwiseV7:
        return family == DensePrefillWeightFamily::Int8Blockwise;
    }
    return false;
}
}
