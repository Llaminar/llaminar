/**
 * @file ROCmCompiledWaveContract.h
 * @brief Typed host/device agreement for the installed ordered HIP reduction family.
 *
 * Native hardware preference may be wave32. These kernels require a separately
 * compiled wave64 program, whose arithmetic order and lane ownership are fixed.
 * This contract does not describe future matrix kernels or their launch modes.
 */
#pragma once
namespace llaminar2 {
    /** @brief Only the currently implemented and certified ordered wave policy. */
    enum class ROCmCompiledWaveContract : unsigned { OrderedWave64 = 64 };
    /** @brief Convert the installed program contract to a launch dimension. */
    inline constexpr unsigned rocmOrderedWaveWidth =
        static_cast<unsigned>(ROCmCompiledWaveContract::OrderedWave64);
}
