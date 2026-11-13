/**
 * @file ROCmDenseProductionPrefillOverlayTypes.h
 * @brief Stable launch and entry schema without measured dense-prefill row data.
 *
 * Native HIP shards consume only this schema and the lookup declaration.
 * Measured row updates belong to the host lookup translation unit and explicit
 * inventory readers, so changing table data does not rebuild device kernels.
 */
#pragma once
#include <cstdint>
#include "kernels/common/NativeQuantisedArchitecture.h"
#include "kernels/rocm/gemm/ROCmVNNIPrefillLaunch.h"
#include "kernels/rocm/gemm/ROCmDensePrefillPreparedWeightFamily.h"

namespace llaminar2::rocm::generated
{
    /** @brief Immutable physical launch value; field order is the generated ABI. */
    struct ROCmDensePrefillOverlayConfig
    {
        int n_tile;
        int m_tile;
        int min_blocks;
        int unroll;
        bool full_tiles;
        VNNIPrefillProducer producer;
    };

    /** @brief Geometry/family/architecture identity and its measured launch. */
    struct ROCmDensePrefillOverlayEntry
    {
        uint8_t codebook;
        int m;
        int n;
        int k;
        DensePrefillWeightFamily family;
        ROCmDensePrefillOverlayConfig config;
        ::llaminar2::native_quantised::NativeQuantisedArchitecture architecture{};
    };
}
