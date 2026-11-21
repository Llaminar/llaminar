/**
 * @file ROCmNativeQuantisedWmmaLaunch.h
 * @brief Typed submission boundary between ordered dispatch and wave64 WMMA.
 *
 * The existing dispatcher owns source arithmetic policy, native target
 * admission and launch publication. This boundary submits that exact policy
 * through a separately compiled wave64 implementation and returns its actual
 * function and geometry for resource inspection. It owns no temporary storage.
 */
#pragma once
#include "ROCmNativeQuantisedWmmaPolicy.h"
#include "ROCmVNNIPrefillLaunch.h"
#include <cstdint>

namespace llaminar2::rocm
{
/** @brief Borrowed persistent operands and the admitted serial arithmetic policy. */
struct NativeQuantisedWmmaArguments
{
    const int8_t *activations;       ///< Row-major signed bytes [M,K].
    const uint8_t *payload;          ///< Block-major compact source payload.
    const uint16_t *scales;          ///< Primary scale words [K/32,N].
    const uint16_t *auxiliary;       ///< Optional secondary scale/correction words.
    const uint32_t *minima_pairs;    ///< Optional Q2_K half-block minima pairs.
    float *output;                   ///< Unique-writer result [M,N].
    const float *row_scales;         ///< Whole-row scales when block_scales is null.
    const float *block_scales;       ///< Optional activation scales [M,K/32].
    int m;                          ///< Positive live row count.
    int n;                          ///< Positive live column count.
    int k;                          ///< Positive contraction width, divisible by 32.
    int serial_partitions;          ///< Admitted serial-M1 partition count.
    uint8_t codebook;                ///< Physical payload and metadata decoder.
};

/**
 * @brief Submit one admitted gfx1100 projection on its exact non-default stream.
 * @param tile Explicit compiled output geometry.
 * @param args Persistent operands and canonical serial arithmetic policy.
 * @param stream Exact producer HIP stream, never null or a default-stream sentinel.
 * @return Actual function identity and launch geometry for dispatcher publication.
 * @throws std::invalid_argument If required operands or geometry are incomplete.
 */
[[nodiscard]] VNNIPrefillLaunch submitNativeQuantisedWmma(
    NativeQuantisedWmmaTile tile, const NativeQuantisedWmmaArguments &args, void *stream);

/**
 * @brief Publish this wave64 translation unit's constant grids during setup.
 * @param device_id Actual device receiving the immutable tables.
 * @param iq3s IQ3_S host table, 512 uint32 entries.
 * @param iq3xxs IQ3_XXS host table, 256 uint32 entries.
 * @param iq2s IQ2_S host table, 1024 uint64 entries.
 * @param iq2xs IQ2_XS host table, 512 uint64 entries.
 * @param iq2xxs IQ2_XXS host table, 256 uint64 entries.
 * @param iq1s Shared IQ1 host table, 2048 uint64 entries.
 * @return True after every table has been published before graph construction.
 */
bool initializeNativeQuantisedWmmaGrids(int device_id, const void *iq3s,
    const void *iq3xxs, const void *iq2s, const void *iq2xs,
    const void *iq2xxs, const void *iq1s);
}
