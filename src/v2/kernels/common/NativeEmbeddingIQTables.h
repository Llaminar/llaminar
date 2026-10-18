/**
 * @file NativeEmbeddingIQTables.h
 * @brief Non-owning bindings to backend-owned immutable source IQ codebooks.
 *
 * Preparation authenticates the backend runtime generation and publishes these
 * addresses before capture. Embedding kernels retain only this immutable view;
 * they never allocate, initialize, download or replace codebooks during replay.
 */
#pragma once
#include "tensors/TensorType.h"
#include <cstdint>

namespace llaminar2
{
    /** @brief Source codebook addresses owned by the native backend generation. */
    struct NativeEmbeddingIQTables
    {
        const std::uint32_t *iq3s = nullptr;   ///< 512 four-byte magnitude grids.
        const std::uint32_t *iq3xxs = nullptr; ///< 256 four-byte magnitude grids.
        const std::uint64_t *iq2s = nullptr;   ///< 1024 eight-byte magnitude grids.
        const std::uint64_t *iq2xs = nullptr;  ///< 512 eight-byte magnitude grids.
        const std::uint64_t *iq2xxs = nullptr; ///< 256 eight-byte magnitude grids.
        const std::uint64_t *iq1s = nullptr;   ///< 2048 signed eight-byte grids.
    };

    /**
     * @brief Bind the existing CUDA source grids before graph recording.
     * @param ordinal Exact CUDA device owning the prepared embedding.
     * @param source_type Original tensor type; non-grid formats need no tables.
     * @param tables Destination immutable view, reset for non-grid formats.
     * @return True after the required grids are published and addresses bound.
     */
    bool prepareNativeEmbeddingTablesCUDA(
        int ordinal, TensorType source_type, NativeEmbeddingIQTables &tables);

    /**
     * @brief Bind the existing ROCm source grids before graph recording.
     * @param ordinal Exact ROCm device owning the prepared embedding.
     * @param source_type Original tensor type; non-grid formats need no tables.
     * @param tables Destination immutable view, reset for non-grid formats.
     * @return True after the required grids are published and addresses bound.
     */
    bool prepareNativeEmbeddingTablesROCm(
        int ordinal, TensorType source_type, NativeEmbeddingIQTables &tables);
}
