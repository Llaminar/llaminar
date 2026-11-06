/**
 * @file W4A4DeviceViews.h
 * @brief Backend-neutral launch ABI for native W4A4/H device arithmetic.
 *
 * These trivially copyable views borrow source, arena and immutable policy
 * addresses. They are not storage owners, source-admission certificates or
 * readiness publications. Cold production binding must supply those authorities
 * and reject invalid geometry, overlap, lifetime or policy before creating a
 * launch. Sharing this C++20-safe ABI lets CUDA, HIP and host binding name the
 * same operands without importing device kernels or C++23 source-admission code.
 */
#pragma once

#include "tensors/W4A4ActivationFormat.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2::w4a4::device
{
    /** @brief Borrowed projection after cold source/operand/arena admission. */
    struct ProjectionView
    {
        const SourceBlock *weights;         ///< Complete original-K rows in the retained GPU pool.
        const ActivationBlock *activations; ///< Producer-owned complete rows, without K padding.
        const std::uint8_t *active_rows;    ///< Sealed validity of original physical activation rows.
        float *output;                     ///< Disjoint admitted FP32 destination.
        const float *bias;                 ///< Optional immutable per-output-column bias.
        std::size_t output_stride;         ///< Physical FP32 row stride, at least columns.
        int rows;                          ///< Physical M, not a downloaded live count.
        int columns;                       ///< Selected output N, not the reduction width.
        int blocks_per_row;                ///< Complete original K/256 with admitted int32 bound.
        float alpha;                       ///< Finite authored post-projection multiplier.
        float beta;                        ///< Finite old-output multiplier; zero forbids a read.
    };

    /** @brief Borrowed whole-token packing operands after transform completion. */
    struct PreparationView
    {
        const float *values;                ///< Original W4A4 or completely transformed W4A4H rows.
        std::size_t input_stride;           ///< Physical FP32 stride, at least complete original K.
        const std::uint8_t *active_rows;    ///< Sealed invocation validity, never mutable request counts.
        ActivationBlock *blocks;           ///< Exact row-major A4 destination in the arena.
        int rows;                          ///< Admitted physical M.
        int columns;                       ///< Complete original K, a multiple of256.
        int rotation_length;               ///< Source-owned original transform width retained in A4.
        float clip;                        ///< Exact resolved source clipping scalar.
    };

    /** @brief Dynamic native shared storage, not a global allocation or second ledger. */
    struct GroupPreparation
    {
        float lower, upper;          ///< Zero-inclusive clipped extrema.
        float inverse;               ///< Rounded reciprocal of multiplier times whole-token unit.
        std::uint32_t quantizer;     ///< Multiplier in low8 bits, zero point in bits8..11.
    };

    /** @brief Immutable transform policy and disjoint admitted invocation banks. */
    struct HadamardView
    {
        const float *input;                ///< Original FP32 rows before the online transform.
        std::size_t input_stride;          ///< Physical FP32 stride, at least complete original K.
        const std::uint8_t *active_rows;    ///< Sealed original-row validity.
        float *butterfly;                  ///< Row-major original-K Sylvester arena bank.
        float *transformed;                ///< Disjoint row-major original-K final transform bank.
        std::uint64_t residues_low;        ///< Immutable residue bits0..63; factorization admits p<=128.
        std::uint64_t residues_high;       ///< Immutable residue bits64..127, not a device allocation.
        int rows;                          ///< Admitted physical M.
        int columns;                       ///< Complete original K, divisible by rotation_length.
        int rotation_length;               ///< Source-owned transform block length.
        int paley;                         ///< Admitted Paley order, or one for pure Sylvester.
        int sylvester;                     ///< Power-of-two width; paley*sylvester=rotation_length.
        float normalization;              ///< Pinned 1/sqrtf(rotation_length), prepared cold.
    };

    static_assert(sizeof(GroupPreparation) == 16);
    static_assert(std::is_trivially_copyable_v<ProjectionView> && std::is_standard_layout_v<ProjectionView>);
    static_assert(std::is_trivially_copyable_v<PreparationView> && std::is_standard_layout_v<PreparationView>);
    static_assert(std::is_trivially_copyable_v<HadamardView> && std::is_standard_layout_v<HadamardView>);
} // namespace llaminar2::w4a4::device
