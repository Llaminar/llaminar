/**
 * @file NEONRMSNormFloatingRows.h
 * @brief Original-storage floating RMSNorm tiles, with shared gamma and no decoded row bank.
 *
 * A tile consumes1..4 original rows in one native reduction/publication program.
 * Fixed reduction trees survive M changes; gamma loads are shared across rows.
 * Exact same-row aliasing is supported after reduction, while cross-row, partial
 * and immutable-gamma aliases reject before stores. Portable FP16 storage rules
 * apply to every lane, including bounded tails. This math layer owns no team,
 * allocation, workspace, tensor lease or physical-memory admission authority.
 */
#pragma once
#include "NEONRMSNormReduction.h"
#include "CPURMSNormBorrow.h"
#include <algorithm>
#include <cmath>

namespace llaminar2::primitives::neon_rmsnorm
{
/** @brief Explicit reduction program; T5 FP32 is not an approximation of FP64. */
enum class Accumulation { FP64, T5FP32 };
/** @brief Distinct staged FP64-mean and per-row FP32-mean inverse programs. */
enum class Inverse { GuardedFP64Mean, FP32Mean };

namespace detail
{
using rmsnorm_operands::overlaps;

/** @brief Publish one bounded four-lane result through its exact physical codec.
 * @tparam Encoding Original/result floating storage kind.
 * @param output Complete original writable row.
 * @param offset Original logical column of this fragment.
 * @param columns Complete logical count, not padded capacity.
 * @param value Four already normalized FP32 values.
 */
template<Storage Encoding>
[[gnu::always_inline]] inline void store(std::span<Unit<Encoding>> output, std::size_t offset,
                                        std::size_t columns, float32x4_t value)
{
    const auto remaining = columns - offset;
    if constexpr (Encoding == Storage::FP32)
    {
        if (remaining >= 4) vst1q_f32(output.data() + offset, value);
        else
        {
            if (remaining > 0) output[offset] = vgetq_lane_f32(value, 0);
            if (remaining > 1) output[offset + 1] = vgetq_lane_f32(value, 1);
            if (remaining > 2) output[offset + 2] = vgetq_lane_f32(value, 2);
        }
    }
    else
    {
        uint16x4_t words;
        if constexpr (Encoding == Storage::FP16) words = neon_fp16::narrow(value);
        else
        {
            const auto bits = vreinterpretq_u32_f32(value);
            const auto lsb = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
            words = vmovn_u32(vshrq_n_u32(vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7fff), lsb)), 16));
        }
        if (remaining >= 4) vst1_u16(output.data() + offset, words);
        else
        {
            if (remaining > 0) output[offset] = vget_lane_u16(words, 0);
            if (remaining > 1) output[offset + 1] = vget_lane_u16(words, 1);
            if (remaining > 2) output[offset + 2] = vget_lane_u16(words, 2);
        }
    }
}
} // namespace detail

/** @brief Normalize one shared-traversal original floating row tile.
 * @tparam Encoding Exact FP32/FP16/BF16 storage program.
 * @tparam Count Number of simultaneously live physical rows1..4.
 * @param input Checked original rows whose external owner retains the lease.
 * @param output Original writable rows, supporting exact same-row aliases only.
 * @param gamma Original immutable FP32 gamma; an empty span explicitly means identity.
 * @param epsilon Finite nonnegative normalization epsilon.
 * @param accumulation Explicit native FP64 or native T5 FP32 reduction program.
 * @param inverse_program Explicit staged FP64-mean or per-row FP32-mean inverse.
 * @throws std::invalid_argument For malformed output/gamma, forbidden aliases or policy.
 * @throws std::domain_error For nonfinite observed summaries or an undefined inverse.
 * @details Staged FP32 retains the guarded FP64-mean inverse; per-row FP32
 * and FP16/BF16 retain the distinct FP32 mean/sqrt/divide program. Undefined FP16/BF16 zero-denominator requests reject
 * before publication instead of hiding the error in a zero-output substitute.
 */
template<Storage Encoding, std::size_t Count>
inline void normalize(const OriginalRows<Encoding, Count> &input,
    std::array<std::span<Unit<Encoding>>, Count> output, std::span<const float> gamma,
    float epsilon, Accumulation accumulation = Accumulation::FP64,
    Inverse inverse_program = Encoding == Storage::FP32 ? Inverse::GuardedFP64Mean : Inverse::FP32Mean)
{
    static_assert(Encoding == Storage::FP32 || Encoding == Storage::FP16 || Encoding == Storage::BF16);
    if (!std::isfinite(epsilon) || epsilon < 0.0f ||
        (accumulation != Accumulation::FP64 && accumulation != Accumulation::T5FP32) ||
        (Encoding != Storage::FP32 && accumulation != Accumulation::FP64) ||
        (inverse_program != Inverse::GuardedFP64Mean && inverse_program != Inverse::FP32Mean) ||
        (Encoding != Storage::FP32 && inverse_program != Inverse::FP32Mean))
        throw std::invalid_argument("Native floating RMS requires a valid epsilon and storage-specific arithmetic policy");
    const auto columns = input.columns();
    if (!gamma.empty() && gamma.size() < columns)
        throw std::invalid_argument("Native RMS immutable gamma is shorter than the original row");
    // Validate the complete tile before reduction or any output mutation. Binding
    // spans never invents capacity; caller-issued spans remain authoritative.
    detail::visit<Count>([&]<std::size_t Row> {
        if (output[Row].size() < columns || (columns && !output[Row].data()))
            throw std::invalid_argument("Native RMS output row is missing or shorter than its original geometry");
        const auto written = columns * sizeof(Unit<Encoding>);
        if (detail::overlaps(output[Row].data(), written, gamma.data(), gamma.size_bytes()))
            throw std::invalid_argument("Native RMS publication aliases immutable gamma");
        detail::visit<Count>([&]<std::size_t Other> {
            const auto original = input.template row<Other>();
            if (detail::overlaps(output[Row].data(), written, original.data(), original.size_bytes()) &&
                !(Row == Other && output[Row].data() == original.data()))
                throw std::invalid_argument("Native RMS publication aliases another original row or a partial source");
            if (Row != Other && detail::overlaps(output[Row].data(), written,
                                                output[Other].data(), written))
                throw std::invalid_argument("Native RMS writable rows overlap");
        });
    });
    if (!columns) return;
    std::array<double, Count> sums;
    if constexpr (Encoding == Storage::FP32)
        sums = accumulation == Accumulation::T5FP32 ? t5SquaredSums(input) : squaredSums(input);
    else sums = squaredSums(input);
    std::array<float, Count> inverse;
    detail::visit<Count>([&]<std::size_t Row> {
        if (!std::isfinite(sums[Row]))
            throw std::domain_error("Native RMS observed a nonfinite original-row squared summary");
        if (inverse_program == Inverse::GuardedFP64Mean)
        {
            const double denominator = sums[Row] / double(columns) + double(epsilon);
            inverse[Row] = denominator > 0.0 ? float(1.0 / std::sqrt(denominator)) : 0.0f;
        }
        else
        {
            const float mean = float(sums[Row] / double(columns));
            inverse[Row] = 1.0f / std::sqrt(mean + epsilon);
        }
        if (!std::isfinite(inverse[Row]))
            throw std::domain_error("Native RMS original-row inverse is not representable");
    });
    for (std::size_t column = 0; column < columns; column += std::min<std::size_t>(4, columns - column))
    {
        const auto weight = gamma.empty() ? vdupq_n_f32(1.0f) : detail::load<Storage::FP32>(gamma, column, columns);
        detail::visit<Count>([&]<std::size_t Row> {
            auto value = detail::load<Encoding>(input.template row<Row>(), column, columns);
            value = vmulq_f32(vmulq_f32(value, vdupq_n_f32(inverse[Row])), weight);
            if constexpr (Encoding == Storage::FP32)
                if (inverse_program == Inverse::GuardedFP64Mean && inverse[Row] == 0.0f)
                    value = vdupq_n_f32(0.0f);
            detail::store<Encoding>(output[Row], column, columns, value);
        });
    }
}
} // namespace llaminar2::primitives::neon_rmsnorm
