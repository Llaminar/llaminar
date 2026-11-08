/**
 * @file W4A4MatrixContract.cpp
 * @brief Fail-closed admission of complete ruquant native matrices.
 *
 * Only the original integer codes and FP32 metadata define the operation.
 * Admission never repairs a header, invents a scale, dequantizes weights or
 * interprets a local K slice as the whole activation row. The validated value
 * has no allocation ledger and no borrowed data pointer; storage ownership
 * remains explicit in the tensor/prepared-weight lifetime.
 */
#include "W4A4MatrixContract.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace llaminar2::w4a4
{
    static_assert(std::endian::native == std::endian::little,
                  "Native W4A4 source blocks require little-endian admission");

    const char *AdmissionError::message() const noexcept
    {
        switch (failure)
        {
        case AdmissionFailure::UnsupportedTransform: return "W4A4 source transform is not supported";
        case AdmissionFailure::InvalidGeometry: return "W4A4 requires positive N and complete K divisible by 256";
        case AdmissionFailure::IntegerRange: return "W4A4 K exceeds the proven int32 accumulation range";
        case AdmissionFailure::SizeOverflow: return "W4A4 native source extent is not representable";
        case AdmissionFailure::SourceExtent: return "W4A4 source must contain exactly the declared complete matrix";
        case AdmissionFailure::InvalidUnit: return "W4A4 row unit must be finite and nonnegative";
        case AdmissionFailure::InconsistentRowUnit: return "W4A4 row unit must repeat bit-identically across its blocks";
        case AdmissionFailure::InvalidRotation: return "W4A4 stored row/rotation geometry is not admitted";
        case AdmissionFailure::InconsistentRotation: return "W4A4 row/rotation length must be uniform across the matrix";
        case AdmissionFailure::InvalidClip: return "W4A4 clip must be finite and nonnegative; zero is the explicit unset value";
        case AdmissionFailure::InconsistentClip: return "W4A4 clip bits must be uniform across the matrix";
        case AdmissionFailure::ZeroMultiplier: return "W4A4 encoded weight multipliers must be in 1..15";
        }
        return "Invalid W4A4 admission error code";
    }

    std::optional<HadamardGeometry> factorHadamard(int length) noexcept
    {
        if (length <= 0) return std::nullopt;
        for (int paley = 1; paley <= 128 && paley <= length; ++paley)
        {
            if (paley > 1)
            {
                if (paley % 4 != 0) continue;
                const int prime = paley - 1;
                bool composite = false;
                for (int divisor = 2; divisor * divisor <= prime; ++divisor)
                    composite |= prime % divisor == 0;
                if (composite) continue;
            }
            if (length % paley != 0) continue;
            const int sylvester = length / paley;
            if ((sylvester & (sylvester - 1)) == 0)
                return HadamardGeometry{length, paley, sylvester};
        }
        return std::nullopt;
    }

    std::expected<std::size_t, AdmissionError> MatrixContract::sourceExtent(
        std::size_t rows, std::size_t columns) noexcept
    {
        if (rows == 0 || columns == 0 || columns % kBlockElements != 0
            || rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return std::unexpected(AdmissionError{AdmissionFailure::InvalidGeometry, std::nullopt});
        if (columns / kGroupElements > static_cast<std::size_t>(kMaximumGroups))
            return std::unexpected(AdmissionError{AdmissionFailure::IntegerRange, std::nullopt});
        const std::size_t bytes_per_row = (columns / kBlockElements) * kSourceBlockBytes;
        if (rows > std::numeric_limits<std::size_t>::max() / bytes_per_row)
            return std::unexpected(AdmissionError{AdmissionFailure::SizeOverflow, std::nullopt});
        return rows * bytes_per_row;
    }

    std::expected<MatrixContract, AdmissionError> MatrixContract::admit(
        Transform transform, std::size_t rows, std::size_t columns,
        std::span<const SourceBlock> source) noexcept
    {
        auto admission = MatrixAdmission::begin(transform, rows, columns);
        if (!admission) return std::unexpected(admission.error());
        const auto extent = sourceExtent(rows, columns);
        if (source.size() != *extent / kSourceBlockBytes || source.data() == nullptr)
            return std::unexpected(AdmissionError{AdmissionFailure::SourceExtent, std::nullopt});
        const auto appended = admission->append(std::as_bytes(source));
        if (!appended) return std::unexpected(appended.error());
        return std::move(*admission).finish();
    }

    std::expected<MatrixAdmission, AdmissionError> MatrixAdmission::begin(
        Transform transform, std::size_t rows, std::size_t columns) noexcept
    {
        if (transform != Transform::Identity && transform != Transform::PaleySylvester)
            return std::unexpected(AdmissionError{AdmissionFailure::UnsupportedTransform, std::nullopt});
        const auto extent = MatrixContract::sourceExtent(rows, columns);
        if (!extent) return std::unexpected(extent.error());
        return MatrixAdmission(State{.rows = static_cast<int>(rows), .columns = static_cast<int>(columns),
            .source_bytes = *extent, .transform = transform});
    }

    std::expected<void, AdmissionError> MatrixAdmission::append(std::span<const std::byte> source) noexcept
    {
        if (!state_) return std::unexpected(AdmissionError{AdmissionFailure::SourceExtent, std::nullopt});
        auto &state = *state_;
        if (state.failure) return std::unexpected(*state.failure);
        const auto fail = [&](AdmissionFailure failure, std::optional<SourceLocation> location)
            -> std::expected<void, AdmissionError> {
            state.failure = AdmissionError{failure, location};
            return std::unexpected(*state.failure);
        };
        if ((!source.empty() && !source.data()) || source.size() % kSourceBlockBytes ||
            source.size() / kSourceBlockBytes > state.source_bytes / kSourceBlockBytes - state.blocks_seen)
            return fail(AdmissionFailure::SourceExtent, std::nullopt);
        const std::size_t blocks_per_row = state.columns / kBlockElements;
        for (std::size_t offset = 0; offset < source.size(); offset += kSourceBlockBytes)
        {
            // Decode only fixed header metadata. This also accepts unaligned
            // transport addresses; neither source codes nor tensor scratch move.
            SourceBlock value{};
            std::memcpy(&value, source.data() + offset, offsetof(SourceBlock, codes));
            const SourceLocation location{state.blocks_seen / blocks_per_row, state.blocks_seen % blocks_per_row};
            if (state.blocks_seen == 0)
            {
                state.rotation = value.rotation_length;
                if (state.transform == Transform::PaleySylvester)
                {
                    state.hadamard = factorHadamard(state.rotation);
                    if (!state.hadamard || state.columns % state.rotation != 0)
                        return fail(AdmissionFailure::InvalidRotation, location);
                }
                else if (state.rotation != state.columns)
                    return fail(AdmissionFailure::InvalidRotation, location);
                state.clip_bits = value.clip_bits;
                const float encoded_clip = std::bit_cast<float>(state.clip_bits);
                if (!std::isfinite(encoded_clip) || encoded_clip < 0.0f)
                    return fail(AdmissionFailure::InvalidClip, location);
                state.clip = encoded_clip > 0.0f ? encoded_clip : 1.0f;
            }
            if (location.block == 0) state.row_unit = std::bit_cast<std::uint32_t>(value.unit);
            if (!std::isfinite(value.unit) || value.unit < 0.0f)
                return fail(AdmissionFailure::InvalidUnit, location);
            if (std::bit_cast<std::uint32_t>(value.unit) != state.row_unit)
                return fail(AdmissionFailure::InconsistentRowUnit, location);
            if (value.rotation_length != state.rotation)
                return fail(AdmissionFailure::InconsistentRotation, location);
            if (value.clip_bits != state.clip_bits)
                return fail(AdmissionFailure::InconsistentClip, location);
            for (unsigned group = 0; group < kGroupsPerBlock; ++group)
                if (weightMultiplier(value.descriptor, group) == 0)
                    return fail(AdmissionFailure::ZeroMultiplier, location);
            ++state.blocks_seen;
        }
        return {};
    }

    std::expected<MatrixContract, AdmissionError> MatrixAdmission::finish() && noexcept
    {
        const auto state = std::exchange(state_, std::nullopt);
        if (!state) return std::unexpected(AdmissionError{AdmissionFailure::SourceExtent, std::nullopt});
        if (state->failure) return std::unexpected(*state->failure);
        if (state->blocks_seen != state->source_bytes / kSourceBlockBytes)
            return std::unexpected(AdmissionError{AdmissionFailure::SourceExtent, std::nullopt});
        return MatrixContract(state->rows, state->columns, state->source_bytes, state->transform,
            state->rotation, state->hadamard, state->clip_bits, state->clip);
    }
} // namespace llaminar2::w4a4
