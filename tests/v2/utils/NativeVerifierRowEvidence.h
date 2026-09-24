/**
 * @file NativeVerifierRowEvidence.h
 * @brief Device-free finite-byte proof for native grouped/serial verifier rows.
 *
 * HF distribution tolerances do not certify batch invariance. This comparison
 * consumes already-published FP32 diagnostic rows, regardless of the expert
 * weight codebook, and rejects missing geometry, nonfinite values and even a
 * signed-zero bit change. It never performs inference or reads a device.
 */
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace llaminar2::test
{
    /** Mutually exclusive dispositions; only Exact can certify a native row. */
    enum class NativeVerifierRowStatus
    {
        Exact,
        Empty,
        GeometryMismatch,
        Nonfinite,
        ByteMismatch,
    };

    /** Compact evidence retained alongside the independent HF numerical CSVs. */
    struct NativeVerifierRowEvidence
    {
        NativeVerifierRowStatus status = NativeVerifierRowStatus::Empty;
        std::size_t first_mismatch = 0;
        std::uint32_t grouped_bits = 0;
        std::uint32_t serial_bits = 0;
        double max_abs_diff = 0.0;

        /** @return True only for nonempty, equal-shaped, finite, identical rows. */
        [[nodiscard]] bool passed() const noexcept
        {
            return status == NativeVerifierRowStatus::Exact;
        }

        /** @return Stable artifact name, not a numerical threshold decision. */
        [[nodiscard]] std::string_view name() const noexcept
        {
            switch (status)
            {
            case NativeVerifierRowStatus::Exact: return "exact";
            case NativeVerifierRowStatus::Empty: return "empty";
            case NativeVerifierRowStatus::GeometryMismatch: return "geometry_mismatch";
            case NativeVerifierRowStatus::Nonfinite: return "nonfinite";
            case NativeVerifierRowStatus::ByteMismatch: return "byte_mismatch";
            }
            return "invalid";
        }
    };

    /**
     * @brief Compare one native FP32 row without numerical tolerance.
     * @param grouped Published grouped row, not retokenized or recomputed data.
     * @param serial Same-input ordinary M=1 production row.
     * @return First offending element/bits and maximum finite absolute error.
     */
    [[nodiscard]] inline NativeVerifierRowEvidence compareNativeVerifierRow(
        std::span<const float> grouped, std::span<const float> serial) noexcept
    {
        if (grouped.empty() || serial.empty())
            return {};
        if (grouped.size() != serial.size())
            return {.status = NativeVerifierRowStatus::GeometryMismatch};

        NativeVerifierRowEvidence evidence{.status = NativeVerifierRowStatus::Exact};
        for (std::size_t i = 0; i < serial.size(); ++i)
        {
            const auto actual_bits = std::bit_cast<std::uint32_t>(grouped[i]);
            const auto expected_bits = std::bit_cast<std::uint32_t>(serial[i]);
            const bool finite = std::isfinite(grouped[i]) && std::isfinite(serial[i]);
            if (finite)
                evidence.max_abs_diff = std::max(evidence.max_abs_diff,
                    std::abs(static_cast<double>(grouped[i]) - serial[i]));

            // Identical NaN/Inf payloads must fail as well. Preserve the first
            // offending index so an earlier ULP change cannot be hidden later.
            if (evidence.passed() && (!finite || actual_bits != expected_bits))
            {
                evidence.status = finite ? NativeVerifierRowStatus::ByteMismatch
                                         : NativeVerifierRowStatus::Nonfinite;
                evidence.first_mismatch = i;
                evidence.grouped_bits = actual_bits;
                evidence.serial_bits = expected_bits;
            }
        }
        return evidence;
    }

    /**
     * @brief Validate row-major matrix geometry before selecting a verifier row.
     * @param grouped Complete published matrix, including physical padding.
     * @param rows Authenticated physical row count, never inferred per stage.
     * @param row Logical row selected by the same-input transaction boundary.
     * @param serial One matching serial row.
     * @return Exact row evidence, or a structural failure without reading OOB.
     */
    [[nodiscard]] inline NativeVerifierRowEvidence compareNativeVerifierMatrixRow(
        std::span<const float> grouped, std::size_t rows, std::size_t row,
        std::span<const float> serial) noexcept
    {
        if (grouped.empty() || serial.empty())
            return {};
        // Division avoids overflowing a rows*width check on malformed metadata.
        if (rows == 0 || row >= rows || grouped.size() % rows != 0 ||
            grouped.size() / rows != serial.size())
            return {.status = NativeVerifierRowStatus::GeometryMismatch};
        return compareNativeVerifierRow(grouped.subspan(row * serial.size(), serial.size()), serial);
    }

    /**
     * @brief Select ordinary main-model row publications, excluding MTP banks.
     * @param key Semantic snapshot identity from the production graph.
     * @return Whether this row-major checkpoint belongs to the serial proof.
     */
    [[nodiscard]] inline bool isNativeVerifierRowCheckpoint(std::string_view key) noexcept
    {
        if (key == "EMBEDDING" || key == "FINAL_NORM" || key == "LM_HEAD")
            return true;
        if (!key.starts_with("layer"))
            return false;
        const auto delimiter = key.find('_');
        if (delimiter == std::string_view::npos || delimiter == 5)
            return false;
        for (std::size_t i = 5; i < delimiter; ++i)
            if (key[i] < '0' || key[i] > '9')
                return false;
        constexpr std::array<std::string_view, 21> stages{
            "ATTENTION_NORM", "QKV_PROJECTION", "Q_PROJECTION", "GDN_Z_PROJECTION",
            "ATTENTION_CONTEXT", "ATTENTION_OUTPUT", "ATTENTION_OUTPUT_ALLREDUCED",
            "FFN_NORM_RESIDUAL_OUT", "FFN_NORM", "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES", "MOE_ROUTING_WEIGHTS", "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT", "MOE_SHARED_GATE_OUTPUT", "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL", "GDN_CONV1D_OUTPUT", "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT", "GDN_OUTPUT"};
        return std::find(stages.begin(), stages.end(), key.substr(delimiter + 1)) != stages.end();
    }
}
