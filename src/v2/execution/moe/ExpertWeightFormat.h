/**
 * @file ExpertWeightFormat.h
 * @brief Exhaustive mathematical source-format identity for routed experts.
 *
 * ExpertOverlay must preserve the checkpoint's arithmetic contract while an
 * expert moves between execution tiers. Quantized tensors retain their exact
 * NativeVNNI source identity, while floating tensors retain their IEEE storage
 * precision. Keeping the distinction in one tagged value prevents a floating
 * projection from being mistaken for an absent quantized codebook.
 */

#pragma once

#include "tensors/NativeVnniFormatInfo.h"
#include "tensors/TensorType.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace llaminar2
{
    /** @brief Representation families accepted by the ExpertOverlay fabric. */
    enum class ExpertWeightFormatKind : std::uint8_t
    {
        Invalid = 0,
        NativeVnni = 1,
        FP16 = 2,
        BF16 = 3,
        FP32 = 4,
    };

    /**
     * @brief Exact checkpoint-level format identity for one expert projection.
     *
     * `native_vnni` is populated only for @ref ExpertWeightFormatKind::NativeVnni.
     * Floating formats carry no codebook because their raw row-major bytes are
     * already the common CPU/CUDA/ROCm transfer representation.
     */
    struct ExpertWeightFormat
    {
        ExpertWeightFormatKind kind = ExpertWeightFormatKind::Invalid;
        NativeVnniSourceIdentity native_vnni;

        /** @brief Construct a validated quantized source-format identity. */
        [[nodiscard]] static ExpertWeightFormat nativeVnni(
            NativeVnniSourceIdentity identity) noexcept
        {
            return {
                .kind = ExpertWeightFormatKind::NativeVnni,
                .native_vnni = identity,
            };
        }

        /**
         * @brief Construct one supported floating source-format identity.
         * @param type FP16, BF16, or FP32 tensor storage type.
         * @return A valid floating identity, or an invalid identity otherwise.
         */
        [[nodiscard]] static ExpertWeightFormat floating(
            TensorType type) noexcept
        {
            switch (type)
            {
            case TensorType::FP16:
                return {.kind = ExpertWeightFormatKind::FP16};
            case TensorType::BF16:
                return {.kind = ExpertWeightFormatKind::BF16};
            case TensorType::FP32:
                return {.kind = ExpertWeightFormatKind::FP32};
            default:
                return {};
            }
        }

        /** @return Whether this identity names one supported source format. */
        [[nodiscard]] bool valid() const noexcept
        {
            if (kind == ExpertWeightFormatKind::NativeVnni)
            {
                return native_vnni.present &&
                       native_vnni_formats::forSourceIdentity(
                           native_vnni.codebook_id,
                           native_vnni.is_superblock) != nullptr;
            }
            return isFloating() && !native_vnni.present;
        }

        /** @return Whether raw floating bytes are the execution representation. */
        [[nodiscard]] bool isFloating() const noexcept
        {
            return kind == ExpertWeightFormatKind::FP16 ||
                   kind == ExpertWeightFormatKind::BF16 ||
                   kind == ExpertWeightFormatKind::FP32;
        }

        /** @return Whether the format uses prepared NativeVNNI representations. */
        [[nodiscard]] bool isNativeVnni() const noexcept
        {
            return kind == ExpertWeightFormatKind::NativeVnni;
        }

        /** @return Floating tensor type, or no value for quantized/invalid data. */
        [[nodiscard]] std::optional<TensorType> floatingTensorType()
            const noexcept
        {
            switch (kind)
            {
            case ExpertWeightFormatKind::FP16:
                return TensorType::FP16;
            case ExpertWeightFormatKind::BF16:
                return TensorType::BF16;
            case ExpertWeightFormatKind::FP32:
                return TensorType::FP32;
            default:
                return std::nullopt;
            }
        }

        /** @return Bytes per scalar for a floating format, or zero otherwise. */
        [[nodiscard]] std::size_t floatingElementBytes() const noexcept
        {
            switch (kind)
            {
            case ExpertWeightFormatKind::FP16:
            case ExpertWeightFormatKind::BF16:
                return sizeof(std::uint16_t);
            case ExpertWeightFormatKind::FP32:
                return sizeof(float);
            default:
                return 0;
            }
        }

        /** @brief Compare the complete tagged source-format identity. */
        bool operator==(const ExpertWeightFormat &) const = default;
    };
} // namespace llaminar2
